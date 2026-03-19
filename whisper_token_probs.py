from __future__ import annotations

from typing import Any

import torch

from whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
from whisper.decoding import DecodingOptions, DecodingTask


def _token_to_text(tokenizer: Any, token_id: int) -> str:
    if token_id >= tokenizer.timestamp_begin:
        return tokenizer.decode_with_timestamps([token_id])

    text = tokenizer.decode([token_id])
    if text:
        return text

    for special_token, special_id in tokenizer.special_tokens.items():
        if special_id == token_id:
            return special_token

    return f"<|{token_id}|>"


def _compute_token_predictions(
    task: DecodingTask,
    mel_segment: torch.Tensor,
    sampled_tokens: list[int],
    topk: int,
) -> list[dict[str, Any]]:
    audio_features = task._get_audio_features(mel_segment.unsqueeze(0))
    tokens = torch.tensor([task.initial_tokens], device=audio_features.device)
    inference = task.inference
    tokenizer = task.tokenizer
    topk = max(1, topk)

    token_predictions: list[dict[str, Any]] = []

    try:
        for token_id in sampled_tokens:
            logits = inference.logits(tokens, audio_features)[:, -1]
            for logit_filter in task.logit_filters:
                logit_filter.apply(logits, tokens)

            probabilities = torch.softmax(logits.float(), dim=-1)[0]
            chosen_probability = probabilities[token_id].item()
            top_probabilities, top_token_ids = probabilities.topk(topk)

            token_predictions.append(
                {
                    "token_id": token_id,
                    "token": _token_to_text(tokenizer, token_id),
                    "probability": chosen_probability,
                    "log_probability": float(torch.log(probabilities[token_id]).item()),
                    "top_candidates": [
                        {
                            "token_id": candidate_id.item(),
                            "token": _token_to_text(tokenizer, candidate_id.item()),
                            "probability": candidate_probability.item(),
                        }
                        for candidate_probability, candidate_id in zip(
                            top_probabilities, top_token_ids
                        )
                    ],
                }
            )

            next_token = torch.tensor([[token_id]], device=tokens.device)
            tokens = torch.cat([tokens, next_token], dim=-1)
    finally:
        inference.cleanup_caching()

    return token_predictions


def attach_token_probabilities(
    *,
    model: Any,
    audio_path: str,
    segments: list[dict[str, Any]],
    language: str,
    task_name: str,
    initial_prompt: str | None,
    carry_initial_prompt: bool,
    condition_on_previous_text: bool,
    topk: int,
    fp16: bool,
    reset_key: str | None = None,
) -> list[dict[str, Any]]:
    if not segments:
        return segments

    full_mel = log_mel_spectrogram(audio_path, model.dims.n_mels)
    tokenizer = DecodingTask(
        model,
        DecodingOptions(language=language, task=task_name, fp16=fp16),
    ).tokenizer
    remaining_prompt_length = model.dims.n_text_ctx // 2 - 1

    if initial_prompt:
        initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
    else:
        initial_prompt_tokens = []

    all_tokens = initial_prompt_tokens.copy()
    prompt_reset_since = 0

    grouped_segments: list[list[dict[str, Any]]] = []
    current_group: list[dict[str, Any]] = []
    current_seek: int | None = None

    for segment in segments:
        seek = int(segment["seek"])
        if current_group and current_seek != seek:
            grouped_segments.append(current_group)
            current_group = []
        current_group.append(segment)
        current_seek = seek

    if current_group:
        grouped_segments.append(current_group)

    current_scope = None

    for group in grouped_segments:
        if reset_key is not None:
            group_scope = group[0].get(reset_key)
            if current_scope is None or group_scope != current_scope:
                all_tokens = initial_prompt_tokens.copy()
                prompt_reset_since = 0
                current_scope = group_scope

        seek = int(group[0]["seek"])
        sampled_tokens = [token for segment in group for token in segment.get("tokens", [])]
        if not sampled_tokens:
            continue

        if carry_initial_prompt:
            nignored = max(len(initial_prompt_tokens), prompt_reset_since)
            remaining_prompt = all_tokens[nignored:][-remaining_prompt_length:]
            prompt_tokens = initial_prompt_tokens + remaining_prompt
        else:
            prompt_tokens = all_tokens[prompt_reset_since:]

        group_temperature = float(group[0].get("temperature", 0.0))
        decode_options = DecodingOptions(
            language=language,
            task=task_name,
            prompt=prompt_tokens,
            temperature=group_temperature,
            fp16=fp16,
        )
        decode_task = DecodingTask(model, decode_options)

        mel_segment = pad_or_trim(full_mel[:, seek : seek + N_FRAMES], N_FRAMES).to(
            model.device
        )
        token_predictions = _compute_token_predictions(
            decode_task,
            mel_segment,
            sampled_tokens,
            topk=topk,
        )

        offset = 0
        for segment in group:
            segment_tokens = segment.get("tokens", [])
            length = len(segment_tokens)
            segment["token_probs"] = token_predictions[offset : offset + length]
            offset += length

        all_tokens.extend(sampled_tokens)
        if not condition_on_previous_text or group_temperature > 0.5:
            prompt_reset_since = len(all_tokens)

    return segments
