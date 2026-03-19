from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def resolve_hf_token(explicit_token: str | None) -> str:
    token = (
        explicit_token
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HF_TOKEN")
    )
    if token:
        return token
    raise SystemExit(
        "Hugging Face token is required. Set --hf-token or HUGGINGFACE_HUB_TOKEN. "
        "You also need to accept the model conditions on Hugging Face first."
    )


def choose_torch_device(device: str | None) -> str:
    if device:
        return device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def annotation_to_turns(annotation: Any) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        turns.append(
            {
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "speaker": str(label),
            }
        )
    return turns


def timeline_to_segments(annotation: Any) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for segment in annotation.get_timeline().support():
        segments.append(
            {
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "duration": round(float(segment.end - segment.start), 3),
            }
        )
    return segments


def merge_speech_regions(
    speech_regions: list[dict[str, Any]],
    *,
    target_duration: float,
    max_duration: float,
) -> list[dict[str, Any]]:
    if not speech_regions:
        return []

    chunks: list[dict[str, Any]] = []
    current_start = speech_regions[0]["start"]
    current_end = speech_regions[0]["end"]
    members = [speech_regions[0]]

    for region in speech_regions[1:]:
        proposed_end = region["end"]
        proposed_duration = proposed_end - current_start
        current_duration = current_end - current_start

        if proposed_duration <= max_duration and current_duration < target_duration:
            current_end = proposed_end
            members.append(region)
            continue

        chunks.append(
            {
                "start": round(current_start, 3),
                "end": round(current_end, 3),
                "duration": round(current_end - current_start, 3),
                "speech_region_count": len(members),
                "speech_regions": members,
            }
        )
        current_start = region["start"]
        current_end = region["end"]
        members = [region]

    chunks.append(
        {
            "start": round(current_start, 3),
            "end": round(current_end, 3),
            "duration": round(current_end - current_start, 3),
            "speech_region_count": len(members),
            "speech_regions": members,
        }
    )
    return chunks


def merge_speech_regions_sliding(
    speech_regions: list[dict[str, Any]],
    *,
    target_duration: float,
    max_duration: float,
    overlap_duration: float,
) -> list[dict[str, Any]]:
    if not speech_regions:
        return []

    chunks: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    timeline_start = speech_regions[0]["start"]
    timeline_end = speech_regions[-1]["end"]
    step = max(1.0, target_duration - overlap_duration)
    cursor = timeline_start

    while cursor < timeline_end:
        window_end = min(cursor + target_duration, timeline_end)
        overlapping = [
            region
            for region in speech_regions
            if not (region["end"] <= cursor or region["start"] >= window_end)
        ]
        if not overlapping:
            cursor += step
            continue

        chunk_start = overlapping[0]["start"]
        chunk_end = min(max(region["end"] for region in overlapping), chunk_start + max_duration)
        key = (round(chunk_start, 3), round(chunk_end, 3))
        if key not in seen and chunk_end > chunk_start:
            seen.add(key)
            chunks.append(
                {
                    "start": round(chunk_start, 3),
                    "end": round(chunk_end, 3),
                    "duration": round(chunk_end - chunk_start, 3),
                    "speech_region_count": len(overlapping),
                    "speech_regions": overlapping,
                }
            )

        if window_end >= timeline_end:
            break
        cursor += step

    return chunks


def run_pyannote_vad(
    *,
    audio_path: str,
    hf_token: str,
    device_name: str,
    segmentation_model: str,
    target_duration: float,
    max_duration: float,
    strategy: str = "greedy",
    overlap_duration: float = 5.0,
) -> dict[str, Any]:
    import torch
    from pyannote.audio.pipelines import VoiceActivityDetection

    pipeline = VoiceActivityDetection(
        segmentation=segmentation_model,
        use_auth_token=hf_token,
    )
    pipeline.to(torch.device(device_name))

    speech = pipeline(audio_path)
    speech_regions = timeline_to_segments(speech)
    if strategy == "greedy":
        chunks = merge_speech_regions(
            speech_regions,
            target_duration=target_duration,
            max_duration=max_duration,
        )
    elif strategy == "sliding":
        chunks = merge_speech_regions_sliding(
            speech_regions,
            target_duration=target_duration,
            max_duration=max_duration,
            overlap_duration=overlap_duration,
        )
    else:
        raise ValueError(f"Unsupported VAD strategy: {strategy}")

    return {
        "device": device_name,
        "segmentation_model": segmentation_model,
        "target_duration": target_duration,
        "max_duration": max_duration,
        "strategy": strategy,
        "overlap_duration": overlap_duration,
        "speech_regions": speech_regions,
        "chunks": chunks,
    }


def run_pyannote_diarization(
    *,
    audio_path: str,
    hf_token: str,
    device_name: str,
    pipeline_name: str,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[dict[str, Any]]:
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(pipeline_name, use_auth_token=hf_token)
    if pipeline is None:
        raise RuntimeError("Failed to load pyannote diarization pipeline.")
    pipeline.to(torch.device(device_name))
    diarization = pipeline(
        audio_path,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    return annotation_to_turns(diarization)


def segment_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def attach_speakers_to_whisper_segments(
    whisper_payload: dict[str, Any],
    speaker_turns: list[dict[str, Any]],
) -> dict[str, Any]:
    segments = whisper_payload.get("segments", [])

    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        overlaps: dict[str, float] = {}

        for turn in speaker_turns:
            overlap = segment_overlap(start, end, turn["start"], turn["end"])
            if overlap <= 0:
                continue
            overlaps[turn["speaker"]] = overlaps.get(turn["speaker"], 0.0) + overlap

        if overlaps:
            best_speaker = max(overlaps, key=overlaps.get)
            segment["speaker"] = best_speaker
            segment["speaker_overlap_seconds"] = round(overlaps[best_speaker], 3)
            segment["speaker_candidates"] = [
                {"speaker": speaker, "overlap_seconds": round(duration, 3)}
                for speaker, duration in sorted(
                    overlaps.items(), key=lambda item: item[1], reverse=True
                )
            ]
        else:
            segment["speaker"] = None
            segment["speaker_overlap_seconds"] = 0.0
            segment["speaker_candidates"] = []

    whisper_payload["speaker_turns"] = speaker_turns
    return whisper_payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
