#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from asr_whisper_large_v3 import (
    choose_device,
    ensure_audio_path,
    run_vad_chunk_transcribe,
)
from prompt_terms import build_initial_prompt, extract_prompt_terms
from pyannote_helpers import (
    attach_speakers_to_whisper_segments,
    choose_torch_device,
    resolve_hf_token,
    run_pyannote_diarization,
)
from transcript_alerts import attach_token_time_ranges, collect_low_confidence_alerts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a two-pass Whisper pipeline with summary-driven initial_prompt and VAD chunking."
    )
    parser.add_argument("audio_path", type=Path, help="Path to an audio or video file.")
    parser.add_argument("--text", default=None, help="Optional context text used to build initial_prompt.")
    parser.add_argument("--text-file", type=Path, default=None, help="Path to a UTF-8 text file used to build initial_prompt.")
    parser.add_argument("--summary-text", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--summary-file", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--language", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--token-topk", type=int, default=5)
    parser.add_argument("--alert-threshold", type=float, default=0.90)
    parser.add_argument("--vad-target-duration", type=float, default=30.0)
    parser.add_argument("--vad-max-duration", type=float, default=35.0)
    parser.add_argument("--vad-overlap-duration", type=float, default=5.0)
    parser.add_argument("--diarization-pipeline", default="pyannote/speaker-diarization-3.1")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "develop" / "output")
    parser.add_argument("--output-stem", default=None)
    return parser.parse_args()


def load_summary_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return args.text_file.expanduser().resolve().read_text(encoding="utf-8")
    if args.summary_file:
        return args.summary_file.expanduser().resolve().read_text(encoding="utf-8")
    return args.text or args.summary_text or ""


def build_transcribe_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "task": "transcribe",
        "language": args.language,
        "temperature": 0.0,
        "initial_prompt": None,
        "carry_initial_prompt": False,
        "word_timestamps": True,
        "condition_on_previous_text": True,
        "verbose": False,
        "beam_size": 5,
        "best_of": 5,
    }


def attach_alerts_and_probs(
    *,
    model: Any,
    audio_path: Path,
    result: dict[str, Any],
    initial_prompt: str | None,
    token_topk: int,
    alert_threshold: float,
    fp16: bool,
) -> dict[str, Any]:
    from whisper_token_probs import attach_token_probabilities

    result["segments"] = attach_token_probabilities(
        model=model,
        audio_path=str(audio_path),
        segments=result.get("segments", []),
        language=result.get("language") or "ja",
        task_name="transcribe",
        initial_prompt=initial_prompt,
        carry_initial_prompt=False,
        condition_on_previous_text=True,
        topk=token_topk,
        fp16=fp16,
        reset_key="transcribe_call_index",
    )
    result["segments"] = attach_token_time_ranges(result["segments"])
    result["alerts"] = collect_low_confidence_alerts(
        result["segments"],
        threshold=alert_threshold,
    )
    result["alert_count"] = len(result["alerts"])
    return result


def build_diff_candidates(
    left_result: dict[str, Any],
    right_result: dict[str, Any],
) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    right_segments = right_result.get("segments", [])

    for left_segment in left_result.get("segments", []):
        left_start = float(left_segment["start"])
        left_end = float(left_segment["end"])
        best_match = None
        best_overlap = 0.0
        for right_segment in right_segments:
            overlap = max(
                0.0,
                min(left_end, float(right_segment["end"]))
                - max(left_start, float(right_segment["start"])),
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = right_segment

        if not best_match or best_overlap == 0.0:
            continue

        left_text = (left_segment.get("text") or "").strip()
        right_text = (best_match.get("text") or "").strip()
        if left_text == right_text:
            continue

        diffs.append(
            {
                "start": min(left_start, float(best_match["start"])),
                "end": max(left_end, float(best_match["end"])),
                "left_text": left_text,
                "right_text": right_text,
                "left_has_alert": bool(left_segment.get("has_alert")),
                "right_has_alert": bool(best_match.get("has_alert")),
            }
        )

    return diffs


def build_alerts(
    greedy_result: dict[str, Any],
    sliding_result: dict[str, Any],
    diff_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []

    for pass_name, result in (("greedy", greedy_result), ("sliding", sliding_result)):
        for alert in result.get("alerts", []):
            alerts.append(
                {
                    "type": "low_confidence",
                    "pass": pass_name,
                    **alert,
                }
            )

    for diff in diff_candidates:
        alerts.append(
            {
                "type": "pass_mismatch",
                **diff,
            }
        )

    return alerts


def main() -> None:
    args = parse_args()
    audio_path = ensure_audio_path(args.audio_path)
    summary_text = load_summary_text(args)
    terms = extract_prompt_terms(summary_text) if summary_text else []
    initial_prompt = build_initial_prompt(terms) if terms else None

    import torch
    import whisper

    device = choose_device(args.device, torch)
    model = whisper.load_model(args.model, device=device)
    transcribe_options = build_transcribe_options(args)
    transcribe_options["initial_prompt"] = initial_prompt

    hf_token = resolve_hf_token(args.hf_token)

    greedy_result = run_vad_chunk_transcribe(
        model=model,
        audio_path=audio_path,
        device=device,
        hf_token=hf_token,
        vad_segmentation_model="pyannote/segmentation-3.0",
        vad_target_duration=args.vad_target_duration,
        vad_max_duration=args.vad_max_duration,
        vad_strategy="greedy",
        vad_overlap_duration=args.vad_overlap_duration,
        transcribe_options=transcribe_options,
    )
    sliding_result = run_vad_chunk_transcribe(
        model=model,
        audio_path=audio_path,
        device=device,
        hf_token=hf_token,
        vad_segmentation_model="pyannote/segmentation-3.0",
        vad_target_duration=args.vad_target_duration,
        vad_max_duration=args.vad_max_duration,
        vad_strategy="sliding",
        vad_overlap_duration=args.vad_overlap_duration,
        transcribe_options=transcribe_options,
    )

    fp16 = device != "cpu"
    greedy_result = attach_alerts_and_probs(
        model=model,
        audio_path=audio_path,
        result=greedy_result,
        initial_prompt=initial_prompt,
        token_topk=args.token_topk,
        alert_threshold=args.alert_threshold,
        fp16=fp16,
    )
    sliding_result = attach_alerts_and_probs(
        model=model,
        audio_path=audio_path,
        result=sliding_result,
        initial_prompt=initial_prompt,
        token_topk=args.token_topk,
        alert_threshold=args.alert_threshold,
        fp16=fp16,
    )

    speaker_turns = run_pyannote_diarization(
        audio_path=str(audio_path),
        hf_token=hf_token,
        device_name=choose_torch_device(args.device),
        pipeline_name=args.diarization_pipeline,
    )
    attach_speakers_to_whisper_segments(greedy_result, speaker_turns)
    attach_speakers_to_whisper_segments(sliding_result, speaker_turns)

    diff_candidates = build_diff_candidates(greedy_result, sliding_result)
    alerts = build_alerts(greedy_result, sliding_result, diff_candidates)

    payload = {
        "audio_path": str(audio_path),
        "mode": "text_guided" if summary_text else "audio_only",
        "context_text": summary_text,
        "prompt_terms": terms,
        "initial_prompt": initial_prompt,
        "alert_threshold": args.alert_threshold,
        "token_topk": args.token_topk,
        "speaker_turns": speaker_turns,
        "passes": {
            "greedy": greedy_result,
            "sliding": sliding_result,
        },
        "diff_candidates": diff_candidates,
        "alerts": alerts,
        "alert_summary": {
            "low_confidence_count": sum(1 for alert in alerts if alert["type"] == "low_confidence"),
            "pass_mismatch_count": sum(1 for alert in alerts if alert["type"] == "pass_mismatch"),
            "total_count": len(alerts),
        },
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_stem or audio_path.stem
    output_path = output_dir / f"{output_stem}.dual_pass.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved dual-pass pipeline json: {output_path}")


if __name__ == "__main__":
    main()
