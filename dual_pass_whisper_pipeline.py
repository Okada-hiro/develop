#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path
from typing import Any

from asr_whisper_large_v3 import (
    choose_device,
    ensure_audio_path,
    run_vad_chunk_transcribe,
)
from prompt_terms import DEFAULT_GEMINI_MODEL, build_prompt_package
from pyannote_helpers import (
    attach_speakers_to_whisper_segments,
    choose_torch_device,
    resolve_hf_token,
    run_pyannote_diarization,
)
from transcript_alerts import (
    attach_token_time_ranges,
    build_low_confidence_spans,
    collect_low_confidence_alerts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a two-pass Whisper pipeline with summary-driven initial_prompt and VAD chunking."
    )
    parser.add_argument("audio_path", type=Path, help="Path to an audio or video file.")
    parser.add_argument("--text", default=None, help="Optional context text used to build initial_prompt.")
    parser.add_argument("--text-file", type=Path, default=None, help="Path to a UTF-8 text file used to build initial_prompt.")
    parser.add_argument("--summary-text", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--summary-file", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gemini-api-key", default=None, help="Optional Gemini API key. Falls back to GEMINI_API_KEY / GOOGLE_API_KEY.")
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL, help=f"Gemini model name. Default: {DEFAULT_GEMINI_MODEL}")
    parser.add_argument("--disable-gemini", action="store_true", help="Disable Gemini-based initial_prompt generation.")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--language", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--token-topk", type=int, default=5)
    parser.add_argument("--alert-threshold", type=float, default=0.90)
    parser.add_argument("--recheck-beam-size", type=int, default=10, help="Beam size for low-confidence span re-ASR. Default: 10")
    parser.add_argument("--recheck-best-of", type=int, default=10, help="Best-of value for low-confidence span re-ASR. Default: 10")
    parser.add_argument("--recheck-padding", type=float, default=1.0, help="Seconds of left/right padding added to each low-confidence span. Default: 1.0")
    parser.add_argument("--recheck-merge-gap", type=float, default=0.6, help="Merge neighboring low-confidence tokens into one span when the gap is within this value. Default: 0.6")
    parser.add_argument("--vad-target-duration", type=float, default=30.0)
    parser.add_argument("--vad-max-duration", type=float, default=35.0)
    parser.add_argument("--vad-overlap-duration", type=float, default=5.0)
    parser.add_argument(
        "--secondary-vad-target-duration",
        type=float,
        default=20.0,
        help="Target duration for the second segmentation strategy. Default: 20.0",
    )
    parser.add_argument(
        "--secondary-vad-max-duration",
        type=float,
        default=25.0,
        help="Max duration for the second segmentation strategy. Default: 25.0",
    )
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


def split_text_units(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[。！？?!])\s*|\n+", text)
    return [part.strip() for part in parts if part and part.strip()]


def build_diff_candidates(
    left_text: str,
    right_text: str,
) -> list[dict[str, Any]]:
    left_units = split_text_units(left_text)
    right_units = split_text_units(right_text)
    matcher = difflib.SequenceMatcher(a=left_units, b=right_units)

    diffs: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        diffs.append(
            {
                "type": "pass_mismatch",
                "operation": tag,
                "left_index_range": [i1, i2],
                "right_index_range": [j1, j2],
                "left_text": " ".join(left_units[i1:i2]).strip(),
                "right_text": " ".join(right_units[j1:j2]).strip(),
            }
        )

    return diffs


def build_segment_diff_candidates(
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
            right_start = float(right_segment["start"])
            right_end = float(right_segment["end"])
            overlap = max(0.0, min(left_end, right_end) - max(left_start, right_start))
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
                "overlap": round(best_overlap, 3),
                "left_segment_id": left_segment.get("id"),
                "right_segment_id": best_match.get("id"),
                "left_text": left_text,
                "right_text": right_text,
                "left_has_alert": bool(left_segment.get("has_alert")),
                "right_has_alert": bool(best_match.get("has_alert")),
                "left_speaker": left_segment.get("speaker"),
                "right_speaker": best_match.get("speaker"),
            }
        )

    return diffs


def build_alerts(
    greedy_result: dict[str, Any],
    sliding_result: dict[str, Any],
    text_diff_candidates: list[dict[str, Any]],
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

    for diff in text_diff_candidates:
        alerts.append(diff)

    return alerts


def write_text_outputs(
    *,
    output_dir: Path,
    output_stem: str,
    greedy_text: str,
    sliding_text: str,
    greedy_speaker_text: str,
    sliding_speaker_text: str,
) -> dict[str, str]:
    greedy_path = output_dir / f"{output_stem}.greedy.txt"
    sliding_path = output_dir / f"{output_stem}.sliding.txt"
    greedy_speaker_path = output_dir / f"{output_stem}.greedy.by_speaker.txt"
    sliding_speaker_path = output_dir / f"{output_stem}.sliding.by_speaker.txt"

    greedy_path.write_text(greedy_text.strip() + "\n", encoding="utf-8")
    sliding_path.write_text(sliding_text.strip() + "\n", encoding="utf-8")
    greedy_speaker_path.write_text(greedy_speaker_text.strip() + "\n", encoding="utf-8")
    sliding_speaker_path.write_text(sliding_speaker_text.strip() + "\n", encoding="utf-8")

    return {
        "greedy": str(greedy_path),
        "sliding": str(sliding_path),
        "greedy_by_speaker": str(greedy_speaker_path),
        "sliding_by_speaker": str(sliding_speaker_path),
    }


def build_recheck_context(segments: list[dict[str, Any]], start: float, end: float) -> dict[str, str]:
    before_parts: list[str] = []
    after_parts: list[str] = []

    for segment in segments:
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        segment_start = float(segment.get("start", 0.0))
        segment_end = float(segment.get("end", segment_start))
        if segment_end <= start:
            before_parts.append(text)
        elif segment_start >= end:
            after_parts.append(text)

    return {
        "before": " ".join(before_parts[-2:]).strip(),
        "after": " ".join(after_parts[:2]).strip(),
    }


def run_low_confidence_recheck(
    *,
    model: Any,
    audio_path: Path,
    result: dict[str, Any],
    transcribe_options: dict[str, Any],
    beam_size: int,
    best_of: int,
    padding: float,
    merge_gap: float,
) -> list[dict[str, Any]]:
    spans = build_low_confidence_spans(
        result.get("segments", []),
        merge_gap=merge_gap,
        padding=padding,
    )
    rechecks: list[dict[str, Any]] = []

    for span in spans:
        recheck_options = dict(transcribe_options)
        recheck_options["clip_timestamps"] = f"{span['start']},{span['end']}"
        recheck_options["beam_size"] = beam_size
        recheck_options["best_of"] = best_of
        recheck_options["condition_on_previous_text"] = False
        recheck_result = model.transcribe(str(audio_path), **recheck_options)
        context = build_recheck_context(result.get("segments", []), span["start"], span["end"])

        rechecks.append(
            {
                "span_index": span["span_index"],
                "start": span["start"],
                "end": span["end"],
                "duration": span["duration"],
                "raw_start": span["raw_start"],
                "raw_end": span["raw_end"],
                "alert_count": span["alert_count"],
                "segment_ids": span["segment_ids"],
                "token_text": span["token_text"],
                "tokens": span["tokens"],
                "alerts": span["alerts"],
                "context_before": context["before"],
                "context_after": context["after"],
                "beam_size": beam_size,
                "best_of": best_of,
                "text": (recheck_result.get("text") or "").strip(),
                "language": recheck_result.get("language"),
                "segment_count": len(recheck_result.get("segments", [])),
            }
        )

    return rechecks


def build_speaker_readable_text(segments: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    current_speaker: str | None = None
    current_start: float | None = None
    current_end: float | None = None
    current_texts: list[str] = []

    def flush_block() -> None:
        nonlocal current_speaker, current_start, current_end, current_texts
        if not current_texts:
            return
        speaker = current_speaker or "UNKNOWN"
        start = 0.0 if current_start is None else current_start
        end = 0.0 if current_end is None else current_end
        merged_text = " ".join(current_texts).strip()
        blocks.append(f"[{speaker}] ({start:.2f}-{end:.2f})\n{merged_text}")
        current_speaker = None
        current_start = None
        current_end = None
        current_texts = []

    for segment in segments:
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        speaker = segment.get("speaker") or "UNKNOWN"
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))

        if current_speaker is None:
            current_speaker = speaker
            current_start = start
            current_end = end
            current_texts = [text]
            continue

        if speaker == current_speaker:
            current_end = end
            current_texts.append(text)
            continue

        flush_block()
        current_speaker = speaker
        current_start = start
        current_end = end
        current_texts = [text]

    flush_block()
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def format_alerts_for_display(alerts: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, alert in enumerate(alerts, start=1):
        if alert["type"] == "low_confidence":
            lines.append(
                f"[{index}] low_confidence pass={alert.get('pass')} "
                f"segment={alert.get('segment_id')} token={alert.get('token')} "
                f"prob={alert.get('probability'):.3f} "
                f"time={alert.get('time_start')}->{alert.get('time_end')}"
            )
            candidates = alert.get("top_candidates", [])[:5]
            if candidates:
                candidate_text = ", ".join(
                    f"{candidate.get('token')}:{candidate.get('probability', 0.0):.3f}"
                    for candidate in candidates
                )
                lines.append(f"    candidates: {candidate_text}")
        elif alert["type"] == "pass_mismatch":
            lines.append(
                f"[{index}] pass_mismatch op={alert.get('operation')} "
                f"left='{alert.get('left_text', '')}' "
                f"right='{alert.get('right_text', '')}'"
            )

    if not lines:
        return "No alerts.\n"

    return "\n".join(lines) + "\n"


def format_rechecks_for_display(pass_name: str, rechecks: list[dict[str, Any]]) -> str:
    if not rechecks:
        return f"[{pass_name} recheck]\nNo low-confidence spans.\n"

    lines = [f"[{pass_name} recheck]"]
    for recheck in rechecks:
        lines.append(
            f"- span={recheck['span_index']} time={recheck['start']:.2f}-{recheck['end']:.2f} "
            f"tokens='{recheck['token_text']}' recheck='{recheck['text']}'"
        )
    return "\n".join(lines) + "\n"


def write_alerts_output(
    *,
    output_dir: Path,
    output_stem: str,
    alerts_text: str,
) -> str:
    alerts_path = output_dir / f"{output_stem}.alerts.txt"
    alerts_path.write_text(alerts_text, encoding="utf-8")
    return str(alerts_path)


def main() -> None:
    args = parse_args()
    audio_path = ensure_audio_path(args.audio_path)
    summary_text = load_summary_text(args)
    prompt_package = (
        build_prompt_package(
            summary_text,
            gemini_api_key=args.gemini_api_key,
            gemini_model=args.gemini_model,
            use_gemini=not args.disable_gemini,
        )
        if summary_text
        else {
            "source": "audio_only",
            "model": None,
            "terms": [],
            "initial_prompt": None,
            "fallback_terms": [],
            "fallback_initial_prompt": "",
            "llm_error": None,
        }
    )
    terms = prompt_package.get("terms") or []
    initial_prompt = prompt_package.get("initial_prompt") or None

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
        vad_target_duration=args.secondary_vad_target_duration,
        vad_max_duration=args.secondary_vad_max_duration,
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
    greedy_rechecks = run_low_confidence_recheck(
        model=model,
        audio_path=audio_path,
        result=greedy_result,
        transcribe_options=transcribe_options,
        beam_size=args.recheck_beam_size,
        best_of=args.recheck_best_of,
        padding=args.recheck_padding,
        merge_gap=args.recheck_merge_gap,
    )
    sliding_rechecks = run_low_confidence_recheck(
        model=model,
        audio_path=audio_path,
        result=sliding_result,
        transcribe_options=transcribe_options,
        beam_size=args.recheck_beam_size,
        best_of=args.recheck_best_of,
        padding=args.recheck_padding,
        merge_gap=args.recheck_merge_gap,
    )

    speaker_turns = run_pyannote_diarization(
        audio_path=str(audio_path),
        hf_token=hf_token,
        device_name=choose_torch_device(args.device),
        pipeline_name=args.diarization_pipeline,
    )
    attach_speakers_to_whisper_segments(greedy_result, speaker_turns)
    attach_speakers_to_whisper_segments(sliding_result, speaker_turns)

    greedy_text = greedy_result.get("text", "")
    sliding_text = sliding_result.get("text", "")
    greedy_speaker_text = build_speaker_readable_text(greedy_result.get("segments", []))
    sliding_speaker_text = build_speaker_readable_text(sliding_result.get("segments", []))
    diff_candidates = build_diff_candidates(greedy_text, sliding_text)
    segment_diff_candidates = build_segment_diff_candidates(greedy_result, sliding_result)
    alerts = build_alerts(greedy_result, sliding_result, diff_candidates)
    alerts_text = format_alerts_for_display(alerts)

    payload = {
        "audio_path": str(audio_path),
        "mode": "text_guided" if summary_text else "audio_only",
        "context_text": summary_text,
        "prompt_generation": prompt_package,
        "prompt_terms": terms,
        "initial_prompt": initial_prompt,
        "alert_threshold": args.alert_threshold,
        "token_topk": args.token_topk,
        "recheck_settings": {
            "beam_size": args.recheck_beam_size,
            "best_of": args.recheck_best_of,
            "padding": args.recheck_padding,
            "merge_gap": args.recheck_merge_gap,
        },
        "speaker_turns": speaker_turns,
        "transcriptions": {
            "greedy": greedy_text,
            "sliding": sliding_text,
            "greedy_by_speaker": greedy_speaker_text,
            "sliding_by_speaker": sliding_speaker_text,
        },
        "passes": {
            "greedy": greedy_result,
            "sliding": sliding_result,
        },
        "low_confidence_rechecks": {
            "greedy": greedy_rechecks,
            "sliding": sliding_rechecks,
        },
        "diff_candidates": diff_candidates,
        "segment_diff_candidates": segment_diff_candidates,
        "alerts": alerts,
        "alert_summary": {
            "low_confidence_count": sum(1 for alert in alerts if alert["type"] == "low_confidence"),
            "pass_mismatch_count": sum(1 for alert in alerts if alert["type"] == "pass_mismatch"),
            "greedy_recheck_count": len(greedy_rechecks),
            "sliding_recheck_count": len(sliding_rechecks),
            "total_count": len(alerts),
        },
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_stem or audio_path.stem
    text_output_paths = write_text_outputs(
        output_dir=output_dir,
        output_stem=output_stem,
        greedy_text=greedy_text,
        sliding_text=sliding_text,
        greedy_speaker_text=greedy_speaker_text,
        sliding_speaker_text=sliding_speaker_text,
    )
    alerts_output_path = write_alerts_output(
        output_dir=output_dir,
        output_stem=output_stem,
        alerts_text=alerts_text,
    )
    payload["transcription_files"] = text_output_paths
    payload["alert_file"] = alerts_output_path
    output_path = output_dir / f"{output_stem}.dual_pass.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved dual-pass pipeline json: {output_path}")
    print(f"Saved greedy transcription: {text_output_paths['greedy']}")
    print(f"Saved sliding transcription: {text_output_paths['sliding']}")
    print(f"Saved greedy transcription by speaker: {text_output_paths['greedy_by_speaker']}")
    print(f"Saved sliding transcription by speaker: {text_output_paths['sliding_by_speaker']}")
    print(f"Saved alerts: {alerts_output_path}")
    if summary_text:
        print("[prompt generation]")
        print(
            json.dumps(
                {
                    "source": prompt_package.get("source"),
                    "model": prompt_package.get("model"),
                    "prompt_terms": terms,
                    "initial_prompt": initial_prompt,
                    "fallback_terms": prompt_package.get("fallback_terms"),
                    "fallback_initial_prompt": prompt_package.get("fallback_initial_prompt"),
                    "llm_error": prompt_package.get("llm_error"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    print("[greedy transcription]")
    print(greedy_text)
    print("[greedy transcription by speaker]")
    print(greedy_speaker_text)
    print("[sliding transcription]")
    print(sliding_text)
    print("[sliding transcription by speaker]")
    print(sliding_speaker_text)
    print("[alerts]")
    print(alerts_text)
    print(format_rechecks_for_display("greedy", greedy_rechecks), end="")
    print(format_rechecks_for_display("sliding", sliding_rechecks), end="")


if __name__ == "__main__":
    main()
