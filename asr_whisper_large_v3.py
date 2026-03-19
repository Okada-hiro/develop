#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
WHISPER_PROJECT_ROOT = REPO_ROOT / "whisper"

if str(WHISPER_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(WHISPER_PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ASR with Whisper large-v3 and save results for downstream processing."
    )
    parser.add_argument("audio_path", type=Path, help="Path to an audio or video file.")
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model name or checkpoint path. Default: large-v3",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device such as cuda, cpu, or mps. Default: auto-detect",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code such as ja, en. Default: auto-detect",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Run transcription or translation. Default: transcribe",
    )
    parser.add_argument(
        "--initial-prompt",
        default=None,
        help="Optional prompt text passed to Whisper.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Default: 0.0",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size when temperature is 0. Default: 5",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=5,
        help="Number of candidates when temperature is above 0. Default: 5",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Include word-level timings and probabilities when available.",
    )
    parser.add_argument(
        "--token-topk",
        type=int,
        default=0,
        help="If greater than 0, attach top-k token candidates and probabilities to each segment.",
    )
    parser.add_argument(
        "--carry-initial-prompt",
        action="store_true",
        help="Prepend the initial prompt to every internal decoding window.",
    )
    parser.add_argument(
        "--vad-chunk-transcribe",
        action="store_true",
        help="Use pyannote VAD chunks and transcribe each chunk independently before merging.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token for pyannote VAD. Default: use env vars such as HUGGINGFACE_HUB_TOKEN or HF_TOKEN",
    )
    parser.add_argument(
        "--vad-segmentation-model",
        default="pyannote/segmentation-3.0",
        help="Segmentation model id used by pyannote VAD.",
    )
    parser.add_argument(
        "--vad-target-duration",
        type=float,
        default=30.0,
        help="Preferred VAD chunk duration in seconds. Default: 30.0",
    )
    parser.add_argument(
        "--vad-max-duration",
        type=float,
        default=35.0,
        help="Hard upper bound when merging VAD speech regions. Default: 35.0",
    )
    parser.add_argument(
        "--vad-strategy",
        choices=["greedy", "sliding"],
        default="greedy",
        help="How pyannote speech regions are merged into chunks. Default: greedy",
    )
    parser.add_argument(
        "--vad-overlap-duration",
        type=float,
        default=5.0,
        help="Overlap duration in seconds for the sliding VAD strategy. Default: 5.0",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether each window should condition on prior decoded text. Default: true",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print decoded segments while processing.",
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        default=None,
        help="Optional directory for model downloads.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "develop" / "output",
        help="Directory where text/json outputs are written.",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Base filename for outputs. Default: audio filename stem",
    )
    return parser.parse_args()


def choose_device(device: str | None, torch_module: Any) -> str:
    if device:
        return device
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_audio_path(audio_path: Path) -> Path:
    resolved = audio_path.expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"Audio file not found: {resolved}")
    return resolved


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(item) for item in obj]
    return obj


def run_plain_transcribe(model: Any, audio_path: Path, transcribe_options: dict[str, Any]) -> dict[str, Any]:
    return model.transcribe(str(audio_path), **transcribe_options)


def run_vad_chunk_transcribe(
    *,
    model: Any,
    audio_path: Path,
    device: str,
    hf_token: str,
    vad_segmentation_model: str,
    vad_target_duration: float,
    vad_max_duration: float,
    vad_strategy: str,
    vad_overlap_duration: float,
    transcribe_options: dict[str, Any],
) -> dict[str, Any]:
    from pyannote_helpers import run_pyannote_vad

    vad_result = run_pyannote_vad(
        audio_path=str(audio_path),
        hf_token=hf_token,
        device_name=device,
        segmentation_model=vad_segmentation_model,
        target_duration=vad_target_duration,
        max_duration=vad_max_duration,
        strategy=vad_strategy,
        overlap_duration=vad_overlap_duration,
    )

    all_segments: list[dict[str, Any]] = []
    chunk_results: list[dict[str, Any]] = []
    detected_languages: list[str] = []
    next_segment_id = 0

    for chunk_index, chunk in enumerate(vad_result["chunks"]):
        chunk_options = dict(transcribe_options)
        chunk_options["clip_timestamps"] = f"{chunk['start']},{chunk['end']}"
        chunk_result = model.transcribe(str(audio_path), **chunk_options)
        detected_language = chunk_result.get("language")
        if detected_language:
            detected_languages.append(detected_language)

        chunk_segments = chunk_result.get("segments", [])
        for segment in chunk_segments:
            segment["id"] = next_segment_id
            segment["vad_chunk_index"] = chunk_index
            segment["vad_chunk_start"] = chunk["start"]
            segment["vad_chunk_end"] = chunk["end"]
            segment["transcribe_call_index"] = chunk_index
            next_segment_id += 1
            all_segments.append(segment)

        chunk_results.append(
            {
                "chunk_index": chunk_index,
                "start": chunk["start"],
                "end": chunk["end"],
                "duration": chunk["duration"],
                "speech_region_count": chunk["speech_region_count"],
                "text": chunk_result.get("text", ""),
                "language": detected_language,
                "segment_count": len(chunk_segments),
            }
        )

    merged_text = "".join(chunk["text"] for chunk in chunk_results).strip()
    merged_language = detected_languages[0] if detected_languages else transcribe_options.get("language")

    return {
        "text": merged_text,
        "segments": all_segments,
        "language": merged_language,
        "vad": vad_result,
        "vad_chunk_results": chunk_results,
    }


def main() -> None:
    args = parse_args()
    audio_path = ensure_audio_path(args.audio_path)

    try:
        import torch
        import whisper
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Failed to import dependencies. Install the project dependencies first, "
            "for example with `pip install -e ./whisper`."
        ) from exc

    device = choose_device(args.device, torch)

    model = whisper.load_model(
        args.model,
        device=device,
        download_root=str(args.download_root) if args.download_root else None,
    )

    transcribe_options: dict[str, Any] = {
        "task": args.task,
        "language": args.language,
        "temperature": args.temperature,
        "initial_prompt": args.initial_prompt,
        "carry_initial_prompt": args.carry_initial_prompt,
        "word_timestamps": args.word_timestamps,
        "condition_on_previous_text": args.condition_on_previous_text,
        "verbose": args.verbose,
        "beam_size": args.beam_size,
        "best_of": args.best_of,
    }

    if device == "cpu":
        transcribe_options["fp16"] = False

    if args.vad_chunk_transcribe:
        from pyannote_helpers import resolve_hf_token

        hf_token = resolve_hf_token(args.hf_token)
        result = run_vad_chunk_transcribe(
            model=model,
            audio_path=audio_path,
            device=device,
            hf_token=hf_token,
            vad_segmentation_model=args.vad_segmentation_model,
            vad_target_duration=args.vad_target_duration,
            vad_max_duration=args.vad_max_duration,
            vad_strategy=args.vad_strategy,
            vad_overlap_duration=args.vad_overlap_duration,
            transcribe_options=transcribe_options,
        )
    else:
        result = run_plain_transcribe(model, audio_path, transcribe_options)

    if args.token_topk > 0:
        from whisper_token_probs import attach_token_probabilities

        result["segments"] = attach_token_probabilities(
            model=model,
            audio_path=str(audio_path),
            segments=result.get("segments", []),
            language=result.get("language") or args.language or "en",
            task_name=args.task,
            initial_prompt=args.initial_prompt,
            carry_initial_prompt=args.carry_initial_prompt,
            condition_on_previous_text=args.condition_on_previous_text,
            topk=args.token_topk,
            fp16=bool(transcribe_options.get("fp16", True)),
            reset_key="transcribe_call_index" if args.vad_chunk_transcribe else None,
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_stem or audio_path.stem
    text_path = output_dir / f"{output_stem}.txt"
    json_path = output_dir / f"{output_stem}.json"

    text_path.write_text(result["text"].strip() + "\n", encoding="utf-8")

    payload = {
        "audio_path": audio_path,
        "model": args.model,
        "device": device,
        "task": args.task,
        "mode": "vad_chunk_transcribe" if args.vad_chunk_transcribe else "plain_transcribe",
        "language_requested": args.language,
        "language_detected": result.get("language"),
        "initial_prompt": args.initial_prompt,
        "carry_initial_prompt": args.carry_initial_prompt,
        "word_timestamps": args.word_timestamps,
        "token_topk": args.token_topk,
        "condition_on_previous_text": args.condition_on_previous_text,
        "vad_chunk_transcribe": args.vad_chunk_transcribe,
        "vad_segmentation_model": args.vad_segmentation_model if args.vad_chunk_transcribe else None,
        "vad_target_duration": args.vad_target_duration if args.vad_chunk_transcribe else None,
        "vad_max_duration": args.vad_max_duration if args.vad_chunk_transcribe else None,
        "vad_strategy": args.vad_strategy if args.vad_chunk_transcribe else None,
        "vad_overlap_duration": args.vad_overlap_duration if args.vad_chunk_transcribe else None,
        "vad": result.get("vad"),
        "vad_chunk_results": result.get("vad_chunk_results"),
        "text": result.get("text", ""),
        "segments": result.get("segments", []),
    }
    json_path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved transcript text: {text_path}")
    print(f"Saved transcript json: {json_path}")
    print(f"Detected language: {result.get('language')}")


if __name__ == "__main__":
    main()
