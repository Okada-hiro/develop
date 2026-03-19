#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyannote_helpers import (
    annotation_to_turns,
    attach_speakers_to_whisper_segments,
    choose_torch_device,
    resolve_hf_token,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pyannote speaker diarization and optionally merge speakers into Whisper segments."
    )
    parser.add_argument("audio_path", type=Path, help="Path to an audio or video file.")
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face access token. Default: use HUGGINGFACE_HUB_TOKEN or HF_TOKEN",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device such as cuda, cpu, or mps. Default: auto-detect",
    )
    parser.add_argument(
        "--pipeline",
        default="pyannote/speaker-diarization-3.1",
        help="Pretrained pyannote pipeline id.",
    )
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument(
        "--whisper-json",
        type=Path,
        default=None,
        help="Existing Whisper output json. If provided, attach speaker labels to its segments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "develop" / "output",
        help="Directory where json output is written.",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Base filename for outputs. Default: audio filename stem",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = args.audio_path.expanduser().resolve()
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    token = resolve_hf_token(args.hf_token)
    device_name = choose_torch_device(args.device)

    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(args.pipeline, use_auth_token=token)
    if pipeline is None:
        raise SystemExit(
            "Failed to load pyannote pipeline. Check Hugging Face access, token, and model gating."
        )
    pipeline.to(torch.device(device_name))

    diarization = pipeline(
        str(audio_path),
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )
    speaker_turns = annotation_to_turns(diarization)

    output_dir = args.output_dir.expanduser().resolve()
    output_stem = args.output_stem or audio_path.stem
    diarization_path = output_dir / f"{output_stem}.diarization.json"

    diarization_payload = {
        "audio_path": str(audio_path),
        "device": device_name,
        "pipeline": args.pipeline,
        "num_speakers": args.num_speakers,
        "min_speakers": args.min_speakers,
        "max_speakers": args.max_speakers,
        "speaker_turns": speaker_turns,
    }
    write_json(diarization_path, diarization_payload)
    print(f"Saved diarization json: {diarization_path}")

    if args.whisper_json:
        whisper_json_path = args.whisper_json.expanduser().resolve()
        whisper_payload = json.loads(whisper_json_path.read_text(encoding="utf-8"))
        merged_payload = attach_speakers_to_whisper_segments(
            whisper_payload=whisper_payload,
            speaker_turns=speaker_turns,
        )
        merged_path = output_dir / f"{output_stem}.whisper_with_speakers.json"
        write_json(merged_path, merged_payload)
        print(f"Saved merged Whisper json: {merged_path}")


if __name__ == "__main__":
    main()
