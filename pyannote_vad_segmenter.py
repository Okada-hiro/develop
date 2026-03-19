#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pyannote_helpers import (
    choose_torch_device,
    merge_speech_regions,
    resolve_hf_token,
    timeline_to_segments,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pyannote VAD and group speech regions into roughly 30-second chunks."
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
        "--segmentation-model",
        default="pyannote/segmentation-3.0",
        help="Segmentation model id used by pyannote VAD.",
    )
    parser.add_argument(
        "--target-duration",
        type=float,
        default=30.0,
        help="Preferred chunk duration in seconds. Default: 30.0",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=35.0,
        help="Hard upper bound when merging VAD speech regions. Default: 35.0",
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
    from pyannote.audio.pipelines import VoiceActivityDetection

    pipeline = VoiceActivityDetection(
        segmentation=args.segmentation_model,
        use_auth_token=token,
    )
    pipeline.to(torch.device(device_name))

    speech = pipeline(str(audio_path))
    speech_regions = timeline_to_segments(speech)
    chunks = merge_speech_regions(
        speech_regions,
        target_duration=args.target_duration,
        max_duration=args.max_duration,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_stem = args.output_stem or audio_path.stem
    output_path = output_dir / f"{output_stem}.vad.json"

    payload = {
        "audio_path": str(audio_path),
        "device": device_name,
        "segmentation_model": args.segmentation_model,
        "target_duration": args.target_duration,
        "max_duration": args.max_duration,
        "speech_regions": speech_regions,
        "chunks": chunks,
    }
    write_json(output_path, payload)

    print(f"Saved VAD json: {output_path}")
    print(f"Detected speech regions: {len(speech_regions)}")
    print(f"Chunk count: {len(chunks)}")


if __name__ == "__main__":
    main()
