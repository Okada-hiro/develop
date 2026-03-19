#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prompt_terms import DEFAULT_GEMINI_MODEL, build_prompt_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an initial_prompt-oriented term list from summary text."
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Context text passed directly on the command line.",
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        default=None,
        help="Path to a UTF-8 text file containing context text.",
    )
    parser.add_argument(
        "--summary-text",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=30,
        help="Maximum number of extracted terms. Default: 30",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional json output path.",
    )
    parser.add_argument(
        "--gemini-api-key",
        default=None,
        help="Optional Gemini API key. Falls back to GEMINI_API_KEY / GOOGLE_API_KEY.",
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_GEMINI_MODEL,
        help=f"Gemini model name. Default: {DEFAULT_GEMINI_MODEL}",
    )
    parser.add_argument(
        "--disable-gemini",
        action="store_true",
        help="Disable Gemini-based prompt generation and use regex fallback only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.text and not args.text_file and not args.summary_text and not args.summary_file:
        raise SystemExit("Provide either --text, --text-file, --summary-text, or --summary-file.")

    if args.text_file:
        summary_text = args.text_file.expanduser().resolve().read_text(encoding="utf-8")
    elif args.summary_file:
        summary_text = args.summary_file.expanduser().resolve().read_text(encoding="utf-8")
    else:
        summary_text = args.text or args.summary_text or ""

    prompt_package = build_prompt_package(
        summary_text,
        max_terms=args.max_terms,
        gemini_api_key=args.gemini_api_key,
        gemini_model=args.gemini_model,
        use_gemini=not args.disable_gemini,
    )
    payload = {
        "summary_text": summary_text,
        **prompt_package,
    }

    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved prompt terms json: {output_path}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
