from __future__ import annotations

import re
from typing import Any


def is_visible_output_token(token_text: str | None) -> bool:
    if not token_text:
        return False
    if token_text.startswith("<|") and token_text.endswith("|>"):
        return False
    return bool(token_text.strip())


def is_substantive_output_token(token_text: str | None) -> bool:
    if not is_visible_output_token(token_text):
        return False
    return bool(re.search(r"[0-9A-Za-zぁ-んァ-ヴ一-龠々]", token_text or ""))


def attach_token_time_ranges(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for segment in segments:
        token_infos = segment.get("token_probs", [])
        if not token_infos:
            continue

        word_tokens: list[tuple[int, float, float]] = []
        for word in segment.get("words", []) or []:
            for token_id in word.get("tokens", []):
                word_tokens.append((token_id, float(word["start"]), float(word["end"])))

        word_index = 0
        for token_info in token_infos:
            token_text = token_info.get("token", "")
            token_id = token_info.get("token_id")

            if token_text.startswith("<|") and token_text.endswith("|>"):
                token_info["time_start"] = float(segment["start"])
                token_info["time_end"] = float(segment["end"])
                continue

            if word_index < len(word_tokens) and word_tokens[word_index][0] == token_id:
                _, start, end = word_tokens[word_index]
                token_info["time_start"] = start
                token_info["time_end"] = end
                word_index += 1
                continue

            token_info["time_start"] = None
            token_info["time_end"] = None

    return segments


def collect_low_confidence_alerts(
    segments: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []

    for segment in segments:
        segment_alerts = []
        for token_info in segment.get("token_probs", []):
            token_text = token_info.get("token")
            if not is_visible_output_token(token_text):
                continue

            probability = float(token_info.get("probability", 1.0))
            if probability >= threshold:
                continue

            alert = {
                "segment_id": segment.get("id"),
                "segment_start": segment.get("start"),
                "segment_end": segment.get("end"),
                "token": token_text,
                "token_id": token_info.get("token_id"),
                "probability": probability,
                "time_start": token_info.get("time_start"),
                "time_end": token_info.get("time_end"),
                "top_candidates": token_info.get("top_candidates", []),
            }
            alerts.append(alert)
            segment_alerts.append(alert)

        if segment_alerts:
            segment["alerts"] = segment_alerts
            segment["has_alert"] = True
        else:
            segment["alerts"] = []
            segment["has_alert"] = False

    return alerts


def build_low_confidence_spans(
    segments: list[dict[str, Any]],
    *,
    merge_gap: float = 0.6,
    padding: float = 1.0,
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    current_span: dict[str, Any] | None = None

    def flush_span() -> None:
        nonlocal current_span
        if current_span is None:
            return
        raw_start = float(current_span["raw_start"])
        raw_end = float(current_span["raw_end"])
        current_span["start"] = max(0.0, round(raw_start - padding, 3))
        current_span["end"] = round(raw_end + padding, 3)
        current_span["raw_start"] = round(raw_start, 3)
        current_span["raw_end"] = round(raw_end, 3)
        current_span["duration"] = round(current_span["end"] - current_span["start"], 3)
        spans.append(current_span)
        current_span = None

    for segment in segments:
        segment_id = segment.get("id")
        segment_start = float(segment.get("start", 0.0))
        segment_end = float(segment.get("end", segment_start))

        for alert in segment.get("alerts", []) or []:
            token = alert.get("token")
            if not is_substantive_output_token(token):
                continue

            token_start = alert.get("time_start")
            token_end = alert.get("time_end")
            start = segment_start if token_start is None else float(token_start)
            end = segment_end if token_end is None else float(token_end)

            if current_span and start - float(current_span["raw_end"]) <= merge_gap:
                current_span["raw_end"] = max(float(current_span["raw_end"]), end)
                current_span["segment_ids"].append(segment_id)
                current_span["tokens"].append(token)
                current_span["alerts"].append(alert)
            else:
                flush_span()
                current_span = {
                    "segment_ids": [segment_id],
                    "raw_start": start,
                    "raw_end": end,
                    "tokens": [token],
                    "alerts": [alert],
                }

    flush_span()

    for span_index, span in enumerate(spans):
        span["span_index"] = span_index
        span["segment_ids"] = list(dict.fromkeys(span["segment_ids"]))
        span["token_text"] = "".join(span["tokens"])
        span["alert_count"] = len(span["alerts"])

    return spans
