from __future__ import annotations

from typing import Any


def is_visible_output_token(token_text: str | None) -> bool:
    if not token_text:
        return False
    if token_text.startswith("<|") and token_text.endswith("|>"):
        return False
    return bool(token_text.strip())


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
