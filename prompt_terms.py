from __future__ import annotations

import re


STOP_TERMS = {
    "内容",
    "確認",
    "説明",
    "連絡",
    "訪問",
    "面談",
    "対応",
    "予定",
    "実施",
    "連携",
    "必要",
    "検討",
    "相談",
    "資料",
    "案内",
    "依頼",
}


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def extract_prompt_terms(summary_text: str, max_terms: int = 30) -> list[str]:
    text = re.sub(r"\s+", " ", summary_text)

    patterns = [
        r"[A-Z]{2,}(?:[-_/][A-Z0-9]+)*",
        r"[A-Za-z][A-Za-z0-9+._/-]{2,}",
        r"[ァ-ヴー]{2,}(?:・[ァ-ヴー]{2,})*",
        r"[一-龠々]{2,8}",
        r"[一-龠々]+[A-Za-z0-9]+",
        r"[A-Za-z0-9]+[一-龠々]+",
    ]

    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(re.findall(pattern, text))

    normalized = []
    for term in candidates:
        cleaned = term.strip(" ,.;:()[]{}<>\"'").strip()
        if len(cleaned) < 2:
            continue
        if cleaned in STOP_TERMS:
            continue
        normalized.append(cleaned)

    return _ordered_unique(normalized)[:max_terms]


def build_initial_prompt(terms: list[str], max_chars: int = 220) -> str:
    prompt = ", ".join(terms)
    if len(prompt) <= max_chars:
        return prompt

    trimmed_terms: list[str] = []
    current_length = 0
    for term in terms:
        addition = len(term) if not trimmed_terms else len(term) + 2
        if current_length + addition > max_chars:
            break
        trimmed_terms.append(term)
        current_length += addition

    return ", ".join(trimmed_terms)
