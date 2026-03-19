from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib import error, request


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

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _normalize_terms(terms: list[str], max_terms: int = 30) -> list[str]:
    normalized: list[str] = []
    for term in terms:
        cleaned = str(term).strip(" ,.;:()[]{}<>\"'").strip()
        if len(cleaned) < 2:
            continue
        if cleaned in STOP_TERMS:
            continue
        normalized.append(cleaned)
    return _ordered_unique(normalized)[:max_terms]


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

    return _normalize_terms(candidates, max_terms=max_terms)


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


def resolve_gemini_api_key(explicit_key: str | None = None) -> str | None:
    if explicit_key:
        return explicit_key
    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def _extract_text_from_gemini_response(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    for candidate in candidates:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            text = part.get("text")
            if text:
                return text
    return ""


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Gemini response text is empty.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def generate_prompt_terms_with_gemini(
    summary_text: str,
    *,
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
    max_terms: int = 30,
    max_chars: int = 220,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    prompt = f"""You are creating an initial_prompt for Whisper ASR.

Task:
- Read the Japanese context text.
- Extract up to {max_terms} high-value terms that are likely to improve transcription.
- Prioritize company names, service names, product names, insurance names, person names, place names, abbreviations, katakana words, and mixed-script brand names.
- Keep the exact surface form from the input whenever possible.
- Exclude generic words like explanation, confirmation, response, schedule.
- Build a short initial_prompt string no longer than {max_chars} characters.
- The initial_prompt should mainly be a comma-separated list of the most useful terms.
- Do not invent terms that are not supported by the input text.

Return JSON only with this schema:
{{
  "terms": ["term1", "term2"],
  "initial_prompt": "term1, term2"
}}

Context text:
{summary_text}
"""

    response_schema = {
        "type": "object",
        "properties": {
            "terms": {
                "type": "array",
                "items": {"type": "string"},
            },
            "initial_prompt": {"type": "string"},
        },
        "required": ["terms", "initial_prompt"],
    }

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
            "responseJsonSchema": response_schema,
        },
    }

    api_url = GEMINI_API_URL_TEMPLATE.format(model=model)
    http_request = request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Gemini API request failed: {exc}") from exc

    response_text = _extract_text_from_gemini_response(response_payload)
    parsed = _extract_json_object(response_text)
    terms = _normalize_terms(parsed.get("terms") or [], max_terms=max_terms)
    initial_prompt = (parsed.get("initial_prompt") or "").strip()
    if not initial_prompt:
        initial_prompt = build_initial_prompt(terms, max_chars=max_chars)
    else:
        initial_prompt = build_initial_prompt(_normalize_terms(initial_prompt.split(","), max_terms=max_terms), max_chars=max_chars)
        if not initial_prompt:
            initial_prompt = build_initial_prompt(terms, max_chars=max_chars)

    return {
        "source": "gemini",
        "model": model,
        "terms": terms,
        "initial_prompt": initial_prompt,
        "raw_response_text": response_text,
    }


def build_prompt_package(
    summary_text: str,
    *,
    max_terms: int = 30,
    max_chars: int = 220,
    gemini_api_key: str | None = None,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    use_gemini: bool = True,
) -> dict[str, Any]:
    fallback_terms = extract_prompt_terms(summary_text, max_terms=max_terms)
    fallback_initial_prompt = build_initial_prompt(fallback_terms, max_chars=max_chars)

    package: dict[str, Any] = {
        "source": "fallback_regex",
        "model": None,
        "terms": fallback_terms,
        "initial_prompt": fallback_initial_prompt,
        "fallback_terms": fallback_terms,
        "fallback_initial_prompt": fallback_initial_prompt,
        "llm_error": None,
    }

    if not use_gemini:
        package["source"] = "fallback_regex_disabled"
        return package

    api_key = resolve_gemini_api_key(gemini_api_key)
    if not api_key:
        package["source"] = "fallback_regex_no_gemini_key"
        return package

    try:
        gemini_result = generate_prompt_terms_with_gemini(
            summary_text,
            api_key=api_key,
            model=gemini_model,
            max_terms=max_terms,
            max_chars=max_chars,
        )
        package.update(gemini_result)
        package["fallback_terms"] = fallback_terms
        package["fallback_initial_prompt"] = fallback_initial_prompt
        return package
    except Exception as exc:
        package["source"] = "fallback_regex_gemini_failed"
        package["model"] = gemini_model
        package["llm_error"] = str(exc)
        return package
