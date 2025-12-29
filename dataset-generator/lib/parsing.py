from __future__ import annotations

import json
import re
from typing import Any, List


def strip_code_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()


def extract_json_object(text: str) -> Any:
    cleaned = strip_code_fences(text)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))

    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(cleaned):
        chunk = cleaned[idx:].lstrip()
        if not chunk:
            break
        try:
            obj, consumed = decoder.raw_decode(chunk)
            return obj
        except json.JSONDecodeError:
            idx += len(cleaned[idx:]) - len(chunk) + 1
            continue
    raise ValueError("No JSON object found in input")


def extract_json_array(text: str) -> List[Any]:
    def try_json_parse(chunk: str) -> Any:
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            return None

    cleaned = strip_code_fences(text)
    for candidate in (cleaned, cleaned.splitlines()[0] if cleaned else ""):
        if not candidate:
            continue
        parsed = try_json_parse(candidate)
        if parsed is not None:
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                for key in ("queries", "items", "responses"):
                    if isinstance(parsed.get(key), list):
                        return parsed[key]

    match = re.search(r"\[(?:[^\[\]]|\n)*\]", cleaned, flags=re.DOTALL)
    if match:
        parsed = try_json_parse(match.group(0))
        if isinstance(parsed, list):
            return parsed

    raise ValueError("No JSON array found in input")
