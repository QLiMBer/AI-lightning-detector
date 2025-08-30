from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


def extract_json_maybe(text: str) -> Optional[str]:
    # try fenced code block first
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if m:
        return m.group(1)
    # fallback: first brace to last brace
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1]
    return None


def parse_detection_json(text: str) -> Dict[str, Any]:
    # try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # try extraction
    blob = extract_json_maybe(text)
    if blob is not None:
        try:
            return json.loads(blob)
        except Exception:
            pass
    # give up: return minimal structure with raw text
    return {
        "error": "failed_to_parse_json",
        "raw": text.strip(),
    }


def to_pretty_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False)

