from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG_PATH = Path("config/defaults.json")


def load_config(path: Path | str | None = None) -> Dict[str, Any]:
    target = Path(path) if path else DEFAULT_CONFIG_PATH
    if not target.exists():
        return {}
    try:
        with target.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def load_section(section: str, path: Path | str | None = None) -> Dict[str, Any]:
    config = load_config(path)
    section_data = config.get(section)
    if isinstance(section_data, dict):
        return section_data
    return {}
