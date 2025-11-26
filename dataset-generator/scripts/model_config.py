"""Shared LLM model configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_MODEL = "gpt-oss:120b"


def default_model() -> str:
    env_model = os.getenv("LLM_MODEL")
    if env_model:
        return env_model.strip()

    root = Path(__file__).resolve().parent
    config_file = root.parent.joinpath(".llmrc")
    if config_file.exists():
        value = config_file.read_text(encoding="utf-8").strip()
        if value:
            return value
    return DEFAULT_MODEL


if __name__ == "__main__":
    print(default_model())
