"""Shared LLM model configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_MODEL = "gpt-oss:120b"


def default_model() -> str:
    env_model = os.getenv("LLM_MODEL")
    if env_model:
        return env_model.strip()
    return DEFAULT_MODEL


if __name__ == "__main__":
    print(default_model())
