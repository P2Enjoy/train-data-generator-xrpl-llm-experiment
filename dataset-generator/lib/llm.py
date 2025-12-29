from __future__ import annotations

import subprocess


def run_ollama(prompt: str, model: str) -> str:
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ollama exited with {result.returncode}: {result.stderr.strip()}")
    return result.stdout.strip()
