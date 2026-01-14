"""Find the longest JSONL line in a dataset and estimate its token count."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import _bootstrap  # noqa: F401
from lib.config import DEFAULT_CONFIG_PATH, load_section


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to defaults JSON (config/defaults.json).",
    )
    config_args, remaining = config_parser.parse_known_args()
    defaults = load_section("dataset_generation", config_args.config)
    dataset_default = defaults.get("dataset_out")
    if not dataset_default:
        raise SystemExit("dataset_generation.dataset_out must be set in the config.")

    parser = argparse.ArgumentParser(
        description="Check longest line length in a JSONL dataset.",
        parents=[config_parser],
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(dataset_default),
        help="Dataset JSONL to scan.",
    )

    parser.add_argument(
        "--tokenizer",
        choices=["chars", "whitespace", "tiktoken"],
        default="chars",
        help="Token estimation strategy. chars is a robust heuristic for minified JSON.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model name used by tiktoken when --tokenizer=tiktoken.",
    )
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=4.0,
        help="Heuristic used when --tokenizer=chars, tokens ~= chars / this_value.",
    )
    return parser.parse_args(remaining)


def _count_tokens_whitespace(text: str) -> int:
    # Not good for minified JSON, but kept for completeness.
    return len(text.split())


def _count_tokens_chars(text: str, chars_per_token: float) -> int:
    if not text:
        return 0
    if chars_per_token <= 0:
        chars_per_token = 4.0
    # Round to nearest int, always at least 1 for non-empty strings.
    est = int((len(text) / chars_per_token) + 0.5)
    return max(est, 1)


def _count_tokens_tiktoken(text: str, model: str) -> Optional[int]:
    try:
        import tiktoken  # type: ignore
    except Exception:
        return None

    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        # Sensible fallback for most modern OpenAI models.
        enc = tiktoken.get_encoding("cl100k_base")

    return len(enc.encode(text))


def estimate_tokens(text: str, tokenizer: str, model: str, chars_per_token: float) -> Tuple[int, str]:
    """
    Returns (token_count, method_used).
    method_used can differ from tokenizer if tiktoken is unavailable and we fallback.
    """
    if tokenizer == "whitespace":
        return _count_tokens_whitespace(text), "whitespace"

    if tokenizer == "tiktoken":
        n = _count_tokens_tiktoken(text, model=model)
        if n is not None:
            return n, f"tiktoken({model})"
        # Fallback when dependency is missing.
        return _count_tokens_chars(text, chars_per_token=chars_per_token), "chars(fallback)"

    return _count_tokens_chars(text, chars_per_token=chars_per_token), "chars"


def longest_line_stats(path: Path, tokenizer: str, model: str, chars_per_token: float) -> Tuple[int, int, int, int, str]:
    """Return (line_number, char_len, byte_len, token_len, token_method) for the longest line."""
    max_line_no = 0
    max_chars = -1
    max_bytes = -1
    max_tokens = -1
    max_method = ""

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for idx, raw in enumerate(handle, start=1):
            line = raw.rstrip("\r\n")
            char_len = len(line)
            if char_len <= max_chars:
                continue

            byte_len = len(line.encode("utf-8"))
            tok_len, method = estimate_tokens(
                line,
                tokenizer=tokenizer,
                model=model,
                chars_per_token=chars_per_token,
            )

            max_line_no = idx
            max_chars = char_len
            max_bytes = byte_len
            max_tokens = tok_len
            max_method = method

    return max_line_no, max_chars, max_bytes, max_tokens, max_method


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    line_no, char_len, byte_len, tok_len, method = longest_line_stats(
        args.dataset,
        tokenizer=args.tokenizer,
        model=args.model,
        chars_per_token=args.chars_per_token,
    )
    print(
        "[seqcheck] "
        f"file={args.dataset} "
        f"longest_line={line_no} "
        f"chars={char_len} "
        f"bytes={byte_len} "
        f"approx_tokens={tok_len} "
        f"token_method={method}"
    )


if __name__ == "__main__":
    main()
