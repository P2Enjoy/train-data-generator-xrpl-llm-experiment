"""Convert JSONL files into pretty-printed JSON arrays for human reading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from lib.io import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prettify JSONL into indented JSON.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Source JSONL file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination JSON file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on records to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} does not exist")
    records = load_jsonl(args.input, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(records, ensure_ascii=True, indent=2))
        handle.write("\n")


if __name__ == "__main__":
    main()
