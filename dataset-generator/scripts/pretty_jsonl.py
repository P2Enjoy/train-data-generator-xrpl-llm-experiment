"""Convert JSONL files into pretty-printed JSON arrays for human reading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List


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


def load_jsonl(path: Path, limit: int | None = None) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


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
