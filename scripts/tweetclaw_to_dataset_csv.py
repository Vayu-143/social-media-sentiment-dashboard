#!/usr/bin/env python3
"""Convert TweetClaw-style exports into the dashboard training CSV format."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


TEXT_FIELDS = (
    "text",
    "tweet_text",
    "tweetText",
    "full_text",
    "fullText",
    "content",
    "message",
)
SENTIMENT_FIELDS = (
    "sentiment",
    "label",
    "sentiment_label",
    "sentimentLabel",
    "polarity",
)
ID_FIELDS = ("id", "tweet_id", "tweetId", "textID")
SENTIMENT_MAP = {
    "positive": "positive",
    "pos": "positive",
    "1": "positive",
    "negative": "negative",
    "neg": "negative",
    "-1": "negative",
    "neutral": "neutral",
    "neu": "neutral",
    "0": "neutral",
}


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    records.append(value)
        return records

    if isinstance(parsed, list):
        return [record for record in parsed if isinstance(record, dict)]
    if isinstance(parsed, dict):
        for key in ("tweets", "items", "results", "data", "records"):
            value = parsed.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
        return [parsed]
    return []


def first_string(record: dict[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        value = record.get(field)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


def normalize_sentiment(value: Any, default: str | None) -> str:
    if value is None or str(value).strip() == "":
        return default or ""
    return SENTIMENT_MAP.get(str(value).strip().lower(), "")


def convert_records(
    records: Iterable[dict[str, Any]], default_sentiment: str | None
) -> list[dict[str, str]]:
    converted: list[dict[str, str]] = []
    seen: set[str] = set()

    for record in records:
        text = first_string(record, TEXT_FIELDS)
        sentiment = normalize_sentiment(
            first_string(record, SENTIMENT_FIELDS), default_sentiment
        )
        if not text or not sentiment:
            continue

        dedupe_key = first_string(record, ID_FIELDS) or text
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        converted.append({"text": text, "sentiment": sentiment})

    return converted


def write_dataset(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("text", "sentiment"))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TweetClaw exports to data/dataset.csv format."
    )
    parser.add_argument("input", type=Path, help="TweetClaw JSON, JSONL, or CSV export")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("data/dataset.csv"),
        help="Output CSV path, defaults to data/dataset.csv",
    )
    parser.add_argument(
        "--default-sentiment",
        choices=("positive", "neutral", "negative"),
        help="Use this sentiment when an export row has text but no label",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    rows = convert_records(records, args.default_sentiment)
    write_dataset(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
