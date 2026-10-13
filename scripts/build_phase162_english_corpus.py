"""Build a bounded English-only Fodci pretraining corpus.

The large corpus is intentionally generated outside Git under artifacts/. Git
stores this builder and its manifest/catalog fingerprints, not tens of millions
of tokens. Sources are English Project Gutenberg text records selected from the
official weekly CSV catalog.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend_ai.english_only import ENGLISH_ONLY_POLICY_VERSION, EnglishOnlyViolation, validate_english_text
from backend_ai.tokenizer import FodciTokenizer

ROOT = Path(__file__).resolve().parents[1]
CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
TEXT_URL = "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
DEFAULT_OUTPUT = ROOT / "artifacts" / "phase162_english_corpus"
DEFAULT_TOKENIZER = ROOT / "tokenizers" / "fodci-english-v4.json"
DEFAULT_TARGET_TOKENS = 20_000_000
MAX_TARGET_TOKENS = 50_000_000
USER_AGENT = "Fodci-English-Corpus/1.0"


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def clean_gutenberg(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    start = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text, flags=re.IGNORECASE | re.DOTALL)
    if start:
        text = text[start.end():]
    end = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text, flags=re.IGNORECASE | re.DOTALL)
    if end:
        text = text[: end.start()]
    lines: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = " ".join(raw_line.strip().split())
        if line and not line.startswith("[Illustration"):
            lines.append(line)
        elif not line and lines and lines[-1] != "":
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def fetch(url: str, *, timeout: int = 120) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def load_english_catalog(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig", errors="strict").splitlines()))
    selected: list[dict[str, str]] = []
    for row in rows:
        if row.get("Type") != "Text" or row.get("Language") != "en":
            continue
        raw_id = (row.get("Text#") or "").strip()
        if not raw_id.isdigit():
            continue
        selected.append({
            "book_id": raw_id,
            "title": (row.get("Title") or "").strip(),
            "authors": (row.get("Authors") or "").strip(),
            "subjects": (row.get("Subjects") or "").strip(),
            "url": TEXT_URL.format(book_id=raw_id),
        })
    return sorted(selected, key=lambda item: int(item["book_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a bounded English-only Fodci pretraining corpus from Gutenberg.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--max-books", type=int, default=500)
    parser.add_argument("--catalog-url", default=CATALOG_URL)
    parser.add_argument("--refresh", action="store_true", help="redownload the official catalog and source texts")
    args = parser.parse_args()
    if not 20_000_000 <= args.target_tokens <= MAX_TARGET_TOKENS:
        raise ValueError("target-tokens must be between 20,000,000 and 50,000,000")
    if args.max_books <= 0:
        raise ValueError("max-books must be positive")

    raw_dir = args.output / "raw"
    clean_dir = args.output / "clean"
    args.output.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = args.output / "pg_catalog.csv"
    if args.refresh or not catalog_path.exists():
        catalog_path.write_bytes(fetch(args.catalog_url))
    catalog = load_english_catalog(catalog_path)
    if not catalog:
        raise RuntimeError("official catalog contained no English text records")

    tokenizer = FodciTokenizer.load(args.tokenizer)
    records: list[dict[str, object]] = []
    total_tokens = 0
    total_chars = 0
    failures: list[dict[str, str]] = []
    started = time.perf_counter()
    for item in catalog[: args.max_books]:
        if total_tokens >= args.target_tokens:
            break
        book_id = item["book_id"]
        raw_path = raw_dir / f"pg{book_id}.txt"
        clean_path = clean_dir / f"pg{book_id}.txt"
        try:
            if args.refresh or not raw_path.exists():
                raw_path.write_bytes(fetch(item["url"]))
            cleaned = clean_gutenberg(raw_path.read_text(encoding="utf-8", errors="strict"))
            validate_english_text(cleaned, field=f"gutenberg:{book_id}")
            clean_path.write_text(cleaned, encoding="utf-8")
            token_count = len(tokenizer.encode(cleaned))
            if token_count < 1_024:
                continue
            total_tokens += token_count
            total_chars += len(cleaned)
            records.append({
                **item,
                "raw_path": str(raw_path.relative_to(ROOT)),
                "clean_path": str(clean_path.relative_to(ROOT)),
                "raw_sha256": sha256_bytes(raw_path.read_bytes()),
                "clean_sha256": sha256_bytes(cleaned.encode("utf-8")),
                "characters": len(cleaned),
                "tokens": token_count,
            })
            print(json.dumps({"book_id": book_id, "tokens": token_count, "total_tokens": total_tokens}, ensure_ascii=False), flush=True)
        except (HTTPError, URLError, TimeoutError, OSError, UnicodeError, UnicodeDecodeError, EnglishOnlyViolation) as exc:
            failures.append({"book_id": book_id, "title": item["title"], "error": str(exc)[:256]})
            continue

    if total_tokens < args.target_tokens:
        raise RuntimeError(f"only collected {total_tokens:,} tokens; target was {args.target_tokens:,}. Increase --max-books or retry downloads.")
    manifest = {
        "format": "fodci.phase162_english_corpus_manifest",
        "schema_version": "1.0",
        "language": "en",
        "language_policy": ENGLISH_ONLY_POLICY_VERSION,
        "purpose": "english_pretraining_only",
        "tokenizer_path": str(args.tokenizer.relative_to(ROOT)),
        "tokenizer_sha256": sha256_bytes(args.tokenizer.read_bytes()),
        "token_count_method": "len(FodciTokenizer.encode(clean_text))",
        "target_tokens": args.target_tokens,
        "total_tokens": total_tokens,
        "total_characters": total_chars,
        "book_count": len(records),
        "catalog_url": args.catalog_url,
        "catalog_sha256": sha256_bytes(catalog_path.read_bytes()),
        "source_license": "Project Gutenberg public-domain catalog; verify local redistribution obligations.",
        "records": records,
        "download_failures": failures,
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "books": len(records), "total_tokens": total_tokens, "total_characters": total_chars}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
