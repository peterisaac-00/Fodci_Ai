from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any

from backend_ai.tokenizer import FodciTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "training_data" / "english_foundation" / "raw"
DEFAULT_OUTPUT = ROOT / "training_data" / "english_foundation"
DEFAULT_TOKENIZER = ROOT / "tokenizers" / "fodci-english-v4.json"
TOKENIZER_VERSION = "fodci-english-v4"
SOURCES = {
    "alice_in_wonderland.txt": {"ebook_id": 11, "title": "Alice's Adventures in Wonderland", "author": "Lewis Carroll", "url": "https://www.gutenberg.org/ebooks/11.txt.utf-8"},
    "pride_and_prejudice.txt": {"ebook_id": 1342, "title": "Pride and Prejudice", "author": "Jane Austen", "url": "https://www.gutenberg.org/ebooks/1342.txt.utf-8"},
    "frankenstein.txt": {"ebook_id": 84, "title": "Frankenstein; or, the Modern Prometheus", "author": "Mary Wollstonecraft Shelley", "url": "https://www.gutenberg.org/ebooks/84.txt.utf-8"},
    "moby_dick.txt": {"ebook_id": 2701, "title": "Moby-Dick; or, The Whale", "author": "Herman Melville", "url": "https://www.gutenberg.org/ebooks/2701.txt.utf-8"},
    "sherlock_holmes.txt": {"ebook_id": 1661, "title": "The Adventures of Sherlock Holmes", "author": "Arthur Conan Doyle", "url": "https://www.gutenberg.org/ebooks/1661.txt.utf-8"},
}


def clean_gutenberg(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    start = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text, flags=re.IGNORECASE | re.DOTALL)
    end = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text, flags=re.IGNORECASE | re.DOTALL)
    if start:
        text = text[start.end():]
    if end:
        text = text[: end.start()]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.strip().split())
        if line and not line.startswith("[Illustration"):
            lines.append(line)
        elif not line and lines and lines[-1] != "":
            lines.append("")
    cleaned = "\n".join(lines).strip()
    return cleaned + "\n"


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Phase 13.13 English-only foundation corpus.")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-merges", type=int, default=512)
    parser.add_argument("--dolly", type=Path, default=ROOT / "training_data" / "english_foundation" / "dolly" / "databricks-dolly-15k.jsonl")
    args = parser.parse_args()
    if args.max_merges < 0:
        raise ValueError("max-merges must be non-negative")

    clean_dir = args.output / "clean"
    train_dir = args.output / "train"
    validation_dir = args.output / "validation"
    for directory in (clean_dir, train_dir, validation_dir, args.tokenizer.parent):
        directory.mkdir(parents=True, exist_ok=True)

    missing = [name for name in SOURCES if not (args.raw / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing downloaded English sources: {missing}")

    cleaned: dict[str, str] = {}
    source_records: list[dict[str, Any]] = []
    for filename, metadata in SOURCES.items():
        raw_path = args.raw / filename
        cleaned_text = clean_gutenberg(raw_path.read_text(encoding="utf-8"))
        clean_path = clean_dir / filename
        clean_path.write_text(cleaned_text, encoding="utf-8")
        cleaned[filename] = cleaned_text
        source_records.append({**metadata, "filename": filename, "raw_sha256": sha256(raw_path), "clean_sha256": sha256(clean_path), "clean_characters": len(cleaned_text)})

    validation_name = "sherlock_holmes.txt"
    for filename, text in cleaned.items():
        target = validation_dir if filename == validation_name else train_dir
        (target / filename).write_text(text, encoding="utf-8")

    tokenizer_material = [text[:50_000] for filename, text in cleaned.items() if filename != validation_name]
    if args.dolly.is_file():
        for line in args.dolly.read_text(encoding="utf-8").splitlines()[:512]:
            row = json.loads(line)
            tokenizer_material.append(" ".join(str(row.get(field, "")) for field in ("instruction", "context", "response"))[:2_000])
    tokenizer = FodciTokenizer(vocab_size=10_000).train(tuple(tokenizer_material), max_merges=args.max_merges)
    tokenizer.save(args.tokenizer)
    manifest = {
        "format": "fodci.english_foundation_manifest",
        "schema_version": "1.0",
        "language": "en",
        "tokenizer_version": TOKENIZER_VERSION,
        "tokenizer_path": str(args.tokenizer.relative_to(ROOT)),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_merges": len(tokenizer.merges),
        "tokenizer_training_sources": [str(path.relative_to(ROOT)) for path in (args.raw, args.dolly) if path.exists()],
        "raw_sources": source_records,
        "train_sources": sorted(path.name for path in train_dir.glob("*.txt")),
        "validation_sources": sorted(path.name for path in validation_dir.glob("*.txt")),
        "license_url": "https://www.gutenberg.org/policy/license.html",
        "terms_note": "Source texts are selected from Project Gutenberg English works and must retain applicable source license notices when redistributed.",
        "quality": {
            "all_sources_english": True,
            "train_documents": len(list(train_dir.glob("*.txt"))),
            "validation_documents": len(list(validation_dir.glob("*.txt"))),
            "total_clean_characters": sum(len(text) for text in cleaned.values()),
            "min_clean_characters": min(len(text) for text in cleaned.values()),
        },
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "tokenizer": str(args.tokenizer), "train_documents": manifest["quality"]["train_documents"], "validation_documents": manifest["quality"]["validation_documents"], "tokenizer_merges": len(tokenizer.merges)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
