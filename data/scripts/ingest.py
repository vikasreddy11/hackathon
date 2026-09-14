"""
data/scripts/ingest.py
======================
Ingests raw documents from data/raw/, cleans text, splits into overlapping
chunks (~500 tokens), and writes structured output to data/processed/chunks.jsonl.

Supported formats: .txt, .md, .pdf, .csv

Usage:
    python data/scripts/ingest.py [--raw-dir PATH] [--output-dir PATH]
                                  [--chunk-tokens INT] [--overlap-tokens INT]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Optional heavy deps (pdf, csv) — imported lazily so plain .txt works without them
# ---------------------------------------------------------------------------

def _try_import(name: str):
    try:
        import importlib
        return importlib.import_module(name)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(raw: str) -> str:
    """Normalise unicode, strip boilerplate whitespace, collapse runs."""
    # Unicode normalisation (NFC)
    text = unicodedata.normalize("NFC", raw)
    # Replace Windows CRLF
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    # Collapse multiple blank lines to a maximum of two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [ln.rstrip() for ln in text.splitlines()]
    text = "\n".join(lines)
    # Strip overall leading/trailing
    return text.strip()


# ---------------------------------------------------------------------------
# Document loaders
# ---------------------------------------------------------------------------

def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_pdf(path: Path) -> str:
    pypdf = _try_import("pypdf")
    if pypdf is None:
        raise ImportError("pypdf is required for PDF ingestion: pip install pypdf")
    reader = pypdf.PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def load_csv(path: Path) -> str:
    """Flatten CSV rows into newline-delimited key=value text."""
    import csv
    rows: list[str] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row_text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
            rows.append(row_text)
    return "\n".join(rows)


LOADERS = {
    ".txt": load_txt,
    ".md": load_md,
    ".pdf": load_pdf,
    ".csv": load_csv,
}


def load_document(path: Path) -> str:
    """Load a document from path using the appropriate loader."""
    suffix = path.suffix.lower()
    loader = LOADERS.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file type: {suffix}")
    return loader(path)


# ---------------------------------------------------------------------------
# Tokenisation (lightweight word-level approximation — no external tokeniser)
# ---------------------------------------------------------------------------

def approx_token_count(text: str) -> int:
    """Approximate token count: split on whitespace (good enough for chunking)."""
    return len(text.split())


def word_tokens(text: str) -> list[str]:
    """Split text into word tokens, preserving whitespace boundaries."""
    return text.split()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_tokens: int = 500,
    overlap_tokens: int = 50,
) -> Iterator[tuple[str, int, int]]:
    """
    Yield (chunk_text, char_start, char_end) tuples.

    Strategy: split on whitespace tokens, window with overlap, map back to
    character offsets via cumulative scan.
    """
    if not text:
        return

    # Build list of (token_str, char_start, char_end) for each whitespace token
    token_spans: list[tuple[str, int, int]] = []
    for m in re.finditer(r"\S+", text):
        token_spans.append((m.group(), m.start(), m.end()))

    if not token_spans:
        return

    total = len(token_spans)
    start_idx = 0

    while start_idx < total:
        end_idx = min(start_idx + chunk_tokens, total)
        chunk_tokens_list = token_spans[start_idx:end_idx]

        chunk_str = " ".join(t for t, _, _ in chunk_tokens_list)
        char_start = chunk_tokens_list[0][1]
        char_end = chunk_tokens_list[-1][2]

        yield chunk_str, char_start, char_end

        if end_idx >= total:
            break

        # Next window: step forward by (chunk_tokens - overlap_tokens)
        step = max(1, chunk_tokens - overlap_tokens)
        start_idx += step


# ---------------------------------------------------------------------------
# Document ID helpers
# ---------------------------------------------------------------------------

def make_doc_id(path: Path) -> str:
    """Stable doc ID: stem of the filename (no extension)."""
    return path.stem


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Chunk ID: <doc_id>__chunk_<index>."""
    return f"{doc_id}__chunk_{chunk_index:04d}"


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------

def ingest(
    raw_dir: Path,
    output_dir: Path,
    chunk_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[dict]:
    """
    Ingest all supported documents from raw_dir, chunk them, and write to
    output_dir/chunks.jsonl.

    Returns the list of chunk dicts written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "chunks.jsonl"

    raw_files = sorted(
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in LOADERS
    )

    if not raw_files:
        print(f"[ingest] No supported files found in {raw_dir}", file=sys.stderr)
        return []

    all_chunks: list[dict] = []

    with open(out_path, "w", encoding="utf-8") as fh:
        for doc_path in raw_files:
            doc_id = make_doc_id(doc_path)
            print(f"[ingest] Loading: {doc_path.name}")

            try:
                raw_text = load_document(doc_path)
            except Exception as exc:
                print(f"[ingest]   ERROR loading {doc_path.name}: {exc}", file=sys.stderr)
                continue

            clean = clean_text(raw_text)
            if not clean:
                print(f"[ingest]   Skipping empty document: {doc_path.name}", file=sys.stderr)
                continue

            chunk_index = 0
            for chunk_str, char_start, char_end in chunk_text(
                clean, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens
            ):
                chunk_id = make_chunk_id(doc_id, chunk_index)
                record = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "source": str(doc_path.resolve()),
                    "text": chunk_str,
                    "char_start": char_start,
                    "char_end": char_end,
                    "token_count": approx_token_count(chunk_str),
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                all_chunks.append(record)
                chunk_index += 1

            print(f"[ingest]   → {chunk_index} chunks written for {doc_id}")

    print(f"\n[ingest] Done. {len(all_chunks)} total chunks → {out_path}")
    return all_chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest raw documents and write chunks.jsonl"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw source documents (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where chunks.jsonl will be written (default: data/processed)",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=500,
        help="Target chunk size in (whitespace-split) tokens (default: 500)",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=50,
        help="Overlap between consecutive chunks in tokens (default: 50)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
    )
