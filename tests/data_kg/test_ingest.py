"""
tests/data_kg/test_ingest.py
============================
Unit tests for data/scripts/ingest.py

Covered:
- clean_text strips boilerplate whitespace
- chunk_text produces non-empty chunks
- chunk size respects configured token limit (±overlap tolerance)
- chunk char offsets are valid (within original text)
- ingest() on sample docs produces chunks with all required fields
- required JSONL schema fields are present and typed correctly
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make project root importable regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.scripts.ingest import (
    approx_token_count,
    chunk_text,
    clean_text,
    ingest,
    make_chunk_id,
    make_doc_id,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RAW_DIR = PROJECT_ROOT / "data" / "raw"
SAMPLE_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

REQUIRED_CHUNK_FIELDS = {
    "doc_id": str,
    "chunk_id": str,
    "source": str,
    "text": str,
    "char_start": int,
    "char_end": int,
    "token_count": int,
}


@pytest.fixture(scope="module")
def sample_chunks(tmp_path_factory) -> list[dict]:
    """Run ingest() on the real sample docs and return the chunk list."""
    out = tmp_path_factory.mktemp("processed")
    return ingest(
        raw_dir=SAMPLE_RAW_DIR,
        output_dir=out,
        chunk_tokens=500,
        overlap_tokens=50,
    )


# ---------------------------------------------------------------------------
# clean_text tests
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_strips_leading_trailing_whitespace(self):
        assert clean_text("  hello world  ") == "hello world"

    def test_collapses_triple_blank_lines(self):
        raw = "line1\n\n\n\nline2"
        result = clean_text(raw)
        assert "\n\n\n" not in result
        assert "line1" in result
        assert "line2" in result

    def test_normalises_crlf(self):
        raw = "line1\r\nline2\r\nline3"
        result = clean_text(raw)
        assert "\r" not in result
        assert "line1" in result

    def test_strips_zero_width_chars(self):
        raw = "hello\u200bworld"
        assert "\u200b" not in clean_text(raw)

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_whitespace_only(self):
        assert clean_text("   \n\n   ") == ""


# ---------------------------------------------------------------------------
# chunk_text tests
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_empty_text_yields_nothing(self):
        result = list(chunk_text("", chunk_tokens=500, overlap_tokens=50))
        assert result == []

    def test_short_text_yields_one_chunk(self):
        text = "Hello world this is a test sentence with a few words."
        result = list(chunk_text(text, chunk_tokens=500, overlap_tokens=50))
        assert len(result) == 1
        chunk_str, char_start, char_end = result[0]
        assert chunk_str.strip() != ""
        assert char_start == 0
        assert char_end == len(text)

    def test_chunks_are_non_empty(self):
        # 2000-word text should produce multiple chunks
        text = " ".join(["word"] * 2000)
        chunks = list(chunk_text(text, chunk_tokens=500, overlap_tokens=50))
        assert len(chunks) > 1
        for chunk_str, _, _ in chunks:
            assert chunk_str.strip() != ""

    def test_chunk_token_count_within_bounds(self):
        text = " ".join([f"token{i}" for i in range(2000)])
        chunks = list(chunk_text(text, chunk_tokens=500, overlap_tokens=50))
        for chunk_str, _, _ in chunks:
            count = approx_token_count(chunk_str)
            # Last chunk may be smaller; all non-last chunks should be ≤ 500
            assert count <= 500

    def test_char_offsets_are_valid(self):
        text = " ".join([f"word{i}" for i in range(1000)])
        chunks = list(chunk_text(text, chunk_tokens=200, overlap_tokens=20))
        for chunk_str, char_start, char_end in chunks:
            assert 0 <= char_start < char_end <= len(text)
            # The extracted slice should roughly match the chunk text
            extracted = text[char_start:char_end]
            assert extracted.strip() != ""

    def test_overlap_produces_repeated_tokens(self):
        text = " ".join([f"tok{i}" for i in range(600)])
        chunks = list(chunk_text(text, chunk_tokens=100, overlap_tokens=20))
        # Consecutive chunks share overlap tokens
        if len(chunks) >= 2:
            end_of_first = chunks[0][0].split()[-20:]
            start_of_second = chunks[1][0].split()[:20]
            overlap = set(end_of_first) & set(start_of_second)
            assert len(overlap) > 0, "Expected some overlapping tokens between consecutive chunks"

    def test_configurable_chunk_size(self):
        text = " ".join(["a"] * 1000)
        chunks_500 = list(chunk_text(text, chunk_tokens=500, overlap_tokens=0))
        chunks_200 = list(chunk_text(text, chunk_tokens=200, overlap_tokens=0))
        assert len(chunks_200) > len(chunks_500)


# ---------------------------------------------------------------------------
# ingest() integration tests (uses real sample docs)
# ---------------------------------------------------------------------------

class TestIngest:
    def test_produces_non_empty_chunks(self, sample_chunks):
        assert len(sample_chunks) > 0, "Expected at least one chunk from sample docs"

    def test_all_required_fields_present(self, sample_chunks):
        for chunk in sample_chunks:
            for field in REQUIRED_CHUNK_FIELDS:
                assert field in chunk, f"Missing field '{field}' in chunk: {chunk.get('chunk_id')}"

    def test_field_types_correct(self, sample_chunks):
        for chunk in sample_chunks:
            for field, expected_type in REQUIRED_CHUNK_FIELDS.items():
                assert isinstance(chunk[field], expected_type), (
                    f"Field '{field}' has wrong type: "
                    f"expected {expected_type}, got {type(chunk[field])}"
                )

    def test_chunk_id_format(self, sample_chunks):
        for chunk in sample_chunks:
            assert "__chunk_" in chunk["chunk_id"], (
                f"chunk_id '{chunk['chunk_id']}' does not follow <doc_id>__chunk_NNNN format"
            )
            assert chunk["chunk_id"].startswith(chunk["doc_id"]), (
                f"chunk_id should start with doc_id"
            )

    def test_char_offsets_non_negative(self, sample_chunks):
        for chunk in sample_chunks:
            assert chunk["char_start"] >= 0
            assert chunk["char_end"] > chunk["char_start"]

    def test_text_non_empty(self, sample_chunks):
        for chunk in sample_chunks:
            assert chunk["text"].strip() != "", f"Empty text in chunk {chunk['chunk_id']}"

    def test_token_count_positive(self, sample_chunks):
        for chunk in sample_chunks:
            assert chunk["token_count"] > 0

    def test_all_sample_docs_represented(self, sample_chunks):
        doc_ids = {chunk["doc_id"] for chunk in sample_chunks}
        # We have 5 sample docs — all should produce at least one chunk
        expected_docs = {
            "doc1_llm_overview",
            "doc2_rag_systems",
            "doc3_knowledge_graphs",
            "doc4_agentic_ai",
            "doc5_evaluation",
        }
        for doc in expected_docs:
            assert doc in doc_ids, f"Expected doc '{doc}' to produce chunks"

    def test_chunks_jsonl_written(self, tmp_path):
        """Verify ingest() actually writes the JSONL file."""
        chunks = ingest(
            raw_dir=SAMPLE_RAW_DIR,
            output_dir=tmp_path,
            chunk_tokens=500,
            overlap_tokens=50,
        )
        out_path = tmp_path / "chunks.jsonl"
        assert out_path.exists(), "chunks.jsonl not written"
        lines = [l for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == len(chunks), "Line count mismatch in chunks.jsonl"

    def test_make_doc_id(self):
        p = Path("data/raw/doc1_llm_overview.txt")
        assert make_doc_id(p) == "doc1_llm_overview"

    def test_make_chunk_id(self):
        cid = make_chunk_id("doc1_llm_overview", 3)
        assert cid == "doc1_llm_overview__chunk_0003"
