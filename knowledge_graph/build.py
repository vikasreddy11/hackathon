"""
knowledge_graph/build.py
========================
CLI entrypoint — runs the full data-ingestion + graph-building pipeline.

Usage
-----
    # From the project root:
    python knowledge_graph/build.py

    # With custom paths / settings:
    python knowledge_graph/build.py \
        --raw-dir      data/raw \
        --output-dir   data/processed \
        --graphml-out  data/processed/graph.graphml \
        --pkl-out      knowledge_graph/graph.pkl \
        --chunk-tokens 500 \
        --overlap-tokens 50

Outputs
-------
    data/processed/chunks.jsonl        — chunked documents with metadata
    data/processed/graph.graphml       — human-readable graph (Gephi-compatible)
    knowledge_graph/graph.pkl          — fast-load binary graph for pipelines
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.scripts.ingest import ingest  # noqa: E402
from knowledge_graph.builder import GraphBuilder  # noqa: E402
from knowledge_graph.extractor import SpacyExtractor  # noqa: E402
from knowledge_graph.graph_store import NetworkXGraphStore  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "M1 pipeline: ingest raw docs → chunk → extract entities "
            "→ build knowledge graph → save."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Source documents directory (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for chunks.jsonl and graph files (default: data/processed)",
    )
    parser.add_argument(
        "--graphml-out",
        type=Path,
        default=None,
        help="Path for GraphML output (default: <output-dir>/graph.graphml)",
    )
    parser.add_argument(
        "--pkl-out",
        type=Path,
        default=Path("knowledge_graph/graph.pkl"),
        help="Path for pickle output (default: knowledge_graph/graph.pkl)",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=500,
        help="Target chunk size in tokens (default: 500)",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=50,
        help="Token overlap between consecutive chunks (default: 50)",
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model name (default: en_core_web_sm)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress messages (default: True)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    graphml_out = args.graphml_out or (output_dir / "graph.graphml")
    pkl_out = args.pkl_out

    print("=" * 60)
    print("M1 Pipeline: Data Ingestion + Knowledge Graph")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Stage 1: Ingest
    # ------------------------------------------------------------------
    print("\n[Stage 1] Ingesting raw documents …")
    chunks = ingest(
        raw_dir=args.raw_dir,
        output_dir=output_dir,
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    if not chunks:
        print("ERROR: No chunks produced. Check data/raw/ for documents.", file=sys.stderr)
        sys.exit(1)

    chunks_path = output_dir / "chunks.jsonl"

    # ------------------------------------------------------------------
    # Stage 2: Build graph
    # ------------------------------------------------------------------
    print(f"\n[Stage 2] Building knowledge graph …")
    print(f"  spaCy model : {args.spacy_model}")

    try:
        extractor = SpacyExtractor(model_name=args.spacy_model)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "  Install spaCy and the model:\n"
            "    pip install spacy\n"
            f"    python -m spacy download {args.spacy_model}",
            file=sys.stderr,
        )
        sys.exit(1)

    store = NetworkXGraphStore()
    builder = GraphBuilder(extractor=extractor, store=store, verbose=args.verbose)
    builder.build_from_chunks(chunks_path)

    # ------------------------------------------------------------------
    # Stage 3: Save
    # ------------------------------------------------------------------
    print(f"\n[Stage 3] Saving graph …")
    store.save(graphml_path=graphml_out, pkl_path=pkl_out)

    summary = store.summary()
    print("\n[Summary]")
    print(f"  Chunks processed : {len(chunks)}")
    print(f"  Total nodes      : {summary['total_nodes']}")
    print(f"  Total edges      : {summary['total_edges']}")
    print(f"  Node types       : {summary['node_type_counts']}")
    print(f"  Edge types       : {summary['edge_type_counts']}")
    print(f"\n  Outputs:")
    print(f"    chunks.jsonl  → {chunks_path}")
    print(f"    graph.graphml → {graphml_out}")
    print(f"    graph.pkl     → {pkl_out}")
    print("\nDone.")


if __name__ == "__main__":
    run_pipeline(parse_args())
