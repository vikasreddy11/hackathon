"""
graphrag/context_builder.py
===========================
Converts retrieved subgraph elements (entities, relationships, and linked chunk text)
into a structured textual context prompt for the LLM.
"""

from __future__ import annotations

from typing import Any


def build_graph_context(subgraph: dict[str, Any], max_chunks: int = 5) -> str:
    """
    Format the retrieved subgraph into structured textual context.

    Parameters
    ----------
    subgraph : dict[str, Any]
        Dictionary returned by GraphRetriever.retrieve_subgraph().
    max_chunks : int
        Maximum number of supporting chunk texts to include.

    Returns
    -------
    str
        Human-readable and LLM-friendly context description.
    """
    entities = subgraph.get("entities", [])
    relationships = subgraph.get("relationships", [])
    chunks = subgraph.get("chunks", [])[:max_chunks]

    sections: list[str] = []

    # 1. Entities section
    if entities:
        ent_lines = []
        for e in entities:
            label = e.get("label", e.get("node_id", ""))
            ent_type = e.get("entity_label", "CONCEPT")
            doc = e.get("source_doc_id", "")
            doc_str = f" [Doc: {doc}]" if doc else ""
            ent_lines.append(f"  * {label} ({ent_type}){doc_str}")
        sections.append("Knowledge Graph Entities:\n" + "\n".join(ent_lines))
    else:
        sections.append("Knowledge Graph Entities:\n  * (No explicit graph entities matched)")

    # 2. Relationships section
    if relationships:
        rel_lines = []
        for r in relationships:
            src = r.get("source", "")
            tgt = r.get("target", "")
            rel_type = r.get("edge_type", "RELATED_TO")
            weight = r.get("weight", 1.0)
            rel_lines.append(f"  * {src} --[{rel_type} (weight={weight:.1f})]--> {tgt}")
        sections.append("Graph Relationships:\n" + "\n".join(rel_lines))
    else:
        sections.append("Graph Relationships:\n  * (No direct relationships found)")

    # 3. Supporting chunk evidence
    if chunks:
        chunk_lines = []
        for idx, c in enumerate(chunks, 1):
            cid = c.get("chunk_id", f"chunk_{idx}")
            doc_id = c.get("doc_id", "unknown_doc")
            text = c.get("text", "").strip()
            chunk_lines.append(f"[Evidence {idx}: {cid} (from {doc_id})]\n{text}")
        sections.append("Supporting Evidence from Document Chunks:\n" + "\n\n".join(chunk_lines))
    else:
        sections.append("Supporting Evidence from Document Chunks:\n  (No direct text chunks linked)")

    return "\n\n".join(sections)
