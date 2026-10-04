"""
data/scripts/create_tg_schema.py
==================================
Recreates the AgenticGraphRAG graph with the correct schema.
Drops the existing empty graph and creates a new one with all vertex/edge types.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pyTigerGraph as tg


def main():
    host = os.environ.get("TG_HOST", "")
    graphname = os.environ.get("TG_GRAPHNAME", "AgenticGraphRAG")
    secret = os.environ.get("TG_SECRET", "")
    username = os.environ.get("TIGERGRAPH_USERNAME", "test1")
    password = os.environ.get("TIGERGRAPH_PASSWORD", "")

    print(f"Connecting to {host} / graph={graphname}")
    conn = tg.TigerGraphConnection(
        host=host,
        graphname=graphname,
        username=username,
        password=password,
        gsqlSecret=secret,
    )
    print("Connected!")

    # Check global state
    ls_global = conn.gsql("USE GLOBAL\nls")
    has_doc = "VERTEX Document" in ls_global
    print(f"Global types exist: {has_doc}")

    # Step 1: Create global vertex/edge types if needed
    if not has_doc:
        print("Creating global vertex/edge types...")
        stmts = [
            'USE GLOBAL',
            'CREATE VERTEX Document (PRIMARY_ID doc_id STRING, title STRING DEFAULT "", url STRING DEFAULT "", wikidata_qid STRING DEFAULT "", wikipedia_pageid STRING DEFAULT "") WITH STATS="OUTDEGREE_BY_EDGETYPE", PRIMARY_ID_AS_ATTRIBUTE="true"',
            'CREATE VERTEX Chunk (PRIMARY_ID chunk_id STRING, doc_id STRING DEFAULT "", title STRING DEFAULT "", text STRING DEFAULT "", token_count INT DEFAULT 0, char_start INT DEFAULT 0, char_end INT DEFAULT 0, source STRING DEFAULT "") WITH STATS="OUTDEGREE_BY_EDGETYPE", PRIMARY_ID_AS_ATTRIBUTE="true"',
            'CREATE VERTEX Entity (PRIMARY_ID entity_id STRING, label STRING DEFAULT "", entity_label STRING DEFAULT "", mention_count INT DEFAULT 1) WITH STATS="OUTDEGREE_BY_EDGETYPE", PRIMARY_ID_AS_ATTRIBUTE="true"',
            'CREATE DIRECTED EDGE PART_OF (FROM Chunk, TO Document, weight FLOAT DEFAULT 1.0) WITH REVERSE_EDGE="PART_OF_REVERSE"',
            'CREATE DIRECTED EDGE MENTIONS (FROM Chunk, TO Entity, weight FLOAT DEFAULT 1.0) WITH REVERSE_EDGE="MENTIONS_REVERSE"',
            'CREATE UNDIRECTED EDGE COOCCURS_WITH (FROM Entity, TO Entity, weight FLOAT DEFAULT 1.0, chunk_id STRING DEFAULT "")',
            'CREATE UNDIRECTED EDGE RELATED_TO (FROM Entity, TO Entity, weight FLOAT DEFAULT 1.0, doc_id STRING DEFAULT "")',
        ]
        for stmt in stmts:
            try:
                r = conn.gsql(stmt)
                print(f"  OK: {stmt[:70]}")
            except Exception as e:
                print(f"  Skip ({e}): {stmt[:40]}")
    else:
        print("Global types already exist.")

    # Step 2: Drop the empty graph and recreate WITH the types
    print(f"\nDropping empty graph {graphname} and recreating with schema...")

    drop_stmt = f"USE GLOBAL\nDROP GRAPH {graphname}"
    try:
        r = conn.gsql(drop_stmt)
        print(f"DROP result: {r}")
    except Exception as e:
        print(f"DROP note: {e}")

    create_stmt = (
        f"USE GLOBAL\n"
        f"CREATE GRAPH {graphname} "
        f"(Document, Chunk, Entity, "
        f"PART_OF, PART_OF_REVERSE, MENTIONS, MENTIONS_REVERSE, "
        f"COOCCURS_WITH, RELATED_TO)"
    )
    try:
        r = conn.gsql(create_stmt)
        print(f"CREATE GRAPH result: {r}")
    except Exception as e:
        print(f"CREATE GRAPH error: {e}")

    # Step 3: Reconnect to new graph and verify
    print("\nReconnecting and verifying...")
    conn2 = tg.TigerGraphConnection(
        host=host,
        graphname=graphname,
        username=username,
        password=password,
        gsqlSecret=secret,
    )
    ls = conn2.gsql(f"USE GRAPH {graphname}\nls")
    print(ls[:3000])

    if "Document" in ls and "Chunk" in ls and "Entity" in ls:
        print("\nSchema setup COMPLETE!")
    else:
        print("\nSomething went wrong — check ls output above.")


if __name__ == "__main__":
    main()
