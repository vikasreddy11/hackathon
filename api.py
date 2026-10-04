"""
api.py — FastAPI backend for Agentic GraphRAG Hackathon
=======================================================
Exposes endpoints for all three pipelines:
  - /ask/rag          → Pipeline 1: Standard RAG
  - /ask/graphrag     → Pipeline 2: GraphRAG
  - /ask/agentic      → Pipeline 3: Agentic GraphRAG
  - /ask/compare      → Run all three and compare
"""

import time
import traceback
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Agentic GraphRAG — Three-Pipeline Benchmark",
    version="2.0.0",
    description="Compare RAG, GraphRAG, and Agentic GraphRAG pipelines",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class QuestionRequest(BaseModel):
    question: str
    backend: Optional[str] = None
    model: Optional[str] = None
    top_k: int = Field(default=3, ge=1, le=20)
    max_steps: int = Field(default=8, ge=1, le=20)


class PipelineResult(BaseModel):
    pipeline: str
    answer: str
    sources: list = []
    metadata: dict = {}
    trace: list = []
    subgraph: dict = {}
    execution_time_s: float = 0.0
    success: bool = True
    error: Optional[str] = None


# ============================================================
# HEALTH / ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Agentic GraphRAG — Three-Pipeline Benchmark",
        "pipelines": ["rag", "graphrag", "agentic"],
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


# ============================================================
# PIPELINE 1: RAG
# ============================================================

@app.post("/ask/rag")
def ask_rag(request: QuestionRequest) -> dict:
    """Run standard RAG pipeline."""
    question = request.question.strip()
    if not question:
        return {"success": False, "error": "Question cannot be empty."}

    start = time.perf_counter()
    try:
        from rag.pipeline import run_rag

        result = run_rag(
            question=question,
            top_k=request.top_k,
            backend=request.backend,
            model=request.model,
        )
        elapsed = round(time.perf_counter() - start, 2)

        return {
            "success": True,
            "pipeline": "rag",
            "question": question,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "scores": result.get("scores", []),
            "metadata": {
                "execution_time_s": elapsed,
                "chunks_retrieved": len(result.get("sources", [])),
                "top_k": request.top_k,
            },
        }
    except Exception as e:
        elapsed = round(time.perf_counter() - start, 2)
        traceback.print_exc()
        return {
            "success": False,
            "pipeline": "rag",
            "question": question,
            "error": str(e),
            "metadata": {"execution_time_s": elapsed},
        }


# ============================================================
# PIPELINE 2: GRAPHRAG
# ============================================================

@app.post("/ask/graphrag")
def ask_graphrag(request: QuestionRequest) -> dict:
    """Run GraphRAG pipeline."""
    question = request.question.strip()
    if not question:
        return {"success": False, "error": "Question cannot be empty."}

    start = time.perf_counter()
    try:
        from graphrag.pipeline import run_graphrag

        result = run_graphrag(
            question=question,
            top_k_entities=10,
            max_chunks=request.top_k,
            backend=request.backend,
            model=request.model,
        )
        elapsed = round(time.perf_counter() - start, 2)

        subgraph = result.get("subgraph", {})
        return {
            "success": True,
            "pipeline": "graphrag",
            "question": question,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "subgraph": subgraph,
            "metadata": {
                "execution_time_s": elapsed,
                "entities_found": len(subgraph.get("entities", [])),
                "relationships_found": len(subgraph.get("relationships", [])),
                "chunks_linked": len(subgraph.get("chunk_ids", [])),
                "query_entities": subgraph.get("query_entities", []),
            },
        }
    except Exception as e:
        elapsed = round(time.perf_counter() - start, 2)
        traceback.print_exc()
        return {
            "success": False,
            "pipeline": "graphrag",
            "question": question,
            "error": str(e),
            "metadata": {"execution_time_s": elapsed},
        }


# ============================================================
# PIPELINE 3: AGENTIC GRAPHRAG
# ============================================================

@app.post("/ask/agentic")
def ask_agentic(request: QuestionRequest) -> dict:
    """Run Agentic GraphRAG pipeline."""
    question = request.question.strip()
    if not question:
        return {"success": False, "error": "Question cannot be empty."}

    start = time.perf_counter()
    try:
        from agents.pipeline import run_agentic_graphrag

        result = run_agentic_graphrag(
            question=question,
            max_steps=request.max_steps,
            backend=request.backend,
            model=request.model,
            verbose=True,
        )
        elapsed = round(time.perf_counter() - start, 2)

        metadata = result.get("metadata", {})
        metadata["execution_time_s"] = elapsed

        return {
            "success": True,
            "pipeline": "agentic",
            "question": question,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "trace": result.get("trace", []),
            "subgraph": result.get("subgraph", {}),
            "metadata": metadata,
        }
    except Exception as e:
        elapsed = round(time.perf_counter() - start, 2)
        traceback.print_exc()
        return {
            "success": False,
            "pipeline": "agentic",
            "question": question,
            "error": str(e),
            "metadata": {"execution_time_s": elapsed},
        }


# ============================================================
# COMPARE ALL THREE
# ============================================================

@app.post("/ask/compare")
def ask_compare(request: QuestionRequest) -> dict:
    """Run all three pipelines on the same question and compare."""
    question = request.question.strip()
    if not question:
        return {"success": False, "error": "Question cannot be empty."}

    results = {}

    # Pipeline 1: RAG
    try:
        rag_response = ask_rag(request)
        results["rag"] = rag_response
    except Exception as e:
        results["rag"] = {"success": False, "error": str(e), "pipeline": "rag"}

    # Pipeline 2: GraphRAG
    try:
        graphrag_response = ask_graphrag(request)
        results["graphrag"] = graphrag_response
    except Exception as e:
        results["graphrag"] = {"success": False, "error": str(e), "pipeline": "graphrag"}

    # Pipeline 3: Agentic GraphRAG
    try:
        agentic_response = ask_agentic(request)
        results["agentic"] = agentic_response
    except Exception as e:
        results["agentic"] = {"success": False, "error": str(e), "pipeline": "agentic"}

    return {
        "success": True,
        "question": question,
        "results": results,
    }