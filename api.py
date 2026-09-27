from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.agent import GraphRAGAgent


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Agentic GraphRAG Fraud API",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):
    question: str


# ============================================================
# AGENT
# ============================================================

agent = GraphRAGAgent()


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Agentic GraphRAG Fraud API",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


# ============================================================
# ASK AGENT
# ============================================================

@app.post("/ask")
def ask_agent(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        return {
            "success": False,
            "error": "Question cannot be empty.",
        }

    try:

        result = agent.run(question)

        return {
            "success": True,
            "question": question,
            "result": result,
        }

    except Exception as e:

        print("API ERROR:", type(e).__name__, str(e))

        return {
            "success": False,
            "question": question,
            "error": str(e),
        }