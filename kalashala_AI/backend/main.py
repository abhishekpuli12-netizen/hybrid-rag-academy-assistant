from contextlib import asynccontextmanager
import os
from time import perf_counter

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from guardrails import MAX_QUESTION_LENGTH, normalize_question
from rag import initialize_rag, answer_question, is_initialized


# ==========================================================
# FastAPI Lifespan
# ==========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 50)
    print("Initializing Kalashala AI Assistant...")
    print("=" * 50)

    initialize_rag()

    print("RAG initialized successfully.")
    print("=" * 50)

    yield

    print("Shutting down application...")


# ==========================================================
# FastAPI App
# ==========================================================

app = FastAPI(
    title="Kalashala AI Assistant",
    description="Hybrid RAG API for Kalashala Beauty & Makeup Academy",
    version="1.0.0",
    lifespan=lifespan,
)


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS", "http://localhost:8501"
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# Request Model
# ==========================================================

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        return normalize_question(value)


@app.middleware("http")
async def add_process_time_header(request, call_next):
    started = perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(perf_counter() - started) * 1000:.1f}"
    return response


# ==========================================================
# Health Check
# ==========================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "application": "Kalashala AI Assistant",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {"status": "ready" if is_initialized() else "starting"}


# ==========================================================
# Chat Endpoint
# ==========================================================
@app.post("/chat")
def chat(request: ChatRequest):
    return answer_question(request.question)
