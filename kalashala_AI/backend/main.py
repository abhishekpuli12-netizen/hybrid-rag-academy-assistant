from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import initialize_rag, answer_question


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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# Request Model
# ==========================================================

class ChatRequest(BaseModel):
    question: str


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


# ==========================================================
# Chat Endpoint
# ==========================================================
@app.post("/chat")
def chat(request: ChatRequest):

    print("=" * 50)
    print("Received Question:")
    print(request.question)

    print("Calling RAG...")

    result = answer_question(request.question)

    print("Answer generated.")

    return result
