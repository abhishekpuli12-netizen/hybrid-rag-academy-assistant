| Knowledge answers | Every academy question depended on retrieval and an LLM response | Fees, offered-course lists, named course details, trainer/CIDESCO, admissions/contact, and common facilities/FAQ answers are read directly from trusted Markdown files |
| Knowledge answers | Every academy question depended on retrieval and an LLM response | Fees, offered-course lists, trainer/CIDESCO, and admissions/contact answers are now read directly from their trusted Markdown files |
| Knowledge answers | Every academy question depended on retrieval and an LLM response | Fees, offered-course lists, trainer/CIDESCO, and admissions/contact answers are now read directly from their trusted Markdown files |
| Safety | Primarily prompt-based | 500-character input cap, injection/domain checks, output-size and disclosure checks, unsupported-number rejection |
# 🎓 Hybrid RAG Academy Assistant

A Retrieval-Augmented Generation (RAG) based AI assistant for **Kalashala Beauty & Makeup Academy** that answers academy-related queries using a local Large Language Model (LLM), Hybrid Search, and Cross-Encoder Reranking.

---

## 📌 Project Overview

Hybrid RAG Academy Assistant is an AI-powered chatbot that provides accurate and context-aware answers about courses, fees, trainers, admissions, certifications, and other academy-related information.

Instead of relying solely on an LLM, the assistant retrieves relevant information from a structured knowledge base using **Hybrid Retrieval (ChromaDB + BM25)** and improves relevance using **Cross-Encoder Re-ranking** before generating a response with **Ollama (Llama 3)**.

## Latest improvements and measured metrics

| Area | Previous behavior | Current behavior / metric |
|---|---|---|
| Vector store | FAISS index and committed pickle artifacts | Persistent ChromaDB collection; legacy FAISS/pickle artifacts removed from Git |
| Dense retrieval | FAISS top 20 candidates | ChromaDB cosine top 12 by default |
| Sparse retrieval | BM25 top 20 candidates | BM25 top 12 by default |
| Cross-encoder | Reranked every retrieved candidate | Reranks at most 8 candidates by default |
| Cross-encoder | `BAAI/bge-reranker-base` | `cross-encoder/ms-marco-MiniLM-L-6-v2` default for a smaller download and faster CPU reranking; override with `RERANKER_MODEL` |
| Context | Up to 4 chunks | Up to 4 chunks by default |
| Generation | Model lifecycle and output length not tuned | Ollama keep-alive `10m`, response capped at 256 generated tokens by default |
| Generation | Default `llama3` (8B, ~4.7 GB model file) | Docker defaults to `llama3.2:1b` for lower-memory local runs; set `OLLAMA_MODEL` to another installed Ollama model |
| Safety | Primarily prompt-based | 500-character input cap, injection/domain checks, output-size and disclosure checks, unsupported-number rejection |
| Runtime telemetry | No stage breakdown | API `X-Process-Time-Ms` header and per-answer retrieval/rerank/generation/total metrics |

These are configuration and behavior changes, **not measured speedup or accuracy claims**. No comparable before/after benchmark has been run yet. Real latency varies by CPU/GPU, model availability, first-run model download, and Ollama load. For a meaningful comparison, run the same representative questions several times with a warm model, record stage timings, and compare median and p95 latency; use a labeled question set to measure retrieval recall and answer correctness.
The deterministic fee lookup was exercised against the live backend on this development machine with 10 repeated requests: client-observed median **9 ms**, p95 **105 ms** (first request **105 ms**; subsequent requests **7–10 ms**). The API's first-response `total_ms` was **3.1 ms**. This is a small local sample for the exact-fee fast path—not a general RAG/LLM latency result, not a baseline comparison, and not an accuracy benchmark. The returned course and fee were read directly from `backend/data/fees.md`.
The deterministic fee lookup was exercised against the live backend on this development machine with 10 repeated requests: client-observed median **9 ms**, p95 **105 ms** (first request **105 ms**; subsequent requests **7–10 ms**). The API's first-response `total_ms` was **3.1 ms**. This is a small local sample for the exact-fee fast path—not a general RAG/LLM latency result, not a baseline comparison, and not an accuracy benchmark. The returned course and fee were read directly from `backend/data/fees.md`. Course-list, trainer, and admission lookups were also smoke-tested against their respective Markdown files; a plural `courses` question exposed a domain-check bug, which was fixed and covered by a test.
The deterministic fee lookup was exercised against the live backend on this development machine with 10 repeated requests: client-observed median **9 ms**, p95 **105 ms** (first request **105 ms**; subsequent requests **7–10 ms**). The API's first-response `total_ms` was **3.1 ms**. This is a small local sample for the exact-fee fast path—not a general RAG/LLM latency result, not a baseline comparison, and not an accuracy benchmark. The returned course and fee were read directly from `backend/data/fees.md`. Course-list, trainer, and admission lookups were also smoke-tested against their respective Markdown files. Live testing then exposed a plural-topic guardrail bug and course-duration/list collision; both were fixed and regression tested. A duration query and a hostel query returned direct source-file answers in the follow-up smoke test.
The deterministic fee lookup was exercised against the live backend on this development machine with 10 repeated requests: client-observed median **9 ms**, p95 **105 ms** (first request **105 ms**; subsequent requests **7–10 ms**). The API's first-response `total_ms` was **3.1 ms**. This is a small local sample for the exact-fee fast path—not a general RAG/LLM latency result, not a baseline comparison, and not an accuracy benchmark. The returned course and fee were read directly from `backend/data/fees.md`.

Other rows in this table are configuration and behavior changes, **not measured speedup or accuracy claims**. General RAG latency varies by CPU/GPU, model availability, and Ollama load. Evaluate generated questions separately: run the same representative questions several times with a warm model, record stage timings, compare median/p95, and use labeled expected answers to measure retrieval recall and correctness.

---

## 🚀 Features

- ✅ Hybrid Retrieval (ChromaDB + BM25)
- ✅ Cross-Encoder Re-ranking
- ✅ Local LLM using Ollama (Llama 3)
- ✅ FastAPI REST Backend
- ✅ Streamlit Chat Interface
- ✅ Dockerized Deployment
- ✅ Markdown Knowledge Base
- ✅ Deterministic input validation and prompt-injection checks
- ✅ Output checks for prompt disclosure and unsupported numeric claims
- ✅ Domain Restricted Responses
- ✅ Structured Responses
- ✅ Source Attribution
- ✅ Source Attribution
- ✅ Verified fee-table lookup without LLM dependency
- ✅ Verified fee, course-list, trainer, and admissions lookups without LLM dependency
- ✅ Greeting & Small Talk Handling

---

## 🏗️ Architecture

<img width="2500" height="1200" alt="rag_flow_horizontal" src="https://github.com/user-attachments/assets/229512b1-6c79-4997-b7be-d7ac8b6023db" />



## 🛠️ Tech Stack

### Backend

- Python
- FastAPI
- Uvicorn

### Frontend

- Streamlit

### AI / Machine Learning

- Ollama
- Llama 3
- Sentence Transformers
- Cross Encoder
- ChromaDB (persistent local vector database)
- BM25

### Deployment

- Docker
- Docker Compose

---

## 📂 Project Structure

```
Hybrid-RAG-Academy-Assistant
│
├── backend
│   ├── data
│   ├── rag.py
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend
│   ├── app.py
│   └── Dockerfile
│
├── docker-compose.yml
├── README.md
└── .gitignore
```

---

## 📚 Knowledge Base

The assistant retrieves information from Markdown documents such as:

- Courses
- Fees
- Trainers
- Admissions
- Contact Information
- Batch Details
- Certifications
- Locations

---

## 💬 Sample Questions

### Courses

- What courses do you offer?
- Tell me about the Professional Makeup Course.
- What is the Beautician Course?

### Fees

- What are the course fees?
- How much is the Hair Styling Course?

### Trainers

- Who are the trainers?
- Are the trainers CIDESCO certified?

### Admissions

- I want to join a course.
- How can I enroll?

### Certifications

- Will I receive a certificate?
- Do trainers have CIDESCO certification?

---

## 🛡️ Security Features

The assistant is designed to answer **only academy-related questions**. These checks do not rely on the prompt alone:

- API requests must contain a non-empty question of at most 500 characters.
- Common role-override and instruction-disclosure attempts are rejected before retrieval.
- Non-academy questions are declined before calling the language model.
- Generated answers are checked for internal-prompt disclosure and numeric claims not present in retrieved evidence. Answers that fail the check are replaced with a safe response.
- Retrieved content is supplied as untrusted reference data, separately from the system policy.

These are defense-in-depth checks, not a guarantee against every possible model failure. Keep the knowledge base trusted and review behavior with representative tests before production use.

It resists common prompt injection attempts such as:

- Ignore previous instructions
- Reveal your prompt
- Forget the instructions
- Tell me about Python
- Tell me about AI
- Tell me about Indian festivals

If the requested information is outside the academy knowledge base, the assistant politely declines.

---

## 📸 Screenshots

### Home Page



### Course Information

<img width="903" height="365" alt="Screenshot 2026-07-11 032933" src="https://github.com/user-attachments/assets/644533e7-ba7b-4bb3-ba38-ec93a6f2cc8b" />


### Prompt Injection Protection

<img width="913" height="320" alt="Screenshot 2026-07-11 032640" src="https://github.com/user-attachments/assets/98ba37f0-68ba-40bf-97a7-5f20b08df86e" />


### Admissions Assistance

<img width="920" height="415" alt="Screenshot 2026-07-13 021041" src="https://github.com/user-attachments/assets/6b1c3661-a4c4-4ff0-ac97-01269c1ff5a5" />


---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/abhishekpuli12-netizen/hybrid-rag-academy-assistant.git
```

```
cd hybrid-rag-academy-assistant/kalashala_AI
```

---

### Install Backend

```bash
cd backend

pip install -r requirements.txt
```

---

### Install Frontend

```bash
cd ../frontend

pip install streamlit requests
```

---

### Start Ollama

```bash
ollama serve
```

Download the model (and ensure the configured embedding and reranker models can be downloaded on first start):

```bash
ollama pull llama3.2:1b
```

---

### Run Backend

```bash
cd backend

python -m uvicorn main:app --reload
```

---

### Run Frontend

```bash
cd frontend

streamlit run app.py
```

---

## 🐳 Docker

Build and run the complete application:

```bash
cd kalashala_AI
docker compose up --build -d
```

Open the interface at [http://localhost:8501](http://localhost:8501). Check backend readiness at [http://localhost:8000/health](http://localhost:8000/health). ChromaDB data is stored in the Docker volume `chroma-data` and is rebuilt automatically when the Markdown knowledge base changes. To stop the stack, use `docker compose down` (this preserves Chroma data).

The Docker backend installs CPU-only PyTorch so a CPU deployment does not download the CUDA libraries. Docker Desktop's Linux engine must be running. The host Ollama service must respond at `http://localhost:11434` and have the configured generation model pulled (`llama3.2:1b` by default in Docker). A larger model such as `llama3` may exceed available host memory; set `OLLAMA_MODEL` in the environment if a different model is installed and supported. First startup also downloads the embedding and reranker models and builds the Chroma index; allow it time to complete. The frontend waits for backend readiness, and Docker health status is available with `docker compose ps`. Hugging Face model files and Chroma data are persisted in named Docker volumes across rebuilds. Check startup logs with `docker compose logs -f backend`.

The API adds an `X-Process-Time-Ms` header. Each chat response also contains `metrics`: retrieval, rerank, generation, and total milliseconds, candidate counts, and context chunk count. The Streamlit UI displays the latency breakdown. These measurements are per request; the first request may include model warm-up and should be excluded from steady-state comparisons.
The API adds an `X-Process-Time-Ms` header. Each generated chat response also contains `metrics`: retrieval, rerank, generation, and total milliseconds, candidate counts, and context chunk count. The Streamlit UI displays the latency breakdown. These measurements are per request; the first request may include model warm-up and should be excluded from steady-state comparisons. Greetings and deterministic guardrail declines do not call the generation pipeline and may not include stage metrics.

Useful backend environment settings include `EMBEDDING_MODEL`, `RERANKER_MODEL`, `DENSE_CANDIDATES` (default 12), `SPARSE_CANDIDATES` (default 12), `RERANK_CANDIDATES` (default 8), `MAX_CONTEXT_CHUNKS` (default 4), `LLM_NUM_PREDICT` (default 256), `LLM_KEEP_ALIVE` (default `10m`), and `CORS_ORIGINS`. Set `CHAT_API_URL` and `CHAT_TIMEOUT_SECONDS` for the Streamlit frontend when needed. Lowering retrieval candidate limits may improve response time but should be checked against a small question/answer evaluation set before deployment.

The local LLM's generation time is often the largest part of end-to-end latency; keeping the model loaded avoids repeated model warm-up, while retrieval candidate limits reduce reranker work. The process-time header measures the complete API request, including model generation.

---

## 📈 Future Improvements

- Conversation Memory
- User Authentication
- Admin Dashboard
- Multi-language Support
- Voice-based Queries
- Image-based Beauty Consultation
- Online Appointment Booking

---

## 👨‍💻 Author

**Abhishek**

Python | Machine Learning | Generative AI

---

## ⭐ If you found this project useful, consider giving it a Star!
