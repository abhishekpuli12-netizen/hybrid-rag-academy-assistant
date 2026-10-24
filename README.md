# 🎓 Hybrid RAG Academy Assistant

A Retrieval-Augmented Generation (RAG) based AI assistant for **Kalashala Beauty & Makeup Academy** that answers academy-related queries using a local Large Language Model (LLM), Hybrid Search, and Cross-Encoder Reranking.

---

## 📌 Project Overview

Hybrid RAG Academy Assistant is an AI-powered chatbot that provides accurate and context-aware answers about courses, fees, trainers, admissions, certifications, and other academy-related information.

Instead of relying solely on an LLM, the assistant retrieves relevant information from a structured knowledge base using **Hybrid Retrieval (ChromaDB + BM25)** and improves relevance using **Cross-Encoder Re-ranking** before generating a response with **Ollama (Llama 3)**.

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
git clone https://github.com/yourusername/hybrid-rag-academy-assistant.git
```

```
cd hybrid-rag-academy-assistant
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

Download the model:

```bash
ollama pull llama3
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
docker compose up --build
```

ChromaDB data is stored in the Docker volume `chroma-data` and is rebuilt automatically when the Markdown knowledge base changes. The API adds an `X-Process-Time-Ms` response header for measuring request latency.

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
