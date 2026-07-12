# 🎓 Hybrid RAG Academy Assistant

A Retrieval-Augmented Generation (RAG) based AI assistant for **Kalashala Beauty & Makeup Academy** that answers academy-related queries using a local Large Language Model (LLM), Hybrid Search, and Cross-Encoder Reranking.

---

## 📌 Project Overview

Hybrid RAG Academy Assistant is an AI-powered chatbot that provides accurate and context-aware answers about courses, fees, trainers, admissions, certifications, and other academy-related information.

Instead of relying solely on an LLM, the assistant retrieves relevant information from a structured knowledge base using **Hybrid Retrieval (FAISS + BM25)** and improves relevance using **Cross-Encoder Re-ranking** before generating a response with **Ollama (Llama 3)**.

---

## 🚀 Features

- ✅ Hybrid Retrieval (FAISS + BM25)
- ✅ Cross-Encoder Re-ranking
- ✅ Local LLM using Ollama (Llama 3)
- ✅ FastAPI REST Backend
- ✅ Streamlit Chat Interface
- ✅ Dockerized Deployment
- ✅ Markdown Knowledge Base
- ✅ Prompt Injection Protection
- ✅ Domain Restricted Responses
- ✅ Structured Responses
- ✅ Source Attribution
- ✅ Greeting & Small Talk Handling

---

## 🏗️ Architecture

<img width="820" height="1400" alt="rag_pipeline" src="https://github.com/user-attachments/assets/35c11766-f183-4dda-8072-41a181285c47" />


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
- FAISS
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

The assistant is designed to answer **only academy-related questions**.

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

*(Add Screenshot 4 here)*

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
