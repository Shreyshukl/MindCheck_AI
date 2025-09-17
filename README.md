# 🧠 MindCheck_AI

MindCheck_AI is an **AI-powered document Q&A assistant** that allows you to upload PDFs, transform them into searchable vector embeddings, and ask natural language questions.  
It uses **LangChain**, **FAISS**, and **Groq LLM** to retrieve context-aware answers while showing the exact **source text snippets** for transparency.  

---

## ✨ Features

- 📄 **PDF ingestion** – Load and chunk academic or clinical documents.  
- 🔍 **Semantic search** – Embeddings generated using `sentence-transformers` and indexed with **FAISS**.  
- 🤖 **LLM-powered answers** – Context-aware responses using Groq’s `llama-3.1-8b-instant`.  
- 🖥️ **Two interfaces**:
  - CLI terminal Q&A loop (`connect_memory_with_llm.py`)
  - Flask Web UI (`app.py`)  
- 📚 **Source transparency** – Displays file name, page number, and snippet of text used in the answer.  
- 🔐 **Secure config** – API keys loaded via `.env` (not hard-coded).  
- 🌐 **Deployable** – Configured for Render/Azure/Heroku with `requirements.txt` and `Procfile`.

---

## 🏗️ Tech Stack

- **Python 3.10+**
- **LangChain** – LLM orchestration  
- **FAISS** – Vector store for semantic search  
- **Groq LLM** – Fast inference with `llama-3.1-8b-instant`  
- **HuggingFace Embeddings** – `all-MiniLM-L6-v2` for vectorization  
- **Flask** – Simple web-based UI  

---

