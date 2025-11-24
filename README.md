# 📘 AI-Powered PDF Analyst  
**RAG • LangChain • FAISS • Groq LLMs • Streamlit**

AI-Powered PDF Analyst is an advanced application that lets you **upload PDFs, chat with them using RAG, generate summaries, create quizzes, and get topic explanations** — all powered by Groq LLMs, FAISS vector search, and LangChain.

Built end-to-end using Python, Sentence Transformers, FAISS, Groq, and Streamlit.  
Developed by **Kriti Tiwari**.

---

## 🚀 Features

### 📂 Upload & Process PDFs
- Upload one or more PDFs  
- Automatic text extraction  
- Page-wise segmentation  
- Smart text chunking  

### 🔍 AI Question Answering (RAG)
- Ask questions directly from your PDFs  
- Retrieved chunks via FAISS  
- Groq LLM generates final answers  
- **Includes citations with page numbers**  

### 🧠 AI Tools Included
- **Summary Generator**  
- **Quiz Generator** (MCQ / Short questions)  
- **Explain Mode** (simple / expert / examples)

### 💬 ChatGPT-style UI
- Smooth, clean chat experience  
- Latest message appears at the bottom  
- Persistent chat history  

### ⚡ Fast Vector Search
- Uses FAISS for high-speed chunk retrieval  
- Sentence Transformer embeddings (MiniLM-L6-v2)

### 🌐 Fully Deployable
- Deploy easily to Streamlit Cloud (free)

---

## 🛠️ Tech Stack

### **Backend**
- Python  
- LangChain  
- FAISS  
- Sentence Transformers  
- Groq LLM API (Llama3.3)  

### **Frontend**
- Streamlit (modern tab UI)

### **Deployment**
- Streamlit Cloud (recommended)

---

## 📁 Folder Structure
AI-Powered-PDF-Analyst/
│
├── app.py # Main Streamlit App
├── requirements.txt
├── README.md
│
├── backend/
│ ├── pdf_loader.py # PDF extraction
│ ├── text_splitter.py # Chunking logic
│ ├── embeddings.py # Embedding model
│ ├── vector_store.py # FAISS DB manager
│ ├── rag_pipeline.py # Retrieval + Generation Pipeline
│
├── data/
│ └── uploaded_pdfs/ # Uploaded PDFs
│
├── vectorstore/
│ └── faiss_index.bin # Saved FAISS index
│
└── chat_history/
└── history.json # Saved chat history


---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/AI-Powered-PDF-Analyst.git
cd AI-Powered-PDF-Analyst

## Activate Virtual Environment
python -m venv venv
# Activate:
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

## Install dependencies
pip install -r requirements.txt


## Create an env file 
GROQ_API_KEY=YOUR_API_KEY_HERE

## Run Locally
streamlit run app.py


## Architecture
Upload PDFs
     │
     ▼
PDF Loader → Text Splitter → Embeddings → FAISS Vector DB
     │                                         │
     └────── Query ──────→ Retrieve Top Chunks ┘
                              │
                              ▼
                       Groq LLM (RAG)
                              │
                              ▼
                       Streamlit Chat UI
