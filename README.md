<<<<<<< HEAD
# RAG-Powered Document Intelligence
A Retrieval-Augmented Generation (RAG) pipeline built to extract contextual insights and answer complex user queries directly from unstructured PDF data using LangChain, Streamlit, and Google's Gemini 2.5 Flash.
=======
# 🧠 RAG-Powered Document Intelligence

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C.svg?logo=langchain&logoColor=white)](https://langchain.com/)
[![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-8E75B2.svg?logo=google&logoColor=white)](https://ai.google.dev/)

An enterprise-grade Retrieval-Augmented Generation (RAG) pipeline designed to extract contextual insights and answer complex user queries directly from unstructured PDF data. 

**[🔗 Try the Live Application Here](https://rag-powered-document-intelligence-hne8pkmr3j5jdfgqmw3xrh.streamlit.app)**

---

## 🏗️ System Architecture

This application leverages modern NLP and Vector Search methodologies, orchestrated entirely via **LangChain Expression Language (LCEL)** for optimized data routing.

1. **Document Ingestion:** `PyPDFLoader` reads the raw unstructured data.
2. **Semantic Chunking:** `RecursiveCharacterTextSplitter` segments the text (1000 chunk size / 200 overlap) to maintain semantic context boundaries.
3. **Embedding Generation:** `Gemini-Embedding-2` converts chunks into dense vector representations.
4. **Vector Storage:** Data is indexed in an `InMemoryVectorStore` for rapid retrieval.
5. **Retrieval & Generation:** A K-Nearest Neighbors (KNN) search retrieves the top `k=2` most relevant chunks, passing them into a custom prompt template. `Gemini 2.5 Flash` synthesizes the final response.

---

## ✨ Key Features

* **Dynamic Semantic Search:** Move beyond simple keyword matching to true contextual understanding of your uploaded documents.
* **Modern LCEL Pipeline:** Built using robust pipe (`|`) operators for scalable, production-ready chaining.
* **Low-Latency Conversational UI:** A fully interactive, stateful chat interface built with Streamlit, featuring real-time system diagnostics.
* **Session Memory:** Retains chat history within the active session for seamless follow-up querying.
* **Secure API Handling:** Environment variables safely managed via `.env` and Streamlit Secrets.

---

## 🛠️ Tech Stack

* **Framework:** LangChain (LCEL)
* **Large Language Model:** Google Gemini 2.5 Flash
* **Embedding Model:** Google Generative AI Embeddings (`gemini-embedding-2-preview`)
* **Frontend UI:** Streamlit
* **Document Processing:** PyPDF

---

## 🚀 Run Locally

Want to test the engine on your own machine? Follow these steps:

### 1. Clone the repository
```bash
git clone [https://github.com/kunalshekhawat/RAG-Powered-Document-Intelligence.git](https://github.com/kunalshekhawat/RAG-Powered-Document-Intelligence.git)
cd RAG-Powered-Document-Intelligence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API Keys
Create a `.env` file in the root directory and add your Google Gemini API key:
```env
GOOGLE_API_KEY="your_api_key_here"
```

### 4. Launch the application
```bash
streamlit run app.py
```

---
>>>>>>> 5efea52c814c19af649c8191557fa8c5faa7f6eb
