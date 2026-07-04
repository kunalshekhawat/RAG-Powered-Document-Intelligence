# 🧠 RAG-Powered Document Intelligence

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C.svg?logo=langchain&logoColor=white)](https://langchain.com/)
[![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-8E75B2.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F00.svg)](https://www.trychroma.com/)
[![RAGAS](https://img.shields.io/badge/RAGAS-Evaluated-green.svg)](https://docs.ragas.io/)

An advanced, production-grade Retrieval-Augmented Generation (RAG) pipeline designed to extract contextual insights and answer complex user queries from unstructured PDF documents — featuring hybrid retrieval, cross-encoder reranking, multi-turn memory, and quantitative RAGAS evaluation.

**[🔗 Try the Live Application Here]([https://rag-powered-document-intelligence-hne8pkmr3j5jdfgqmw3xrh.streamlit.app](https://rag-powered-document-intelligence-hne8pkmr3j5jdfgqmw3xrh.streamlit.app/))**

---

## 🏗️ System Architecture

The pipeline is orchestrated entirely via **LangChain Expression Language (LCEL)** and consists of five key stages:

```
PDF Upload
    ↓
PyPDFLoader → RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
    ↓
Gemini-Embedding-2 → ChromaDB (persistent vector store)
    ↓
Query → BM25 Retriever (k=10) ─┐
      → Dense Vector Retriever  ─┤ EnsembleRetriever (weights: 0.5/0.5)
                                 ↓
                    Cross-Encoder Reranker (top_n=6)
                                 ↓
                    Gemini 2.5 Flash + Chat History
                                 ↓
                           Final Answer
```

---

## ✨ Key Features

- **Hybrid Retrieval** — Combines BM25 sparse keyword search with Gemini dense semantic embeddings via `EnsembleRetriever` for superior recall over either method alone
- **Cross-Encoder Reranking** — `ms-marco-MiniLM-L-6-v2` reranks the top 20 hybrid candidates to surface only the most relevant chunks before LLM inference
- **Persistent ChromaDB Storage** — Vector embeddings persisted to disk with session-isolated collections, replacing volatile in-memory storage
- **Multi-turn Conversational Memory** — Full chat history injected into each prompt so the LLM resolves follow-up references correctly
- **Strict Context-Grounded Answers** — Prompt engineering enforces answers only from retrieved context, minimizing hallucination
- **RAGAS Evaluation Pipeline** — Standalone `evaluate_ragas.py` script with auto-generated golden dataset for quantitative pipeline assessment
- **Real-time System Diagnostics** — Live sidebar displaying the full architecture stack

---

## 📊 RAGAS Evaluation Results

The pipeline was evaluated using the [RAGAS framework](https://docs.ragas.io/) on a 100-sample golden dataset auto-generated from the Attention Is All You Need paper using `TestsetGenerator`.

| Metric | Score | Description |
|--------|-------|-------------|
| **Faithfulness** | **0.98** | Answers grounded in retrieved context |
| **Context Precision** | **0.97** | Retrieved chunks are relevant |
| **Context Recall** | **0.95** | Pipeline captures needed information |
| **Answer Relevancy** | **0.80** | Answers address the question asked |

> **Note on Answer Relevancy:** Scored by Mistral (local judge) evaluating Gemini 2.5 Flash answers. The judge-answer model mismatch causes conservative scoring — retrieval metrics (Faithfulness, Context Precision, Context Recall) which are more objective all score above 0.95.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM Engine | Google Gemini 2.5 Flash |
| Embedding Model | Gemini-Embedding-2-preview |
| Vector Store | ChromaDB (persistent) |
| Sparse Retriever | BM25 (rank-bm25) |
| Hybrid Retriever | LangChain EnsembleRetriever |
| Reranker | Cross-Encoder ms-marco-MiniLM-L-6-v2 |
| RAG Framework | LangChain LCEL |
| Evaluation | RAGAS Framework |
| Judge LLM (eval) | Mistral via Ollama (GPU-accelerated) |
| Frontend | Streamlit |

---

## 📁 Repository Structure

```
RAG-Powered-Document-Intelligence/
│
├── app.py                  # Main Streamlit chatbot application
├── evaluate_ragas.py       # RAGAS evaluation pipeline
├── golden_dataset.csv      # 100-sample Q&A evaluation dataset
├── requirements.txt        # Python dependencies
├── Modelfile               # Ollama GPU config for Mistral
├── chroma_db/              # Persisted ChromaDB vector store
└── Attention_Is_All_You_Need.pdf  # Sample evaluation document
```

---

## 🚀 Run Locally

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) (for running evaluation only)
- Google Gemini API key (free tier at [ai.google.dev](https://ai.google.dev))

### 1. Clone the repository

```bash
git clone https://github.com/kunalshekhawat/RAG-Powered-Document-Intelligence.git
cd RAG-Powered-Document-Intelligence
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up API keys

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. Launch the chatbot

```bash
streamlit run app.py
```

Upload any PDF and start asking questions.

---

## 🧪 Running the RAGAS Evaluation

### 1. Pull Ollama models (for local judge LLM)

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

### 2. Start Ollama

```bash
ollama serve
```

### 3. Run evaluation

```bash
python evaluate_ragas.py
```

Results are printed to console and saved to `ragas_results.csv`.

> The golden dataset (`golden_dataset.csv`) is pre-generated from the Attention Is All You Need paper.

---

## 💡 How It Works

### Hybrid Retrieval
Traditional RAG uses only dense vector search (semantic similarity). This pipeline combines:
- **BM25** — captures exact keyword matches that embeddings miss
- **Dense search** — captures semantic meaning even when exact words differ
- **EnsembleRetriever** — merges and deduplicates both result sets with equal weighting

### Cross-Encoder Reranking
The hybrid retriever returns 20 candidate chunks (10 from each source). A cross-encoder model then scores each `(query, chunk)` pair jointly — much more accurate than the dot-product similarity used in first-stage retrieval — and keeps only the top 6 for LLM context.

### Multi-turn Memory
Chat history from all previous turns is injected into the prompt before the current query, allowing the LLM to correctly resolve pronouns and follow-up references across conversation turns.

---

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Google Gemini API key for LLM and embeddings |

---
