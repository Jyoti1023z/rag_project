# RAG Project


This project implements a Retrieval-Augmented Generation (RAG) pipeline for legal document question answering using LlamaIndex, ChromaDB, and Gemini (Google Generative AI) via LangChain. It provides ingestion, retrieval, and evaluation of legal case PDFs.

## File Structure
```
rag_project/
│
├── app.py                  # Streamlit RAG chat app
├── ingest.py               # Script to ingest and embed documents
├── evaluate.py             # Script to evaluate RAG performance
├── evaluation_set.py       # Evaluation questions and answers
├── requirements.txt        # Python dependencies
├── rag_eval_results.json   # (Optional) Stores evaluation results
├── .env                    # (Not committed) API keys and secrets
├── data/                   # PDF legal documents
│   └── *.pdf.PDF           # Example: Haryana_Power_Purchase_Centre_...pdf.PDF
├── chroma_db/              # Persistent ChromaDB vector store
│   ├── chroma.sqlite3
│   └── ...
└── data-*.zip              # (Optional) Zipped data backup
```

## Features
- **Document Ingestion:** Loads and embeds PDF documents from the `data/` directory into a persistent ChromaDB vector store using BAAI/bge-m3 embeddings.
- **RAG Chat App:** Streamlit web app for querying the vector store with Gemini LLM, reranking, and context-aware chat.
- **Evaluation:** Automated evaluation of RAG answers using semantic similarity and exact match against a set of legal questions and ground truths.

## Project Structure
- `app.py` — Streamlit app for interactive RAG chat and retrieval.
- `ingest.py` — Script to ingest and embed documents into ChromaDB.
- `evaluate.py` — Script to evaluate RAG performance on a set of legal questions.
- `evaluation_set.py` — Contains evaluation questions and ground truth answers.
- `requirements.txt` — Python dependencies.
- `data/` — Directory for PDF legal documents.
- `chroma_db/` — Persistent ChromaDB vector store.

## Setup
1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables:**
   - Create a `.env` file with your Google API key:
     ```env
     GOOGLE_API_KEY=your_google_api_key_here
     ```
4. **Ingest documents:**
   ```bash
   python ingest.py
   ```
5. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
6. **Evaluate the system:**
   ```bash
   python evaluate.py
   ```

## Requirements
- Python 3.10+
- See `requirements.txt` for all dependencies

## Notes
- Place your legal PDF files in the `data/` directory before running ingestion.
- The vector store is persisted in `chroma_db/`.
- The evaluation set is defined in `evaluation_set.py`.

## License
MIT License
