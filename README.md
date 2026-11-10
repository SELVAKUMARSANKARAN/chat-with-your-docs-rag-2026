# Chat With Your Docs — Local RAG MVP

A local PDF/TXT question-answering Streamlit app with source references. No paid API keys required.

## Windows setup

Install Python 3.11 and Ollama from https://ollama.com/download/windows . Then in PowerShell:

```powershell
ollama pull llama3.2:3b
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501. Upload sample.txt, process documents, and ask a question. Ollama must be running in the background. First run downloads the embedding model.

## Architecture

Streamlit UI -> PyMuPDF/pdfplumber PDF extraction -> overlapping character chunks -> SentenceTransformer embeddings -> FAISS cosine similarity -> top-4 retrieved chunks -> local Ollama llama3.2:3b -> grounded response with source IDs.

## Engineering decisions and trade-offs

This starter uses a local small model to avoid API costs, a simple in-memory FAISS index for speed of development, and explicit source metadata for auditability. Chunk size 1000 characters, overlap 150, retrieval top 4. These are initial defaults to evaluate, not guaranteed optimal choices. Document excerpts are treated as untrusted data. The system asks the model to decline unsupported questions, but hallucinations remain possible.

## Limitations and productionization

No OCR for scanned PDFs; PDF table extraction is best-effort. Local in-memory indexes disappear on restart; no authentication, multi-user isolation, or persistent document storage. Local inference performance depends on hardware. Production would add object storage, durable vector DB, background ingestion jobs, auth, observability (latency, retrieval relevance, answer quality), regression evaluations, CI/CD, and deployment to managed containers with model hosting. Do not upload sensitive documents to a public demo without access controls.


