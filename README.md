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

1. Chunking Strategy

Current implementation: Documents are split into overlapping text chunks using a default chunk size of 1,000 characters and an overlap of 150 characters. The splitting logic attempts to preserve newline boundaries where possible. Each chunk retains its filename, page number, and content type.

Rationale: Chunking helps retrieve relevant portions of large documents without passing the entire document to the language model. Overlap reduces the risk of losing information across chunk boundaries.

2. Embedding Model & LLM Selection

Embedding model: SentenceTransformers generates dense vector embeddings locally. Embeddings are L2-normalized before indexing.

LLM: Ollama with llama3.2:3b provides local language-model inference.
Rationale: The selected stack avoids dependency on paid external inference APIs and is practical for a lightweight proof of concept.    

Trade-offs: Smaller local LLMs may have weaker reasoning capabilities than larger models. SentenceTransformer models also differ in retrieval accuracy, speed, dimensionality, and memory requirements.

3. Retrieval Approach

Current implementation: FAISS IndexFlatIP performs exact inner-product similarity search over normalized vectors, effectively ranking by cosine similarity. The top four matching chunks are retrieved for each question.

Rationale: FAISS is lightweight, fast, and suitable for a small local document collection. Exact search also provides a straightforward baseline before introducing approximate nearest-neighbor indexing.

Future improvement: Introduce metadata filtering, hybrid keyword-plus-vector retrieval, reranking, and relevance thresholds. For larger collections, evaluate persistent vector databases and scalable indexing strategies.

4. Prompt Engineering

Current implementation: The system prompt instructs the LLM to answer using only retrieved document excerpts, avoid inventing facts, and decline unsupported questions. Retrieved excerpts include source metadata. The model uses temperature zero to favor more consistent responses.

5. Context Management

Current implementation: The system retrieves the four most relevant document chunks rather than sending entire documents to the LLM. It also includes up to four recent chat-history messages. Retrieval uses the current user question rather than the entire conversation.

Rationale: This keeps the prompt smaller and reduces irrelevant context.

Trade-offs: Follow-up questions that depend on earlier conversation context may retrieve less relevant information because the retrieval query is not rewritten using chat history.

6. Guardrails

Current implementation: The application supports PDF and TXT documents, and the system prompt treats retrieved document text as untrusted content. It instructs the model not to follow instructions embedded in documents and to decline questions unsupported by the retrieved context.

Limitations: Prompt instructions alone cannot guarantee protection against prompt injection or hallucinations.

Future improvement: Add document validation, content and access policies, prompt-injection testing, relevance-based abstention, and safeguards against exposing information between users.

7. Quality Controls

Current implementation: The project includes a basic automated test for text chunking. Retrieved chunks are returned alongside generated answers, allowing inspection of supporting context.

Future improvement: Build a representative evaluation dataset containing questions, expected answers, and relevant document passages. Measure:

Recall@K: Whether relevant evidence appears in the retrieved results.

Precision@K: How much retrieved content is relevant.

Faithfulness: Whether answers are supported by retrieved passages.

Answer correctness: Whether the response addresses the question accurately.

Abstention accuracy: Whether the chatbot declines questions that the documents cannot answer.

8. Observability

Current implementation: Basic Python logging records the number of retrieved matches for answered questions.

Future improvement: Introduce structured monitoring for document extraction, chunk counts, embedding time, retrieval latency, LLM response latency, error rates, and resource utilization.

Track quality trends across prompt and model versions. Avoid recording sensitive document content or personal information in application logs.

Architectural Trade-offs and Next Step

The solution intentionally prioritizes simplicity, local execution, transparency, and ease of evaluation.

## Limitations and productionization

No OCR for scanned PDFs; PDF table extraction is best-effort. Local in-memory indexes disappear on restart; no authentication, multi-user isolation, or persistent document storage. Local inference performance depends on hardware. Production would add object storage, durable vector DB, background ingestion jobs, auth, observability (latency, retrieval relevance, answer quality), regression evaluations, CI/CD, and deployment to managed containers with model hosting. Do not upload sensitive documents to a public demo without access controls.


