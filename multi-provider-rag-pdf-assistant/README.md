# Multi-Provider PDF RAG Assistant

A portfolio-focused **Retrieval-Augmented Generation (RAG)** application for asking questions about PDF documents using either:

- **IBM watsonx.ai** hosted foundation models and embeddings
- **Ollama** local open-source language models and embeddings

The goal of this project is not just to build a PDF chatbot. It is to demonstrate how the **same RAG architecture can be evaluated across multiple model providers**.

> **Project origin:** The first version was developed through hands-on Coursera coursework using IBM watsonx.ai. I then reorganized and extended the project into a provider-agnostic RAG implementation so the same retrieval workflow can be explored with both enterprise-hosted and local open-source models.

---

## Why This Project Matters

Many introductory RAG projects work with only one model provider.

This project intentionally separates:

1. document ingestion
2. chunking
3. embedding generation
4. vector search
5. retrieval
6. prompt construction
7. language-model generation

That separation makes it possible to change the LLM or embedding model without redesigning the entire application.

This is useful for studying trade-offs such as:

- hosted vs local inference
- privacy
- latency
- cost
- model quality
- embedding quality
- hardware requirements
- reproducibility

---

## Architecture

```text
                       PDF DOCUMENT
                            |
                            v
                    PyPDFLoader
                            |
                            v
             Recursive Text Splitter
                            |
                            v
                  Document Chunks
                            |
             +--------------+--------------+
             |                             |
             v                             v
    watsonx Embeddings             Ollama Embeddings
             |                             |
             +--------------+--------------+
                            |
                            v
                        Chroma DB
                            |
                            v
                         Retriever
                            |
                            v
                    Top-K Relevant Chunks
                            |
                            v
                  Grounded Prompt Template
                            |
             +--------------+--------------+
             |                             |
             v                             v
      IBM watsonx.ai LLM             Ollama Local LLM
             |                             |
             +--------------+--------------+
                            |
                            v
                     Generated Answer
                            |
                            v
                     Gradio Interface
```

---

## Supported Profiles

The starter implementation includes three selectable profiles:

| Profile | LLM | Embeddings | Execution |
|---|---|---|---|
| IBM watsonx.ai | Mistral Medium | IBM Granite Embeddings | Hosted |
| Ollama - Llama + Nomic | Llama | Nomic Embed Text | Local |
| Ollama - Qwen + Nomic | Qwen | Nomic Embed Text | Local |

The exact Ollama model names are configurable using environment variables, so users can substitute any compatible locally installed model.

---

## Technology Stack

- Python
- IBM watsonx.ai
- LangChain
- Ollama
- ChromaDB
- Gradio
- PyPDF
- Mistral
- Llama
- Qwen
- IBM Granite Embeddings
- Nomic embeddings

---

## Repository Structure

```text
multi-provider-rag-pdf-assistant/
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
└── src/
    ├── __init__.py
    ├── config.py
    ├── providers.py
    └── rag_pipeline.py
```

---

## RAG Workflow

### 1. PDF ingestion

`PyPDFLoader` extracts text and page metadata from the uploaded PDF.

### 2. Chunking

`RecursiveCharacterTextSplitter` divides the document into overlapping chunks.

Default settings:

```text
chunk_size = 1000
chunk_overlap = 150
```

### 3. Embeddings

The project supports both:

- IBM watsonx.ai embeddings
- Ollama local embeddings

### 4. Vector database

The chunks are stored in ChromaDB as embedding vectors.

### 5. Retrieval

A similarity-based retriever selects the most relevant document chunks.

Default:

```text
top_k = 4
```

### 6. Grounded prompting

Retrieved text is passed to the LLM with explicit instructions to answer only from the supplied document context.

### 7. Answer generation

The selected IBM or Ollama model generates the final answer.

The interface also displays the pages from which the retrieved context originated.

---

## Installation

### Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/multi-provider-rag-pdf-assistant.git
cd multi-provider-rag-pdf-assistant
```

### Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

---

# Option A — IBM watsonx.ai

Configure the required watsonx.ai environment and credentials.

Copy `.env.example` and update the IBM variables appropriate to your account/project.

Example configuration:

```env
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT_ID=your_project_id
```

---

# Option B — Ollama

Install Ollama separately on the machine running the application.

Then download local models.

Example:

```bash
ollama pull llama3.2
ollama pull qwen2.5
ollama pull nomic-embed-text
```

You can then run the entire RAG workflow locally without sending document text to a hosted LLM provider.

---

## Run the Application

```bash
python app.py
```

Open the local Gradio address displayed in the terminal.

---

## Experiments to Try

The repository is designed to encourage experimentation.

### Experiment 1 — LLM comparison

Keep the embeddings fixed and compare:

```text
Llama
Qwen
Mistral
Granite
Gemma
Phi
```

Questions to record:

- Which model follows the retrieved context most closely?
- Which produces the shortest or clearest answer?
- Which hallucinates unsupported information?
- Which is fastest?

### Experiment 2 — Embedding comparison

Keep the LLM fixed while changing the embedding model.

Measure whether the retrieved chunks become more relevant.

### Experiment 3 — Chunk size

Compare:

```text
500
750
1000
1500
```

Observe how chunk size affects retrieval quality.

### Experiment 4 — Chunk overlap

Compare different overlap settings.

### Experiment 5 — Top-K retrieval

Compare:

```text
k = 2
k = 4
k = 6
k = 8
```

### Experiment 6 — Hosted vs local inference

Compare IBM watsonx.ai and Ollama across:

- response quality
- latency
- privacy
- hardware requirements
- cost
- setup complexity

---

## Suggested Evaluation Framework

Create a small benchmark with 10–20 questions for the same PDF.

For every model profile record:

| Metric | Description |
|---|---|
| Groundedness | Is the answer supported by the retrieved text? |
| Answer relevance | Does it directly answer the question? |
| Retrieval relevance | Were the retrieved chunks useful? |
| Latency | How long did the pipeline take? |
| Hallucination count | Did it introduce unsupported facts? |
| Resource usage | CPU/GPU/RAM requirements |
| Deployment mode | Hosted or local |

This turns the project from a demo into an actual **RAG experimentation project**.

---

## Improvements Over the Original Coursework Version

The original learning exercise demonstrated the core IBM watsonx.ai RAG flow.

This portfolio version adds:

- provider-independent architecture
- multiple selectable model profiles
- Ollama local LLM support
- Ollama local embedding support
- explicit grounded prompt template
- source-page reporting
- modular project structure
- environment-based configuration
- easier model experimentation
- clearer GitHub documentation
- evaluation roadmap
- hosted-vs-local comparison scope

---

## Important Fixes From the Starter Code

A few issues were corrected while restructuring the project.

### File handling

When Gradio uses:

```python
type="filepath"
```

the uploaded file is normally passed as a file path string.

Therefore the loader should use:

```python
PyPDFLoader(file_path)
```

rather than relying on:

```python
file.name
```

### RetrievalQA import

The starter code contains:

```python
from langchain.chains import RetrievalQAI
```

The intended class is:

```python
RetrievalQA
```

This implementation avoids coupling the application tightly to that legacy convenience chain and makes the retrieval/prompt/generation steps visible instead.

### Embedding truncation

A truncation value of `3` tokens is far too small for meaningful document embeddings.

The portfolio implementation uses a more realistic configurable value.

---

## Future Roadmap

### Phase 1 — Baseline RAG
- PDF ingestion
- chunking
- embeddings
- ChromaDB
- retrieval
- grounded generation

### Phase 2 — Multi-model RAG
- IBM watsonx.ai
- Ollama
- configurable LLM profiles
- configurable embedding profiles

### Phase 3 — Evaluation
- latency logging
- groundedness scoring
- retrieval precision
- answer-quality benchmark
- experiment result tables

### Phase 4 — Advanced Retrieval
- hybrid search
- reranking
- metadata filtering
- parent-child retrieval
- multi-query retrieval

### Phase 5 — Advanced Product Features
- multi-document collections
- conversational memory
- streaming responses
- source citations
- PDF preview
- model comparison view

### Phase 6 — Production
- persistent vector database
- Docker
- API layer
- caching
- authentication
- observability
- cloud deployment

---

## Portfolio Summary

**Multi-Provider PDF RAG Assistant**

Designed and implemented a provider-agnostic Retrieval-Augmented Generation application using Python, LangChain, ChromaDB, Gradio, IBM watsonx.ai, and Ollama. Built a reusable document-processing pipeline for PDF ingestion, recursive chunking, embedding generation, semantic retrieval, grounded prompting, and LLM response generation.

Extended the original IBM watsonx.ai implementation to support locally hosted open-source LLMs and embedding models through Ollama, enabling comparative experimentation across hosted and local RAG architectures.

---

## Skills Demonstrated

`Python` · `Generative AI` · `Retrieval-Augmented Generation` · `LLMs` · `LangChain` · `IBM watsonx.ai` · `Ollama` · `ChromaDB` · `Vector Embeddings` · `Semantic Search` · `Prompt Engineering` · `Gradio` · `Local AI`

---

## Acknowledgment

The initial IBM watsonx.ai implementation was developed through Coursera hands-on coursework. The repository was subsequently reorganized and extended into a multi-provider RAG experimentation project for independent learning and portfolio development.
