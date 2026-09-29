# Semantic Similarity Search & Embedding Benchmark

A portfolio project that explains and demonstrates the mathematics behind **vector similarity search**.

The project starts with the core calculations used in embedding search:

- Euclidean / L2 distance
- dot-product similarity
- vector normalization
- cosine similarity
- cosine distance

It then applies those ideas to **SentenceTransformer embeddings** and builds a small semantic search engine that ranks documents against a natural-language query.

> **Project origin:** The mathematical foundations were studied in a Coursera/IBM Skills Network lab. This repository is an independently reorganized and extended implementation intended for experimentation and portfolio presentation. The original course notebook is not redistributed here.

---

## Why This Project Matters

Vector similarity is a foundational concept behind:

- semantic search
- vector databases
- Retrieval-Augmented Generation (RAG)
- recommendation systems
- clustering
- duplicate detection
- nearest-neighbor retrieval

Rather than treating vector search as a black box, this project implements the most important similarity calculations manually and then connects them to a working search application.

---

## Project Architecture

```text
                    TEXT DOCUMENTS
                          |
                          v
              SentenceTransformer
                          |
                          v
                   Dense Embeddings
                          |
          +---------------+---------------+
          |               |               |
          v               v               v
    Euclidean L2      Dot Product       Cosine
      Distance         Similarity      Similarity
          |               |               |
          +---------------+---------------+
                          |
                          v
                   Ranked Results
                          |
                          v
                     Gradio UI
```

---

## Key Learning Layers

### Layer 1 — Manual Mathematics

The project implements similarity calculations directly from their mathematical definitions.

### Euclidean distance

\[
d(a,b)=\sqrt{\sum_i(a_i-b_i)^2}
\]

### Dot product

\[
a \cdot b = \sum_i a_i b_i
\]

### Cosine similarity

\[
\cos(a,b)=\frac{a \cdot b}{\|a\|\|b\|}
\]

### Cosine distance

\[
d_{cos}=1-\cos(a,b)
\]

---

### Layer 2 — Vectorized Computation

The same operations are implemented using NumPy matrix operations to show why vectorized search is much more efficient than nested Python loops.

---

### Layer 3 — Embeddings

Documents are converted into dense numerical vectors using SentenceTransformers.

Default model:

```text
sentence-transformers/paraphrase-MiniLM-L6-v2
```

The interactive application also exposes other embedding models for comparison.

---

### Layer 4 — Semantic Search

A query is embedded into the same vector space as the documents.

Each document is scored against the query and ranked according to:

- cosine similarity
- dot-product similarity
- Euclidean distance

---

## Ambiguous-Language Test

The starter dataset intentionally uses multiple meanings of the word **"bugs."**

Two sentences refer to software bugs, while two refer to insects/arthropods.

This makes the dataset useful for demonstrating that semantic embeddings can use context rather than relying only on shared keywords.

Example query:

```text
Who is responsible for a coding project and fixing others' mistakes?
```

A useful semantic-search system should favor the software-development sentence even though the wording of the query differs substantially from the document.

---

## Repository Structure

```text
semantic-similarity-search-benchmark/
├── app.py
├── README.md
├── PORTFOLIO.md
├── EXPERIMENTS.md
├── requirements.txt
├── .gitignore
├── data/
│   └── sample_documents.txt
└── src/
    ├── __init__.py
    ├── metrics.py
    ├── search.py
    └── benchmark.py
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/semantic-similarity-search-benchmark.git
cd semantic-similarity-search-benchmark
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it and install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Interactive Search App

```bash
python app.py
```

The Gradio application lets you change:

- the document collection
- search query
- embedding model
- similarity metric
- top-k result count

---

## Model Experiments

The application is structured so you can compare several SentenceTransformer models, for example:

```text
paraphrase-MiniLM-L6-v2
all-MiniLM-L6-v2
multi-qa-MiniLM-L6-cos-v1
```

Do not simply claim that all models were tested. Record actual runs in `EXPERIMENTS.md`.

---

## Recommended Experiments

### 1. Metric comparison

Keep the embedding model fixed and compare:

```text
Cosine similarity
Dot product
Euclidean distance
```

### 2. Normalization experiment

Compare dot-product rankings before and after unit normalization.

For normalized vectors:

```text
dot product ≈ cosine similarity
```

This provides a useful mathematical demonstration of why many embedding systems normalize vectors before retrieval.

### 3. Embedding-model comparison

Keep the same dataset and queries while changing the SentenceTransformer model.

### 4. Query paraphrasing

Rewrite the same information need in several ways and observe ranking stability.

### 5. Ambiguous vocabulary

Create documents where the same words have different meanings.

Examples:

```text
bank → financial institution / river bank
python → programming language / snake
java → programming language / coffee / island
apple → fruit / technology company
```

### 6. Corpus scaling

Expand from four documents to:

```text
10
100
1,000
10,000
```

Measure search latency and discuss when a dedicated vector index becomes useful.

---

## From This Project to RAG

This project directly explains the retrieval foundation of a RAG system:

```text
User Query
    |
    v
Query Embedding
    |
    v
Vector Similarity Search
    |
    v
Top-K Documents
    |
    v
LLM Context
    |
    v
Generated Answer
```

Understanding this layer makes systems such as Chroma, FAISS, Pinecone, Milvus, Weaviate, and other vector-search engines easier to reason about.

---

## Future Roadmap

### Phase 1 — Mathematical foundations
- manual Euclidean distance
- manual dot product
- manual normalization
- manual cosine similarity

### Phase 2 — Semantic search engine
- document embeddings
- query embeddings
- ranking
- top-k retrieval

### Phase 3 — Model benchmark
- multiple embedding models
- evaluation queries
- Hit@1
- Hit@K
- latency comparison

### Phase 4 — Larger-scale retrieval
- FAISS
- Chroma
- approximate nearest neighbors
- persistent indexes

### Phase 5 — Retrieval evaluation
- Recall@K
- Precision@K
- Mean Reciprocal Rank
- nDCG

### Phase 6 — RAG integration
- retrieve context
- pass top-k documents to an LLM
- grounded answers
- citations

---

## Skills Demonstrated

`Python` · `Vector Embeddings` · `Semantic Search` · `Information Retrieval` · `Sentence Transformers` · `NumPy` · `SciPy` · `PyTorch` · `Cosine Similarity` · `Vector Mathematics` · `Gradio`

---

## Acknowledgment

The foundational similarity-search concepts were studied through a Coursera/IBM Skills Network exercise. This repository is an independently structured implementation and extension for learning, experimentation, and portfolio use.
