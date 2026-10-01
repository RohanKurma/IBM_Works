# ChromaDB Semantic Grocery Search

An interactive semantic-search project showing how **embeddings + vector databases** can retrieve information based on meaning rather than exact keywords.

> **Project origin:** The initial collection/query workflow came from hands-on coursework. I reorganized and extended it into a reusable semantic-search application for independent experimentation and portfolio presentation.

## Why I built this

After implementing vector similarity manually, I wanted to understand the next layer:

**What changes when embeddings are stored inside an actual vector database?**

This project explores:

- SentenceTransformer embeddings
- ChromaDB collections
- HNSW vector indexing
- cosine distance
- Top-K semantic retrieval
- metadata filtering
- persistent local storage
- interactive Gradio search

## Architecture

```text
Natural-Language Query
        |
        v
SentenceTransformer
        |
        v
Query Embedding
        |
        v
ChromaDB
        |
        v
HNSW Vector Search
        |
        v
Cosine Distance
        |
        v
Top-K Grocery Items
```

## Application Preview

Save your screenshot as:

```text
assets/app-screenshot.png
```

Then add:

```html
<p align="center">
  <img src="assets/app-screenshot.png"
       alt="ChromaDB Semantic Grocery Search"
       width="900">
</p>
```

## Example queries

```text
red
fresh
something fresh and red
healthy yellow fruit
protein for dinner
something for breakfast
```

## Repository Structure

```text
chromadb-semantic-grocery-search/
├── app.py
├── cli.py
├── README.md
├── PORTFOLIO.md
├── EXPERIMENTS.md
├── requirements.txt
├── .gitignore
├── assets/
└── src/
    ├── __init__.py
    ├── data.py
    ├── vector_store.py
    └── search_service.py
```

## Run

```bash
pip install -r requirements.txt
python app.py
```

## Why HNSW?

HNSW is an approximate-nearest-neighbor indexing approach used to efficiently search vector collections as the dataset grows.

This project uses:

```python
configuration={
    "hnsw": {
        "space": "cosine"
    }
}
```

## Metadata filtering

The search can combine semantic retrieval with structured constraints:

```text
Query: "healthy food"
Filter: category = fruit
```

## Connection to RAG

This project is the retrieval layer behind many RAG systems:

```text
Documents
   |
Embeddings
   |
Vector Database
   |
Top-K Retrieval
   |
LLM Context
   |
Grounded Answer
```

## Future Roadmap

- compare cosine, L2, and inner-product spaces
- compare embedding models
- benchmark HNSW settings
- expand to 1,000+ records
- compare ChromaDB with FAISS
- add hybrid keyword + semantic search
- connect results to an LLM for RAG

## Skills Demonstrated

`Python` · `ChromaDB` · `Vector Databases` · `SentenceTransformers` · `Embeddings` · `HNSW` · `Cosine Similarity` · `Semantic Search` · `Metadata Filtering` · `Gradio`

## Acknowledgment

The original ChromaDB exercise was completed through coursework. This repository extends it into a modular semantic-search application for experimentation and portfolio development.
