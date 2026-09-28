# Portfolio Copy

## Multi-Provider PDF RAG Assistant

Built a provider-agnostic Retrieval-Augmented Generation application that lets users upload PDF documents and ask grounded natural-language questions.

The project implements a complete RAG workflow using Python, LangChain, ChromaDB, PyPDF, and Gradio. It supports IBM watsonx.ai for hosted enterprise inference and Ollama for running open-source LLMs and embedding models locally.

I extended the original coursework implementation into a modular experimentation framework that makes it possible to compare different LLMs, embedding models, chunking strategies, and retrieval parameters while keeping the underlying document pipeline consistent.

### Key capabilities

- PDF document ingestion and preprocessing
- recursive text chunking
- dense vector embeddings
- ChromaDB semantic retrieval
- grounded prompt construction
- IBM watsonx.ai inference
- local Ollama inference
- configurable LLM and embedding profiles
- page-level retrieval references
- interactive Gradio interface
- architecture designed for model comparison

### Technologies

Python · LangChain · IBM watsonx.ai · Ollama · ChromaDB · Gradio · PyPDF · Mistral · Llama · Qwen · Granite Embeddings · Nomic Embeddings
