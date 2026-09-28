import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RAGProfile:
    name: str
    provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str


PROFILES = {
    "IBM watsonx.ai": RAGProfile(
        name="IBM watsonx.ai",
        provider="watsonx",
        llm_model=os.getenv("WATSONX_LLM_MODEL", "mistralai/mistral-medium-2505"),
        embedding_provider="watsonx",
        embedding_model=os.getenv(
            "WATSONX_EMBED_MODEL",
            "ibm/granite-embedding-278m-multilingual",
        ),
    ),
    "Ollama - Llama + Nomic": RAGProfile(
        name="Ollama - Llama + Nomic",
        provider="ollama",
        llm_model=os.getenv("OLLAMA_LLM_MODEL", "llama3.2"),
        embedding_provider="ollama",
        embedding_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    ),
    "Ollama - Qwen + Nomic": RAGProfile(
        name="Ollama - Qwen + Nomic",
        provider="ollama",
        llm_model=os.getenv("OLLAMA_QWEN_MODEL", "qwen2.5"),
        embedding_provider="ollama",
        embedding_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    ),
}


CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
