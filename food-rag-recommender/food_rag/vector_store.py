"""Chroma DB collection creation and population."""

from __future__ import annotations

from typing import List

from . import config
from .data import FoodItem, build_document, build_metadata, unique_ids


def get_client():
    import chromadb

    if config.CHROMA_PERSIST_DIR:
        return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    return chromadb.EphemeralClient()


def create_collection(name: str = config.COLLECTION_NAME, client=None, description: str = ""):
    """(Re)create a cosine-similarity collection backed by a SentenceTransformer embedder."""
    from chromadb.utils import embedding_functions

    client = client or get_client()
    try:
        client.delete_collection(name)
    except Exception:
        pass  # collection did not exist yet

    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL)
    return client.create_collection(
        name=name,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine", "description": description or "Food recommendation index"},
    )


def populate_collection(collection, foods: List[FoodItem], batch_size: int = 256) -> int:
    """Embed every food item and add it (with filterable metadata) to the collection."""
    ids = unique_ids(foods)
    documents = [build_document(f) for f in foods]
    metadatas = [build_metadata(f) for f in foods]
    for start in range(0, len(foods), batch_size):
        end = start + batch_size
        collection.add(ids=ids[start:end], documents=documents[start:end], metadatas=metadatas[start:end])
    return len(foods)


def build_index(foods: List[FoodItem], name: str = config.COLLECTION_NAME):
    """Convenience: create + populate in one call."""
    collection = create_collection(name)
    populate_collection(collection, foods)
    return collection
