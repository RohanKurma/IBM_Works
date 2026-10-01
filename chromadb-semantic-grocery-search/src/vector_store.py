from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from .data import GROCERY_ITEMS

class GroceryVectorStore:
    def __init__(self, persist_directory="chroma_data",
                 collection_name="grocery_semantic_search",
                 model_name="all-MiniLM-L6-v2"):
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Semantic grocery-search collection"},
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=self.embedding_function,
        )

    def seed(self):
        existing = set(self.collection.get(include=[])["ids"])
        new_items = [x for x in GROCERY_ITEMS if x["id"] not in existing]
        if new_items:
            self.collection.add(
                ids=[x["id"] for x in new_items],
                documents=[x["text"] for x in new_items],
                metadatas=[{"source": "grocery_store", "category": x["category"]} for x in new_items],
            )
        return self.collection.count()

    def search(self, query, n_results=5, category=None):
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        where = {"category": category} if category else None
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        rows = []
        for rank, (doc_id, doc, meta, distance) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ), start=1):
            rows.append({
                "rank": rank,
                "id": doc_id,
                "document": doc,
                "category": meta.get("category"),
                "cosine_distance": float(distance),
                "similarity_proxy": float(1.0 - distance),
            })
        return rows
