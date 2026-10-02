from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from .data import EMPLOYEES, employee_document

class EmployeeVectorStore:
    def __init__(self, persist_directory="chroma_data", collection_name="employee_talent_search", model_name="all-MiniLM-L6-v2"):
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Semantic employee and talent discovery collection"},
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=self.embedding_function,
        )

    def seed(self):
        existing = set(self.collection.get(include=[])["ids"])
        new_items = [e for e in EMPLOYEES if e["id"] not in existing]
        if new_items:
            self.collection.add(
                ids=[e["id"] for e in new_items],
                documents=[employee_document(e) for e in new_items],
                metadatas=[{
                    "name": e["name"], "department": e["department"], "role": e["role"],
                    "experience": e["experience"], "location": e["location"],
                    "employment_type": e["employment_type"]
                } for e in new_items],
            )
        return self.collection.count()

    def semantic_search(self, query, top_k=5, where=None):
        kwargs = {
            "query_texts": [query], "n_results": int(top_k),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        results = self.collection.query(**kwargs)
        rows = []
        for rank, (doc_id, document, metadata, distance) in enumerate(zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        ), start=1):
            rows.append({
                "rank": rank, "id": doc_id, "name": metadata["name"], "role": metadata["role"],
                "department": metadata["department"], "experience": metadata["experience"],
                "location": metadata["location"], "employment_type": metadata["employment_type"],
                "document": document, "cosine_distance": float(distance),
                "similarity_proxy": float(1.0 - distance),
            })
        return rows
