from .vector_store import GroceryVectorStore

class SemanticGrocerySearch:
    def __init__(self, model_name="all-MiniLM-L6-v2", persist_directory="chroma_data"):
        self.store = GroceryVectorStore(
            model_name=model_name,
            persist_directory=persist_directory,
        )
        self.store.seed()

    def search(self, query, top_k=5, category=None):
        return self.store.search(query=query, n_results=top_k, category=category)
