from .vector_store import EmployeeVectorStore

class TalentSearchEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2", persist_directory="chroma_data"):
        self.store = EmployeeVectorStore(model_name=model_name, persist_directory=persist_directory)
        self.store.seed()

    def search(self, query, top_k=5, department=None, min_experience=None, locations=None, employment_type=None):
        clauses = []
        if department:
            clauses.append({"department": department})
        if min_experience is not None:
            clauses.append({"experience": {"$gte": int(min_experience)}})
        if locations:
            clauses.append({"location": {"$in": locations}})
        if employment_type:
            clauses.append({"employment_type": employment_type})
        if len(clauses) == 1:
            where = clauses[0]
        elif len(clauses) > 1:
            where = {"$and": clauses}
        else:
            where = None
        return self.store.semantic_search(query=query, top_k=top_k, where=where)
