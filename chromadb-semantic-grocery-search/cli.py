from src.search_service import SemanticGrocerySearch

def main():
    engine = SemanticGrocerySearch()
    for query in ["red", "fresh", "something fresh and red", "protein for dinner"]:
        print(f"\nQuery: {query}")
        for row in engine.search(query, top_k=3):
            print(
                f"{row['rank']}. {row['document']:<25} "
                f"category={row['category']:<12} "
                f"distance={row['cosine_distance']:.4f}"
            )

if __name__ == "__main__":
    main()
