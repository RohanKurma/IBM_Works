from src.search_service import TalentSearchEngine

def main():
    engine = TalentSearchEngine()
    cases = [
        {"query": "Python developer with web development experience"},
        {"query": "team leader manager with experience", "min_experience": 8},
        {"query": "senior Python developer full-stack", "min_experience": 8, "locations": ["San Francisco", "New York", "Seattle"]},
    ]
    for case in cases:
        print(f"\nQuery: {case['query']}")
        print("-" * 72)
        for row in engine.search(top_k=5, **case):
            print(f"{row['rank']}. {row['name']} — {row['role']} | {row['experience']} yrs | {row['location']} | similarity={row['similarity_proxy']:.4f}")

if __name__ == "__main__":
    main()
