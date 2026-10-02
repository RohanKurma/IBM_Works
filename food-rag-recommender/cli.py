"""Terminal version of the three systems from the original lab.

    python cli.py search     # interactive similarity search
    python cli.py advanced   # search with cuisine / calorie filters
    python cli.py chat       # RAG chatbot (type 'compare' to compare two queries)
"""

from __future__ import annotations

import argparse

from food_rag import FoodRAG, SearchFilters, load_food_data, semantic_search
from food_rag import config
from food_rag.llm import get_llm
from food_rag.vector_store import build_index


def print_results(results, title: str) -> None:
    print(f"\n📋 {title}\n" + "=" * 60)
    if not results:
        print("❌ No matching foods found. Try different keywords or relax your filters.")
        return
    for i, r in enumerate(results, 1):
        print(f"{i}. 🍽️  {r['food_name']}  ({r['similarity_score'] * 100:.1f}% match)")
        print(f"   🏷️  {r['cuisine_type']}  |  🔥 {r['food_calories_per_serving']} kcal")
        print(f"   📝 {r['food_description']}")


def run_search(collection) -> None:
    print("Type a dish, ingredient or mood. 'quit' to exit.")
    while (query := input("\n🔍 Search for food: ").strip()).lower() not in {"quit", "exit", "q"}:
        if query:
            print_results(semantic_search(collection, query, 5), f"Results for '{query}'")


def run_advanced(collection, cuisines) -> None:
    print("Available cuisines: " + ", ".join(cuisines))
    while True:
        query = input("\n🔍 Query ('quit' to exit): ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            break
        cuisine = input("   Cuisine (optional): ").strip()
        max_cal = input("   Max calories (optional): ").strip()
        filters = SearchFilters(cuisines=[cuisine] if cuisine else [],
                                max_calories=int(max_cal) if max_cal.isdigit() else None)
        print_results(semantic_search(collection, query, 5, filters), f"'{query}' with {filters.describe()}")


def run_chat(rag: FoodRAG) -> None:
    print(f"🤖 RAG chatbot · LLM: {rag.llm_name}\nAsk naturally, type 'compare' or 'quit'.")
    history = []
    while True:
        query = input("\n👤 You: ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            break
        if query.lower() == "compare":
            q1, q2 = input("First query: ").strip(), input("Second query: ").strip()
            analysis, _, _ = rag.compare(q1, q2)
            print(f"\n🤖 {analysis}")
            continue
        if query:
            result = rag.recommend(query, history)
            print(f"\n🤖 {result.answer}")
            print_results(result.results, f"Retrieved context ({result.filters.describe()})")
            history.append((query, result.answer))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["search", "advanced", "chat"])
    args = parser.parse_args()

    foods = load_food_data(config.DATA_PATH)
    print(f"✅ Loaded {len(foods)} food items, building vector index...")
    collection = build_index(foods)
    cuisines = sorted({f["cuisine_type"] for f in foods})

    try:
        if args.mode == "search":
            run_search(collection)
        elif args.mode == "advanced":
            run_advanced(collection, cuisines)
        else:
            run_chat(FoodRAG(collection, get_llm(), cuisines))
    except (KeyboardInterrupt, EOFError):
        pass
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
