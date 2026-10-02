"""Retrieval-Augmented Generation: retrieve foods, build context, generate an answer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .llm import LLM
from .search import SearchFilters, SearchResult, parse_query_filters, semantic_search

log = logging.getLogger(__name__)

RECOMMEND_PROMPT = """You are a friendly food recommendation assistant.
Only recommend dishes that appear in the retrieved food information below. Never invent dishes.
{history}
User Query: "{query}"

Retrieved Food Information:
{context}

Please provide a helpful, short response that:
1. Acknowledges the user's request
2. Recommends 2-3 specific food items from the retrieved options
3. Explains why these recommendations match their request
4. Includes relevant details like cuisine type, calories, or health benefits
5. Uses a friendly, conversational tone
6. Keeps the response concise but informative

Response:"""

COMPARE_PROMPT = """You are analyzing and comparing two different food preferences.

Query 1: "{query1}"
Top Results for Query 1:
{context1}

Query 2: "{query2}"
Top Results for Query 2:
{context2}

Please provide a short comparison that:
1. Highlights the key differences between these two food preferences
2. Notes any similarities or overlaps
3. Explains which query might be better for different situations
4. Recommends the best option from each query
5. Keeps the analysis concise but insightful

Comparison:"""


def prepare_context(search_results: Sequence[SearchResult], limit: int = 3) -> str:
    """Turn search hits into a structured block of text for the LLM."""
    if not search_results:
        return "No relevant food items found in the database."
    lines = []
    for i, r in enumerate(search_results[:limit], 1):
        lines.append(f"Option {i}: {r['food_name']}")
        lines.append(f"  - Description: {r['food_description']}")
        lines.append(f"  - Cuisine: {r['cuisine_type']}")
        lines.append(f"  - Calories: {r['food_calories_per_serving']} per serving")
        if r.get("protein_g"):
            lines.append(f"  - Macros: {r['protein_g']:g}g protein, {r['carbs_g']:g}g carbs, {r['fat_g']:g}g fat")
        if r.get("food_ingredients"):
            lines.append(f"  - Key ingredients: {', '.join(r['food_ingredients'][:6])}")
        if r.get("food_health_benefits"):
            lines.append(f"  - Health benefits: {r['food_health_benefits']}")
        if r.get("cooking_method"):
            lines.append(f"  - Cooking method: {r['cooking_method']}")
        if r.get("taste_profile"):
            lines.append(f"  - Taste profile: {r['taste_profile']}")
        lines.append(f"  - Similarity score: {r['similarity_score'] * 100:.1f}%")
        lines.append("")
    return "\n".join(lines).strip()


def fallback_response(query: str, search_results: Sequence[SearchResult]) -> str:
    """Template answer used when no LLM is configured or the LLM call fails."""
    if not search_results:
        return ("I couldn't find any dishes matching that. Try describing a cuisine, an ingredient, "
                "or a mood, for example 'something warm and spicy'.")
    top = search_results[0]
    text = (f"Based on your request for '{query}', I'd recommend **{top['food_name']}**. "
            f"This {top['cuisine_type']} dish has {top['food_calories_per_serving']} calories per serving. "
            f"{top['food_description']}")
    if len(search_results) > 1:
        second = search_results[1]
        text += (f"\n\nAnother great option would be **{second['food_name']}** "
                 f"({second['cuisine_type']}, {second['food_calories_per_serving']} kcal).")
    return text


def simple_comparison(q1: str, q2: str, r1: Sequence[SearchResult], r2: Sequence[SearchResult]) -> str:
    if not r1 and not r2:
        return "No results found for either query."
    if not r1:
        return f"Found results for '{q2}' but none for '{q1}'."
    if not r2:
        return f"Found results for '{q1}' but none for '{q2}'."
    a, b = r1[0], r2[0]
    diff = a["food_calories_per_serving"] - b["food_calories_per_serving"]
    lighter = a if diff < 0 else b
    return (f"For **{q1}**, the top pick is **{a['food_name']}** ({a['cuisine_type']}, "
            f"{a['food_calories_per_serving']} kcal). For **{q2}**, it's **{b['food_name']}** "
            f"({b['cuisine_type']}, {b['food_calories_per_serving']} kcal). "
            f"**{lighter['food_name']}** is the lighter choice by {abs(diff)} calories.")


@dataclass
class RAGAnswer:
    answer: str
    results: List[SearchResult]
    filters: SearchFilters
    used_llm: bool


class FoodRAG:
    """Ties the vector store and the LLM together."""

    def __init__(self, collection, llm: Optional[LLM], known_cuisines: Sequence[str], top_k: int = 3):
        self.collection = collection
        self.llm = llm
        self.known_cuisines = list(known_cuisines)
        self.top_k = top_k

    @property
    def llm_name(self) -> str:
        return self.llm.name if self.llm else "template fallback (no LLM key set)"

    def _generate(self, prompt: str) -> Optional[str]:
        if not self.llm:
            return None
        try:
            text = self.llm.generate(prompt)
            return text if len(text) >= 50 else None
        except Exception as exc:
            log.warning("LLM call failed, using fallback: %s", exc)
            return None

    def retrieve(self, query: str) -> Tuple[List[SearchResult], SearchFilters]:
        """Self-querying retrieval: apply constraints found in the text, relax them if nothing matches."""
        filters = parse_query_filters(query, self.known_cuisines)
        results = semantic_search(self.collection, query, self.top_k, filters)
        if not results and not filters.is_empty():
            filters = SearchFilters()
            results = semantic_search(self.collection, query, self.top_k)
        return results, filters

    def recommend(self, query: str, history: Sequence[Tuple[str, str]] = ()) -> RAGAnswer:
        results, filters = self.retrieve(query)
        history_text = ""
        if history:
            turns = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in list(history)[-3:])
            history_text = f"\nRecent conversation (for context only):\n{turns}\n"
        prompt = RECOMMEND_PROMPT.format(history=history_text, query=query, context=prepare_context(results))
        generated = self._generate(prompt) if results else None
        answer = generated or fallback_response(query, results)
        return RAGAnswer(answer=answer, results=results, filters=filters, used_llm=generated is not None)

    def compare(self, query1: str, query2: str):
        r1, _ = self.retrieve(query1)
        r2, _ = self.retrieve(query2)
        prompt = COMPARE_PROMPT.format(query1=query1, query2=query2,
                                       context1=prepare_context(r1), context2=prepare_context(r2))
        generated = self._generate(prompt) if (r1 and r2) else None
        return generated or simple_comparison(query1, query2, r1, r2), r1, r2
