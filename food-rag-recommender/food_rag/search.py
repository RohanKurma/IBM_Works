"""Semantic search over the food collection, with optional metadata filters."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

SearchResult = Dict[str, Any]


@dataclass
class SearchFilters:
    cuisines: List[str] = field(default_factory=list)
    max_calories: Optional[int] = None
    min_protein_g: Optional[float] = None
    cooking_method: Optional[str] = None
    must_include_ingredient: Optional[str] = None

    def is_empty(self) -> bool:
        return not (self.cuisines or self.max_calories or self.min_protein_g
                    or self.cooking_method or self.must_include_ingredient)

    def describe(self) -> str:
        parts = []
        if self.cuisines:
            parts.append("cuisine: " + " / ".join(self.cuisines))
        if self.max_calories:
            parts.append(f"≤ {self.max_calories} kcal")
        if self.min_protein_g:
            parts.append(f"≥ {self.min_protein_g:g} g protein")
        if self.cooking_method:
            parts.append(f"method: {self.cooking_method}")
        if self.must_include_ingredient:
            parts.append(f"contains: {self.must_include_ingredient}")
        return ", ".join(parts) if parts else "no filters"


def build_where_clause(filters: SearchFilters) -> Optional[Dict[str, Any]]:
    """Translate filters into Chroma's `where` syntax, combining conditions with $and."""
    conditions: List[Dict[str, Any]] = []
    if len(filters.cuisines) == 1:
        conditions.append({"cuisine_type": filters.cuisines[0]})
    elif len(filters.cuisines) > 1:
        conditions.append({"cuisine_type": {"$in": list(filters.cuisines)}})
    if filters.max_calories:
        conditions.append({"calories": {"$lte": int(filters.max_calories)}})
    if filters.min_protein_g:
        conditions.append({"protein_g": {"$gte": float(filters.min_protein_g)}})
    if filters.cooking_method:
        conditions.append({"cooking_method": filters.cooking_method})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def format_results(raw: Dict[str, Any]) -> List[SearchResult]:
    """Flatten Chroma's nested query response into a list of result dicts."""
    if not raw or not raw.get("ids") or not raw["ids"][0]:
        return []
    results = []
    for i, food_id in enumerate(raw["ids"][0]):
        meta = raw["metadatas"][0][i]
        distance = raw["distances"][0][i]
        results.append({
            "food_id": food_id,
            "food_name": meta.get("name", ""),
            "food_description": meta.get("description", ""),
            "cuisine_type": meta.get("cuisine_type", "Unknown"),
            "food_calories_per_serving": meta.get("calories", 0),
            "food_ingredients": [s.strip() for s in meta.get("ingredients", "").split(",") if s.strip()],
            "food_health_benefits": meta.get("health_benefits", ""),
            "cooking_method": meta.get("cooking_method", ""),
            "taste_profile": meta.get("taste_profile", ""),
            "protein_g": meta.get("protein_g", 0.0),
            "carbs_g": meta.get("carbs_g", 0.0),
            "fat_g": meta.get("fat_g", 0.0),
            "similarity_score": round(1 - distance, 4),  # cosine distance -> similarity
            "distance": distance,
        })
    return results


def _ingredient_match(result: SearchResult, ingredient: str) -> bool:
    needle = ingredient.lower().strip()
    return any(needle in ing.lower() for ing in result["food_ingredients"])


def semantic_search(collection, query: str, n_results: int = 5,
                    filters: Optional[SearchFilters] = None) -> List[SearchResult]:
    """Similarity search with optional metadata filters.

    Ingredient matching is done as a case-insensitive post-filter, so we over-fetch
    candidates when it is active.
    """
    filters = filters or SearchFilters()
    where = build_where_clause(filters)
    fetch_n = n_results * 5 if filters.must_include_ingredient else n_results
    fetch_n = max(1, min(fetch_n, collection.count()))

    kwargs: Dict[str, Any] = {"query_texts": [query], "n_results": fetch_n}
    if where:
        kwargs["where"] = where
    results = format_results(collection.query(**kwargs))

    if filters.must_include_ingredient:
        results = [r for r in results if _ingredient_match(r, filters.must_include_ingredient)]
    return results[:n_results]


# --------------------------------------------------------------------------------------
# "Self-query" parsing: pull structured constraints out of a natural-language request
# --------------------------------------------------------------------------------------
_CALORIE_PATTERN = re.compile(
    r"(?:under|below|less than|fewer than|max(?:imum)?|at most|<=?|up to)\s*(\d{2,4})\s*(?:k?cal(?:ories)?)?",
    re.IGNORECASE,
)


def parse_query_filters(query: str, known_cuisines: Iterable[str]) -> SearchFilters:
    """Extract a calorie cap and cuisine names mentioned in free text.

    "Italian pasta under 400 calories" -> cuisines=["Italian"], max_calories=400
    """
    filters = SearchFilters()
    match = _CALORIE_PATTERN.search(query)
    if match:
        filters.max_calories = int(match.group(1))
    lowered = query.lower()
    filters.cuisines = [c for c in known_cuisines if re.search(rf"\b{re.escape(c.lower())}\b", lowered)]
    return filters
