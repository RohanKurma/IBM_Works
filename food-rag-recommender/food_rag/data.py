"""Loading, normalising and serialising the food dataset."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

FoodItem = Dict[str, Any]


def _grams(value: Any) -> float:
    """Turn values like '42g', '3.5 g' or 12 into a float (0.0 if unparseable)."""
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[\d.]+", str(value or ""))
    return float(match.group()) if match else 0.0


def normalize_food_item(item: FoodItem, index: int) -> FoodItem:
    """Fill missing fields and derive a flat taste profile so every item has the same shape."""
    item = dict(item)
    item["food_id"] = str(item.get("food_id", index + 1))
    item.setdefault("food_name", f"Food {item['food_id']}")
    item.setdefault("food_description", "")
    item.setdefault("food_ingredients", [])
    item.setdefault("cuisine_type", "Unknown")
    item.setdefault("cooking_method", "")
    item.setdefault("food_health_benefits", "")
    item["food_calories_per_serving"] = int(item.get("food_calories_per_serving") or 0)

    if isinstance(item["food_ingredients"], str):
        item["food_ingredients"] = [i.strip() for i in item["food_ingredients"].split(",") if i.strip()]

    features = item.get("food_features")
    if isinstance(features, dict):
        item["taste_profile"] = ", ".join(str(v) for v in features.values() if v)
    else:
        item["taste_profile"] = item.get("taste_profile", "")

    nutrition = item.get("food_nutritional_factors")
    item["food_nutritional_factors"] = nutrition if isinstance(nutrition, dict) else {}
    return item


def load_food_data(file_path: str | Path) -> List[FoodItem]:
    """Load the JSON dataset and normalise every record."""
    with open(file_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return [normalize_food_item(item, i) for i, item in enumerate(raw)]


def build_document(food: FoodItem) -> str:
    """Create the rich text that gets embedded for semantic search."""
    parts = [
        f"Name: {food['food_name']}.",
        f"Description: {food.get('food_description', '')}.",
        f"Ingredients: {', '.join(food.get('food_ingredients', []))}.",
        f"Cuisine: {food.get('cuisine_type', 'Unknown')}.",
        f"Cooking method: {food.get('cooking_method', '')}.",
    ]
    if food.get("taste_profile"):
        parts.append(f"Taste and features: {food['taste_profile']}.")
    if food.get("food_health_benefits"):
        parts.append(f"Health benefits: {food['food_health_benefits']}.")
    nutrition = food.get("food_nutritional_factors") or {}
    if nutrition:
        parts.append("Nutrition: " + ", ".join(f"{k}: {v}" for k, v in nutrition.items()) + ".")
    return " ".join(parts)


def build_metadata(food: FoodItem) -> Dict[str, Any]:
    """Flat metadata (Chroma only accepts str/int/float/bool values) used for filtering and display."""
    nutrition = food.get("food_nutritional_factors") or {}
    return {
        "name": food["food_name"],
        "cuisine_type": food.get("cuisine_type", "Unknown"),
        "ingredients": ", ".join(food.get("food_ingredients", [])),
        "calories": int(food.get("food_calories_per_serving", 0)),
        "description": food.get("food_description", ""),
        "cooking_method": food.get("cooking_method", ""),
        "health_benefits": food.get("food_health_benefits", ""),
        "taste_profile": food.get("taste_profile", ""),
        "protein_g": _grams(nutrition.get("protein")),
        "carbs_g": _grams(nutrition.get("carbohydrates")),
        "fat_g": _grams(nutrition.get("fat")),
    }


def unique_ids(foods: List[FoodItem]) -> List[str]:
    """Return one unique ID per item, suffixing duplicates (e.g. '7', '7_1')."""
    seen: set[str] = set()
    ids = []
    for food in foods:
        base = candidate = str(food["food_id"])
        counter = 1
        while candidate in seen:
            candidate = f"{base}_{counter}"
            counter += 1
        seen.add(candidate)
        ids.append(candidate)
    return ids
