"""Food recommendation system: semantic search, metadata filtering and RAG over Chroma DB."""

from .data import load_food_data
from .rag import FoodRAG
from .search import SearchFilters, semantic_search

__all__ = ["load_food_data", "FoodRAG", "SearchFilters", "semantic_search"]
__version__ = "1.0.0"
