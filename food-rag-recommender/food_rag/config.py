"""Central configuration, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from pathlib import Path

try:  # python-dotenv is optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.getenv("FOOD_DATA_PATH", PROJECT_ROOT / "data" / "FoodDataSet.json"))

# Embeddings / vector store
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "food_recommendations")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")  # unset -> in-memory index

# LLM: "auto" picks watsonx if its keys exist, then OpenAI-compatible, else template fallback
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "400"))

WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID", "ibm/granite-4-h-small")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
