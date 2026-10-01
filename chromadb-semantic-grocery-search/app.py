import pandas as pd
import gradio as gr
from src.search_service import SemanticGrocerySearch

MODEL_CHOICES = ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"]
CATEGORY_CHOICES = [
    "All", "fruit", "bakery", "dairy_eggs", "vegetables",
    "meat", "seafood", "beverages", "pantry"
]

def run_search(query, model_name, category, top_k):
    engine = SemanticGrocerySearch(
        model_name=model_name,
        persist_directory=f"chroma_data/{model_name.replace('/', '_')}",
    )
    selected_category = None if category == "All" else category
    rows = engine.search(query, int(top_k), selected_category)
    return pd.DataFrame([
        {
            "Rank": r["rank"],
            "Item": r["document"],
            "Category": r["category"],
            "Cosine Distance": round(r["cosine_distance"], 4),
            "Similarity": round(r["similarity_proxy"], 4),
        }
        for r in rows
    ])

demo = gr.Interface(
    fn=run_search,
    inputs=[
        gr.Textbox(label="What are you looking for?",
                   value="something fresh and red"),
        gr.Dropdown(MODEL_CHOICES, value=MODEL_CHOICES[0],
                    label="Embedding Model"),
        gr.Dropdown(CATEGORY_CHOICES, value="All",
                    label="Metadata Filter"),
        gr.Slider(1, 8, value=5, step=1, label="Top-K Results"),
    ],
    outputs=gr.Dataframe(label="Semantic Search Results"),
    title="ChromaDB Semantic Grocery Search",
    description="Explore embeddings, HNSW indexing, cosine retrieval, and metadata filters with ChromaDB.",
)

if __name__ == "__main__":
    demo.launch()
