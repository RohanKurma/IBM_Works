"""Gradio web app for the Food Recommendation RAG system.

Run locally:   python app.py   (then open http://127.0.0.1:7860)
"""

from __future__ import annotations

import inspect
import logging

import gradio as gr

from food_rag import FoodRAG, SearchFilters, config, load_food_data, semantic_search
from food_rag.llm import get_llm
from food_rag.render import CSS, render_results
from food_rag.vector_store import build_index

logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------------------------------
# One-time startup: load data, embed it into Chroma, connect the LLM
# ------------------------------------------------------------------------------------------
FOODS = load_food_data(config.DATA_PATH)
COLLECTION = build_index(FOODS)
CUISINES = sorted({f["cuisine_type"] for f in FOODS})
METHODS = sorted({f["cooking_method"] for f in FOODS if f["cooking_method"]})
MAX_KCAL = max(f["food_calories_per_serving"] for f in FOODS)
RAG = FoodRAG(COLLECTION, get_llm(), CUISINES)


# ------------------------------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------------------------------
def basic_search(query: str, n_results: int):
    if not query.strip():
        return render_results([], "Type something to search, e.g. 'chocolate dessert'.")
    return render_results(semantic_search(COLLECTION, query, int(n_results)))


def advanced_search(query, cuisines, max_kcal, min_protein, method, ingredient, n_results):
    if not query.strip():
        return render_results([], "Enter a search query."), ""
    filters = SearchFilters(
        cuisines=list(cuisines or []),
        max_calories=int(max_kcal) if max_kcal and max_kcal < MAX_KCAL else None,
        min_protein_g=float(min_protein) if min_protein else None,
        cooking_method=method or None,
        must_include_ingredient=ingredient.strip() or None,
    )
    results = semantic_search(COLLECTION, query, int(n_results), filters)
    status = f"**Filters applied:** {filters.describe()} · **{len(results)}** result(s)"
    return render_results(results), status


def _text(content) -> str:
    """Message content is a str in Gradio 5 and may be a list of parts in Gradio 6."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def chat(message: str, history: list):
    history = history or []
    if not message.strip():
        return history, "", gr.update()
    # Turn the messages history into (user, assistant) pairs for the prompt
    pairs, pending = [], None
    for m in history:
        if m["role"] == "user":
            pending = _text(m["content"])
        elif pending is not None:
            pairs.append((pending, _text(m["content"])))
            pending = None

    result = RAG.recommend(message, pairs)
    source = "🧠 LLM" if result.used_llm else "📋 template"
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result.answer},
    ]
    context_html = (
        f'<div class="status-line">Retrieved with <b>{result.filters.describe()}</b> · answer by {source}</div>'
        + render_results(result.results)
    )
    return history, "", context_html


def compare(q1: str, q2: str):
    if not q1.strip() or not q2.strip():
        return "Please enter both queries.", "", ""
    analysis, r1, r2 = RAG.compare(q1, q2)
    return analysis, render_results(r1), render_results(r2)


# ------------------------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------------------------
HOW_IT_WORKS = f"""
### Architecture

```
FoodDataSet.json ─▶ normalise ─▶ rich text document + flat metadata
                                       │
                         all-MiniLM-L6-v2 embeddings
                                       ▼
                     Chroma DB collection (cosine / HNSW)
                                       │
   user query ─▶ self-query parser ─▶ similarity search + `where` filters
                                       │ top-k foods
                                       ▼
                 prompt = instructions + retrieved context + query
                                       ▼
                 LLM (IBM Granite / OpenAI-compatible / template)
                                       ▼
                       grounded recommendation + sources
```

| Tab | Technique |
|---|---|
| 🔍 Discover | Pure vector similarity search |
| 🎛️ Advanced Search | Similarity search + Chroma metadata filters (`$and`, `$in`, `$lte`, `$gte`) + ingredient post-filter |
| 🤖 AI Food Chat | RAG: retrieval → context building → augmented generation, with constraints parsed from the question |
| ⚖️ Compare | Two retrievals + one LLM call that contrasts them |

**Current setup:** {len(FOODS)} dishes · {len(CUISINES)} cuisines · embeddings `{config.EMBEDDING_MODEL}` · LLM `{RAG.llm_name}`
"""


def _supports(fn, name: str) -> bool:
    return name in inspect.signature(fn).parameters


theme = gr.themes.Soft(primary_hue="orange", secondary_hue="green")
blocks_kwargs = {"title": "Food RAG Recommender"}
launch_kwargs = {}
# Gradio 5 takes theme/css on Blocks; Gradio 6 moved them to launch()
target = blocks_kwargs if _supports(gr.Blocks.__init__, "css") else launch_kwargs
target.update(theme=theme, css=CSS)
chatbot_kwargs = {"type": "messages"} if _supports(gr.Chatbot.__init__, "type") else {}

with gr.Blocks(**blocks_kwargs) as demo:
    gr.Markdown(
        "# 🍽️ Food Recommendation RAG\n"
        "Semantic food search, metadata filtering and a retrieval-augmented chatbot, powered by "
        f"**Chroma DB** + **Sentence Transformers**. \n"
        f'<span class="status-line">{len(FOODS)} dishes indexed · {len(CUISINES)} cuisines · '
        f"LLM: {RAG.llm_name}</span>"
    )

    with gr.Tab("🔍 Discover"):
        with gr.Row():
            q_basic = gr.Textbox(label="What are you craving?", placeholder="e.g. creamy pasta, sweet treats",
                                 scale=4)
            n_basic = gr.Slider(1, 10, value=5, step=1, label="Results", scale=1)
        btn_basic = gr.Button("Search", variant="primary")
        out_basic = gr.HTML(render_results([], "Results will appear here."))
        gr.Examples(["chocolate dessert", "Italian food", "sweet treats", "baked goods", "low calorie",
                     "spicy noodles"], inputs=q_basic)
        btn_basic.click(basic_search, [q_basic, n_basic], out_basic)
        q_basic.submit(basic_search, [q_basic, n_basic], out_basic)

    with gr.Tab("🎛️ Advanced Search"):
        with gr.Row():
            with gr.Column(scale=1):
                q_adv = gr.Textbox(label="Search query", placeholder="e.g. healthy meal")
                cuisines_in = gr.Dropdown(CUISINES, multiselect=True, label="Cuisine(s)")
                kcal_in = gr.Slider(50, MAX_KCAL, value=MAX_KCAL, step=10,
                                    label="Max calories per serving (far right = no limit)")
                protein_in = gr.Slider(0, 50, value=0, step=1, label="Min protein (g)")
                method_in = gr.Dropdown([""] + METHODS, value="", label="Cooking method")
                ingredient_in = gr.Textbox(label="Must include ingredient", placeholder="e.g. chicken")
                n_adv = gr.Slider(1, 10, value=5, step=1, label="Results")
                btn_adv = gr.Button("Search with filters", variant="primary")
            with gr.Column(scale=2):
                status_adv = gr.Markdown()
                out_adv = gr.HTML(render_results([], "Set your filters and search."))
        gr.Examples(
            [["creamy pasta", ["Italian"], MAX_KCAL, 0, "", "", 3],
             ["healthy meal", [], 300, 0, "", "", 3],
             ["light fresh meal", ["Japanese"], 250, 0, "", "", 3],
             ["high protein dinner", [], 600, 25, "", "", 5]],
            inputs=[q_adv, cuisines_in, kcal_in, protein_in, method_in, ingredient_in, n_adv],
            label="Demo searches (from the original lab's demonstration mode)",
        )
        adv_inputs = [q_adv, cuisines_in, kcal_in, protein_in, method_in, ingredient_in, n_adv]
        btn_adv.click(advanced_search, adv_inputs, [out_adv, status_adv])
        q_adv.submit(advanced_search, adv_inputs, [out_adv, status_adv])

    with gr.Tab("🤖 AI Food Chat"):
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Food assistant", height=460, **chatbot_kwargs)
                with gr.Row():
                    msg = gr.Textbox(placeholder="I want something spicy and healthy for dinner",
                                     show_label=False, scale=5)
                    send = gr.Button("Send", variant="primary", scale=1)
                clear = gr.Button("Clear conversation", size="sm")
                gr.Examples(["I want something healthy and light for lunch",
                             "What Italian dishes do you recommend under 400 calories?",
                             "I'm craving comfort food for a cold evening",
                             "Protein-rich options for workout recovery",
                             "Celebratory foods for a party"], inputs=msg)
            with gr.Column(scale=2):
                gr.Markdown("#### 📚 Retrieved context")
                context_out = gr.HTML(render_results([], "The dishes the answer is grounded in show up here."))
        send.click(chat, [msg, chatbot], [chatbot, msg, context_out])
        msg.submit(chat, [msg, chatbot], [chatbot, msg, context_out])
        clear.click(lambda: ([], "", render_results([], "Conversation cleared.")), None,
                    [chatbot, msg, context_out])

    with gr.Tab("⚖️ Compare"):
        with gr.Row():
            cq1 = gr.Textbox(label="First craving", value="chocolate dessert")
            cq2 = gr.Textbox(label="Second craving", value="healthy breakfast")
        btn_cmp = gr.Button("Compare", variant="primary")
        cmp_text = gr.Markdown()
        with gr.Row():
            cmp_left = gr.HTML()
            cmp_right = gr.HTML()
        btn_cmp.click(compare, [cq1, cq2], [cmp_text, cmp_left, cmp_right])

    with gr.Tab("ℹ️ How it works"):
        gr.Markdown(HOW_IT_WORKS)


if __name__ == "__main__":
    demo.launch(**launch_kwargs)
