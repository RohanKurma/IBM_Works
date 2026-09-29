import gradio as gr
import pandas as pd

from src.search import SemanticSearchEngine


DEFAULT_DOCUMENTS = """Bugs introduced by the intern had to be squashed by the lead developer.
Bugs found by the quality assurance engineer were difficult to debug.
Bugs are common throughout the warm summer months, according to the entomologist.
Bugs, in particular spiders, are extensively studied by arachnologists."""

MODEL_CHOICES = [
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
]

METRIC_CHOICES = ["cosine", "dot", "euclidean"]


def run_search(documents_text, query, model_name, metric, top_k):
    documents = [
        line.strip()
        for line in documents_text.splitlines()
        if line.strip()
    ]

    engine = SemanticSearchEngine(
        model_name=model_name,
        metric=metric,
    )
    engine.index(documents)
    results = engine.search(query, int(top_k))

    return pd.DataFrame(
        [
            {
                "Rank": result.rank,
                "Score": round(result.score, 6),
                "Document": result.document,
            }
            for result in results
        ]
    )


demo = gr.Interface(
    fn=run_search,
    inputs=[
        gr.Textbox(
            value=DEFAULT_DOCUMENTS,
            label="Documents (one per line)",
            lines=8,
        ),
        gr.Textbox(
            label="Search Query",
            value="Who is responsible for a coding project and fixing others' mistakes?",
            lines=2,
        ),
        gr.Dropdown(
            choices=MODEL_CHOICES,
            value=MODEL_CHOICES[0],
            label="Embedding Model",
        ),
        gr.Radio(
            choices=METRIC_CHOICES,
            value="cosine",
            label="Similarity / Distance Metric",
        ),
        gr.Slider(
            minimum=1,
            maximum=10,
            value=4,
            step=1,
            label="Top-K Results",
        ),
    ],
    outputs=gr.Dataframe(label="Ranked Search Results"),
    title="Semantic Similarity Search Benchmark",
    description=(
        "Explore how embedding models and vector similarity metrics change "
        "semantic-search rankings."
    ),
)


if __name__ == "__main__":
    demo.launch()
