import pandas as pd
import gradio as gr

from src.search_service import TalentSearchEngine


MODEL_CHOICES = [
    "all-MiniLM-L6-v2",
    "paraphrase-MiniLM-L6-v2",
]

DEPARTMENTS = ["Any", "Engineering", "Marketing", "HR"]
EMPLOYMENT_TYPES = ["Any", "Full-time", "Part-time"]
LOCATION_CHOICES = [
    "New York", "Los Angeles", "Chicago", "San Francisco", "Austin",
    "Seattle", "Boston", "Miami", "Denver", "Portland", "Phoenix",
    "Atlanta", "Dallas",
]

QUICK_SEARCHES = {
    "Python Leader": "senior Python developer with leadership experience",
    "Cloud Architect": "experienced cloud architect with distributed systems knowledge",
    "Engineering Manager": "engineering manager with mentoring and project leadership",
    "DevOps Expert": "DevOps engineer with AWS Kubernetes and infrastructure automation",
}


CUSTOM_CSS = """
.gradio-container {
    max-width: 1450px !important;
    margin: 0 auto !important;
    background:
        radial-gradient(circle at top left, rgba(99,102,241,.09), transparent 28%),
        radial-gradient(circle at top right, rgba(14,165,233,.08), transparent 24%);
}

.hero {
    padding: 30px 34px;
    border-radius: 24px;
    margin-bottom: 18px;
    background: linear-gradient(135deg, #0f172a 0%, #172554 48%, #312e81 100%);
    color: white;
    box-shadow: 0 18px 45px rgba(15, 23, 42, .18);
}

.hero h1 {
    margin: 0;
    font-size: 2.35rem;
    line-height: 1.05;
    letter-spacing: -0.04em;
}

.hero p {
    margin: 10px 0 0;
    color: #cbd5e1;
    font-size: 1.02rem;
    max-width: 850px;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 18px;
}

.badge {
    display: inline-block;
    padding: 6px 11px;
    border-radius: 999px;
    background: rgba(255,255,255,.10);
    border: 1px solid rgba(255,255,255,.14);
    color: #e2e8f0;
    font-size: .83rem;
}

.panel {
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 16px;
    background: rgba(255,255,255,.88);
    box-shadow: 0 8px 28px rgba(15, 23, 42, .05);
}

.metric-card {
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 18px;
    background: white;
    min-height: 104px;
    box-shadow: 0 6px 22px rgba(15, 23, 42, .05);
}

.metric-label {
    font-size: .76rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: #64748b;
    font-weight: 700;
}

.metric-value {
    margin-top: 6px;
    font-size: 1.75rem;
    font-weight: 800;
    color: #0f172a;
}

.metric-sub {
    font-size: .82rem;
    color: #64748b;
    margin-top: 2px;
}

.section-title h3 {
    margin-bottom: 4px;
}

.section-title p {
    margin-top: 0;
    color: #64748b;
}

#search-button {
    min-height: 48px;
    font-weight: 750;
    font-size: 1rem;
}

#query-box textarea {
    font-size: 1.02rem !important;
}

footer {
    display: none !important;
}
"""


def build_engine(model_name):
    return TalentSearchEngine(
        model_name=model_name,
        persist_directory=f"chroma_data/{model_name.replace('/', '_')}",
    )


def run_search(
    query,
    model_name,
    department,
    min_experience,
    locations,
    employment_type,
    top_k,
):
    if not query or not query.strip():
        empty = pd.DataFrame(
            columns=[
                "Rank", "Name", "Role", "Department",
                "Experience", "Location", "Employment", "Similarity"
            ]
        )
        return empty, metric_html(0, "—", "—"), "Enter a search query to begin."

    engine = build_engine(model_name)

    rows = engine.search(
        query=query.strip(),
        top_k=int(top_k),
        department=None if department == "Any" else department,
        min_experience=None if int(min_experience) == 0 else int(min_experience),
        locations=locations or None,
        employment_type=None if employment_type == "Any" else employment_type,
    )

    df = pd.DataFrame(
        [
            {
                "Rank": r["rank"],
                "Name": r["name"],
                "Role": r["role"],
                "Department": r["department"],
                "Experience": f"{r['experience']} yrs",
                "Location": r["location"],
                "Employment": r["employment_type"],
                "Similarity": round(r["similarity_proxy"], 4),
            }
            for r in rows
        ]
    )

    best_name = rows[0]["name"] if rows else "—"
    best_score = f"{rows[0]['similarity_proxy']:.3f}" if rows else "—"

    summary = (
        f"Found **{len(rows)}** semantic matches for "
        f"**“{query.strip()}”** using **{model_name}**."
    )

    return df, metric_html(len(rows), best_name, best_score), summary


def metric_html(count, best_match, best_score):
    return f"""
    <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;">
        <div class="metric-card">
            <div class="metric-label">Matches returned</div>
            <div class="metric-value">{count}</div>
            <div class="metric-sub">Top-K semantic results</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Best match</div>
            <div class="metric-value" style="font-size:1.18rem">{best_match}</div>
            <div class="metric-sub">Highest semantic relevance</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Top similarity</div>
            <div class="metric-value">{best_score}</div>
            <div class="metric-sub">1 − cosine distance</div>
        </div>
    </div>
    """


def use_quick_search(label):
    return QUICK_SEARCHES[label]


with gr.Blocks(css=CUSTOM_CSS, title="TalentIQ — Semantic Talent Search") as demo:

    gr.HTML("""
    <div class="hero">
        <h1>TalentIQ</h1>
        <p>
            Semantic talent discovery powered by SentenceTransformers + ChromaDB.
            Describe the person you need in natural language, then refine results
            with real business constraints.
        </p>
        <div class="badge-row">
            <span class="badge">Vector Search</span>
            <span class="badge">ChromaDB</span>
            <span class="badge">HNSW</span>
            <span class="badge">SentenceTransformers</span>
            <span class="badge">Metadata Filtering</span>
        </div>
    </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=320):
            with gr.Group(elem_classes="panel"):
                gr.Markdown(
                    "### Search talent\n"
                    "Describe the skills, experience, or type of professional you need."
                )

                query = gr.Textbox(
                    label="Talent requirement",
                    value="senior Python developer with leadership experience",
                    placeholder="Example: cloud architect with mentoring experience",
                    lines=3,
                    elem_id="query-box",
                )

                with gr.Row():
                    quick = gr.Dropdown(
                        choices=list(QUICK_SEARCHES.keys()),
                        label="Quick searches",
                        value=None,
                    )
                    model = gr.Dropdown(
                        choices=MODEL_CHOICES,
                        value=MODEL_CHOICES[0],
                        label="Embedding model",
                    )

                gr.Markdown("#### Refine results")

                department = gr.Dropdown(
                    DEPARTMENTS,
                    value="Any",
                    label="Department",
                )

                min_experience = gr.Slider(
                    minimum=0,
                    maximum=20,
                    value=0,
                    step=1,
                    label="Minimum experience",
                )

                locations = gr.Dropdown(
                    LOCATION_CHOICES,
                    multiselect=True,
                    label="Locations",
                    info="Optional — leave empty to search all locations.",
                )

                employment_type = gr.Dropdown(
                    EMPLOYMENT_TYPES,
                    value="Any",
                    label="Employment type",
                )

                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of results",
                )

                search_button = gr.Button(
                    "Find Talent",
                    variant="primary",
                    elem_id="search-button",
                )

        with gr.Column(scale=8, min_width=600):
            gr.Markdown(
                "### Talent Intelligence\n"
                "Ranked candidates based on semantic relevance and selected constraints.",
                elem_classes="section-title",
            )

            metrics = gr.HTML(metric_html(0, "—", "—"))

            search_summary = gr.Markdown(
                "Enter a query and click **Find Talent**."
            )

            results = gr.Dataframe(
                headers=[
                    "Rank", "Name", "Role", "Department",
                    "Experience", "Location", "Employment", "Similarity"
                ],
                interactive=False,
                wrap=True,
                label="Ranked candidates",
                max_height=480,
            )

            with gr.Accordion("How this search works", open=False):
                gr.Markdown("""
**1. Query embedding**  
Your natural-language request is converted into a dense vector using a SentenceTransformer model.

**2. Vector retrieval**  
ChromaDB searches the employee-profile embeddings using cosine-based HNSW retrieval.

**3. Metadata constraints**  
Department, experience, location, and employment type are applied as structured filters.

**4. Ranked results**  
The app returns the Top-K candidates that satisfy both semantic relevance and business constraints.
                """)

            with gr.Accordion("Example real-world use cases", open=False):
                gr.Markdown("""
- Internal talent discovery
- Project staffing
- Recruiting candidate retrieval
- Expert finder systems
- Mentorship matching
- Workforce planning
- Internal mobility
                """)

    quick.change(
        fn=use_quick_search,
        inputs=quick,
        outputs=query,
    )

    search_button.click(
        fn=run_search,
        inputs=[
            query,
            model,
            department,
            min_experience,
            locations,
            employment_type,
            top_k,
        ],
        outputs=[results, metrics, search_summary],
    )

    query.submit(
        fn=run_search,
        inputs=[
            query,
            model,
            department,
            min_experience,
            locations,
            employment_type,
            top_k,
        ],
        outputs=[results, metrics, search_summary],
    )

    gr.HTML("""
    <div style="text-align:center;color:#64748b;padding:22px 0 8px;font-size:.86rem;">
        TalentIQ • Semantic Talent Discovery • ChromaDB + SentenceTransformers
    </div>
    """)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Glass())
