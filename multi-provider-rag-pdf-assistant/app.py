import gradio as gr

from src.rag_pipeline import answer_question, available_profiles


def run_rag(file_path, query, profile):
    if not file_path:
        return "Please upload a PDF document."
    if not query or not query.strip():
        return "Please enter a question."
    return answer_question(file_path, query.strip(), profile)


profiles = available_profiles()

demo = gr.Interface(
    fn=run_rag,
    inputs=[
        gr.File(
            label="Upload PDF",
            file_count="single",
            file_types=[".pdf"],
            type="filepath",
        ),
        gr.Textbox(
            label="Question",
            lines=2,
            placeholder="Ask a question about the uploaded PDF...",
        ),
        gr.Dropdown(
            choices=profiles,
            value=profiles[0],
            label="Model / Embedding Profile",
        ),
    ],
    outputs=gr.Textbox(label="Answer", lines=10),
    title="Multi-Provider PDF RAG Assistant",
    description=(
        "Compare Retrieval-Augmented Generation using IBM watsonx.ai "
        "and local open-source Ollama models."
    ),
    examples=[
        [None, "Summarize the document in five bullet points.", profiles[0]],
        [None, "What are the main conclusions?", profiles[0]],
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
