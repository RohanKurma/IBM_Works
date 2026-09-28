"""
Watsonx.ai + LangChain + Gradio Chatbot

A lightweight generative AI web application that connects an IBM watsonx.ai
foundation model to a Gradio user interface through LangChain's WatsonxLLM wrapper.
"""

import os
import gradio as gr

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import WatsonxLLM


MODEL_ID = os.getenv(
    "WATSONX_MODEL_ID",
    "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
)

WATSONX_URL = os.getenv(
    "WATSONX_URL",
    "https://us-south.ml.cloud.ibm.com",
)

PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "skills-network")

PARAMETERS = {
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.5,
}


def build_llm() -> WatsonxLLM:
    """Create and return the IBM watsonx.ai LangChain LLM wrapper."""
    return WatsonxLLM(
        model_id=MODEL_ID,
        url=WATSONX_URL,
        project_id=PROJECT_ID,
        params=PARAMETERS,
    )


watsonx_llm = build_llm()


def generate_response(prompt_text: str) -> str:
    """Generate a response for a user prompt using IBM watsonx.ai."""
    if not prompt_text or not prompt_text.strip():
        return "Please enter a question."

    try:
        return watsonx_llm.invoke(prompt_text.strip())
    except Exception as exc:
        return (
            "The model could not generate a response. "
            f"Please verify your watsonx.ai configuration. Error: {exc}"
        )


demo = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(
        label="Your Question",
        lines=3,
        placeholder="Ask the AI assistant something...",
    ),
    outputs=gr.Textbox(label="Model Response", lines=8),
    title="IBM watsonx.ai Generative AI Chatbot",
    description=(
        "A simple LLM-powered chatbot built with IBM watsonx.ai, "
        "LangChain, and Gradio."
    ),
    flagging_mode="never",
    examples=[
        ["Explain retrieval-augmented generation in simple terms."],
        ["What is the difference between supervised and unsupervised learning?"],
        ["Give me three practical use cases for generative AI in finance."],
    ],
)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
