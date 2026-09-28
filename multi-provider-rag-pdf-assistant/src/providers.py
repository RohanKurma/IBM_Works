import os

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings


def build_llm(profile):
    if profile.provider == "watsonx":
        return WatsonxLLM(
            model_id=profile.llm_model,
            url=os.getenv(
                "WATSONX_URL",
                "https://us-south.ml.cloud.ibm.com",
            ),
            project_id=os.getenv("WATSONX_PROJECT_ID", "skills-network"),
            params={
                GenParams.MAX_NEW_TOKENS: 512,
                GenParams.TEMPERATURE: 0.2,
            },
        )

    if profile.provider == "ollama":
        return ChatOllama(
            model=profile.llm_model,
            temperature=0.2,
        )

    raise ValueError(f"Unsupported LLM provider: {profile.provider}")


def build_embeddings(profile):
    if profile.embedding_provider == "watsonx":
        return WatsonxEmbeddings(
            model_id=profile.embedding_model,
            url=os.getenv(
                "WATSONX_URL",
                "https://us-south.ml.cloud.ibm.com",
            ),
            project_id=os.getenv("WATSONX_PROJECT_ID", "skills-network"),
            params={
                EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 512,
                EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": False},
            },
        )

    if profile.embedding_provider == "ollama":
        return OllamaEmbeddings(model=profile.embedding_model)

    raise ValueError(
        f"Unsupported embedding provider: {profile.embedding_provider}"
    )
