from pathlib import Path
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

from .config import PROFILES, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from .providers import build_llm, build_embeddings


PROMPT = PromptTemplate.from_template(
    """You are a document-grounded assistant.

Use only the supplied context to answer the question.
If the answer is not supported by the context, say:
"I could not find that information in the uploaded document."

Context:
{context}

Question:
{question}

Answer:"""
)


def available_profiles():
    return list(PROFILES.keys())


def _load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()


def _split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(documents)


def _collection_name(file_path: str, profile_name: str):
    key = f"{Path(file_path).name}:{profile_name}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()[:12]
    return f"rag_{digest}"


def _format_docs(docs):
    parts = []
    for doc in docs:
        page = doc.metadata.get("page")
        page_label = f"Page {page + 1}" if isinstance(page, int) else "Unknown page"
        parts.append(f"[{page_label}]\n{doc.page_content}")
    return "\n\n".join(parts)


def answer_question(file_path: str, query: str, profile_name: str):
    profile = PROFILES[profile_name]

    documents = _load_pdf(file_path)
    chunks = _split_documents(documents)

    embeddings = build_embeddings(profile)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=_collection_name(file_path, profile_name),
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": TOP_K})
    retrieved_docs = retriever.invoke(query)

    context = _format_docs(retrieved_docs)
    prompt = PROMPT.format(context=context, question=query)

    llm = build_llm(profile)
    response = llm.invoke(prompt)

    if hasattr(response, "content"):
        answer = response.content
    else:
        answer = str(response)

    source_pages = []
    for doc in retrieved_docs:
        page = doc.metadata.get("page")
        if isinstance(page, int):
            page_number = page + 1
            if page_number not in source_pages:
                source_pages.append(page_number)

    if source_pages:
        answer += "\n\nRetrieved pages: " + ", ".join(map(str, source_pages))

    return answer
