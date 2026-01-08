from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ragx.config import settings


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.gemini_api_key,
    )
