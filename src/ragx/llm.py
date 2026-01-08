from langchain_google_genai import ChatGoogleGenerativeAI

from ragx.config import settings


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.3,
        convert_system_message_to_human=True,
    )
