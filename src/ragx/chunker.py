from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragx.config import settings


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = get_text_splitter()
    return splitter.split_documents(documents)
