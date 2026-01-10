import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from ragx.config import settings
from ragx.embeddings import get_embeddings

COLLECTION_NAME = "ragx_documents"
BATCH_SIZE = 20
RATE_LIMIT_DELAY = 1.5


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def add_documents(documents: list[Document], on_progress: callable = None) -> None:
    vectorstore = get_vectorstore()
    total = len(documents)

    for i in range(0, total, BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        retries = 0
        max_retries = 5

        while retries < max_retries:
            try:
                vectorstore.add_documents(batch)
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    retries += 1
                    wait_time = min(60, 2**retries * 5)
                    if on_progress:
                        on_progress(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

        if on_progress:
            on_progress(f"Processed {min(i + BATCH_SIZE, total)}/{total} chunks")

        if i + BATCH_SIZE < total:
            time.sleep(RATE_LIMIT_DELAY)


def clear_vectorstore() -> None:
    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection
        ids = collection.get()["ids"]
        if ids:
            collection.delete(ids=ids)
    except Exception:
        pass


def get_document_count() -> int:
    try:
        vectorstore = get_vectorstore()
        return vectorstore._collection.count()
    except Exception:
        return 0
