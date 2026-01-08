from langchain_core.vectorstores import VectorStoreRetriever

from ragx.vectorstore import get_vectorstore


def get_retriever(k: int = 4) -> VectorStoreRetriever:
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})
