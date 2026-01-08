from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from ragx.chunker import chunk_documents
from ragx.document_loader import load_directory, load_document
from ragx.llm import get_llm
from ragx.retriever import get_retriever
from ragx.vectorstore import add_documents, clear_vectorstore, get_document_count

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer. If the answer is not in the context, say so.
Be concise and accurate in your responses.""",
        ),
        (
            "human",
            """Context:
{context}

Question: {question}

Answer:""",
        ),
    ]
)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def ingest(path: str | Path, on_progress: callable = None) -> int:
    path = Path(path)

    if path.is_file():
        documents = load_document(path)
    elif path.is_dir():
        documents = load_directory(path)
    else:
        raise ValueError(f"Path does not exist: {path}")

    if not documents:
        return 0

    chunks = chunk_documents(documents)
    add_documents(chunks, on_progress=on_progress)

    return len(chunks)


def query(question: str) -> str:
    retriever = get_retriever()
    llm = get_llm()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)


def clear() -> None:
    clear_vectorstore()


def get_stats() -> dict:
    return {
        "document_count": get_document_count(),
    }
