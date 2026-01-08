from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def load_document(file_path: str | Path) -> list[Document]:
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix in {".txt", ".md", ".markdown"}:
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    
    return loader.load()


def load_directory(dir_path: str | Path) -> list[Document]:
    path = Path(dir_path)
    
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    
    documents: list[Document] = []
    supported_extensions = {".pdf", ".txt", ".md", ".markdown"}
    
    for file_path in path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                documents.extend(load_document(file_path))
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
    
    return documents
