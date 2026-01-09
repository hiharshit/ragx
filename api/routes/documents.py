import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from ragx import rag_chain
from api.schemas import DocumentStats, IngestResponse

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get("/stats", response_model=DocumentStats)
async def get_stats():
    try:
        stats = rag_chain.get_stats()
        return DocumentStats(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    try:
        # Create a temporary file to store the upload
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename).suffix
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        try:
            chunks_count = rag_chain.ingest(tmp_path)
            return IngestResponse(
                message=f"Successfully ingested {file.filename}",
                chunks_added=chunks_count,
                files_processed=1,
            )
        finally:
            # Clean up the temp file
            if tmp_path.exists():
                tmp_path.unlink()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("", status_code=204)
async def clear_documents():
    try:
        rag_chain.clear()
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
