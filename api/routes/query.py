from fastapi import APIRouter, HTTPException

from ragx import rag_chain
from api.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        stats = rag_chain.get_stats()
        if stats["document_count"] == 0:
            raise HTTPException(
                status_code=400,
                detail="No documents indexed. Please ingest documents first.",
            )

        answer = rag_chain.query(request.question)
        return QueryResponse(answer=answer, question=request.question)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
