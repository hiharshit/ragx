from pydantic import BaseModel, ConfigDict


class QueryRequest(BaseModel):
    question: str
    model_config = ConfigDict(extra="forbid")


class QueryResponse(BaseModel):
    answer: str
    question: str


class DocumentStats(BaseModel):
    document_count: int


class IngestResponse(BaseModel):
    message: str
    chunks_added: int
    files_processed: int
