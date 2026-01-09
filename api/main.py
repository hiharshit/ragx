from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import query, documents

app = FastAPI(
    title="RAGx API",
    description="API for RAGx Retrieval-Augmented Generation pipeline",
    version="0.1.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(query.router)
app.include_router(documents.router)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
