"""FastAPI REST endpoints for RAG operations."""
import os
import time
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, status
from fastapi.responses import JSONResponse
from loguru import logger

from config.settings import settings
from src.models.api_models import (
    DocumentUploadResponse,
    SearchRequest,
    SearchResponse,
    QuestionRequest,
    QuestionResponse,
    DocumentStats,
    HealthResponse,
    ConfigResponse,
    MilvusConfig,
    EmbeddingConfig,
    ErrorResponse,
)
from src.services.rag_service import rag_service
from src.core.milvus_client import milvus_client


# Create API router
router = APIRouter()


# Dependency to ensure Milvus connection
async def ensure_milvus_connection():
    """Ensure Milvus client is connected."""
    try:
        if not milvus_client._connected:
            await milvus_client.connect()
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector database is not available"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check Milvus connection
        milvus_connected = False
        total_documents = 0
        
        try:
            if not milvus_client._connected:
                await milvus_client.connect()
            stats = await milvus_client.get_collection_stats()
            milvus_connected = True
            total_documents = stats.get("total_entities", 0)
        except Exception as e:
            logger.warning(f"Milvus health check failed: {e}")
        
        return HealthResponse(
            status="healthy",
            version=settings.app_version,
            milvus_connected=milvus_connected,
            total_documents=total_documents,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not healthy"
        )


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current configuration."""
    return ConfigResponse(
        milvus=MilvusConfig(
            host=settings.milvus_host,
            port=settings.milvus_port,
            collection_name=settings.milvus_collection_name,
            embedding_dimension=settings.embedding_dimension
        ),
        embedding=EmbeddingConfig(
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        ),
        supported_file_types=[".pdf", ".docx", ".doc", ".txt", ".md", ".csv"],
        max_file_size_mb=settings.max_file_size
    )


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    _: None = Depends(ensure_milvus_connection)
):
    """Upload and process a document."""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No filename provided"
            )
        
        # Check file extension
        file_extension = Path(file.filename).suffix.lower()
        supported_types = [".pdf", ".docx", ".doc", ".txt", ".md", ".csv"]
        if file_extension not in supported_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type. Supported: {supported_types}"
            )
        
        # Check file size
        file.file.seek(0, 2)  # Go to end of file
        file_size = file.file.tell()
        file.file.seek(0)  # Reset file pointer
        
        max_size = settings.max_file_size * 1024 * 1024  # Convert MB to bytes
        if file_size > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds {settings.max_file_size}MB limit"
            )
        
        # Save uploaded file
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process document
        try:
            document_id = await rag_service.process_and_store_document(
                file_path=str(file_path),
                metadata={
                    "filename": file.filename,
                    "file_size": file_size,
                    "upload_timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # Get stats to determine number of chunks
            stats = await milvus_client.get_collection_stats()
            
            return DocumentUploadResponse(
                document_id=document_id,
                filename=file.filename,
                size=file_size,
                chunks_created=0,  # Could query Milvus for exact count if needed
                message=f"Document uploaded and processed successfully"
            )
            
        finally:
            # Clean up uploaded file
            try:
                os.unlink(file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up uploaded file: {e}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {str(e)}"
        )


@router.post("/documents/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    _: None = Depends(ensure_milvus_connection)
):
    """Search for similar document chunks."""
    try:
        start_time = time.time()
        
        results = await rag_service.search_documents(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold
        )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search documents: {str(e)}"
        )


@router.post("/chat/completions", response_model=QuestionResponse)
async def answer_question(
    request: QuestionRequest,
    _: None = Depends(ensure_milvus_connection)
):
    """Answer a question using RAG."""
    try:
        start_time = time.time()
        
        response = await rag_service.answer_question(
            question=request.question,
            top_k=request.top_k,
            ai_provider=request.ai_provider,
            ai_model=request.ai_model,
            score_threshold=request.score_threshold
        )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return QuestionResponse(
            question=response["question"],
            answer=response["answer"],
            source_documents=response["source_documents"],
            metadata=response["metadata"],
            processing_time_ms=processing_time
        )
        
    except ValueError as e:
        # Handle configuration errors (e.g., missing API keys)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to answer question: {str(e)}"
        )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    _: None = Depends(ensure_milvus_connection)
):
    """Delete a document and all its chunks."""
    try:
        success = await rag_service.delete_document(document_id)
        
        if success:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found or could not be deleted"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


@router.get("/documents/stats", response_model=DocumentStats)
async def get_document_stats(
    _: None = Depends(ensure_milvus_connection)
):
    """Get document collection statistics."""
    try:
        stats = await rag_service.get_document_stats()
        
        return DocumentStats(
            collection_name=stats.get("collection_name", "unknown"),
            total_entities=stats.get("total_entities", 0),
            is_loaded=stats.get("is_loaded", False),
            indexes=stats.get("indexes", [])
        )
        
    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document stats: {str(e)}"
        )


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "RAG API with LangChain and Milvus",
        "docs_url": "/docs",
        "health_url": "/health"
    }
