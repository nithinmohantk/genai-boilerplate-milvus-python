"""RAG (Retrieval-Augmented Generation) service using LangChain and Milvus."""
from typing import List, Dict, Any, Optional, Tuple
import os
import uuid
from pathlib import Path
from loguru import logger

# LangChain imports
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
)
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.schema.retriever import BaseRetriever
from langchain.schema import BaseLanguageModel
from langchain.schema.vectorstore import VectorStore
from langchain_openai import ChatOpenAI
from langchain_community.llms import Anthropic

from config.settings import settings
from src.core.milvus_client import milvus_client


class MilvusRetriever(BaseRetriever):
    """Custom LangChain retriever using Milvus vector database."""
    
    def __init__(self, milvus_client, top_k: int = 5, score_threshold: float = 0.0):
        self.milvus_client = milvus_client
        self.top_k = top_k
        self.score_threshold = score_threshold
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query."""
        try:
            # Search similar chunks in Milvus
            similar_chunks = self.milvus_client.search_similar(
                query_text=query,
                top_k=self.top_k,
                score_threshold=self.score_threshold
            )
            
            # Convert to LangChain Document format
            documents = []
            for chunk in similar_chunks:
                doc = Document(
                    page_content=chunk["text"],
                    metadata={
                        "id": chunk["id"],
                        "document_id": chunk["document_id"],
                        "chunk_index": chunk["chunk_index"],
                        "similarity_score": chunk["similarity_score"],
                        **chunk.get("metadata", {})
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents."""
        try:
            # Search similar chunks in Milvus
            similar_chunks = await self.milvus_client.search_similar(
                query_text=query,
                top_k=self.top_k,
                score_threshold=self.score_threshold
            )
            
            # Convert to LangChain Document format
            documents = []
            for chunk in similar_chunks:
                doc = Document(
                    page_content=chunk["text"],
                    metadata={
                        "id": chunk["id"],
                        "document_id": chunk["document_id"],
                        "chunk_index": chunk["chunk_index"],
                        "similarity_score": chunk["similarity_score"],
                        **chunk.get("metadata", {})
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []


class RAGService:
    """RAG service for document processing and question answering."""
    
    def __init__(self):
        self.upload_dir = Path(settings.upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Document loaders mapping
        self.loaders = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.doc': Docx2txtLoader,
            '.txt': TextLoader,
            '.md': TextLoader,
            '.csv': CSVLoader,
        }
        
        # Initialize retriever
        self.retriever = None
    
    def _get_llm(self, provider: str = "openai", model: str = "gpt-3.5-turbo") -> BaseLanguageModel:
        """Get language model based on provider."""
        if provider == "openai" and settings.openai_api_key:
            return ChatOpenAI(
                api_key=settings.openai_api_key,
                model_name=model,
                temperature=0.1
            )
        elif provider == "anthropic" and settings.anthropic_api_key:
            return Anthropic(
                api_key=settings.anthropic_api_key,
                model=model or "claude-3-sonnet-20240229",
                temperature=0.1
            )
        else:
            raise ValueError(f"Unsupported AI provider: {provider} or API key not configured")
    
    def _get_retriever(self, top_k: int = 5, score_threshold: float = 0.0) -> MilvusRetriever:
        """Get or create Milvus retriever."""
        if not self.retriever:
            self.retriever = MilvusRetriever(
                milvus_client=milvus_client,
                top_k=top_k,
                score_threshold=score_threshold
            )
        return self.retriever
    
    async def load_document(self, file_path: str) -> List[Document]:
        """Load and parse a document using appropriate loader.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of LangChain Document objects
        """
        try:
            file_path = Path(file_path)
            file_extension = file_path.suffix.lower()
            
            if file_extension not in self.loaders:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Load document using appropriate loader
            loader_class = self.loaders[file_extension]
            loader = loader_class(str(file_path))
            documents = loader.load()
            
            logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of chunked Document objects
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split documents into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise
    
    async def process_and_store_document(
        self, 
        file_path: str, 
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process a document and store it in Milvus.
        
        Args:
            file_path: Path to the document file
            document_id: Optional document ID (will generate if not provided)
            metadata: Optional metadata to store with the document
            
        Returns:
            Document ID
        """
        try:
            # Generate document ID if not provided
            if not document_id:
                document_id = str(uuid.uuid4())
            
            # Load document
            documents = await self.load_document(file_path)
            
            # Split into chunks
            chunks = self.split_documents(documents)
            
            # Prepare chunks for Milvus storage
            milvus_documents = []
            for i, chunk in enumerate(chunks):
                # Combine original metadata with chunk metadata
                chunk_metadata = {
                    "source": str(file_path),
                    "page": chunk.metadata.get("page", 0),
                    "total_chunks": len(chunks),
                    **(metadata or {}),
                    **chunk.metadata
                }
                
                milvus_doc = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "text": chunk.page_content,
                    "metadata": chunk_metadata
                }
                milvus_documents.append(milvus_doc)
            
            # Store in Milvus
            chunk_ids = await milvus_client.insert_documents(milvus_documents)
            
            logger.info(f"Processed and stored document {document_id} with {len(chunk_ids)} chunks")
            return document_id
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    async def search_documents(
        self, 
        query: str, 
        top_k: int = 5, 
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for relevant document chunks.
        
        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of relevant document chunks
        """
        try:
            results = await milvus_client.search_similar(
                query_text=query,
                top_k=top_k,
                score_threshold=score_threshold
            )
            
            logger.info(f"Found {len(results)} relevant chunks for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise
    
    async def answer_question(
        self,
        question: str,
        top_k: int = 5,
        ai_provider: str = "openai",
        ai_model: str = "gpt-3.5-turbo",
        score_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Answer a question using RAG.
        
        Args:
            question: Question to answer
            top_k: Number of document chunks to retrieve
            ai_provider: AI provider to use
            ai_model: AI model to use
            score_threshold: Minimum similarity score for retrieved chunks
            
        Returns:
            Dictionary with answer and source information
        """
        try:
            # Get LLM
            llm = self._get_llm(provider=ai_provider, model=ai_model)
            
            # Get retriever
            retriever = self._get_retriever(top_k=top_k, score_threshold=score_threshold)
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            # Get answer
            result = qa_chain({"query": question})
            
            # Format response
            response = {
                "question": question,
                "answer": result["result"],
                "source_documents": [],
                "metadata": {
                    "ai_provider": ai_provider,
                    "ai_model": ai_model,
                    "top_k": top_k,
                    "score_threshold": score_threshold
                }
            }
            
            # Add source document information
            for doc in result.get("source_documents", []):
                source_info = {
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": doc.metadata.get("similarity_score", 0.0)
                }
                response["source_documents"].append(source_info)
            
            logger.info(f"Generated answer for question with {len(response['source_documents'])} sources")
            return response
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            raise
    
    async def create_custom_chain(
        self,
        chain_type: str = "stuff",
        ai_provider: str = "openai",
        ai_model: str = "gpt-3.5-turbo"
    ) -> Any:
        """Create a custom LangChain chain with Milvus retriever.
        
        Args:
            chain_type: Type of chain to create
            ai_provider: AI provider to use
            ai_model: AI model to use
            
        Returns:
            LangChain chain object
        """
        try:
            # Get LLM
            llm = self._get_llm(provider=ai_provider, model=ai_model)
            
            # Get retriever
            retriever = self._get_retriever()
            
            # Create chain based on type
            if chain_type == "stuff":
                chain = load_qa_chain(llm, chain_type="stuff")
            elif chain_type == "retrieval_qa":
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever
                )
            else:
                raise ValueError(f"Unsupported chain type: {chain_type}")
            
            logger.info(f"Created custom {chain_type} chain with {ai_provider}")
            return chain
            
        except Exception as e:
            logger.error(f"Error creating custom chain: {e}")
            raise
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks from Milvus.
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            success = await milvus_client.delete_documents(document_id)
            if success:
                logger.info(f"Successfully deleted document: {document_id}")
            return success
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    async def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about stored documents.
        
        Returns:
            Document statistics
        """
        try:
            stats = await milvus_client.get_collection_stats()
            return stats
            
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return {}


# Global RAG service instance
rag_service = RAGService()
