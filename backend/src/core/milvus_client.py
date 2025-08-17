"""Milvus vector database client with collection management."""
from typing import List, Dict, Any, Optional, Tuple
import uuid
from loguru import logger
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from pymilvus.exceptions import MilvusException
from sentence_transformers import SentenceTransformer

from config.settings import settings


class MilvusClient:
    """Milvus vector database client for document embeddings."""
    
    def __init__(self):
        self.collection_name = settings.milvus_collection_name
        self.embedding_dim = settings.embedding_dimension
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.collection: Optional[Collection] = None
        self._connected = False
    
    async def connect(self) -> None:
        """Connect to Milvus server."""
        try:
            # Check if already connected
            if self._connected:
                return
            
            # Connect to Milvus
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port,
                user=settings.milvus_user,
                password=settings.milvus_password,
            )
            self._connected = True
            logger.info(f"Connected to Milvus at {settings.milvus_host}:{settings.milvus_port}")
            
            # Initialize collection
            await self._initialize_collection()
            
        except MilvusException as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Milvus server."""
        try:
            if self._connected:
                connections.disconnect("default")
                self._connected = False
                logger.info("Disconnected from Milvus")
        except MilvusException as e:
            logger.error(f"Error disconnecting from Milvus: {e}")
    
    async def _initialize_collection(self) -> None:
        """Initialize or create the documents collection."""
        try:
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                logger.info(f"Collection '{self.collection_name}' already exists")
                self.collection = Collection(self.collection_name)
            else:
                # Create collection schema
                schema = self._create_collection_schema()
                
                # Create collection
                self.collection = Collection(
                    name=self.collection_name,
                    schema=schema,
                )
                logger.info(f"Created collection '{self.collection_name}'")
            
            # Create index for vector field
            await self._create_vector_index()
            
            # Load collection into memory
            self.collection.load()
            logger.info(f"Collection '{self.collection_name}' loaded into memory")
            
        except MilvusException as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise
    
    def _create_collection_schema(self) -> CollectionSchema:
        """Create the collection schema for documents."""
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=64,
                description="Unique document chunk ID"
            ),
            FieldSchema(
                name="document_id",
                dtype=DataType.VARCHAR,
                max_length=64,
                description="Original document ID"
            ),
            FieldSchema(
                name="chunk_index",
                dtype=DataType.INT64,
                description="Index of chunk within document"
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="Text content of the chunk"
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
                description="Additional metadata as JSON"
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.embedding_dim,
                description="Text embedding vector"
            ),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Document chunks with embeddings for RAG"
        )
        return schema
    
    async def _create_vector_index(self) -> None:
        """Create an index on the vector field for fast similarity search."""
        try:
            # Check if index already exists
            indexes = self.collection.indexes
            if len(indexes) > 0:
                logger.info("Vector index already exists")
                return
            
            # Create IVF_FLAT index
            index_params = {
                "metric_type": "L2",  # L2 distance (Euclidean)
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}  # Number of clusters
            }
            
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params,
                timeout=30
            )
            logger.info("Created vector index for similarity search")
            
        except MilvusException as e:
            logger.error(f"Failed to create vector index: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    async def insert_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Insert document chunks with embeddings into Milvus.
        
        Args:
            documents: List of document dictionaries with keys:
                - document_id: str
                - chunk_index: int  
                - text: str
                - metadata: dict
        
        Returns:
            List of generated chunk IDs
        """
        try:
            if not documents:
                return []
            
            # Generate unique IDs for each chunk
            chunk_ids = [str(uuid.uuid4()) for _ in documents]
            
            # Extract texts for embedding
            texts = [doc["text"] for doc in documents]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} document chunks")
            embeddings = self.generate_embeddings(texts)
            
            # Prepare data for insertion
            entities = [
                chunk_ids,  # id
                [doc["document_id"] for doc in documents],  # document_id
                [doc["chunk_index"] for doc in documents],  # chunk_index
                texts,  # text
                [doc.get("metadata", {}) for doc in documents],  # metadata
                embeddings,  # embedding
            ]
            
            # Insert into Milvus
            insert_result = self.collection.insert(entities)
            
            # Flush to ensure data is written
            self.collection.flush()
            
            logger.info(f"Inserted {len(chunk_ids)} document chunks into Milvus")
            return chunk_ids
            
        except MilvusException as e:
            logger.error(f"Failed to insert documents: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during document insertion: {e}")
            raise
    
    async def search_similar(
        self, 
        query_text: str, 
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar document chunks.
        
        Args:
            query_text: Text to search for
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            filter_expr: Optional filter expression
        
        Returns:
            List of similar document chunks with metadata
        """
        try:
            # Generate embedding for query
            query_embedding = self.generate_embeddings([query_text])[0]
            
            # Search parameters
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}  # Number of clusters to search
            }
            
            # Perform vector search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["id", "document_id", "chunk_index", "text", "metadata"]
            )
            
            # Process results
            similar_chunks = []
            for result in results[0]:  # results[0] contains the matches for our single query
                # Convert L2 distance to similarity score (0-1, where 1 is most similar)
                similarity_score = 1.0 / (1.0 + result.distance)
                
                if similarity_score >= score_threshold:
                    similar_chunks.append({
                        "id": result.entity.get("id"),
                        "document_id": result.entity.get("document_id"),
                        "chunk_index": result.entity.get("chunk_index"),
                        "text": result.entity.get("text"),
                        "metadata": result.entity.get("metadata", {}),
                        "similarity_score": similarity_score,
                        "distance": result.distance,
                    })
            
            logger.info(f"Found {len(similar_chunks)} similar chunks for query")
            return similar_chunks
            
        except MilvusException as e:
            logger.error(f"Failed to search similar documents: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            raise
    
    async def delete_documents(self, document_id: str) -> bool:
        """Delete all chunks belonging to a document.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            # Delete by document_id
            expr = f'document_id == "{document_id}"'
            self.collection.delete(expr)
            
            # Flush to ensure deletion
            self.collection.flush()
            
            logger.info(f"Deleted all chunks for document: {document_id}")
            return True
            
        except MilvusException as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            stats = {
                "collection_name": self.collection_name,
                "total_entities": self.collection.num_entities,
                "is_loaded": utility.loading_progress(self.collection_name),
                "indexes": [index.to_dict() for index in self.collection.indexes],
            }
            return stats
            
        except MilvusException as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}


# Global Milvus client instance
milvus_client = MilvusClient()
