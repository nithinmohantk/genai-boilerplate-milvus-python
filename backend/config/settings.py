"""Application configuration using Pydantic Settings."""
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class MilvusSettings(BaseModel):
    """Milvus vector database configuration."""
    host: str = "localhost"
    port: int = 19530
    user: str = ""
    password: str = ""
    db_name: str = "default"
    collection_name: str = "documents"


class EmbeddingSettings(BaseModel):
    """Embedding model configuration."""
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    chunk_size: int = 1000
    chunk_overlap: int = 200


class AIProviders(BaseModel):
    """AI provider API keys."""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application
    app_name: str = "GenAI RAG Boilerplate"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Security
    secret_key: str = Field(..., description="Secret key for JWT tokens")
    access_token_expire_minutes: int = 30
    
    # Vector Database
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_user: str = ""
    milvus_password: str = ""
    milvus_db_name: str = "default"
    milvus_collection_name: str = "documents"
    
    # AI Providers
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    
    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # File Upload
    upload_dir: str = "./uploads"
    max_file_size: int = 50  # MB
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @property
    def milvus_settings(self) -> MilvusSettings:
        """Get Milvus configuration."""
        return MilvusSettings(
            host=self.milvus_host,
            port=self.milvus_port,
            user=self.milvus_user,
            password=self.milvus_password,
            db_name=self.milvus_db_name,
            collection_name=self.milvus_collection_name,
        )
    
    @property
    def embedding_settings(self) -> EmbeddingSettings:
        """Get embedding configuration."""
        return EmbeddingSettings(
            model=self.embedding_model,
            dimension=self.embedding_dimension,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
    
    @property
    def ai_providers(self) -> AIProviders:
        """Get AI provider configurations."""
        return AIProviders(
            openai_api_key=self.openai_api_key,
            anthropic_api_key=self.anthropic_api_key,
            google_api_key=self.google_api_key,
        )


# Global settings instance
settings = Settings()
