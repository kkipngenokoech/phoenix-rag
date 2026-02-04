"""
Configuration management for Phoenix RAG system.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


class LLMConfig(BaseModel):
    """Configuration for the LLM provider."""

    provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "auto"))
    model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "llama3.2"))
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    groq_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    temperature: float = 0.0
    max_tokens: int = 4096


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""

    model_name: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    device: str = "cpu"  # or "cuda" for GPU


class VectorDBConfig(BaseModel):
    """Configuration for ChromaDB vector database."""

    persist_directory: Path = Field(
        default_factory=lambda: Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db"))
    )
    collection_name: str = "phoenix_refactoring_knowledge"


class ChunkingConfig(BaseModel):
    """Configuration for document chunking strategy."""

    # Semantic chunking parameters
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Code-specific chunking
    code_chunk_size: int = 500
    code_chunk_overlap: int = 100

    # Separators for different content types
    text_separators: list[str] = ["\n\n", "\n", ". ", " "]
    code_separators: list[str] = ["\nclass ", "\ndef ", "\n\n", "\n"]


class AgentConfig(BaseModel):
    """Configuration for the Phoenix agent."""

    max_iterations: int = Field(
        default_factory=lambda: int(os.getenv("MAX_ITERATIONS", "10"))
    )
    groundedness_threshold: float = Field(
        default_factory=lambda: float(os.getenv("GROUNDEDNESS_THRESHOLD", "0.7"))
    )
    verbose: bool = True


class PhoenixConfig(BaseModel):
    """Main configuration container for Phoenix RAG system."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)

    # Paths
    data_dir: Path = Path("./data")
    documents_dir: Path = Path("./data/documents")
    logs_dir: Path = Path("./logs")

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for dir_path in [self.data_dir, self.documents_dir, self.logs_dir, self.vector_db.persist_directory]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = PhoenixConfig()
