"""
Retrieval Module for Phoenix RAG system.

This module handles:
- Document ingestion with domain-specific processing
- Advanced chunking strategies (semantic and code-aware)
- Vector database operations with ChromaDB
"""

from phoenix_rag.retrieval.module import RetrievalModule
from phoenix_rag.retrieval.chunking import (
    SemanticChunker,
    CodeAwareChunker,
    HybridChunker,
)
from phoenix_rag.retrieval.ingestion import DocumentIngestionPipeline

__all__ = [
    "RetrievalModule",
    "SemanticChunker",
    "CodeAwareChunker",
    "HybridChunker",
    "DocumentIngestionPipeline",
]
