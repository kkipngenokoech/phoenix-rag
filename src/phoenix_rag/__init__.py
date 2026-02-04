"""
Phoenix RAG: A RAG-enabled agent for code refactoring assistance.

This package implements:
1. Retrieval Module - Vector database with domain-specific ingestion
2. Tool-Calling Module - ReAct-style reasoning loop with custom tools
3. Verification Module - Self-evaluation and groundedness scoring
"""

__version__ = "0.1.0"

from phoenix_rag.agent import PhoenixAgent
from phoenix_rag.retrieval import RetrievalModule
from phoenix_rag.tools import ToolRegistry
from phoenix_rag.verification import VerificationModule

__all__ = [
    "PhoenixAgent",
    "RetrievalModule",
    "ToolRegistry",
    "VerificationModule",
]
