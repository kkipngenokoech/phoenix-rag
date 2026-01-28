"""
Verification Module for Phoenix RAG system.

This module provides guardrails for the agent:
- Self-evaluation against retrieved sources
- Groundedness scoring for hallucination detection
- Response validation and correction
"""

from phoenix_rag.verification.module import VerificationModule
from phoenix_rag.verification.groundedness import GroundednessEvaluator
from phoenix_rag.verification.self_evaluation import SelfEvaluationNode

__all__ = [
    "VerificationModule",
    "GroundednessEvaluator",
    "SelfEvaluationNode",
]
