"""
Main Verification Module for Phoenix RAG system.

Provides the unified interface for all verification functionality.
"""

import logging
from typing import Optional

from phoenix_rag.config import PhoenixConfig, config as default_config
from phoenix_rag.verification.groundedness import (
    GroundednessEvaluator,
    GroundednessResult,
    quick_groundedness_check,
)
from phoenix_rag.verification.self_evaluation import (
    SelfEvaluationNode,
    EvaluationResult,
)

logger = logging.getLogger(__name__)


class VerificationModule:
    """
    Unified verification module for Phoenix RAG system.

    Provides:
    - Quick groundedness checks (keyword-based)
    - Full groundedness evaluation (LLM-based)
    - Self-evaluation with automatic correction
    - Clarification request detection

    This module acts as the guardrail for agent responses,
    ensuring they are grounded in retrieved knowledge.
    """

    def __init__(
        self,
        config: Optional[PhoenixConfig] = None,
        groundedness_threshold: float = 0.7,
        auto_correct: bool = True,
    ):
        """
        Initialize the verification module.

        Args:
            config: Phoenix configuration
            groundedness_threshold: Minimum groundedness score (0-1)
            auto_correct: Whether to auto-correct low-groundedness responses
        """
        self.config = config or default_config
        self.groundedness_threshold = groundedness_threshold
        self.auto_correct = auto_correct

        # Initialize components
        self.groundedness_evaluator = GroundednessEvaluator(self.config)
        self.self_evaluation_node = SelfEvaluationNode(
            self.config,
            groundedness_threshold=groundedness_threshold,
        )

        # Tracking
        self.verification_history: list[dict] = []

        logger.info(
            f"VerificationModule initialized (threshold: {groundedness_threshold:.0%})"
        )

    def quick_check(
        self,
        response: str,
        sources: list[dict],
    ) -> float:
        """
        Perform quick groundedness check using keyword overlap.

        This is fast but less accurate. Use for initial filtering
        before full evaluation.

        Args:
            response: Agent response to check
            sources: Retrieved source documents

        Returns:
            Quick groundedness score (0-1)
        """
        return quick_groundedness_check(response, sources)

    def evaluate_groundedness(
        self,
        response: str,
        sources: list[dict],
    ) -> GroundednessResult:
        """
        Perform full groundedness evaluation.

        Uses LLM to extract claims and verify them against sources.

        Args:
            response: Agent response to evaluate
            sources: Retrieved source documents

        Returns:
            Detailed groundedness result
        """
        result = self.groundedness_evaluator.evaluate(
            response=response,
            retrieved_sources=sources,
            threshold=self.groundedness_threshold,
        )

        # Track evaluation
        self._track_verification("groundedness", result.score, response[:100])

        return result

    def verify_and_correct(
        self,
        response: str,
        sources: list[dict],
        original_query: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Full verification with optional correction.

        This is the main method for ensuring response quality:
        1. Evaluates groundedness
        2. Corrects unsupported claims if needed
        3. Determines if clarification is needed

        Args:
            response: Agent response to verify
            sources: Retrieved source documents
            original_query: Original user query (for clarification check)

        Returns:
            Complete evaluation result with optional corrected response
        """
        result = self.self_evaluation_node.evaluate(
            response=response,
            retrieved_sources=sources,
            original_query=original_query,
            auto_correct=self.auto_correct,
        )

        # Track
        self._track_verification(
            "full_evaluation",
            result.groundedness.score,
            response[:100],
            corrected=result.was_corrected,
        )

        return result

    def get_verified_response(
        self,
        response: str,
        sources: list[dict],
        original_query: Optional[str] = None,
    ) -> tuple[str, float, dict]:
        """
        Get the best verified response.

        Returns the corrected response if corrections were made,
        otherwise returns the original with verification metadata.

        Args:
            response: Agent response
            sources: Retrieved source documents
            original_query: Original user query

        Returns:
            (final_response, confidence_score, metadata)
        """
        result = self.verify_and_correct(response, sources, original_query)

        final_response = result.corrected_response or result.original_response

        metadata = {
            "groundedness_score": result.groundedness.score,
            "confidence_score": result.confidence_score,
            "was_corrected": result.was_corrected,
            "claims_verified": f"{result.groundedness.supported_claims}/{result.groundedness.total_claims}",
            "confidence_level": result.groundedness.confidence_level,
        }

        if result.should_request_clarification:
            metadata["clarification_needed"] = True
            metadata["clarification_question"] = result.clarification_question

        return final_response, result.confidence_score, metadata

    def format_response_with_confidence(
        self,
        response: str,
        sources: list[dict],
        include_sources: bool = True,
    ) -> str:
        """
        Format response with confidence information and source citations.

        Args:
            response: Agent response
            sources: Retrieved source documents
            include_sources: Whether to include source citations

        Returns:
            Formatted response with confidence indicator
        """
        result = self.verify_and_correct(response, sources)

        # Build formatted response
        parts = []

        # Add confidence indicator
        confidence_emoji = {
            "high": "[High Confidence]",
            "medium": "[Medium Confidence]",
            "low": "[Low Confidence - Please Verify]",
        }.get(result.groundedness.confidence_level, "[Unknown Confidence]")

        parts.append(confidence_emoji)
        parts.append("")

        # Add response (corrected if available)
        final_response = result.corrected_response or result.original_response
        parts.append(final_response)

        # Add sources if requested
        if include_sources and sources:
            parts.append("")
            parts.append("---")
            parts.append("Sources:")
            seen_sources = set()
            for s in sources[:3]:
                source_name = s.get("metadata", {}).get("source", "Unknown")
                if source_name not in seen_sources:
                    parts.append(f"  - {source_name}")
                    seen_sources.add(source_name)

        # Add confidence score
        parts.append("")
        parts.append(f"Groundedness Score: {result.groundedness.score:.0%}")

        return "\n".join(parts)

    def _track_verification(
        self,
        verification_type: str,
        score: float,
        response_preview: str,
        corrected: bool = False,
    ) -> None:
        """Track verification for analytics."""
        self.verification_history.append({
            "type": verification_type,
            "score": score,
            "response_preview": response_preview,
            "corrected": corrected,
            "met_threshold": score >= self.groundedness_threshold,
        })

    def get_verification_stats(self) -> dict:
        """Get statistics about verifications performed."""
        if not self.verification_history:
            return {"total_verifications": 0}

        total = len(self.verification_history)
        avg_score = sum(v["score"] for v in self.verification_history) / total
        met_threshold = sum(1 for v in self.verification_history if v["met_threshold"])
        corrections = sum(1 for v in self.verification_history if v["corrected"])

        return {
            "total_verifications": total,
            "average_groundedness": round(avg_score, 3),
            "met_threshold_rate": met_threshold / total,
            "correction_rate": corrections / total,
            "threshold": self.groundedness_threshold,
        }

    def reset_stats(self) -> None:
        """Reset verification statistics."""
        self.verification_history = []
