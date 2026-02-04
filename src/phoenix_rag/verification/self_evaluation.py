"""
Self-Evaluation Node for Phoenix RAG system.

Implements a secondary processing step where the agent evaluates
its own output against retrieved sources to flag and correct
potential hallucinations.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from phoenix_rag.config import PhoenixConfig, config as default_config
from phoenix_rag.verification.groundedness import GroundednessEvaluator, GroundednessResult

logger = logging.getLogger(__name__)


def _create_llm(config: PhoenixConfig):
    """Create LLM instance based on provider configuration."""
    if config.llm.provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.llm.model,
            base_url=config.llm.base_url,
            temperature=0.0,
        )
    elif config.llm.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.llm.model,
            api_key=config.llm.api_key,
            temperature=0.0,
            max_tokens=2048,
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.2", temperature=0.0)


@dataclass
class EvaluationResult:
    """Result from self-evaluation."""
    original_response: str
    corrected_response: Optional[str]
    groundedness: GroundednessResult
    was_corrected: bool
    corrections_made: list[str]
    confidence_score: float
    should_request_clarification: bool
    clarification_question: Optional[str] = None


class SelfEvaluationNode:
    """
    Self-Evaluation Node for hallucination detection and correction.

    This node:
    1. Takes the agent's response and retrieved sources
    2. Evaluates groundedness of the response
    3. Identifies potential hallucinations
    4. Attempts to correct unsupported claims
    5. Flags responses that need human clarification

    This is a guardrail that ensures agent responses are grounded
    in the retrieved knowledge base.
    """

    CORRECTION_PROMPT = """You are a response correction assistant. Your task is to revise a response
to remove or correct claims that are not supported by the source material.

ORIGINAL RESPONSE:
{response}

UNSUPPORTED CLAIMS (these need to be removed or qualified):
{unsupported_claims}

SOURCE MATERIAL (use this to verify and correct):
{sources}

Instructions:
1. Keep all claims that ARE supported by the sources
2. Remove claims that cannot be verified
3. Qualify uncertain claims with phrases like "Based on available information..." or "This may require verification..."
4. Do not add new information not in the sources
5. Maintain the helpful tone and structure of the original

Provide the corrected response:
"""

    CLARIFICATION_PROMPT = """Based on the following query and the available source material,
determine if we need to ask the user for clarification.

USER QUERY: {query}

AVAILABLE SOURCES:
{sources}

CURRENT RESPONSE GROUNDEDNESS: {groundedness_score}%

If the sources don't adequately address the query, suggest a clarification question.
If the sources are sufficient, respond with "NO CLARIFICATION NEEDED".

Format: Either "NO CLARIFICATION NEEDED" or "CLARIFICATION: [your question]"
"""

    def __init__(
        self,
        config: Optional[PhoenixConfig] = None,
        groundedness_threshold: float = 0.7,
    ):
        self.config = config or default_config
        self.groundedness_threshold = groundedness_threshold

        # Initialize components
        self.groundedness_evaluator = GroundednessEvaluator(self.config)

        self.llm = _create_llm(self.config)

    def evaluate(
        self,
        response: str,
        retrieved_sources: list[dict],
        original_query: Optional[str] = None,
        auto_correct: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate and optionally correct the agent's response.

        Args:
            response: The agent's generated response
            retrieved_sources: List of retrieved documents
            original_query: The original user query (for clarification check)
            auto_correct: Whether to automatically correct unsupported claims

        Returns:
            EvaluationResult with evaluation details and optional correction
        """
        # Step 1: Evaluate groundedness
        groundedness = self.groundedness_evaluator.evaluate(
            response=response,
            retrieved_sources=retrieved_sources,
            threshold=self.groundedness_threshold,
        )

        logger.info(
            f"Groundedness evaluation: {groundedness.score:.2%} "
            f"({groundedness.supported_claims}/{groundedness.total_claims} claims)"
        )

        # Step 2: Determine if correction is needed
        needs_correction = groundedness.score < self.groundedness_threshold
        corrected_response = None
        corrections_made = []

        if needs_correction and auto_correct and groundedness.unsupported_claims:
            # Attempt to correct the response
            corrected_response, corrections_made = self._correct_response(
                response=response,
                unsupported_claims=groundedness.unsupported_claims,
                sources=retrieved_sources,
            )

        # Step 3: Check if clarification is needed
        should_clarify = False
        clarification_question = None

        if groundedness.score < 0.5 and original_query:
            # Low groundedness might mean sources are insufficient
            should_clarify, clarification_question = self._check_clarification_needed(
                query=original_query,
                sources=retrieved_sources,
                groundedness_score=groundedness.score,
            )

        # Calculate final confidence
        confidence = self._calculate_confidence(
            groundedness=groundedness,
            was_corrected=corrected_response is not None,
        )

        return EvaluationResult(
            original_response=response,
            corrected_response=corrected_response,
            groundedness=groundedness,
            was_corrected=corrected_response is not None,
            corrections_made=corrections_made,
            confidence_score=confidence,
            should_request_clarification=should_clarify,
            clarification_question=clarification_question,
        )

    def _correct_response(
        self,
        response: str,
        unsupported_claims: list[str],
        sources: list[dict],
    ) -> tuple[Optional[str], list[str]]:
        """Attempt to correct unsupported claims in the response."""
        try:
            # Format sources
            source_text = "\n\n---\n\n".join(
                s.get("content", "")[:2000] for s in sources
            )

            # Format unsupported claims
            claims_text = "\n".join(f"- {claim}" for claim in unsupported_claims)

            messages = [
                SystemMessage(content="You correct responses to be factually accurate."),
                HumanMessage(content=self.CORRECTION_PROMPT.format(
                    response=response,
                    unsupported_claims=claims_text,
                    sources=source_text[:6000],
                )),
            ]

            result = self.llm.invoke(messages)
            corrected = result.content.strip()

            # Track corrections
            corrections = [f"Addressed unsupported claim: {c[:50]}..." for c in unsupported_claims]

            logger.info(f"Response corrected. {len(corrections)} issues addressed.")
            return corrected, corrections

        except Exception as e:
            logger.error(f"Response correction failed: {e}")
            return None, []

    def _check_clarification_needed(
        self,
        query: str,
        sources: list[dict],
        groundedness_score: float,
    ) -> tuple[bool, Optional[str]]:
        """Check if user clarification is needed."""
        try:
            source_text = "\n\n".join(
                f"[{s.get('metadata', {}).get('source', 'Source')}]: {s.get('content', '')[:500]}"
                for s in sources[:3]
            )

            messages = [
                SystemMessage(content="You determine if clarification is needed."),
                HumanMessage(content=self.CLARIFICATION_PROMPT.format(
                    query=query,
                    sources=source_text,
                    groundedness_score=int(groundedness_score * 100),
                )),
            ]

            result = self.llm.invoke(messages)
            content = result.content.strip()

            if "NO CLARIFICATION NEEDED" in content.upper():
                return False, None
            elif content.startswith("CLARIFICATION:"):
                question = content.replace("CLARIFICATION:", "").strip()
                return True, question
            else:
                return False, None

        except Exception as e:
            logger.error(f"Clarification check failed: {e}")
            return False, None

    def _calculate_confidence(
        self,
        groundedness: GroundednessResult,
        was_corrected: bool,
    ) -> float:
        """Calculate overall confidence in the response."""
        base_confidence = groundedness.score

        # Penalty for correction (indicates initial issues)
        if was_corrected:
            base_confidence *= 0.9

        # Bonus for high claim support
        if groundedness.total_claims > 0:
            support_ratio = groundedness.supported_claims / groundedness.total_claims
            if support_ratio > 0.9:
                base_confidence = min(base_confidence * 1.1, 1.0)

        return round(base_confidence, 3)

    def get_verification_summary(self, result: EvaluationResult) -> str:
        """Generate a human-readable verification summary."""
        summary_parts = [
            "=" * 50,
            "VERIFICATION SUMMARY",
            "=" * 50,
            f"Groundedness Score: {result.groundedness.score:.1%}",
            f"Claims Verified: {result.groundedness.supported_claims}/{result.groundedness.total_claims}",
            f"Confidence Level: {result.groundedness.confidence_level}",
            f"Was Corrected: {'Yes' if result.was_corrected else 'No'}",
        ]

        if result.corrections_made:
            summary_parts.append("\nCorrections Made:")
            for correction in result.corrections_made:
                summary_parts.append(f"  - {correction}")

        if result.should_request_clarification:
            summary_parts.append(f"\nClarification Needed: {result.clarification_question}")

        if result.groundedness.unsupported_claims:
            summary_parts.append("\nUnsupported Claims:")
            for claim in result.groundedness.unsupported_claims[:3]:
                summary_parts.append(f"  - {claim[:80]}...")

        summary_parts.append("=" * 50)

        return "\n".join(summary_parts)
