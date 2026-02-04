"""
Groundedness Evaluator for Phoenix RAG system.

Evaluates how well agent responses are grounded in retrieved sources.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from phoenix_rag.config import PhoenixConfig, config as default_config

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
            max_tokens=1024,
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.2", temperature=0.0)


@dataclass
class Claim:
    """A factual claim extracted from the response."""
    text: str
    supported: bool = False
    supporting_source: Optional[str] = None
    confidence: float = 0.0


@dataclass
class GroundednessResult:
    """Result of groundedness evaluation."""
    score: float  # 0.0 to 1.0
    claims: list[Claim] = field(default_factory=list)
    supported_claims: int = 0
    total_claims: int = 0
    unsupported_claims: list[str] = field(default_factory=list)
    explanation: str = ""
    needs_clarification: bool = False
    confidence_level: str = "unknown"  # low, medium, high

    def is_grounded(self, threshold: float = 0.7) -> bool:
        """Check if response meets groundedness threshold."""
        return self.score >= threshold


class GroundednessEvaluator:
    """
    Evaluates the groundedness of agent responses.

    The Groundedness Score (0 to 1) represents the fraction of the
    response's key factual claims that are explicitly supported by
    retrieved passages or tool outputs.

    Process:
    1. Extract factual claims from the response
    2. Check each claim against retrieved sources
    3. Calculate the ratio of supported claims
    4. Provide explanation and flag unsupported claims
    """

    EXTRACTION_PROMPT = """You are a claim extraction assistant. Extract the key factual claims from the following response.
A claim is a specific, verifiable statement of fact (not opinions or recommendations).

Response to analyze:
{response}

List each factual claim on a new line, prefixed with "CLAIM: ".
Only extract concrete, specific claims that could be verified against source material.
If there are no factual claims, respond with "NO CLAIMS".
"""

    VERIFICATION_PROMPT = """You are a fact-checking assistant. Determine if the following claim is supported by the provided source material.

CLAIM: {claim}

SOURCE MATERIAL:
{sources}

Respond with exactly one of:
- "SUPPORTED" if the claim is directly or strongly implied by the sources
- "PARTIALLY_SUPPORTED" if the claim is partially supported but missing key details
- "NOT_SUPPORTED" if the claim cannot be verified from the sources
- "CONTRADICTED" if the sources contradict the claim

Then on the next line, briefly explain your reasoning (1-2 sentences).
"""

    def __init__(self, config: Optional[PhoenixConfig] = None):
        self.config = config or default_config

        # Initialize LLM for claim extraction and verification
        self.llm = _create_llm(self.config)

    def evaluate(
        self,
        response: str,
        retrieved_sources: list[dict],
        threshold: float = 0.7,
    ) -> GroundednessResult:
        """
        Evaluate the groundedness of a response.

        Args:
            response: The agent's response to evaluate
            retrieved_sources: List of retrieved documents with 'content' key
            threshold: Minimum score to consider response grounded

        Returns:
            GroundednessResult with score and detailed analysis
        """
        # Extract claims from response
        claims = self._extract_claims(response)

        if not claims:
            # No factual claims to verify
            return GroundednessResult(
                score=1.0,
                claims=[],
                supported_claims=0,
                total_claims=0,
                explanation="No factual claims found in response to verify.",
                confidence_level="high",
            )

        # Format sources for verification
        source_text = self._format_sources(retrieved_sources)

        if not source_text.strip():
            # No sources to verify against
            return GroundednessResult(
                score=0.0,
                claims=claims,
                supported_claims=0,
                total_claims=len(claims),
                unsupported_claims=[c.text for c in claims],
                explanation="No source material available to verify claims.",
                needs_clarification=True,
                confidence_level="low",
            )

        # Verify each claim
        supported_count = 0
        for claim in claims:
            is_supported, confidence, source = self._verify_claim(claim.text, source_text)
            claim.supported = is_supported
            claim.confidence = confidence
            claim.supporting_source = source
            if is_supported:
                supported_count += 1

        # Calculate score
        score = supported_count / len(claims) if claims else 1.0

        # Collect unsupported claims
        unsupported = [c.text for c in claims if not c.supported]

        # Determine confidence level
        if score >= 0.9:
            confidence_level = "high"
        elif score >= 0.7:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        # Generate explanation
        explanation = self._generate_explanation(claims, score, threshold)

        return GroundednessResult(
            score=round(score, 3),
            claims=claims,
            supported_claims=supported_count,
            total_claims=len(claims),
            unsupported_claims=unsupported,
            explanation=explanation,
            needs_clarification=score < threshold,
            confidence_level=confidence_level,
        )

    def _extract_claims(self, response: str) -> list[Claim]:
        """Extract factual claims from the response using LLM."""
        try:
            messages = [
                SystemMessage(content="You extract factual claims from text."),
                HumanMessage(content=self.EXTRACTION_PROMPT.format(response=response)),
            ]

            result = self.llm.invoke(messages)
            content = result.content

            if "NO CLAIMS" in content:
                return []

            # Parse claims
            claims = []
            for line in content.split("\n"):
                if line.strip().startswith("CLAIM:"):
                    claim_text = line.replace("CLAIM:", "").strip()
                    if claim_text:
                        claims.append(Claim(text=claim_text))

            return claims

        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            # Fallback: simple sentence extraction
            return self._fallback_claim_extraction(response)

    def _fallback_claim_extraction(self, response: str) -> list[Claim]:
        """Simple fallback claim extraction without LLM."""
        # Extract sentences that look like factual claims
        sentences = re.split(r'[.!?]+', response)
        claims = []

        for sentence in sentences:
            sentence = sentence.strip()
            # Filter for sentences that look like factual claims
            if len(sentence) > 20 and not sentence.endswith('?'):
                # Check for indicators of factual statements
                if any(word in sentence.lower() for word in
                       ['is', 'are', 'was', 'were', 'has', 'have', 'should', 'will', 'can']):
                    claims.append(Claim(text=sentence))

        return claims[:10]  # Limit to 10 claims

    def _format_sources(self, sources: list[dict]) -> str:
        """Format retrieved sources for verification."""
        formatted = []
        for i, source in enumerate(sources, 1):
            content = source.get("content", "")
            source_name = source.get("metadata", {}).get("source", f"Source {i}")
            formatted.append(f"[{source_name}]\n{content}")

        return "\n\n---\n\n".join(formatted)

    def _verify_claim(
        self,
        claim: str,
        source_text: str,
    ) -> tuple[bool, float, Optional[str]]:
        """
        Verify a single claim against source material.

        Returns:
            (is_supported, confidence, supporting_source)
        """
        try:
            messages = [
                SystemMessage(content="You verify claims against source material."),
                HumanMessage(content=self.VERIFICATION_PROMPT.format(
                    claim=claim,
                    sources=source_text[:8000],  # Limit source length
                )),
            ]

            result = self.llm.invoke(messages)
            content = result.content.strip()

            # Parse result
            lines = content.split("\n")
            verdict = lines[0].strip().upper() if lines else "NOT_SUPPORTED"

            if "SUPPORTED" in verdict and "NOT" not in verdict:
                return True, 0.9, "Retrieved sources"
            elif "PARTIALLY" in verdict:
                return True, 0.6, "Partially from sources"
            elif "CONTRADICTED" in verdict:
                return False, 0.0, None
            else:
                return False, 0.3, None

        except Exception as e:
            logger.error(f"Claim verification failed: {e}")
            # Fallback: simple keyword matching
            return self._fallback_verification(claim, source_text)

    def _fallback_verification(
        self,
        claim: str,
        source_text: str,
    ) -> tuple[bool, float, Optional[str]]:
        """Simple fallback verification using keyword matching."""
        claim_words = set(claim.lower().split())
        source_words = set(source_text.lower().split())

        # Remove common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'or', 'and', 'but', 'if', 'then', 'else', 'when',
                     'that', 'this', 'it', 'they', 'we', 'you', 'i'}

        claim_keywords = claim_words - stopwords
        overlap = len(claim_keywords & source_words)
        ratio = overlap / len(claim_keywords) if claim_keywords else 0

        if ratio > 0.5:
            return True, ratio, "Keyword match"
        else:
            return False, ratio, None

    def _generate_explanation(
        self,
        claims: list[Claim],
        score: float,
        threshold: float,
    ) -> str:
        """Generate human-readable explanation of groundedness result."""
        total = len(claims)
        supported = sum(1 for c in claims if c.supported)
        unsupported = [c for c in claims if not c.supported]

        explanation_parts = [
            f"Groundedness Score: {score:.1%} ({supported}/{total} claims supported)"
        ]

        if score >= threshold:
            explanation_parts.append(
                f"Response meets groundedness threshold ({threshold:.0%})."
            )
        else:
            explanation_parts.append(
                f"Response is below groundedness threshold ({threshold:.0%})."
            )
            if unsupported:
                explanation_parts.append("\nUnsupported claims that need verification:")
                for c in unsupported[:3]:  # Show top 3
                    explanation_parts.append(f"  - {c.text[:100]}...")

        return "\n".join(explanation_parts)


def quick_groundedness_check(response: str, sources: list[dict]) -> float:
    """
    Quick groundedness check without full LLM evaluation.

    Uses keyword overlap as a proxy for groundedness.
    Useful for fast filtering before detailed evaluation.
    """
    if not sources:
        return 0.0

    # Combine all source content
    source_text = " ".join(s.get("content", "") for s in sources).lower()
    response_lower = response.lower()

    # Extract key terms from response (simple approach)
    words = re.findall(r'\b[a-z]{4,}\b', response_lower)
    if not words:
        return 1.0  # No significant words to check

    # Count matches
    matches = sum(1 for w in words if w in source_text)
    return min(matches / len(words), 1.0)
