"""
Advanced chunking strategies for Phoenix RAG system.

Implements:
- SemanticChunker: Preserves semantic meaning across chunk boundaries
- CodeAwareChunker: Understands code structure (functions, classes)
- HybridChunker: Combines both for refactoring documentation
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    """Represents a document chunk with metadata."""

    content: str
    metadata: dict
    chunk_id: str
    source: str
    chunk_type: str  # "text", "code", "pattern", "example"
    start_index: int
    end_index: int


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Chunk]:
        """Split text into chunks."""
        pass


class SemanticChunker(BaseChunker):
    """
    Semantic chunking strategy that preserves context.

    Uses recursive character splitting with smart separators
    to maintain semantic coherence across chunks.

    Rationale: For refactoring documentation, we need chunks that
    contain complete concepts (e.g., full pattern descriptions,
    complete code smell explanations) rather than arbitrary splits.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[list[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n## ",      # Major section headers
            "\n### ",     # Subsection headers
            "\n\n",       # Paragraph breaks
            "\n",         # Line breaks
            ". ",         # Sentence endings
            " ",          # Word breaks
        ]
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Chunk]:
        """Split text semantically while preserving context."""
        metadata = metadata or {}
        documents = self._splitter.create_documents([text], [metadata])

        chunks = []
        current_index = 0

        for i, doc in enumerate(documents):
            content = doc.page_content
            start_idx = text.find(content, current_index)
            if start_idx == -1:
                start_idx = current_index
            end_idx = start_idx + len(content)

            chunk = Chunk(
                content=content,
                metadata={**metadata, "chunk_index": i},
                chunk_id=f"{metadata.get('source', 'unknown')}_{i}",
                source=metadata.get("source", "unknown"),
                chunk_type="text",
                start_index=start_idx,
                end_index=end_idx,
            )
            chunks.append(chunk)
            current_index = start_idx + 1

        return chunks


class CodeAwareChunker(BaseChunker):
    """
    Code-aware chunking that respects code structure.

    Identifies and preserves:
    - Complete function definitions
    - Complete class definitions
    - Import blocks
    - Docstrings with their associated code

    Rationale: When refactoring code, we need complete code units
    to understand patterns. Splitting a function in half loses
    critical context for refactoring suggestions.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        language: str = "python",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language = language

        # Language-specific patterns for code structure detection
        self.patterns = {
            "python": {
                "class": r"^class\s+\w+.*?(?=\nclass\s|\ndef\s(?!\s)|\Z)",
                "function": r"^def\s+\w+.*?(?=\ndef\s|\nclass\s|\Z)",
                "import": r"^(?:from\s+\S+\s+)?import\s+.+$",
            },
            "javascript": {
                "class": r"^class\s+\w+.*?(?=\nclass\s|\nfunction\s|\Z)",
                "function": r"^(?:async\s+)?function\s+\w+.*?(?=\nfunction\s|\nclass\s|\Z)",
                "import": r"^import\s+.+$",
            },
        }

    def _detect_code_blocks(self, text: str) -> list[dict]:
        """Detect structural code blocks in text."""
        blocks = []
        patterns = self.patterns.get(self.language, self.patterns["python"])

        # Find code blocks (indicated by triple backticks or indentation)
        code_block_pattern = r"```(?:\w+)?\n(.*?)```"
        for match in re.finditer(code_block_pattern, text, re.DOTALL):
            blocks.append({
                "type": "code_block",
                "content": match.group(1),
                "start": match.start(),
                "end": match.end(),
                "full_match": match.group(0),
            })

        return blocks

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Chunk]:
        """Split code-aware content into logical chunks."""
        metadata = metadata or {}
        chunks = []

        # Detect code blocks
        code_blocks = self._detect_code_blocks(text)

        if not code_blocks:
            # Fall back to semantic chunking for non-code content
            semantic = SemanticChunker(self.chunk_size, self.chunk_overlap)
            return semantic.chunk(text, metadata)

        # Process text with code block awareness
        current_pos = 0
        chunk_index = 0

        for block in code_blocks:
            # Handle text before code block
            if block["start"] > current_pos:
                pre_text = text[current_pos:block["start"]]
                if pre_text.strip():
                    semantic = SemanticChunker(self.chunk_size, self.chunk_overlap)
                    text_chunks = semantic.chunk(pre_text, {
                        **metadata,
                        "section": "text",
                    })
                    for tc in text_chunks:
                        tc.chunk_id = f"{metadata.get('source', 'unknown')}_{chunk_index}"
                        tc.metadata["chunk_index"] = chunk_index
                        chunks.append(tc)
                        chunk_index += 1

            # Handle the code block itself
            code_content = block["full_match"]
            chunk = Chunk(
                content=code_content,
                metadata={
                    **metadata,
                    "chunk_index": chunk_index,
                    "section": "code",
                    "language": self.language,
                },
                chunk_id=f"{metadata.get('source', 'unknown')}_{chunk_index}",
                source=metadata.get("source", "unknown"),
                chunk_type="code",
                start_index=block["start"],
                end_index=block["end"],
            )
            chunks.append(chunk)
            chunk_index += 1
            current_pos = block["end"]

        # Handle remaining text after last code block
        if current_pos < len(text):
            remaining = text[current_pos:]
            if remaining.strip():
                semantic = SemanticChunker(self.chunk_size, self.chunk_overlap)
                text_chunks = semantic.chunk(remaining, {
                    **metadata,
                    "section": "text",
                })
                for tc in text_chunks:
                    tc.chunk_id = f"{metadata.get('source', 'unknown')}_{chunk_index}"
                    tc.metadata["chunk_index"] = chunk_index
                    chunks.append(tc)
                    chunk_index += 1

        return chunks


class HybridChunker(BaseChunker):
    """
    Hybrid chunking strategy combining semantic and code-aware approaches.

    Automatically detects content type and applies appropriate strategy:
    - Pure documentation → Semantic chunking
    - Code files → Code-aware chunking
    - Mixed content (patterns with examples) → Hybrid approach

    Rationale: Refactoring knowledge bases contain both explanatory text
    and code examples. We need to handle both appropriately to maintain
    context for both "what to do" (text) and "how to do it" (code).
    """

    def __init__(
        self,
        text_chunk_size: int = 1000,
        text_chunk_overlap: int = 200,
        code_chunk_size: int = 500,
        code_chunk_overlap: int = 100,
    ):
        self.semantic_chunker = SemanticChunker(
            chunk_size=text_chunk_size,
            chunk_overlap=text_chunk_overlap,
        )
        self.code_chunker = CodeAwareChunker(
            chunk_size=code_chunk_size,
            chunk_overlap=code_chunk_overlap,
        )

    def _detect_content_type(self, text: str) -> str:
        """Detect whether content is primarily text, code, or mixed."""
        code_indicators = [
            r"```",           # Markdown code blocks
            r"def\s+\w+",     # Python functions
            r"class\s+\w+",   # Class definitions
            r"import\s+",     # Import statements
            r"function\s+",   # JavaScript functions
        ]

        code_matches = sum(
            len(re.findall(pattern, text)) for pattern in code_indicators
        )

        text_length = len(text)
        code_ratio = code_matches * 50 / text_length if text_length > 0 else 0

        if code_ratio > 0.3:
            return "code"
        elif code_ratio > 0.1:
            return "mixed"
        else:
            return "text"

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Chunk]:
        """Apply appropriate chunking strategy based on content type."""
        metadata = metadata or {}
        content_type = self._detect_content_type(text)
        metadata["content_type"] = content_type

        if content_type == "text":
            return self.semantic_chunker.chunk(text, metadata)
        elif content_type == "code":
            return self.code_chunker.chunk(text, metadata)
        else:  # mixed
            # Use code-aware chunker which handles both
            return self.code_chunker.chunk(text, metadata)
