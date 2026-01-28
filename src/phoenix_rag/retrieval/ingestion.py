"""
Document ingestion pipeline for Phoenix RAG system.

Handles loading, processing, and preparing documents for
vector database storage.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from phoenix_rag.retrieval.chunking import Chunk, HybridChunker

logger = logging.getLogger(__name__)


class Document:
    """Represents a document to be ingested."""

    def __init__(
        self,
        content: str,
        source: str,
        doc_type: str = "general",
        metadata: Optional[dict] = None,
    ):
        self.content = content
        self.source = source
        self.doc_type = doc_type  # "refactoring_pattern", "style_guide", "best_practice", "code_smell"
        self.metadata = metadata or {}
        self.doc_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique document ID based on content hash."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:12]
        return f"{self.source.replace('/', '_')}_{content_hash}"


class DocumentIngestionPipeline:
    """
    Pipeline for ingesting documents into the vector database.

    Supports:
    - Multiple file formats (txt, md, json)
    - Document categorization by type
    - Metadata enrichment
    - Batch processing
    """

    def __init__(
        self,
        chunker: Optional[HybridChunker] = None,
    ):
        self.chunker = chunker or HybridChunker()
        self.processed_docs: list[Document] = []
        self.all_chunks: list[Chunk] = []

    def load_from_file(self, file_path: Path, doc_type: str = "general") -> Document:
        """Load a single document from file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        content = file_path.read_text(encoding="utf-8")

        # Determine document type from file extension or explicit parameter
        if file_path.suffix == ".json":
            data = json.loads(content)
            content = data.get("content", json.dumps(data, indent=2))
            metadata = data.get("metadata", {})
        else:
            metadata = {}

        doc = Document(
            content=content,
            source=str(file_path),
            doc_type=doc_type,
            metadata={
                "file_name": file_path.name,
                "file_type": file_path.suffix,
                **metadata,
            },
        )

        logger.info(f"Loaded document: {file_path.name} ({len(content)} chars)")
        return doc

    def load_from_directory(
        self,
        directory: Path,
        doc_type: str = "general",
        extensions: Optional[list[str]] = None,
    ) -> list[Document]:
        """Load all documents from a directory."""
        directory = Path(directory)
        extensions = extensions or [".txt", ".md", ".json"]

        documents = []
        for ext in extensions:
            for file_path in directory.glob(f"**/*{ext}"):
                try:
                    doc = self.load_from_file(file_path, doc_type)
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")

        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents

    def process_document(self, document: Document) -> list[Chunk]:
        """Process a single document into chunks."""
        metadata = {
            "doc_id": document.doc_id,
            "source": document.source,
            "doc_type": document.doc_type,
            **document.metadata,
        }

        chunks = self.chunker.chunk(document.content, metadata)

        logger.info(
            f"Processed {document.source}: {len(chunks)} chunks created"
        )

        return chunks

    def process_documents(self, documents: list[Document]) -> list[Chunk]:
        """Process multiple documents into chunks."""
        all_chunks = []

        for doc in documents:
            chunks = self.process_document(doc)
            all_chunks.extend(chunks)
            self.processed_docs.append(doc)

        self.all_chunks = all_chunks
        logger.info(
            f"Total: {len(self.processed_docs)} docs → {len(all_chunks)} chunks"
        )

        return all_chunks

    def create_document(
        self,
        content: str,
        source: str,
        doc_type: str = "general",
        metadata: Optional[dict] = None,
    ) -> Document:
        """Create a document from raw content."""
        return Document(
            content=content,
            source=source,
            doc_type=doc_type,
            metadata=metadata,
        )

    def get_statistics(self) -> dict:
        """Get ingestion statistics."""
        return {
            "total_documents": len(self.processed_docs),
            "total_chunks": len(self.all_chunks),
            "doc_types": {
                doc.doc_type: sum(1 for d in self.processed_docs if d.doc_type == doc.doc_type)
                for doc in self.processed_docs
            },
            "avg_chunk_size": (
                sum(len(c.content) for c in self.all_chunks) / len(self.all_chunks)
                if self.all_chunks else 0
            ),
        }
