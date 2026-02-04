"""
Main Retrieval Module for Phoenix RAG system.

Integrates ChromaDB vector database with the ingestion pipeline
and provides retrieval functionality for the agent.
"""

import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from phoenix_rag.config import PhoenixConfig, config as default_config
from phoenix_rag.retrieval.chunking import Chunk, HybridChunker
from phoenix_rag.retrieval.ingestion import Document, DocumentIngestionPipeline

logger = logging.getLogger(__name__)


class RetrievalModule:
    """
    The Memory component of Phoenix RAG system.

    Responsibilities:
    - Manage vector database (ChromaDB)
    - Handle document ingestion with advanced chunking
    - Perform similarity search for relevant context
    - Track retrieval metadata for verification
    """

    def __init__(self, config: Optional[PhoenixConfig] = None):
        self.config = config or default_config
        self.config.ensure_directories()

        # Initialize embedding model
        self._init_embeddings()

        # Initialize ChromaDB
        self._init_vector_store()

        # Initialize ingestion pipeline
        self.pipeline = DocumentIngestionPipeline(
            chunker=HybridChunker(
                text_chunk_size=self.config.chunking.chunk_size,
                text_chunk_overlap=self.config.chunking.chunk_overlap,
                code_chunk_size=self.config.chunking.code_chunk_size,
                code_chunk_overlap=self.config.chunking.code_chunk_overlap,
            )
        )

        logger.info("RetrievalModule initialized successfully")

    def _init_embeddings(self) -> None:
        """Initialize the embedding model."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding.model_name,
            model_kwargs={"device": self.config.embedding.device},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(f"Embedding model loaded: {self.config.embedding.model_name}")

    def _init_vector_store(self) -> None:
        """Initialize ChromaDB vector store."""
        persist_dir = str(self.config.vector_db.persist_directory)

        # Initialize Chroma client with persistence
        self.chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Create or get collection via LangChain
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=self.config.vector_db.collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )

        logger.info(f"ChromaDB initialized at: {persist_dir}")

    def ingest_document(
        self,
        content: str,
        source: str,
        doc_type: str = "general",
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Ingest a single document into the vector store.

        Args:
            content: Document text content
            source: Source identifier (e.g., filename, URL)
            doc_type: Type of document (refactoring_pattern, style_guide, etc.)
            metadata: Additional metadata

        Returns:
            Number of chunks created
        """
        doc = self.pipeline.create_document(content, source, doc_type, metadata)
        chunks = self.pipeline.process_document(doc)

        # Add to vector store
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]

        self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        logger.info(f"Ingested {len(chunks)} chunks from {source}")
        return len(chunks)

    def ingest_documents(self, documents: list[Document]) -> int:
        """Ingest multiple documents into the vector store."""
        all_chunks = self.pipeline.process_documents(documents)

        texts = [chunk.content for chunk in all_chunks]
        metadatas = [chunk.metadata for chunk in all_chunks]
        ids = [chunk.chunk_id for chunk in all_chunks]

        self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        logger.info(f"Ingested {len(all_chunks)} total chunks")
        return len(all_chunks)

    def ingest_from_directory(
        self,
        directory: Path,
        doc_type: str = "general",
        extensions: Optional[list[str]] = None,
    ) -> int:
        """Ingest all documents from a directory."""
        documents = self.pipeline.load_from_directory(directory, doc_type, extensions)
        return self.ingest_documents(documents)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict] = None,
        score_threshold: Optional[float] = None,
    ) -> list[dict]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Metadata filters (e.g., {"doc_type": "refactoring_pattern"})
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of retrieved documents with content, metadata, and scores
        """
        # Perform similarity search with scores
        if filter_dict:
            results = self.vector_store.similarity_search_with_relevance_scores(
                query, k=k, filter=filter_dict
            )
        else:
            results = self.vector_store.similarity_search_with_relevance_scores(
                query, k=k
            )

        # Format results
        retrieved_docs = []
        for doc, score in results:
            if score_threshold and score < score_threshold:
                continue

            retrieved_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": score,
            })

        logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query[:50]}...")
        return retrieved_docs

    def retrieve_for_refactoring(
        self,
        code_description: str,
        code_smell: Optional[str] = None,
        k: int = 5,
    ) -> list[dict]:
        """
        Specialized retrieval for refactoring suggestions.

        Combines multiple queries to get comprehensive context:
        1. Code smell patterns
        2. Refactoring techniques
        3. Best practices

        Args:
            code_description: Description of the code to refactor
            code_smell: Identified code smell (if any)
            k: Number of results per query type

        Returns:
            Combined list of relevant documents
        """
        results = []

        # Query 1: General refactoring patterns for the code
        general_results = self.retrieve(
            query=f"How to refactor: {code_description}",
            k=k,
            filter_dict={"doc_type": "refactoring_pattern"},
        )
        results.extend(general_results)

        # Query 2: Specific code smell remediation (if identified)
        if code_smell:
            smell_results = self.retrieve(
                query=f"Fix {code_smell} code smell: {code_description}",
                k=k // 2,
                filter_dict={"doc_type": "code_smell"},
            )
            results.extend(smell_results)

        # Query 3: Best practices
        best_practice_results = self.retrieve(
            query=f"Best practices for: {code_description}",
            k=k // 2,
            filter_dict={"doc_type": "best_practice"},
        )
        results.extend(best_practice_results)

        # Deduplicate by content
        seen = set()
        unique_results = []
        for r in results:
            content_hash = hash(r["content"][:100])
            if content_hash not in seen:
                seen.add(content_hash)
                unique_results.append(r)

        return unique_results

    def get_collection_stats(self) -> dict:
        """Get statistics about the vector store collection."""
        collection = self.chroma_client.get_collection(
            self.config.vector_db.collection_name
        )
        return {
            "collection_name": self.config.vector_db.collection_name,
            "total_documents": collection.count(),
            "pipeline_stats": self.pipeline.get_statistics(),
        }

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        self.chroma_client.delete_collection(self.config.vector_db.collection_name)
        self._init_vector_store()  # Recreate empty collection
        logger.info("Collection cleared")
