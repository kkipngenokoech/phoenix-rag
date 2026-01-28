"""
Retrieval Tool for Phoenix RAG system.

Wraps the RetrievalModule for use in the agent's tool-calling loop.
"""

import time
from typing import Optional

from phoenix_rag.tools.base import BaseTool, ToolCategory, ToolResult


class RetrievalTool(BaseTool):
    """
    Retrieves relevant refactoring knowledge from the vector database.

    This tool allows the agent to search the knowledge base for:
    - Refactoring patterns
    - Code smell remediation strategies
    - Best practices
    - Style guidelines

    The agent decides when to use this tool based on the query.
    """

    name = "knowledge_retrieval"
    description = """Retrieves relevant refactoring knowledge, patterns, and best practices from the knowledge base.
    Use this tool when you need information about HOW to refactor code, specific refactoring patterns,
    or best practices. Input should describe what refactoring knowledge you need."""
    category = ToolCategory.RETRIEVAL

    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query describing what knowledge is needed",
            },
            "doc_type": {
                "type": "string",
                "enum": ["refactoring_pattern", "code_smell", "best_practice", "style_guide", "all"],
                "description": "Type of document to search for",
                "default": "all",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 5,
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
    }

    def __init__(self, retrieval_module=None):
        """
        Initialize with a retrieval module.

        Args:
            retrieval_module: The RetrievalModule instance to use for searches
        """
        self.retrieval_module = retrieval_module

    def set_retrieval_module(self, retrieval_module) -> None:
        """Set the retrieval module after initialization."""
        self.retrieval_module = retrieval_module

    def execute(
        self,
        query: str,
        doc_type: str = "all",
        num_results: int = 5,
    ) -> ToolResult:
        """Execute knowledge retrieval."""
        start_time = time.time()

        if not self.retrieval_module:
            return ToolResult(
                success=False,
                output=None,
                error="Retrieval module not initialized",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        try:
            # Build filter if doc_type specified
            filter_dict = None
            if doc_type != "all":
                filter_dict = {"doc_type": doc_type}

            # Perform retrieval
            results = self.retrieval_module.retrieve(
                query=query,
                k=num_results,
                filter_dict=filter_dict,
            )

            if not results:
                return ToolResult(
                    success=True,
                    output={
                        "query": query,
                        "results": [],
                        "message": "No relevant documents found",
                    },
                    metadata={"num_results": 0},
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

            # Format results for LLM consumption
            formatted_results = []
            for i, r in enumerate(results, 1):
                formatted_results.append({
                    "rank": i,
                    "content": r["content"],
                    "source": r["metadata"].get("source", "unknown"),
                    "doc_type": r["metadata"].get("doc_type", "unknown"),
                    "relevance_score": round(r["relevance_score"], 3),
                })

            execution_time = (time.time() - start_time) * 1000

            return ToolResult(
                success=True,
                output={
                    "query": query,
                    "doc_type_filter": doc_type,
                    "num_results": len(formatted_results),
                    "results": formatted_results,
                },
                metadata={
                    "num_results": len(formatted_results),
                    "avg_relevance": sum(r["relevance_score"] for r in formatted_results) / len(formatted_results),
                },
                execution_time_ms=execution_time,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Retrieval failed: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    def format_for_context(self, result: ToolResult) -> str:
        """
        Format retrieval results for inclusion in LLM context.

        This creates a clean string representation that the LLM can use
        to ground its responses.
        """
        if not result.success:
            return f"Retrieval Error: {result.error}"

        output = result.output
        if not output.get("results"):
            return "No relevant knowledge found for the query."

        context_parts = [f"Retrieved Knowledge for: '{output['query']}'", "=" * 50]

        for r in output["results"]:
            context_parts.append(f"\n[Source: {r['source']} | Relevance: {r['relevance_score']}]")
            context_parts.append(r["content"])
            context_parts.append("-" * 30)

        return "\n".join(context_parts)
