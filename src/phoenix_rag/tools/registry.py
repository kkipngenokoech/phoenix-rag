"""
Tool Registry for Phoenix RAG system.

Manages available tools and provides them to the agent.
"""

import logging
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from phoenix_rag.tools.base import BaseTool, ToolResult
from phoenix_rag.tools.code_analyzer import CodeAnalyzerTool
from phoenix_rag.tools.complexity_calculator import ComplexityCalculatorTool
from phoenix_rag.tools.retrieval_tool import RetrievalTool
from phoenix_rag.tools.test_suggester import TestSuggesterTool

logger = logging.getLogger(__name__)


# Pydantic schemas for LangChain tool integration
class CodeAnalyzerInput(BaseModel):
    """Input schema for code analyzer tool."""
    code: str = Field(description="The Python code to analyze")
    analysis_type: str = Field(
        default="full",
        description="Type of analysis: full, smells, structure, or complexity"
    )


class ComplexityCalculatorInput(BaseModel):
    """Input schema for complexity calculator tool."""
    code: str = Field(description="The Python code to analyze")
    metrics: list[str] = Field(
        default=["all"],
        description="Metrics to calculate: cyclomatic, maintainability, halstead, raw, or all"
    )


class KnowledgeRetrievalInput(BaseModel):
    """Input schema for knowledge retrieval tool."""
    query: str = Field(description="The search query for refactoring knowledge")
    doc_type: str = Field(
        default="all",
        description="Type of document: refactoring_pattern, code_smell, best_practice, style_guide, or all"
    )
    num_results: int = Field(
        default=5,
        description="Number of results to return (1-10)"
    )


class TestSuggesterInput(BaseModel):
    """Input schema for test suggester tool."""
    code: str = Field(description="The Python code to generate tests for")
    test_style: str = Field(
        default="unit",
        description="Style of tests: unit, integration, or both"
    )
    include_edge_cases: bool = Field(
        default=True,
        description="Whether to include edge case tests"
    )


class ToolRegistry:
    """
    Registry for managing Phoenix agent tools.

    Provides:
    - Tool registration and discovery
    - LangChain tool conversion
    - Tool execution tracking
    """

    def __init__(self, retrieval_module=None):
        """
        Initialize the tool registry.

        Args:
            retrieval_module: Optional RetrievalModule for the retrieval tool
        """
        self.tools: dict[str, BaseTool] = {}
        self.execution_history: list[dict] = []

        # Register default tools
        self._register_default_tools(retrieval_module)

    def _register_default_tools(self, retrieval_module=None) -> None:
        """Register the default Phoenix tools."""
        # Code Analyzer
        self.register(CodeAnalyzerTool())

        # Complexity Calculator
        self.register(ComplexityCalculatorTool())

        # Retrieval Tool (with module if provided)
        retrieval_tool = RetrievalTool(retrieval_module)
        self.register(retrieval_tool)

        # Test Suggester Tool
        self.register(TestSuggesterTool())

        logger.info(f"Registered {len(self.tools)} default tools")

    def register(self, tool: BaseTool) -> None:
        """Register a tool with the registry."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> list[dict]:
        """List all available tools with descriptions."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "category": tool.category.value,
            }
            for tool in self.tools.values()
        ]

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool and track the execution."""
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool not found: {tool_name}",
            )

        # Validate parameters
        valid, error = tool.validate_parameters(**kwargs)
        if not valid:
            return ToolResult(
                success=False,
                output=None,
                error=error,
            )

        # Execute
        result = tool.execute(**kwargs)

        # Track execution
        self.execution_history.append({
            "tool": tool_name,
            "parameters": kwargs,
            "success": result.success,
            "execution_time_ms": result.execution_time_ms,
        })

        return result

    def set_retrieval_module(self, retrieval_module) -> None:
        """Update the retrieval module for the retrieval tool."""
        retrieval_tool = self.get("knowledge_retrieval")
        if retrieval_tool and isinstance(retrieval_tool, RetrievalTool):
            retrieval_tool.set_retrieval_module(retrieval_module)
            logger.info("Updated retrieval module for knowledge_retrieval tool")

    def set_config(self, config) -> None:
        """Set configuration for tools that require it (e.g., LLM-based tools)."""
        test_suggester = self.get("test_suggester")
        if test_suggester and isinstance(test_suggester, TestSuggesterTool):
            test_suggester.set_config(config)
            logger.info("Set config for test_suggester tool")

    def get_langchain_tools(self) -> list[StructuredTool]:
        """
        Convert tools to LangChain StructuredTool format.

        This allows seamless integration with LangChain agents.
        """
        langchain_tools = []

        # Code Analyzer
        code_analyzer = self.get("code_analyzer")
        if code_analyzer:
            langchain_tools.append(
                StructuredTool.from_function(
                    func=lambda code, analysis_type="full": code_analyzer.execute(
                        code=code, analysis_type=analysis_type
                    ).to_string(),
                    name="code_analyzer",
                    description=code_analyzer.description,
                    args_schema=CodeAnalyzerInput,
                )
            )

        # Complexity Calculator
        complexity_calc = self.get("complexity_calculator")
        if complexity_calc:
            langchain_tools.append(
                StructuredTool.from_function(
                    func=lambda code, metrics=["all"]: complexity_calc.execute(
                        code=code, metrics=metrics
                    ).to_string(),
                    name="complexity_calculator",
                    description=complexity_calc.description,
                    args_schema=ComplexityCalculatorInput,
                )
            )

        # Knowledge Retrieval
        retrieval = self.get("knowledge_retrieval")
        if retrieval:
            langchain_tools.append(
                StructuredTool.from_function(
                    func=lambda query, doc_type="all", num_results=5: retrieval.execute(
                        query=query, doc_type=doc_type, num_results=num_results
                    ).to_string(),
                    name="knowledge_retrieval",
                    description=retrieval.description,
                    args_schema=KnowledgeRetrievalInput,
                )
            )

        # Test Suggester
        test_suggester = self.get("test_suggester")
        if test_suggester:
            langchain_tools.append(
                StructuredTool.from_function(
                    func=lambda code, test_style="unit", include_edge_cases=True: test_suggester.execute(
                        code=code, test_style=test_style, include_edge_cases=include_edge_cases
                    ).to_string(),
                    name="test_suggester",
                    description=test_suggester.description,
                    args_schema=TestSuggesterInput,
                )
            )

        return langchain_tools

    def get_tool_descriptions_for_prompt(self) -> str:
        """Get formatted tool descriptions for inclusion in prompts."""
        descriptions = ["Available Tools:", "=" * 40]

        for tool in self.tools.values():
            descriptions.append(f"\n{tool.name}")
            descriptions.append(f"  Category: {tool.category.value}")
            descriptions.append(f"  Description: {tool.description}")
            descriptions.append(f"  Parameters: {tool.parameters_schema}")

        return "\n".join(descriptions)

    def get_execution_stats(self) -> dict:
        """Get statistics about tool executions."""
        if not self.execution_history:
            return {"total_executions": 0}

        total = len(self.execution_history)
        successful = sum(1 for e in self.execution_history if e["success"])

        by_tool = {}
        for e in self.execution_history:
            tool = e["tool"]
            if tool not in by_tool:
                by_tool[tool] = {"count": 0, "success": 0, "total_time_ms": 0}
            by_tool[tool]["count"] += 1
            if e["success"]:
                by_tool[tool]["success"] += 1
            by_tool[tool]["total_time_ms"] += e["execution_time_ms"]

        return {
            "total_executions": total,
            "successful_executions": successful,
            "success_rate": successful / total,
            "by_tool": by_tool,
        }
