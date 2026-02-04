"""
Base classes for Phoenix tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class ToolCategory(Enum):
    """Categories of tools available to the agent."""
    RETRIEVAL = "retrieval"      # Knowledge retrieval from vector store
    ANALYSIS = "analysis"        # Code analysis tools
    EXTERNAL = "external"        # External API/service tools
    UTILITY = "utility"          # Utility functions


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    execution_time_ms: float = 0.0

    def to_string(self) -> str:
        """Convert result to string for LLM consumption."""
        if not self.success:
            return f"Error: {self.error}"
        if isinstance(self.output, str):
            return self.output
        return str(self.output)


class BaseTool(ABC):
    """
    Abstract base class for Phoenix tools.

    All tools must implement:
    - name: Unique identifier for the tool
    - description: What the tool does (used by LLM for selection)
    - execute: The actual tool logic
    """

    name: str
    description: str
    category: ToolCategory
    parameters_schema: dict  # JSON schema for parameters

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def get_langchain_tool_schema(self) -> dict:
        """Get tool schema in LangChain format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    def validate_parameters(self, **kwargs) -> tuple[bool, Optional[str]]:
        """Validate input parameters against schema."""
        required = self.parameters_schema.get("required", [])
        for param in required:
            if param not in kwargs:
                return False, f"Missing required parameter: {param}"
        return True, None
