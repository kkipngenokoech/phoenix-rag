"""
Test Suggester Tool for Phoenix RAG

This tool analyzes Python code and generates pytest test stubs that:
- Cover main functionality with descriptive test names
- Follow Arrange/Act/Assert (AAA) pattern
- Include edge cases (None, empty, boundary values)
- Provide docstrings explaining test purpose
"""

import ast
import re
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from phoenix_rag.tools.base import BaseTool, ToolCategory, ToolResult


class TestStyle(Enum):
    """Test generation styles"""
    UNIT = "unit"
    INTEGRATION = "integration"
    BOTH = "both"


@dataclass
class FunctionInfo:
    """Information about a function extracted from code"""
    name: str
    args: List[str]
    arg_types: Dict[str, str]
    return_type: Optional[str]
    docstring: Optional[str]
    raises: List[str]
    is_async: bool
    decorators: List[str]
    body_complexity: int


@dataclass
class TestCase:
    """A single test case"""
    name: str
    description: str
    test_type: str
    code: str


class TestSuggesterTool(BaseTool):
    """
    Tool for generating pytest test stubs for Python functions/classes.

    Analyzes code structure and generates comprehensive test cases including:
    - Happy path tests
    - Edge case tests (None, empty, boundary values)
    - Error condition tests
    """

    name = "test_suggester"
    description = (
        "Generate pytest test stubs for Python code. Analyzes functions and classes "
        "to create comprehensive tests including happy path, edge cases, and error conditions. "
        "Use this when you need to create tests for code."
    )
    category = ToolCategory.ANALYSIS
    parameters_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to generate tests for"
            },
            "test_style": {
                "type": "string",
                "enum": ["unit", "integration", "both"],
                "default": "unit",
                "description": "Style of tests to generate"
            },
            "include_edge_cases": {
                "type": "boolean",
                "default": True,
                "description": "Whether to include edge case tests"
            }
        },
        "required": ["code"]
    }

    def __init__(self, config=None):
        """
        Initialize the test suggester tool.

        Args:
            config: Optional PhoenixConfig instance with LLM settings
        """
        self.config = config
        self._llm = None

    @property
    def llm(self):
        """Lazy initialization of LLM."""
        if self._llm is None and self.config is not None:
            self._llm = self._initialize_llm()
        return self._llm

    def set_config(self, config):
        """Set configuration and reinitialize LLM."""
        self.config = config
        self._llm = None

    def _initialize_llm(self):
        """Initialize LLM based on config settings."""
        from phoenix_rag.agent import create_llm
        return create_llm(self.config)

    def execute(
        self,
        code: str,
        test_style: str = "unit",
        include_edge_cases: bool = True,
        **kwargs
    ) -> ToolResult:
        """
        Execute the test suggester tool.

        Args:
            code: Python source code to generate tests for
            test_style: "unit", "integration", or "both"
            include_edge_cases: Whether to include edge case tests

        Returns:
            ToolResult with generated test code
        """
        start_time = time.time()

        if not code or not code.strip():
            return ToolResult(
                success=False,
                output=None,
                error="No code provided to generate tests for",
                execution_time_ms=(time.time() - start_time) * 1000
            )

        if self.llm is None:
            return ToolResult(
                success=False,
                output=None,
                error="LLM not configured. Set config with set_config() first.",
                execution_time_ms=(time.time() - start_time) * 1000
            )

        try:
            # Parse test style
            try:
                style = TestStyle(test_style.lower())
            except ValueError:
                style = TestStyle.UNIT

            # Extract functions from code
            functions = self._extract_functions(code)

            if not functions:
                return ToolResult(
                    success=True,
                    output="# No functions found to test\n# Add functions to your code to generate tests",
                    metadata={"functions_found": 0, "tests_generated": 0},
                    execution_time_ms=(time.time() - start_time) * 1000
                )

            # Build prompt and generate tests
            prompt = self._build_prompt(code, functions, style, include_edge_cases)
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse response
            test_code, summary = self._parse_response(response_text)

            if not test_code:
                return ToolResult(
                    success=False,
                    output=None,
                    error="Failed to generate test code from LLM response",
                    execution_time_ms=(time.time() - start_time) * 1000
                )

            # Count generated tests
            test_count = len(re.findall(r'def test_', test_code))

            # Format output
            output = self._format_output(test_code, functions, test_count, summary)

            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "functions_found": len(functions),
                    "tests_generated": test_count,
                    "test_style": style.value,
                    "include_edge_cases": include_edge_cases
                },
                execution_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Error generating tests: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def _extract_functions(self, code: str) -> List[FunctionInfo]:
        """Extract function information from code using AST."""
        functions = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return functions

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = self._analyze_function(node)
                functions.append(func_info)

        return functions

    def _analyze_function(self, node: ast.FunctionDef) -> FunctionInfo:
        """Analyze a function AST node to extract detailed information."""
        args = [arg.arg for arg in node.args.args]

        arg_types = {}
        for arg in node.args.args:
            if arg.annotation:
                arg_types[arg.arg] = ast.unparse(arg.annotation)

        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)

        docstring = ast.get_docstring(node)

        raises = []
        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                if child.exc:
                    if isinstance(child.exc, ast.Call):
                        if isinstance(child.exc.func, ast.Name):
                            raises.append(child.exc.func.id)
                    elif isinstance(child.exc, ast.Name):
                        raises.append(child.exc.id)

        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    decorators.append(decorator.func.id)

        complexity = self._calculate_complexity(node)

        return FunctionInfo(
            name=node.name,
            args=args,
            arg_types=arg_types,
            return_type=return_type,
            docstring=docstring,
            raises=raises,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            decorators=decorators,
            body_complexity=complexity
        )

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate a simple complexity metric for a function."""
        complexity = 1

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.Try):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1

        return complexity

    def _build_prompt(
        self,
        code: str,
        functions: List[FunctionInfo],
        test_style: TestStyle,
        include_edge_cases: bool
    ) -> str:
        """Build the prompt for LLM to generate tests."""
        func_summaries = []
        for func in functions:
            summary = f"- {func.name}({', '.join(func.args)})"
            if func.return_type:
                summary += f" -> {func.return_type}"
            if func.raises:
                summary += f" [raises: {', '.join(func.raises)}]"
            if func.docstring:
                summary += f"\n  Docstring: {func.docstring[:100]}..."
            func_summaries.append(summary)

        edge_case_instruction = ""
        if include_edge_cases:
            edge_case_instruction = """
EDGE CASES TO INCLUDE:
- None/null inputs for each parameter
- Empty strings, empty lists, empty dicts where applicable
- Boundary values (0, -1, max values)
- Invalid types
"""

        prompt = f"""You are an expert Python test engineer. Generate comprehensive pytest test cases for the following code.

SOURCE CODE:
```python
{code}
```

FUNCTIONS TO TEST:
{chr(10).join(func_summaries)}

TEST STYLE: {test_style.value}
{edge_case_instruction}

REQUIREMENTS:
1. Use pytest framework with proper imports
2. Follow Arrange/Act/Assert (AAA) pattern with comments
3. Use descriptive test function names (test_<function>_<scenario>)
4. Include docstrings explaining what each test verifies
5. Use pytest.raises() for exception testing
6. Use @pytest.mark.parametrize where appropriate
7. Make tests independent and isolated

OUTPUT FORMAT:
Generate a complete, runnable pytest test file.

=== TEST_CODE ===
```python
[Your complete test file here]
```

=== TEST_SUMMARY ===
[Brief summary of tests generated]
"""
        return prompt

    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse LLM response to extract test code and summary."""
        test_code = ""
        summary = ""

        code_match = re.search(
            r'=== TEST_CODE ===\s*```python\s*(.*?)\s*```',
            response,
            re.DOTALL
        )
        if code_match:
            test_code = code_match.group(1).strip()
        else:
            code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
            if code_match:
                test_code = code_match.group(1).strip()

        summary_match = re.search(
            r'=== TEST_SUMMARY ===\s*(.*?)(?:===|$)',
            response,
            re.DOTALL
        )
        if summary_match:
            summary = summary_match.group(1).strip()

        return test_code, summary

    def _format_output(
        self,
        test_code: str,
        functions: List[FunctionInfo],
        test_count: int,
        summary: str
    ) -> str:
        """Format the output for display."""
        output = []

        output.append("=" * 60)
        output.append("GENERATED PYTEST TESTS")
        output.append("=" * 60)

        output.append(f"\nFunctions analyzed: {len(functions)}")
        for func in functions:
            output.append(f"  - {func.name}({', '.join(func.args)})")

        output.append(f"\nTests generated: {test_count}")

        if summary:
            output.append(f"\nSummary: {summary}")

        output.append("\n" + "-" * 60)
        output.append("TEST CODE:")
        output.append("-" * 60)
        output.append(test_code)
        output.append("-" * 60)

        return "\n".join(output)
