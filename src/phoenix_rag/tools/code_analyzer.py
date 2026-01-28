"""
Code Analyzer Tool for Phoenix RAG system.

Analyzes code for structural issues, code smells, and
refactoring opportunities.
"""

import ast
import re
import time
from typing import Optional

from phoenix_rag.tools.base import BaseTool, ToolCategory, ToolResult


class CodeAnalyzerTool(BaseTool):
    """
    Analyzes Python code for structural issues and code smells.

    This tool provides:
    - Detection of common code smells
    - Structural analysis (functions, classes, complexity)
    - Refactoring opportunity identification

    This is a custom tool that allows the agent to analyze
    code before making refactoring suggestions.
    """

    name = "code_analyzer"
    description = """Analyzes Python code for structural issues, code smells, and refactoring opportunities.
    Use this tool when you need to understand the structure of code or identify what needs to be refactored.
    Input should be Python code as a string."""
    category = ToolCategory.ANALYSIS

    parameters_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to analyze",
            },
            "analysis_type": {
                "type": "string",
                "enum": ["full", "smells", "structure", "complexity"],
                "description": "Type of analysis to perform",
                "default": "full",
            },
        },
        "required": ["code"],
    }

    # Code smell detection patterns
    CODE_SMELLS = {
        "long_method": {
            "threshold": 20,
            "description": "Method has too many lines",
        },
        "long_parameter_list": {
            "threshold": 5,
            "description": "Function has too many parameters",
        },
        "deep_nesting": {
            "threshold": 4,
            "description": "Code has deeply nested blocks",
        },
        "god_class": {
            "threshold": 10,
            "description": "Class has too many methods",
        },
        "duplicate_code": {
            "description": "Similar code patterns detected",
        },
        "magic_numbers": {
            "description": "Unexplained numeric literals in code",
        },
    }

    def execute(self, code: str, analysis_type: str = "full") -> ToolResult:
        """Execute code analysis."""
        start_time = time.time()

        try:
            # Parse the code
            tree = ast.parse(code)

            results = {
                "analysis_type": analysis_type,
                "line_count": len(code.split("\n")),
            }

            if analysis_type in ["full", "structure"]:
                results["structure"] = self._analyze_structure(tree)

            if analysis_type in ["full", "smells"]:
                results["code_smells"] = self._detect_code_smells(code, tree)

            if analysis_type in ["full", "complexity"]:
                results["complexity"] = self._calculate_complexity(tree)

            # Generate summary
            results["summary"] = self._generate_summary(results)

            execution_time = (time.time() - start_time) * 1000

            return ToolResult(
                success=True,
                output=results,
                metadata={"lines_analyzed": results["line_count"]},
                execution_time_ms=execution_time,
            )

        except SyntaxError as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Syntax error in code: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Analysis failed: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    def _analyze_structure(self, tree: ast.AST) -> dict:
        """Analyze code structure."""
        classes = []
        functions = []
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [
                    n.name for n in node.body if isinstance(n, ast.FunctionDef)
                ]
                classes.append({
                    "name": node.name,
                    "methods": methods,
                    "method_count": len(methods),
                    "line": node.lineno,
                })
            elif isinstance(node, ast.FunctionDef) and not isinstance(
                getattr(node, "parent", None), ast.ClassDef
            ):
                # Top-level function
                params = [arg.arg for arg in node.args.args]
                functions.append({
                    "name": node.name,
                    "parameters": params,
                    "param_count": len(params),
                    "line": node.lineno,
                    "body_lines": len(node.body),
                })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                else:
                    imports.append(node.module or "")

        return {
            "classes": classes,
            "functions": functions,
            "imports": imports,
            "total_classes": len(classes),
            "total_functions": len(functions),
        }

    def _detect_code_smells(self, code: str, tree: ast.AST) -> list[dict]:
        """Detect code smells in the code."""
        smells = []

        # Check for long methods
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                body_lines = node.end_lineno - node.lineno if hasattr(node, "end_lineno") else len(node.body)
                if body_lines > self.CODE_SMELLS["long_method"]["threshold"]:
                    smells.append({
                        "type": "long_method",
                        "location": f"Function '{node.name}' at line {node.lineno}",
                        "severity": "medium",
                        "description": f"Method has {body_lines} lines (threshold: {self.CODE_SMELLS['long_method']['threshold']})",
                        "suggestion": "Consider breaking this method into smaller, focused functions",
                    })

                # Check for long parameter list
                param_count = len(node.args.args)
                if param_count > self.CODE_SMELLS["long_parameter_list"]["threshold"]:
                    smells.append({
                        "type": "long_parameter_list",
                        "location": f"Function '{node.name}' at line {node.lineno}",
                        "severity": "low",
                        "description": f"Function has {param_count} parameters (threshold: {self.CODE_SMELLS['long_parameter_list']['threshold']})",
                        "suggestion": "Consider using a configuration object or dataclass",
                    })

        # Check for god classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                method_count = sum(1 for n in node.body if isinstance(n, ast.FunctionDef))
                if method_count > self.CODE_SMELLS["god_class"]["threshold"]:
                    smells.append({
                        "type": "god_class",
                        "location": f"Class '{node.name}' at line {node.lineno}",
                        "severity": "high",
                        "description": f"Class has {method_count} methods (threshold: {self.CODE_SMELLS['god_class']['threshold']})",
                        "suggestion": "Consider splitting into smaller, focused classes (Single Responsibility Principle)",
                    })

        # Check for magic numbers
        magic_numbers = re.findall(r"(?<![a-zA-Z_])\b\d+\b(?!\s*[=:])", code)
        # Filter out common acceptable numbers (0, 1, 2, common indices)
        magic_numbers = [n for n in magic_numbers if int(n) > 2 and n not in ["10", "100", "1000"]]
        if len(magic_numbers) > 3:
            smells.append({
                "type": "magic_numbers",
                "location": "Throughout the code",
                "severity": "low",
                "description": f"Found {len(magic_numbers)} magic numbers",
                "suggestion": "Consider extracting magic numbers into named constants",
            })

        # Check for deep nesting
        max_depth = self._calculate_max_nesting(tree)
        if max_depth > self.CODE_SMELLS["deep_nesting"]["threshold"]:
            smells.append({
                "type": "deep_nesting",
                "location": "Code structure",
                "severity": "medium",
                "description": f"Maximum nesting depth is {max_depth} (threshold: {self.CODE_SMELLS['deep_nesting']['threshold']})",
                "suggestion": "Consider using early returns or extracting nested logic into functions",
            })

        return smells

    def _calculate_max_nesting(self, tree: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth."""
        max_depth = current_depth

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                child_depth = self._calculate_max_nesting(node, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._calculate_max_nesting(node, current_depth)
                max_depth = max(max_depth, child_depth)

        return max_depth

    def _calculate_complexity(self, tree: ast.AST) -> dict:
        """Calculate code complexity metrics."""
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            # Add complexity for branches and loops
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, ast.comprehension):
                complexity += 1

        return {
            "cyclomatic_complexity": complexity,
            "complexity_rating": (
                "low" if complexity <= 5 else
                "moderate" if complexity <= 10 else
                "high" if complexity <= 20 else
                "very high"
            ),
        }

    def _generate_summary(self, results: dict) -> str:
        """Generate a human-readable summary of the analysis."""
        summary_parts = []

        if "structure" in results:
            s = results["structure"]
            summary_parts.append(
                f"Structure: {s['total_classes']} classes, {s['total_functions']} functions"
            )

        if "code_smells" in results:
            smell_count = len(results["code_smells"])
            if smell_count == 0:
                summary_parts.append("No code smells detected")
            else:
                high = sum(1 for s in results["code_smells"] if s["severity"] == "high")
                medium = sum(1 for s in results["code_smells"] if s["severity"] == "medium")
                low = sum(1 for s in results["code_smells"] if s["severity"] == "low")
                summary_parts.append(
                    f"Code smells: {smell_count} total ({high} high, {medium} medium, {low} low)"
                )

        if "complexity" in results:
            c = results["complexity"]
            summary_parts.append(
                f"Complexity: {c['cyclomatic_complexity']} ({c['complexity_rating']})"
            )

        return " | ".join(summary_parts)
