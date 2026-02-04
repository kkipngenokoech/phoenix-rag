"""
Complexity Calculator Tool for Phoenix RAG system.

Calculates detailed code complexity metrics using radon library.
"""

import time
from typing import Optional

from phoenix_rag.tools.base import BaseTool, ToolCategory, ToolResult


class ComplexityCalculatorTool(BaseTool):
    """
    Calculates detailed code complexity metrics.

    Uses the radon library to compute:
    - Cyclomatic Complexity (CC)
    - Maintainability Index (MI)
    - Halstead metrics
    - Raw metrics (LOC, LLOC, etc.)

    This is a specialized tool for when detailed metrics are needed
    to make refactoring decisions.
    """

    name = "complexity_calculator"
    description = """Calculates detailed code complexity metrics including Cyclomatic Complexity,
    Maintainability Index, and Halstead metrics. Use this tool when you need precise metrics
    to justify or prioritize refactoring decisions."""
    category = ToolCategory.ANALYSIS

    parameters_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to analyze",
            },
            "metrics": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["cyclomatic", "maintainability", "halstead", "raw", "all"],
                },
                "description": "Which metrics to calculate",
                "default": ["all"],
            },
        },
        "required": ["code"],
    }

    def execute(self, code: str, metrics: Optional[list[str]] = None) -> ToolResult:
        """Execute complexity calculation."""
        start_time = time.time()
        metrics = metrics or ["all"]

        try:
            results = {}

            # Import radon modules
            from radon.complexity import cc_visit, cc_rank
            from radon.metrics import mi_visit, h_visit
            from radon.raw import analyze

            if "all" in metrics or "cyclomatic" in metrics:
                results["cyclomatic_complexity"] = self._calculate_cyclomatic(
                    code, cc_visit, cc_rank
                )

            if "all" in metrics or "maintainability" in metrics:
                results["maintainability_index"] = self._calculate_maintainability(
                    code, mi_visit
                )

            if "all" in metrics or "halstead" in metrics:
                results["halstead_metrics"] = self._calculate_halstead(code, h_visit)

            if "all" in metrics or "raw" in metrics:
                results["raw_metrics"] = self._calculate_raw(code, analyze)

            # Generate recommendations
            results["recommendations"] = self._generate_recommendations(results)
            results["summary"] = self._generate_summary(results)

            execution_time = (time.time() - start_time) * 1000

            return ToolResult(
                success=True,
                output=results,
                metadata={"metrics_calculated": list(results.keys())},
                execution_time_ms=execution_time,
            )

        except ImportError:
            # Fallback if radon is not installed
            return self._fallback_analysis(code, start_time)
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Complexity calculation failed: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    def _calculate_cyclomatic(self, code: str, cc_visit, cc_rank) -> dict:
        """Calculate cyclomatic complexity for each function/class."""
        blocks = cc_visit(code)

        results = []
        for block in blocks:
            results.append({
                "name": block.name,
                "type": block.letter,  # F=function, M=method, C=class
                "complexity": block.complexity,
                "rank": cc_rank(block.complexity),
                "start_line": block.lineno,
                "end_line": block.endline,
            })

        total_complexity = sum(b["complexity"] for b in results)
        avg_complexity = total_complexity / len(results) if results else 0

        return {
            "blocks": results,
            "total_complexity": total_complexity,
            "average_complexity": round(avg_complexity, 2),
            "overall_rank": self._get_overall_rank(avg_complexity),
        }

    def _calculate_maintainability(self, code: str, mi_visit) -> dict:
        """Calculate Maintainability Index."""
        mi_score = mi_visit(code, True)

        return {
            "score": round(mi_score, 2),
            "rating": self._get_mi_rating(mi_score),
            "interpretation": self._interpret_mi(mi_score),
        }

    def _calculate_halstead(self, code: str, h_visit) -> dict:
        """Calculate Halstead complexity metrics."""
        try:
            halstead = h_visit(code)

            if not halstead:
                return {"error": "No functions found for Halstead analysis"}

            # Aggregate metrics
            total_h1 = sum(h.h1 for h in halstead)
            total_h2 = sum(h.h2 for h in halstead)
            total_N1 = sum(h.N1 for h in halstead)
            total_N2 = sum(h.N2 for h in halstead)

            return {
                "distinct_operators": total_h1,
                "distinct_operands": total_h2,
                "total_operators": total_N1,
                "total_operands": total_N2,
                "vocabulary": total_h1 + total_h2,
                "length": total_N1 + total_N2,
                "functions_analyzed": len(halstead),
            }
        except Exception as e:
            return {"error": str(e)}

    def _calculate_raw(self, code: str, analyze) -> dict:
        """Calculate raw code metrics."""
        raw = analyze(code)

        return {
            "loc": raw.loc,           # Lines of code
            "lloc": raw.lloc,         # Logical lines of code
            "sloc": raw.sloc,         # Source lines of code
            "comments": raw.comments, # Comment lines
            "multi": raw.multi,       # Multi-line strings
            "blank": raw.blank,       # Blank lines
            "comment_ratio": round(
                raw.comments / raw.loc * 100 if raw.loc > 0 else 0, 2
            ),
        }

    def _get_overall_rank(self, avg_complexity: float) -> str:
        """Get overall complexity rank."""
        if avg_complexity <= 5:
            return "A (low risk)"
        elif avg_complexity <= 10:
            return "B (moderate risk)"
        elif avg_complexity <= 20:
            return "C (high risk)"
        else:
            return "F (very high risk)"

    def _get_mi_rating(self, mi_score: float) -> str:
        """Get maintainability rating."""
        if mi_score >= 20:
            return "Highly maintainable"
        elif mi_score >= 10:
            return "Moderately maintainable"
        else:
            return "Difficult to maintain"

    def _interpret_mi(self, mi_score: float) -> str:
        """Interpret maintainability index."""
        if mi_score >= 20:
            return "Code is clean and well-structured. Low refactoring priority."
        elif mi_score >= 10:
            return "Code has some complexity. Consider targeted refactoring."
        else:
            return "Code is complex and hard to maintain. High refactoring priority."

    def _generate_recommendations(self, results: dict) -> list[str]:
        """Generate recommendations based on metrics."""
        recommendations = []

        if "cyclomatic_complexity" in results:
            cc = results["cyclomatic_complexity"]
            if cc["average_complexity"] > 10:
                recommendations.append(
                    "High cyclomatic complexity - break down complex functions"
                )

            # Find specific high-complexity blocks
            high_cc = [b for b in cc["blocks"] if b["complexity"] > 10]
            for block in high_cc[:3]:  # Top 3
                recommendations.append(
                    f"Refactor '{block['name']}' (complexity: {block['complexity']})"
                )

        if "maintainability_index" in results:
            mi = results["maintainability_index"]
            if mi["score"] < 20:
                recommendations.append(
                    f"Maintainability Index is {mi['score']} - consider restructuring"
                )

        if "raw_metrics" in results:
            raw = results["raw_metrics"]
            if raw["comment_ratio"] < 5:
                recommendations.append(
                    "Low comment ratio - consider adding documentation"
                )

        return recommendations

    def _generate_summary(self, results: dict) -> str:
        """Generate summary of complexity analysis."""
        parts = []

        if "cyclomatic_complexity" in results:
            cc = results["cyclomatic_complexity"]
            parts.append(f"Cyclomatic: {cc['overall_rank']}")

        if "maintainability_index" in results:
            mi = results["maintainability_index"]
            parts.append(f"Maintainability: {mi['score']} ({mi['rating']})")

        if "raw_metrics" in results:
            raw = results["raw_metrics"]
            parts.append(f"LOC: {raw['loc']}")

        return " | ".join(parts)

    def _fallback_analysis(self, code: str, start_time: float) -> ToolResult:
        """Fallback analysis when radon is not available."""
        import ast

        try:
            tree = ast.parse(code)

            # Basic complexity calculation
            complexity = 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For)):
                    complexity += 1
                elif isinstance(node, ast.ExceptHandler):
                    complexity += 1

            lines = len(code.split("\n"))

            return ToolResult(
                success=True,
                output={
                    "cyclomatic_complexity": {
                        "total_complexity": complexity,
                        "note": "Basic calculation (radon not available)",
                    },
                    "raw_metrics": {"loc": lines},
                    "summary": f"Basic complexity: {complexity} | LOC: {lines}",
                },
                metadata={"fallback": True},
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )
