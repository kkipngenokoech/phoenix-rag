"""
Command-line interface for Phoenix RAG agent.
"""

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def cmd_ingest(args):
    """Ingest documents into the knowledge base."""
    from phoenix_rag.retrieval import RetrievalModule

    console.print("[bold blue]Phoenix RAG - Document Ingestion[/bold blue]")

    retrieval = RetrievalModule()
    path = Path(args.path)

    if path.is_file():
        chunks = retrieval.ingest_document(
            content=path.read_text(),
            source=str(path),
            doc_type=args.doc_type,
        )
    else:
        chunks = retrieval.ingest_from_directory(
            directory=path,
            doc_type=args.doc_type,
        )

    console.print(f"[green]Successfully ingested {chunks} chunks[/green]")

    stats = retrieval.get_collection_stats()
    console.print(f"Total documents in collection: {stats['total_documents']}")


def cmd_chat(args):
    """Interactive chat with the agent."""
    from phoenix_rag.agent import PhoenixAgent

    console.print(Panel.fit(
        "[bold blue]Phoenix RAG Agent[/bold blue]\n"
        "Type your questions about code refactoring.\n"
        "Type 'quit' to exit, 'trace' to see last execution trace.",
        title="Welcome",
    ))

    agent = PhoenixAgent()

    while True:
        try:
            user_input = console.input("\n[bold green]You:[/bold green] ")

            if user_input.lower() in ('quit', 'exit', 'q'):
                console.print("[yellow]Goodbye![/yellow]")
                break

            if user_input.lower() == 'trace':
                console.print(Panel(agent.get_trace_log(), title="Last Execution Trace"))
                continue

            if not user_input.strip():
                continue

            console.print("\n[bold blue]Phoenix:[/bold blue] ", end="")

            response, trace = agent.run(user_input)
            console.print(Markdown(response))

            # Show groundedness score
            console.print(
                f"\n[dim]Groundedness: {trace.groundedness_score:.0%} | "
                f"Iterations: {trace.total_iterations} | "
                f"Tools: {', '.join(trace.tools_used) or 'None'}[/dim]"
            )

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'quit' to exit.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def cmd_analyze(args):
    """Analyze a code file."""
    from phoenix_rag.tools import CodeAnalyzerTool, ComplexityCalculatorTool

    console.print("[bold blue]Phoenix RAG - Code Analysis[/bold blue]")

    code_path = Path(args.file)
    if not code_path.exists():
        console.print(f"[red]File not found: {args.file}[/red]")
        return

    code = code_path.read_text()

    analyzer = CodeAnalyzerTool()
    result = analyzer.execute(code, analysis_type="full")

    if result.success:
        output = result.output
        console.print(Panel(f"[bold]Summary:[/bold] {output['summary']}", title="Analysis Results"))

        if output.get("code_smells"):
            console.print("\n[bold]Code Smells Detected:[/bold]")
            for smell in output["code_smells"]:
                console.print(f"  - [{smell['severity'].upper()}] {smell['type']}: {smell['description']}")
                console.print(f"    Location: {smell['location']}")
                console.print(f"    Suggestion: {smell['suggestion']}")
    else:
        console.print(f"[red]Analysis failed: {result.error}[/red]")


def cmd_demo(args):
    """Run a demonstration of the Phoenix RAG agent."""
    from phoenix_rag.agent import PhoenixAgent
    from phoenix_rag.retrieval import RetrievalModule
    import json

    console.print(Panel.fit(
        "[bold blue]Phoenix RAG Agent Demonstration[/bold blue]\n"
        "This demo shows the full RAG pipeline with execution trace.",
        title="Demo",
    ))

    # Step 1: Initialize and ingest documents
    console.print("\n[bold]Step 1: Initializing agent and ingesting knowledge...[/bold]")

    agent = PhoenixAgent()

    # Ingest sample documents
    docs_path = Path(__file__).parent.parent.parent / "data" / "documents"
    if docs_path.exists():
        for subdir in docs_path.iterdir():
            if subdir.is_dir():
                doc_type = subdir.name.replace("_", " ").replace("refactoring patterns", "refactoring_pattern").replace("code smells", "code_smell").replace("best practices", "best_practice")
                # Map to correct doc_type
                if "pattern" in subdir.name:
                    doc_type = "refactoring_pattern"
                elif "smell" in subdir.name:
                    doc_type = "code_smell"
                elif "practice" in subdir.name:
                    doc_type = "best_practice"
                else:
                    doc_type = "general"

                count = agent.retrieval.ingest_from_directory(subdir, doc_type=doc_type)
                console.print(f"  Ingested {count} chunks from {subdir.name}")

    stats = agent.retrieval.get_collection_stats()
    console.print(f"  [green]Total documents in knowledge base: {stats['total_documents']}[/green]")

    # Step 2: Demo queries
    console.print("\n[bold]Step 2: Running demo queries...[/bold]")

    demo_queries = [
        "What is the Extract Method refactoring pattern and when should I use it?",
        "How do I identify and fix a God Class code smell?",
    ]

    demo_code = '''
class OrderProcessor:
    def process(self, order):
        # Validate
        if not order.items:
            return None
        if not order.customer:
            return None

        # Calculate total
        total = 0
        for item in order.items:
            total += item.price * item.quantity

        # Apply discount
        if order.customer.is_premium:
            total = total * 0.9

        # Process payment
        if order.payment_method == "card":
            result = self.charge_card(order.customer.card, total)
        elif order.payment_method == "paypal":
            result = self.charge_paypal(order.customer.paypal, total)

        # Send email
        self.send_email(order.customer.email, f"Order total: {total}")

        return result
'''

    for i, query in enumerate(demo_queries, 1):
        console.print(f"\n[bold cyan]Query {i}:[/bold cyan] {query}")
        console.print("-" * 60)

        response, trace = agent.run(query)

        console.print(f"\n[bold green]Response:[/bold green]")
        console.print(Markdown(response))

        console.print(f"\n[bold yellow]Execution Trace:[/bold yellow]")
        console.print(f"  Iterations: {trace.total_iterations}")
        console.print(f"  Tools Used: {', '.join(trace.tools_used) or 'None'}")
        console.print(f"  Groundedness Score: {trace.groundedness_score:.1%}")

        if trace.steps:
            console.print(f"\n  [dim]Reasoning Steps:[/dim]")
            for step in trace.steps[:3]:
                console.print(f"    Step {step.step_number}: {step.action.value} - {step.thought[:80]}...")

    # Step 3: Code analysis demo
    console.print("\n[bold]Step 3: Analyzing sample code...[/bold]")

    query_with_code = "Analyze this code and suggest refactoring improvements."
    response, trace = agent.run(query_with_code, code=demo_code)

    console.print(f"\n[bold green]Analysis Response:[/bold green]")
    console.print(Markdown(response))

    # Step 4: Show full trace
    console.print("\n[bold]Step 4: Full Execution Trace (JSON):[/bold]")
    console.print(Panel(json.dumps(trace.to_dict(), indent=2), title="Trace Log"))

    console.print("\n[bold green]Demo complete![/bold green]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Phoenix RAG - Code Refactoring Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into knowledge base")
    ingest_parser.add_argument("path", help="Path to file or directory to ingest")
    ingest_parser.add_argument(
        "--doc-type",
        default="general",
        choices=["refactoring_pattern", "code_smell", "best_practice", "style_guide", "general"],
        help="Type of document",
    )

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat with the agent")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a code file")
    analyze_parser.add_argument("file", help="Path to code file to analyze")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demonstration")

    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
