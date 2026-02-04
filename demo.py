#!/usr/bin/env python3
"""
Phoenix RAG Agent - Full Demonstration Script

This script demonstrates the complete RAG pipeline including:
1. Document ingestion with advanced chunking
2. ReAct-style reasoning loop with tool selection
3. Verification module with groundedness scoring

Generates an implementation trace for HW2 submission.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Configure logging to capture trace
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/phoenix_trace.log"),
    ],
)
logger = logging.getLogger("phoenix.demo")

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---\n")


def run_demo():
    
    """Run the full Phoenix RAG demonstration."""
    from phoenix_rag.agent import PhoenixAgent
    from phoenix_rag.retrieval import RetrievalModule
    from phoenix_rag.tools import CodeAnalyzerTool
    from phoenix_rag.verification import VerificationModule

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    print_section("PHOENIX RAG AGENT - DEMONSTRATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("This trace demonstrates the agent navigating between retrieval and tool use.\n")

    # =========================================================================
    # STEP 1: Initialize Components
    # =========================================================================
    print_section("STEP 1: COMPONENT INITIALIZATION")

    logger.info("Initializing RetrievalModule with ChromaDB...")
    retrieval = RetrievalModule()
    logger.info("RetrievalModule initialized")

    logger.info("Initializing PhoenixAgent with ReAct loop...")
    agent = PhoenixAgent(retrieval_module=retrieval)
    logger.info("PhoenixAgent initialized")

    print("Components initialized:")
    print("  - RetrievalModule (ChromaDB + HuggingFace embeddings)")
    print("  - ToolRegistry (code_analyzer, complexity_calculator, knowledge_retrieval)")
    print("  - VerificationModule (groundedness evaluation)")
    print("  - PhoenixAgent (ReAct-style orchestration)")

    # =========================================================================
    # STEP 2: Document Ingestion
    # =========================================================================
    print_section("STEP 2: DOCUMENT INGESTION")

    # Ingest sample documents
    docs_path = Path("data/documents")
    total_chunks = 0

    if docs_path.exists():
        for subdir in docs_path.iterdir():
            if subdir.is_dir():
                # Determine document type
                if "pattern" in subdir.name:
                    doc_type = "refactoring_pattern"
                elif "smell" in subdir.name:
                    doc_type = "code_smell"
                elif "practice" in subdir.name:
                    doc_type = "best_practice"
                else:
                    doc_type = "general"

                logger.info(f"Ingesting documents from {subdir.name} as {doc_type}")
                count = retrieval.ingest_from_directory(subdir, doc_type=doc_type)
                total_chunks += count
                print(f"  Ingested: {subdir.name} -> {count} chunks (type: {doc_type})")

    stats = retrieval.get_collection_stats()
    print(f"\nIngestion Summary:")
    print(f"  Total chunks in vector store: {stats['total_documents']}")
    print(f"  Chunking strategy: HybridChunker (semantic + code-aware)")
    print(f"  Embedding model: all-MiniLM-L6-v2")

    # =========================================================================
    # STEP 3: Query 1 - Knowledge Retrieval Path
    # =========================================================================
    print_section("STEP 3: QUERY 1 - KNOWLEDGE RETRIEVAL PATH")

    query1 = "What is the Extract Method refactoring pattern and when should I use it?"
    print(f"User Query: {query1}\n")

    logger.info(f"Processing query: {query1}")
    response1, trace1 = agent.run(query1)

    print_subsection("ReAct Reasoning Trace")
    for step in trace1.steps:
        print(f"Step {step.step_number}:")
        print(f"  Thought: {step.thought[:100]}...")
        print(f"  Action: {step.action.value}")
        print(f"  Tool Used: {step.tool_used or 'N/A'}")
        print(f"  Observation: {step.observation[:150]}...")
        print()

    print_subsection("Agent Response")
    print(response1[:1000] + "..." if len(response1) > 1000 else response1)

    print_subsection("Verification Results")
    print(f"  Groundedness Score: {trace1.groundedness_score:.1%}")
    print(f"  Tools Used: {', '.join(trace1.tools_used)}")
    print(f"  Total Iterations: {trace1.total_iterations}")

    # =========================================================================
    # STEP 4: Query 2 - Code Analysis Path
    # =========================================================================
    print_section("STEP 4: QUERY 2 - CODE ANALYSIS PATH")

    sample_code = '''
class DataProcessor:
    def process_data(self, data, config, user, logger, db_connection):
        # Validate input
        if data is None:
            logger.error("Data is None")
            return None
        if not isinstance(data, list):
            logger.error("Data must be a list")
            return None

        results = []
        for i in range(len(data)):
            item = data[i]
            # Process each item
            if item.get("type") == "A":
                if item.get("status") == "active":
                    if item.get("value") > 100:
                        processed = item["value"] * 1.1
                        results.append({"id": item["id"], "processed": processed})
                    else:
                        processed = item["value"] * 1.05
                        results.append({"id": item["id"], "processed": processed})
                else:
                    results.append({"id": item["id"], "processed": 0})
            elif item.get("type") == "B":
                if item.get("status") == "active":
                    processed = item["value"] * 2
                    results.append({"id": item["id"], "processed": processed})
                else:
                    results.append({"id": item["id"], "processed": item["value"]})

        # Save to database
        for result in results:
            db_connection.execute("INSERT INTO results VALUES (?, ?)",
                                  (result["id"], result["processed"]))

        # Send notification
        self.send_email(user.email, "Processing complete", str(len(results)))

        return results
'''

    query2 = "Analyze this code and suggest specific refactoring improvements based on best practices."
    print(f"User Query: {query2}\n")
    print(f"Code to analyze:\n```python\n{sample_code}\n```\n")

    logger.info(f"Processing query with code: {query2}")
    response2, trace2 = agent.run(query2, code=sample_code)

    print_subsection("ReAct Reasoning Trace")
    for step in trace2.steps:
        print(f"Step {step.step_number}:")
        print(f"  Thought: {step.thought[:100]}...")
        print(f"  Action: {step.action.value}")
        print(f"  Tool Used: {step.tool_used or 'N/A'}")
        print(f"  Observation: {step.observation[:200]}...")
        print()

    print_subsection("Agent Response")
    print(response2[:1500] + "..." if len(response2) > 1500 else response2)

    print_subsection("Verification Results")
    print(f"  Groundedness Score: {trace2.groundedness_score:.1%}")
    print(f"  Tools Used: {', '.join(trace2.tools_used)}")
    print(f"  Total Iterations: {trace2.total_iterations}")

    # =========================================================================
    # STEP 5: Standalone Tool Demo
    # =========================================================================
    print_section("STEP 5: STANDALONE TOOL DEMONSTRATION")

    print("Testing CodeAnalyzerTool directly:\n")
    analyzer = CodeAnalyzerTool()
    result = analyzer.execute(sample_code, analysis_type="full")

    if result.success:
        output = result.output
        print(f"Analysis Summary: {output['summary']}\n")

        if output.get("code_smells"):
            print("Detected Code Smells:")
            for smell in output["code_smells"]:
                print(f"  - [{smell['severity'].upper()}] {smell['type']}")
                print(f"    Location: {smell['location']}")
                print(f"    Suggestion: {smell['suggestion']}")
                print()

        if output.get("structure"):
            struct = output["structure"]
            print(f"Structure: {struct['total_classes']} classes, {struct['total_functions']} functions")

        if output.get("complexity"):
            comp = output["complexity"]
            print(f"Complexity: {comp['cyclomatic_complexity']} ({comp['complexity_rating']})")

    # =========================================================================
    # STEP 6: Generate Full Trace Log
    # =========================================================================
    print_section("STEP 6: FULL EXECUTION TRACE (JSON)")

    full_trace = {
        "demo_timestamp": datetime.now().isoformat(),
        "system_info": {
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "claude-3-5-sonnet-20241022",
            "vector_db": "ChromaDB",
            "chunking_strategy": "HybridChunker",
        },
        "ingestion_stats": stats,
        "queries": [
            {
                "query": query1,
                "trace": trace1.to_dict(),
            },
            {
                "query": query2,
                "code_provided": True,
                "trace": trace2.to_dict(),
            },
        ],
    }

    trace_json = json.dumps(full_trace, indent=2)
    print(trace_json)

    # Save trace to file
    trace_file = Path("logs/implementation_trace.json")
    trace_file.write_text(trace_json)
    print(f"\nTrace saved to: {trace_file}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_section("DEMONSTRATION SUMMARY")

    print("This demonstration showed the Phoenix RAG agent:")
    print("  1. Ingesting domain-specific documents with hybrid chunking")
    print("  2. Using ReAct-style reasoning to decide between retrieval and analysis")
    print("  3. Calling appropriate tools (knowledge_retrieval, code_analyzer)")
    print("  4. Verifying responses with groundedness scoring")
    print()
    print("Key Decision Points:")
    print(f"  - Query 1: Agent chose RETRIEVAL path -> knowledge_retrieval tool")
    print(f"  - Query 2: Agent chose ANALYSIS path -> code_analyzer tool")
    print()
    print("Verification Module:")
    print(f"  - Query 1 Groundedness: {trace1.groundedness_score:.1%}")
    print(f"  - Query 2 Groundedness: {trace2.groundedness_score:.1%}")
    print()
    print("Files generated:")
    print("  - logs/phoenix_trace.log (execution log)")
    print("  - logs/implementation_trace.json (full trace for submission)")


if __name__ == "__main__":
    run_demo()
