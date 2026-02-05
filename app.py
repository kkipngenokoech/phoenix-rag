"""
Phoenix RAG - Streamlit Web Interface

A web-based UI for the Phoenix code refactoring assistant.
Run with: streamlit run app.py
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path for Streamlit Cloud deployment
src_path = Path(__file__).parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import streamlit as st
import json
import time

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Phoenix RAG",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }

    .metric-card h3 {
        margin: 0 0 0.5rem 0;
        color: #374151;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }

    /* Code smell card */
    .smell-card {
        background: #fff;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .smell-card.high {
        border-left: 4px solid #ef4444;
    }

    .smell-card.medium {
        border-left: 4px solid #f59e0b;
    }

    .smell-card.low {
        border-left: 4px solid #10b981;
    }

    /* Trace step styling */
    .trace-step {
        background: #f9fafb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }

    .trace-action {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .trace-action.retrieve {
        background: #dbeafe;
        color: #1d4ed8;
    }

    .trace-action.analyze {
        background: #fef3c7;
        color: #92400e;
    }

    .trace-action.respond {
        background: #d1fae5;
        color: #065f46;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        border-radius: 8px 8px 0 0;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }

    /* Input styling */
    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
    }

    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
</style>
""", unsafe_allow_html=True)


def load_secrets():
    """Load secrets from Streamlit Cloud or environment."""
    if hasattr(st, "secrets") and len(st.secrets) > 0:
        for key in ["LLM_PROVIDER", "LLM_MODEL", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]:
            if key in st.secrets:
                os.environ[key] = st.secrets[key]


@st.cache_resource
def initialize_agent():
    """Initialize the Phoenix agent (cached to avoid reloading)."""
    load_secrets()

    from phoenix_rag.agent import PhoenixAgent
    from phoenix_rag.config import PhoenixConfig

    config = PhoenixConfig()
    config.ensure_directories()

    agent = PhoenixAgent(config)
    return agent


@st.cache_resource
def load_knowledge_base(_agent):
    """Load documents into the knowledge base."""
    docs_dir = Path("data/documents")

    if not docs_dir.exists():
        return 0, "No documents directory found"

    total_chunks = 0
    sources = []

    doc_types = {
        "refactoring_patterns": "refactoring_pattern",
        "code_smells": "code_smell",
        "best_practices": "best_practice",
        "style_guides": "style_guide",
    }

    for folder, doc_type in doc_types.items():
        folder_path = docs_dir / folder
        if folder_path.exists():
            chunks = _agent.retrieval.ingest_from_directory(
                folder_path,
                doc_type=doc_type,
            )
            total_chunks += chunks
            sources.append(f"{folder}: {chunks} chunks")

    return total_chunks, sources


def display_trace(trace):
    """Display agent execution trace."""
    if not trace:
        return

    with st.expander("View Agent Reasoning Trace", expanded=False):
        if trace.steps:
            for step in trace.steps:
                action_value = step.action.value if step.action else "UNKNOWN"
                tool_info = f" | Tool: {step.tool_used}" if step.tool_used else ""

                st.markdown(f"**Step {step.step_number}** - `{action_value}`{tool_info}")

                thought_text = step.thought[:300] if step.thought else "No thought recorded"
                if step.thought and len(step.thought) > 300:
                    thought_text += "..."
                st.caption(thought_text)
                st.divider()
        else:
            st.info("No reasoning steps recorded.")

        col1, col2 = st.columns(2)
        with col1:
            score = trace.groundedness_score if trace.groundedness_score is not None else 0.0
            st.metric("Groundedness Score", f"{score:.0%}")
        with col2:
            st.metric("Iterations", trace.total_iterations or 0)


def display_code_smells(smells):
    """Display code smells in a formatted way."""
    if not smells:
        st.info("No code smells detected.")
        return

    for smell in smells:
        severity = smell.get("severity", "medium").lower()
        st.markdown(f"""
        <div class="smell-card {severity}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <strong>{smell.get('name', 'Unknown Smell')}</strong>
                <span style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase;">{severity}</span>
            </div>
            <p style="color: #4b5563; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                {smell.get('description', '')}
            </p>
            {f'<p style="color: #667eea; margin: 0.5rem 0 0 0; font-size: 0.85rem;"><strong>Location:</strong> {smell.get("location", "")}</p>' if smell.get("location") else ''}
        </div>
        """, unsafe_allow_html=True)


def display_metrics(metrics):
    """Display complexity metrics in a clean format."""
    # Extract values from nested structure
    cc_data = metrics.get('cyclomatic_complexity', {})
    mi_data = metrics.get('maintainability_index', {})
    raw_data = metrics.get('raw_metrics', {})

    # Summary
    if metrics.get('summary'):
        st.info(metrics['summary'])

    # Main metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        cc_value = cc_data.get('average_complexity', 'N/A') if isinstance(cc_data, dict) else 'N/A'
        cc_rank = cc_data.get('overall_rank', '') if isinstance(cc_data, dict) else ''
        st.metric("Cyclomatic Complexity", f"{cc_value}", help=cc_rank)

    with col2:
        mi_value = mi_data.get('score', 'N/A') if isinstance(mi_data, dict) else 'N/A'
        mi_rating = mi_data.get('rating', '') if isinstance(mi_data, dict) else ''
        st.metric("Maintainability Index", f"{mi_value}", help=mi_rating)

    with col3:
        loc = raw_data.get('loc', 'N/A') if isinstance(raw_data, dict) else 'N/A'
        st.metric("Lines of Code", loc)

    with col4:
        sloc = raw_data.get('sloc', 'N/A') if isinstance(raw_data, dict) else 'N/A'
        st.metric("Source Lines", sloc)

    # Recommendations
    if metrics.get('recommendations'):
        st.subheader("Recommendations")
        for rec in metrics['recommendations']:
            st.warning(rec)

    # Function-level complexity breakdown
    if isinstance(cc_data, dict) and cc_data.get('blocks'):
        st.subheader("Function Complexity Breakdown")
        for block in cc_data['blocks']:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.text(f"{block.get('name', 'unknown')}")
            with col2:
                st.text(f"CC: {block.get('complexity', '?')}")
            with col3:
                st.text(f"Rank: {block.get('rank', '?')}")


def main():
    # Header
    st.markdown('<h1 class="main-header">Phoenix RAG</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered code refactoring assistant</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        # Initialize agent
        with st.spinner("Loading Phoenix agent..."):
            try:
                agent = initialize_agent()
                st.success("Agent loaded successfully")
            except Exception as e:
                st.error(f"Failed to load agent: {e}")
                st.stop()

        # Load knowledge base
        if st.button("Load Knowledge Base", use_container_width=True):
            with st.spinner("Ingesting documents..."):
                total, sources = load_knowledge_base(agent)
                if total > 0:
                    st.success(f"Loaded {total} chunks")
                    for s in sources:
                        st.caption(s)
                else:
                    st.warning("No documents loaded")

        st.divider()

        # Info
        st.header("About")
        st.markdown("""
        **Phoenix RAG** helps you:
        - Identify code smells
        - Suggest refactoring patterns
        - Apply best practices
        - Analyze code complexity
        - Generate test stubs
        """)

        st.divider()

        # Model info
        st.caption(f"LLM: {agent.config.llm.model}")
        st.caption(f"Embeddings: {agent.config.embedding.model_name}")

    # Main content area with 4 tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Code Analysis", "Generate Tests", "Quick Tools"])

    # Chat tab
    with tab1:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "traces" not in st.session_state:
            st.session_state.traces = []

        # Display chat history
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and i < len(st.session_state.traces):
                    trace = st.session_state.traces[i // 2]
                    if trace:
                        display_trace(trace)

        # Chat input
        if prompt := st.chat_input("Ask about refactoring patterns, code smells, or best practices..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response, trace = agent.run(prompt)

                        # Ensure response is displayed
                        if response and isinstance(response, str):
                            st.markdown(response)
                        else:
                            st.warning("No response generated. Check the trace below for details.")

                        # Display trace if available
                        if trace and trace.steps:
                            display_trace(trace)

                        st.session_state.messages.append({"role": "assistant", "content": response or "No response"})
                        st.session_state.traces.append(trace)
                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())

        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.traces = []
            st.rerun()

    # Code Analysis tab
    with tab2:
        st.subheader("Analyze Your Code")

        code_input = st.text_area(
            "Paste your code here:",
            height=300,
            placeholder="""# Paste your Python code here
class Example:
    def method_with_issues(self, a, b, c, d, e, f):
        # Long method with many parameters...
        pass
""",
            key="analysis_code"
        )

        col1, col2 = st.columns(2)

        with col1:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Full Analysis", "Code Smells Only", "Complexity Only", "Structure Only"],
            )

        with col2:
            include_suggestions = st.checkbox("Include refactoring suggestions", value=True)

        if st.button("Analyze Code", type="primary", use_container_width=True):
            if not code_input.strip():
                st.warning("Please enter some code to analyze.")
            else:
                with st.spinner("Analyzing code..."):
                    try:
                        if include_suggestions:
                            query = "Analyze this code and suggest specific refactoring improvements based on best practices."
                        else:
                            query = "Analyze this code for structure, complexity, and code smells."

                        response, trace = agent.run(query, code=code_input)

                        st.subheader("Analysis Results")
                        st.markdown(response)

                        display_trace(trace)

                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

    # Generate Tests tab
    with tab3:
        st.subheader("Generate Test Stubs")
        st.markdown("Generate pytest test stubs for your Python code with comprehensive coverage.")

        test_code_input = st.text_area(
            "Paste your code here:",
            height=300,
            placeholder="""# Paste the Python code you want to generate tests for
def calculate_total(items, tax_rate=0.1):
    \"\"\"Calculate total price with tax.\"\"\"
    subtotal = sum(item['price'] * item['quantity'] for item in items)
    return subtotal * (1 + tax_rate)
""",
            key="test_code"
        )

        col1, col2 = st.columns(2)

        with col1:
            test_style = st.selectbox(
                "Test Style",
                ["unit", "integration", "both"],
                format_func=lambda x: x.title()
            )

        with col2:
            include_edge_cases = st.checkbox("Include edge case tests", value=True)

        if st.button("Generate Tests", type="primary", use_container_width=True):
            if not test_code_input.strip():
                st.warning("Please enter some code to generate tests for.")
            else:
                with st.spinner("Generating test stubs..."):
                    try:
                        result = agent.tools.execute(
                            "test_suggester",
                            code=test_code_input,
                            test_style=test_style,
                            include_edge_cases=include_edge_cases,
                        )

                        if result.success:
                            st.subheader("Generated Tests")
                            st.code(result.output, language="python")

                            if result.metadata:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Functions Found", result.metadata.get("functions_found", 0))
                                with col2:
                                    st.metric("Tests Generated", result.metadata.get("tests_generated", 0))
                        else:
                            st.error(f"Failed to generate tests: {result.error}")

                    except Exception as e:
                        st.error(f"Test generation failed: {e}")

    # Quick Tools tab
    with tab4:
        st.subheader("Quick Analysis Tools")
        st.markdown("Run individual analysis tools directly without the full agent reasoning loop.")

        quick_code_input = st.text_area(
            "Paste your code here:",
            height=250,
            placeholder="""# Paste your Python code here for quick analysis
def example_function():
    pass
""",
            key="quick_code"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Detect Code Smells", use_container_width=True):
                if quick_code_input.strip():
                    with st.spinner("Detecting code smells..."):
                        result = agent.tools.execute(
                            "code_analyzer",
                            code=quick_code_input,
                            analysis_type="smells",
                        )
                        if result.success:
                            if isinstance(result.output, dict) and "code_smells" in result.output:
                                display_code_smells(result.output["code_smells"])
                            else:
                                st.json(result.output)
                        else:
                            st.error(result.error)
                else:
                    st.warning("Please enter code first.")

        with col2:
            if st.button("Calculate Complexity", use_container_width=True):
                if quick_code_input.strip():
                    with st.spinner("Calculating metrics..."):
                        result = agent.tools.execute(
                            "complexity_calculator",
                            code=quick_code_input,
                            metrics=["all"],
                        )
                        if result.success:
                            if isinstance(result.output, dict):
                                display_metrics(result.output)
                                st.divider()
                                st.subheader("Detailed Metrics")
                                st.json(result.output)
                            else:
                                st.json(result.output)
                        else:
                            st.error(result.error)
                else:
                    st.warning("Please enter code first.")

        st.divider()

        # Structure analysis
        if st.button("Analyze Structure", use_container_width=True):
            if quick_code_input.strip():
                with st.spinner("Analyzing structure..."):
                    result = agent.tools.execute(
                        "code_analyzer",
                        code=quick_code_input,
                        analysis_type="structure",
                    )
                    if result.success:
                        st.json(result.output)
                    else:
                        st.error(result.error)
            else:
                st.warning("Please enter code first.")


if __name__ == "__main__":
    main()
