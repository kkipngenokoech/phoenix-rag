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
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_secrets():
    """Load secrets from Streamlit Cloud or environment."""
    # Check for Streamlit secrets (cloud deployment)
    if hasattr(st, "secrets") and len(st.secrets) > 0:
        for key in ["LLM_PROVIDER", "LLM_MODEL", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]:
            if key in st.secrets:
                os.environ[key] = st.secrets[key]


@st.cache_resource
def initialize_agent():
    """Initialize the Phoenix agent (cached to avoid reloading)."""
    # Load secrets before importing config
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

    # Ingest different document types
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
    """Display agent execution trace in an expander."""
    with st.expander("🔍 View Agent Reasoning Trace", expanded=False):
        for step in trace.steps:
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"**Step {step.step_number}**")
                action_colors = {
                    "RETRIEVE": "🔎",
                    "ANALYZE": "🔬",
                    "RESPOND": "💬",
                    "CLARIFY": "❓",
                }
                st.markdown(f"{action_colors.get(step.action.value, '•')} {step.action.value}")
            with col2:
                st.markdown(f"*{step.thought[:200]}{'...' if len(step.thought) > 200 else ''}*")
                if step.tool_used:
                    st.caption(f"Tool: `{step.tool_used}`")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Groundedness Score", f"{trace.groundedness_score:.0%}")
        with col2:
            st.metric("Iterations", trace.total_iterations)


def main():
    # Header
    st.title("🔥 Phoenix RAG")
    st.markdown("*AI-powered code refactoring assistant*")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Initialize agent
        with st.spinner("Loading Phoenix agent..."):
            try:
                agent = initialize_agent()
                st.success("✅ Agent loaded")
            except Exception as e:
                st.error(f"Failed to load agent: {e}")
                st.stop()

        # Load knowledge base
        if st.button("📚 Load Knowledge Base", use_container_width=True):
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
        st.header("📖 About")
        st.markdown("""
        **Phoenix RAG** helps you:
        - Identify code smells
        - Suggest refactoring patterns
        - Apply best practices
        - Analyze code complexity
        """)

        st.divider()

        # Model info
        st.caption(f"LLM: {agent.config.llm.model}")
        st.caption(f"Embeddings: {agent.config.embedding.model_name}")

    # Main content area
    tab1, tab2 = st.tabs(["💬 Chat", "🔬 Code Analysis"])

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
                # Show trace for assistant messages
                if message["role"] == "assistant" and i < len(st.session_state.traces):
                    trace = st.session_state.traces[i // 2]  # Every other message is assistant
                    if trace:
                        display_trace(trace)

        # Chat input
        if prompt := st.chat_input("Ask about refactoring patterns, code smells, or best practices..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response, trace = agent.run(prompt)
                        st.markdown(response)
                        display_trace(trace)

                        # Save to history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.session_state.traces.append(trace)
                    except Exception as e:
                        st.error(f"Error: {e}")

        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.traces = []
            st.rerun()

    # Code Analysis tab
    with tab2:
        st.subheader("Analyze Your Code")

        # Code input
        code_input = st.text_area(
            "Paste your code here:",
            height=300,
            placeholder="""# Paste your Python code here
class Example:
    def method_with_issues(self, a, b, c, d, e, f):
        # Long method with many parameters...
        pass
""",
        )

        col1, col2 = st.columns(2)

        with col1:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Full Analysis", "Code Smells Only", "Complexity Only", "Structure Only"],
            )

        with col2:
            include_suggestions = st.checkbox("Include refactoring suggestions", value=True)

        if st.button("🔍 Analyze Code", type="primary", use_container_width=True):
            if not code_input.strip():
                st.warning("Please enter some code to analyze.")
            else:
                with st.spinner("Analyzing code..."):
                    try:
                        # Build the query
                        if include_suggestions:
                            query = f"Analyze this code and suggest specific refactoring improvements based on best practices."
                        else:
                            query = f"Analyze this code for structure, complexity, and code smells."

                        response, trace = agent.run(query, code=code_input)

                        st.subheader("Analysis Results")
                        st.markdown(response)

                        display_trace(trace)

                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

        # Quick analysis using direct tool
        st.divider()
        st.subheader("Quick Tool Analysis")

        if st.button("⚡ Quick Code Smell Detection"):
            if code_input.strip():
                with st.spinner("Detecting code smells..."):
                    result = agent.tools.execute(
                        "code_analyzer",
                        code=code_input,
                        analysis_type="smells",
                    )
                    if result.success:
                        st.json(result.output)
                    else:
                        st.error(result.error)
            else:
                st.warning("Please enter code first.")

        if st.button("📊 Quick Complexity Metrics"):
            if code_input.strip():
                with st.spinner("Calculating metrics..."):
                    result = agent.tools.execute(
                        "complexity_calculator",
                        code=code_input,
                        metrics=["all"],
                    )
                    if result.success:
                        st.json(result.output)
                    else:
                        st.error(result.error)
            else:
                st.warning("Please enter code first.")


if __name__ == "__main__":
    main()
