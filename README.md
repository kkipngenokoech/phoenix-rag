# Phoenix RAG

A Retrieval-Augmented Generation (RAG) system for code refactoring assistance. Phoenix helps developers identify code smells, suggests refactoring patterns, and provides best practices grounded in established software engineering principles.

## Features

- **Knowledge Retrieval**: Search a curated knowledge base of refactoring patterns, code smells, and best practices
- **Code Analysis**: Analyze Python code for structural issues, complexity metrics, and code smells
- **ReAct-Style Reasoning**: Agent uses a think-act-observe loop to gather information before responding
- **Groundedness Verification**: Responses are verified against retrieved sources to reduce hallucination
- **Hybrid Chunking**: Intelligent document chunking that preserves semantic meaning and code structure
- **Multiple LLM Support**: Works with Ollama (local), Groq, Anthropic, and OpenAI

## Architecture

```
phoenix-rag/
├── src/phoenix_rag/
│   ├── agent.py              # Main ReAct agent orchestrator
│   ├── config.py             # Configuration management
│   ├── retrieval/
│   │   ├── module.py         # ChromaDB vector store integration
│   │   ├── chunking.py       # Semantic and code-aware chunking
│   │   └── ingestion.py      # Document ingestion pipeline
│   ├── tools/
│   │   ├── registry.py       # Tool management
│   │   ├── code_analyzer.py  # Code smell detection
│   │   ├── complexity_calculator.py  # Cyclomatic complexity metrics
│   │   └── retrieval_tool.py # Knowledge base search
│   └── verification/
│       ├── groundedness.py   # Response verification
│       └── self_evaluation.py # Self-correction
├── data/documents/           # Knowledge base documents
├── app.py                    # Streamlit web interface
└── demo.py                   # CLI demonstration
```

## Installation

### Prerequisites

- Python 3.10+
- Conda (recommended) or pip

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/phoenix-rag.git
cd phoenix-rag
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate phoenix-rag
```

3. Install the package in development mode:
```bash
pip install -e .
```

4. Configure your environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

### LLM Configuration

Phoenix supports multiple LLM providers. Set `LLM_PROVIDER` in your `.env` file:

| Provider | Value | Requirements |
|----------|-------|--------------|
| Auto (recommended) | `auto` | Tries Ollama first, falls back to Groq |
| Ollama (local) | `ollama` | Ollama running locally |
| Groq | `groq` | `GROQ_API_KEY` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | `OPENAI_API_KEY` |

#### Using Ollama (Local, Free)

1. Install Ollama: https://ollama.ai
2. Pull a model:
```bash
ollama pull llama3.2
```
3. Start Ollama:
```bash
ollama serve
```

#### Using Groq (Cloud, Free Tier)

1. Get an API key from https://console.groq.com
2. Add to `.env`:
```
GROQ_API_KEY=gsk_your_key_here
```

## Usage

### Web Interface

Run the Streamlit app:
```bash
streamlit run app.py
```

The web interface provides:
- Chat interface for asking questions about refactoring
- Code analysis tab for pasting and analyzing code
- Agent trace viewer showing reasoning steps
- Quick tools for direct code smell detection and complexity metrics

### CLI Demo

Run the demonstration script:
```bash
python demo.py
```

This demonstrates:
1. Document ingestion with hybrid chunking
2. Knowledge retrieval queries
3. Code analysis with refactoring suggestions
4. Groundedness verification

### Programmatic Usage

```python
from phoenix_rag.agent import PhoenixAgent
from phoenix_rag.config import PhoenixConfig

# Initialize
config = PhoenixConfig()
agent = PhoenixAgent(config)

# Ingest knowledge base
agent.ingest_knowledge("data/documents")

# Ask a question
response, trace = agent.run("What is the Extract Method refactoring pattern?")
print(response)

# Analyze code
code = '''
def process_data(a, b, c, d, e, f):
    # Long method with many parameters
    result = a + b + c + d + e + f
    return result
'''
response, trace = agent.run("Analyze this code for code smells", code=code)
print(response)
```

## Tools

Phoenix provides three built-in tools:

### knowledge_retrieval

Searches the vector database for relevant refactoring knowledge.

Parameters:
- `query`: Search query string
- `doc_type`: Filter by document type (refactoring_pattern, code_smell, best_practice, style_guide, all)
- `num_results`: Number of results to return (1-10)

### code_analyzer

Analyzes Python code for structure and code smells.

Parameters:
- `code`: Python code to analyze
- `analysis_type`: Type of analysis (full, smells, structure, complexity)

Detects code smells:
- Long methods
- Long parameter lists
- God classes
- Deep nesting
- Complex conditionals

### complexity_calculator

Calculates detailed code complexity metrics.

Parameters:
- `code`: Python code to analyze
- `metrics`: List of metrics (cyclomatic, maintainability, halstead, raw, all)

## Knowledge Base

The knowledge base is stored in `data/documents/` with the following structure:

```
data/documents/
├── refactoring_patterns/    # Extract Method, Extract Class, etc.
├── code_smells/             # Long Method, God Class, etc.
├── best_practices/          # SOLID principles, etc.
└── style_guides/            # Python style guidelines
```

### Adding Custom Documents

1. Create a markdown file in the appropriate subdirectory
2. Run ingestion:
```python
agent.retrieval.ingest_from_directory(
    Path("data/documents/your_folder"),
    doc_type="your_type"
)
```

## Deployment

### Streamlit Cloud

1. Push your code to GitHub

2. Go to https://share.streamlit.io and connect your repository

3. Add secrets in the Streamlit Cloud dashboard:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```

4. Deploy

The app automatically detects if Ollama is unavailable and falls back to Groq.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider to use | `auto` |
| `LLM_MODEL` | Model name | `llama3.2` |
| `GROQ_API_KEY` | Groq API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `CHROMA_PERSIST_DIRECTORY` | ChromaDB storage path | `./data/chroma_db` |
| `MAX_ITERATIONS` | Max agent reasoning iterations | `10` |
| `GROUNDEDNESS_THRESHOLD` | Minimum groundedness score | `0.7` |

## Configuration

### PhoenixConfig

The main configuration class with nested configs:

- `LLMConfig`: LLM provider settings
- `EmbeddingConfig`: Embedding model settings
- `VectorDBConfig`: ChromaDB settings
- `ChunkingConfig`: Document chunking parameters
- `AgentConfig`: Agent behavior settings

### Chunking Strategy

Phoenix uses a hybrid chunking strategy:

1. **SemanticChunker**: Preserves paragraph and section boundaries for documentation
2. **CodeAwareChunker**: Keeps code blocks intact, respects function/class boundaries
3. **HybridChunker**: Automatically detects content type and applies appropriate strategy

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest
```

### Code Formatting

```bash
black src/
ruff check src/
```

### Type Checking

```bash
mypy src/
```

## License

MIT License

## Acknowledgments

- Built with LangChain, ChromaDB, and Sentence Transformers
- Refactoring patterns based on Martin Fowler's catalog
- SOLID principles documentation from Robert C. Martin
