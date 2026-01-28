#!/usr/bin/env python3
"""
Setup script for Phoenix RAG system.
Run: python setup.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    print("=" * 60)
    print("Phoenix RAG - Setup Script")
    print("=" * 60)

    # Check Python version
    if sys.version_info < (3, 10):
        print("Error: Python 3.10+ is required")
        sys.exit(1)

    print(f"\nPython version: {sys.version}")

    # Create directories
    print("\nCreating directories...")
    dirs = [
        "data/documents/refactoring_patterns",
        "data/documents/code_smells",
        "data/documents/best_practices",
        "data/documents/style_guides",
        "data/chroma_db",
        "logs",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {d}")

    # Create .env from example if not exists
    env_file = Path(".env")
    env_example = Path(".env.example")
    if not env_file.exists() and env_example.exists():
        print("\nCreating .env from .env.example...")
        env_file.write_text(env_example.read_text())
        print("  Created: .env")
        print("  Please edit .env and add your ANTHROPIC_API_KEY")

    # Install dependencies
    print("\nInstalling dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "."
        ], check=True)
        print("  Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"  Error installing dependencies: {e}")
        print("  Try running: pip install -e . manually")

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Edit .env and add your ANTHROPIC_API_KEY")
    print("2. Run the demo: python demo.py")
    print("3. Or use the CLI: phoenix chat")
    print("4. Or import in Python: from phoenix_rag import PhoenixAgent")


if __name__ == "__main__":
    main()
