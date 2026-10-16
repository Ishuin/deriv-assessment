"""Regenerate all derived artifacts: python run.py [--llm auto|anthropic|stub|replay]."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag_eval.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
