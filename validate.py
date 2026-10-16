"""Validate generated artifacts: python validate.py [--data-dir DIR] [--out-dir DIR]."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag_eval import config  # noqa: E402
from rag_eval.validation import validate_artifacts  # noqa: E402


def main(argv=None):
    """Print every failure; return 0 when all checks pass, 1 otherwise."""
    parser = argparse.ArgumentParser(description="Validate evaluation artifacts.")
    parser.add_argument("--data-dir", default=str(config.DEFAULT_DIR))
    parser.add_argument("--out-dir", default=str(config.DEFAULT_DIR))
    args = parser.parse_args(argv)
    errors = validate_artifacts(args.data_dir, args.out_dir)
    for error in errors:
        print(f"FAIL: {error}")
    if errors:
        print(f"{len(errors)} validation failure(s)")
        return config.EXIT_VALIDATION_FAILED
    print("OK: all artifacts present, valid, and reproducible")
    return config.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
