"""Read and write JSON artifacts in a stable, diff-friendly format."""

import json
import os
from pathlib import Path


def write_json(path, data):
    """Write `data` as sorted-key, 2-space JSON via temp file + rename (atomic)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_text(path, text):
    """Write UTF-8 text via temp file + rename (atomic)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)
