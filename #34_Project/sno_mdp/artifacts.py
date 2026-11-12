"""Shared serialization and provenance for reproducible experiment artifacts."""

import gzip
import hashlib
import json
import platform
from pathlib import Path

import numpy as np


def save_json(path, data, compact=False):
    options = {"separators": (",", ":")} if compact else {"indent": 2}
    value = (json.dumps(data, allow_nan=False, **options) + "\n").encode("utf-8")
    path.write_bytes(gzip.compress(value, mtime=0) if path.suffix == ".gz" else value)


def read_json(path):
    """Read ordinary JSON or a losslessly compressed experiment trace."""
    path = Path(path)
    value = path.read_bytes()
    if path.suffix == ".gz":
        value = gzip.decompress(value)
    return json.loads(value)


def build_manifest(config, map_data):
    map_bytes = json.dumps(map_data, sort_keys=True, allow_nan=False).encode()
    source_dir = Path(__file__).parent
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "config": config.as_dict(),
        "map_sha256": hashlib.sha256(map_bytes).hexdigest(),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source_dir.glob("*.py"))
        },
        "upstream": json.loads(
            (source_dir.parent / "third_party/upstream.json").read_text()
        ),
        "artifact_access": "evaluator artifacts include ground truth; use only observations to fit learners",
    }
