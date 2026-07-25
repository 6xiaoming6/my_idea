"""Pure-Python fingerprints for reproducible formal experiment reuse."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def source_tree_sha256(root: str | Path) -> str:
    """Hash the shared trainer, model package, and V14/V18 run wrappers."""
    root = Path(root).resolve()
    files = [root / "scripts" / "train.py"]
    files.extend((root / "src" / "stmoe_imputer").rglob("*.py"))
    files.extend((root / "scripts" / "v14-single").glob("*.py"))
    files.extend((root / "scripts" / "v18-single").glob("*.py"))
    files.append(root / "scripts" / "experiment_fingerprint.py")

    digest = hashlib.sha256()
    for path in sorted({path.resolve() for path in files if path.is_file()}):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, byteorder="big"))
        digest.update(relative)
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()

