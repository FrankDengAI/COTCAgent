"""KB digest for reproducibility logs (path, counts, optional SHA-256)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

from .kb_loader import SymptomDiseaseKB


def kb_digest(kb: SymptomDiseaseKB, sha256: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "kb_path": str(kb.kb_path),
        "n_diseases": len(kb.diseases),
        "n_symptoms": len(kb.symptoms),
    }
    if sha256 and kb.kb_path.is_file():
        h = hashlib.sha256()
        with kb.kb_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out["sha256"] = h.hexdigest()
    return out
