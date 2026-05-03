"""Optional per-edge phi(s_j, d_i) table (JSON) layered on top of KB rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


class PhiOverlay:
    """
    JSON shapes supported:
      - {"D000001": {"S002108_014": 0.91, ...}, ...}
      - {"edges": [["D000001","S002108_014",0.91], ...]}
    Values are clamped to [0.05, 0.99].
    """

    def __init__(self, path: Optional[Union[str, Path]] = None):
        self._pair: Dict[tuple[str, str], float] = {}
        if path is None:
            return
        p = Path(path)
        if not p.is_file():
            return
        raw: Any = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "edges" in raw:
            for row in raw["edges"]:
                if isinstance(row, (list, tuple)) and len(row) >= 3:
                    self._set(str(row[0]), str(row[1]), float(row[2]))
        elif isinstance(raw, dict):
            for did, inner in raw.items():
                if not isinstance(inner, dict):
                    continue
                for sid, val in inner.items():
                    self._set(str(did), str(sid), float(val))

    def _set(self, did: str, sid: str, val: float) -> None:
        v = min(0.99, max(0.05, val))
        self._pair[(did, sid)] = v

    def get(self, disease_id: str, symptom_id: str) -> Optional[float]:
        return self._pair.get((disease_id, symptom_id))
