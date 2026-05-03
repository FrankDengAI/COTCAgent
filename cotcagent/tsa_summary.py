"""TSA-style summaries from longitudinal patient JSON (typed cues for COTC, paper Sec.~3.1)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class TSAToken:
    """Contract tuple (span, estimand, value, qual) as short string token for matching."""

    span: str
    estimand: str
    value: str
    qual: str

    def as_evidence_string(self) -> str:
        parts = [self.estimand, self.value, self.qual]
        return "|".join(p for p in parts if p and p != "NA")


def _lin_slope(y: List[float]) -> Tuple[float, str]:
    n = len(y)
    if n < 2:
        return 0.0, "SPARSE"
    x = list(range(n))
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = sum((xi - mx) ** 2 for xi in x) or 1e-9
    slope = num / den
    resid = math.sqrt(sum((yi - (my + slope * (xi - mx))) ** 2 for xi, yi in zip(x, y)) / max(1, n - 1))
    qual = "UNSTABLE" if resid > (abs(my) * 0.35 + 1e-6) else "STABLE"
    return slope, qual


def summarize_patient_json(path: str | Path) -> Tuple[List[TSAToken], Set[str]]:
    """
    Reads patient bundle (基础体征, indicators, etc.), emits TSATokens + initial symptom evidence names.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    tokens: List[TSAToken] = []
    evidence_names: Set[str] = set()

    base = data.get("基础体征") or {}
    for name, block in base.items():
        if isinstance(block, dict):
            evidence_names.add(str(name).strip())
            vals = block.get("测量值")
            if isinstance(vals, list) and vals and all(isinstance(v, (int, float)) for v in vals):
                slope, qual = _lin_slope([float(v) for v in vals])
                direction = "up" if slope > 1e-6 else "down" if slope < -1e-6 else "flat"
                tokens.append(
                    TSAToken(
                        span=name[:32],
                        estimand="lab_slope",
                        value=f"{direction}:{slope:.4f}",
                        qual=qual,
                    )
                )
            sev = block.get("严重程度")
            if isinstance(sev, list) and sev:
                tokens.append(
                    TSAToken(span=name[:32], estimand="severity_series", value=str(sev[-1]), qual="ordinal")
                )

    ind_block = data.get("指标") or data.get("indicators") or {}
    if isinstance(ind_block, dict):
        for k, v in ind_block.items():
            if isinstance(v, dict):
                nm = v.get("指标名称") or k
                if nm:
                    evidence_names.add(str(nm).strip())

    db_sym = data.get("database_symptoms") or data.get("symptoms_from_database") or []
    if isinstance(db_sym, list):
        for row in db_sym:
            if isinstance(row, dict):
                nm = row.get("症状名称") or row.get("name")
                if nm:
                    evidence_names.add(str(nm).strip())

    return tokens, evidence_names


def tsa_tokens_to_predicates(tokens: List[TSAToken]) -> Set[str]:
    """Predicates usable alongside symptom names in soft string overlap (router output channel)."""
    out: Set[str] = set()
    for t in tokens:
        out.add(t.as_evidence_string())
        if t.estimand == "lab_slope":
            out.add(f"trend:{t.value.split(':')[0]}")
    return out
