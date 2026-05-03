"""Hyperparameters aligned with paper_body.tex Reproducibility (Sec.~\\ref{sec:reproducibility})."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class COTCConfig:
    """Dual-threshold scan: energy gate T before softmax, mass gate theta after softmax."""

    T: float = 0.3
    theta: float = 0.9
    R_max: int = 6
    gamma: float = 0.08
    top_k: int = 5
    gap_arity: int = 1
    # Entropy stop: H below tau_H * log(K) with tau_H as a fraction of the uniform ceiling.
    tau_h_frac: float = 0.22
    default_phi: float = 0.78
    kb_path: str = "disease_symptom_database.json"
    # Optional JSON: per (disease_id, symptom_id) phi overrides (see phi_table.py).
    phi_path: Optional[str] = None
    # Cap diseases loaded for fast dev/tests (None = all).
    max_diseases: Optional[int] = None
