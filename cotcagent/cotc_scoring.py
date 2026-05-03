"""Gibbs energies R_i, softmax pseudo-posterior, entropy H, gap priorities (paper Eqs.~\\ref{eq:diag_score}--\\ref{eq:gap_priority})."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .config import COTCConfig
from .kb_loader import SymptomDiseaseKB
from .phi_table import PhiOverlay


def _log_clamp(x: float, floor: float = 1e-9) -> float:
    return math.log(max(floor, x))


@dataclass
class ScoringState:
    disease_ids: List[str]
    R: np.ndarray
    P_tilde: np.ndarray
    H: float
    kept_mask: np.ndarray


def _edge_phi(
    disease_id: str,
    row: Dict[str, Any],
    default_phi: float,
    phi_overlay: Optional[PhiOverlay],
) -> float:
    if phi_overlay is not None:
        o = phi_overlay.get(disease_id, str(row["symptom_id"]))
        if o is not None:
            return o
    v = row.get("phi")
    if v is not None:
        return float(v)
    return default_phi


def compute_R_i(
    kb: SymptomDiseaseKB,
    disease_id: str,
    evidence_symptom_names: Set[str],
    evidence_predicates: Set[str],
    gamma: float,
    default_phi: float,
    phi_overlay: Optional[PhiOverlay] = None,
) -> float:
    """Eq.~\\ref{eq:diag_score}: per-edge phi from KB row, optional PhiOverlay, or default."""
    d = kb.diseases.get(disease_id)
    if not d:
        return float("-inf")
    total = 0.0
    ev_norm = {str(s).strip().lower() for s in evidence_symptom_names}
    pred_norm = {str(p).lower() for p in evidence_predicates}

    for s in d.symptoms:
        sid = s["symptom_id"]
        sname = str(s["symptom_name"]).strip().lower()
        w_raw = kb.idf_weight(sid)
        w_raw = min(max(w_raw, 1e-6), 10.0)
        phi = _edge_phi(disease_id, s, default_phi, phi_overlay)
        present = sname in ev_norm
        if not present:
            for p in pred_norm:
                if p and (p in sname or sname in p):
                    present = True
                    break
        if present:
            total += _log_clamp(w_raw) + _log_clamp(phi)
        else:
            neg = 1.0 - gamma * w_raw
            total += _log_clamp(neg)
    return total


def softmax_from_R(R: np.ndarray) -> np.ndarray:
    z = R - np.max(R)
    ex = np.exp(z)
    s = ex.sum()
    if s <= 0 or not np.isfinite(s):
        return np.ones_like(R) / max(1, len(R))
    return ex / s


def entropy_P(P: np.ndarray) -> float:
    P = np.clip(P, 1e-12, 1.0)
    return float(-np.sum(P * np.log(P)))


def score_all_diseases(
    kb: SymptomDiseaseKB,
    evidence_symptom_names: Set[str],
    evidence_predicates: Set[str],
    cfg: COTCConfig,
    phi_overlay: Optional[PhiOverlay] = None,
) -> ScoringState:
    ids = list(kb.disease_ids())
    R = np.array(
        [
            compute_R_i(
                kb,
                did,
                evidence_symptom_names,
                evidence_predicates,
                cfg.gamma,
                cfg.default_phi,
                phi_overlay,
            )
            for did in ids
        ],
        dtype=float,
    )
    kept = R >= cfg.T
    if not np.any(kept):
        kept = R >= np.percentile(R, 85)
    R_f = np.where(kept, R, -np.inf)
    P = softmax_from_R(np.where(np.isfinite(R_f), R_f, -1e9))
    H = entropy_P(P)
    return ScoringState(disease_ids=ids, R=R, P_tilde=P, H=H, kept_mask=kept)


def topk_gap_priority(
    kb: SymptomDiseaseKB,
    state: ScoringState,
    evidence_symptom_names: Set[str],
    cfg: COTCConfig,
    phi_overlay: Optional[PhiOverlay] = None,
) -> List[Tuple[str, float]]:
    """Simplified Eq.~\\ref{eq:gap_priority}: gaps are missing KB symptoms for top-k diseases by P_tilde."""
    order = np.argsort(-state.P_tilde)
    k = min(cfg.top_k, len(order))
    ev_norm = {str(s).strip().lower() for s in evidence_symptom_names}
    gap_scores: Dict[str, float] = {}

    def mine(ks: int) -> None:
        nonlocal gap_scores
        gap_scores = {}
        for idx in order[:ks]:
            did = state.disease_ids[int(idx)]
            p_mass = float(state.P_tilde[int(idx)])
            d = kb.diseases.get(did)
            if not d:
                continue
            for s in d.symptoms:
                sname = str(s["symptom_name"]).strip().lower()
                if sname in ev_norm:
                    continue
                sid = s["symptom_id"]
                phi = _edge_phi(did, s, cfg.default_phi, phi_overlay)
                w = kb.idf_weight(sid)
                psi = phi * (1.0 + 0.1 * min(w, 5.0))
                gap_scores[sname] = gap_scores.get(sname, 0.0) + p_mass * psi

    mine(k)
    if not gap_scores and len(order) > 0:
        mine(min(40, len(order)))
    ranked = sorted(gap_scores.items(), key=lambda x: -x[1])
    return ranked
