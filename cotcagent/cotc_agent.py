"""COTCAgent: Algorithm~\\ref{alg:cotc_loop} — TSA summary, COTC scoring, targeted consultation rounds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

import numpy as np

from .config import COTCConfig
from .cotc_scoring import score_all_diseases, topk_gap_priority
from .kb_loader import SymptomDiseaseKB
from .kb_repro import kb_digest
from .phi_table import PhiOverlay
from .tsa_summary import summarize_patient_json, tsa_tokens_to_predicates


class UserTurnFn(Protocol):
    def __call__(self, question: str) -> str: ...


@dataclass
class ConsultationResult:
    ranked_disease_ids: List[str]
    ranked_names: List[str]
    P_tilde: List[float]
    cot_log: List[Tuple[str, str]]
    low_kb_coverage: bool
    uncertainty: bool
    tsa_predicates: List[str]
    final_entropy: float


def _entropy_stop(H: float, k: int, cfg: COTCConfig) -> bool:
    if k <= 1:
        return True
    ceiling = float(np.log(k))
    return H < cfg.tau_h_frac * max(ceiling, 1e-6)


def render_question(gap_symptom_name: str) -> str:
    return (
        f"To refine the differential: have you experienced or been told about "
        f"「{gap_symptom_name}」? Please answer yes / no / unsure."
    )


def parse_answer(answer: str, gap_name: str, evidence: Set[str]) -> None:
    a = answer.strip().lower()
    yes = any(x in a for x in ("yes", "y", "是", "有", "true"))
    if yes:
        evidence.add(gap_name.strip().lower())


class COTCAgent:
    """
    End-to-end agent: load KB, ingest patient record for TSA tokens + seed evidence,
    then run probabilistic CoT completion with optional user turns.
    """

    def __init__(self, cfg: Optional[COTCConfig] = None, kb_root: Optional[Path] = None):
        self.cfg = cfg or COTCConfig()
        root = kb_root or Path(__file__).resolve().parent.parent
        kb_path = root / self.cfg.kb_path
        if not kb_path.is_file():
            raise FileNotFoundError(f"KB not found: {kb_path}")
        self.kb = SymptomDiseaseKB(kb_path, max_diseases=self.cfg.max_diseases)
        phi_p = self.cfg.phi_path
        if phi_p:
            pp = Path(phi_p)
            self._phi_overlay = PhiOverlay(pp if pp.is_absolute() else (root / pp))
        else:
            self._phi_overlay = PhiOverlay(None)
        self._root = root

    def repro_metadata(self, sha256: bool = False) -> Dict[str, Any]:
        meta = kb_digest(self.kb, sha256=sha256)
        meta["config"] = {
            "T": self.cfg.T,
            "theta": self.cfg.theta,
            "R_max": self.cfg.R_max,
            "gamma": self.cfg.gamma,
            "tau_h_frac": self.cfg.tau_h_frac,
            "default_phi": self.cfg.default_phi,
        }
        return meta

    def rank_only(
        self,
        evidence_symptoms: Set[str],
        tsa_predicates: Optional[Set[str]] = None,
    ) -> ConsultationResult:
        """Single-shot ranking (no user turns), useful for benchmarks."""
        return self.run_loop(evidence_symptoms, tsa_predicates or set(), user_turn=None)

    def run_from_patient_file(
        self,
        patient_json: str | Path,
        user_turn: Optional[UserTurnFn] = None,
        extra_evidence: Optional[Set[str]] = None,
    ) -> ConsultationResult:
        patient_json = Path(patient_json)
        tokens, seed_evidence = summarize_patient_json(patient_json)
        evidence: Set[str] = {s.strip().lower() for s in seed_evidence}
        if extra_evidence:
            evidence |= {s.strip().lower() for s in extra_evidence}
        predicates = tsa_tokens_to_predicates(tokens)
        return self.run_loop(evidence, predicates, user_turn=user_turn)

    def run_loop(
        self,
        evidence_symptoms: Set[str],
        tsa_predicates: Set[str],
        user_turn: Optional[UserTurnFn] = None,
    ) -> ConsultationResult:
        cfg = self.cfg
        log: List[Tuple[str, str]] = []
        low_cov = False
        uncertainty = False
        ev = set(evidence_symptoms)
        pred = set(tsa_predicates)

        for r in range(cfg.R_max):
            state = score_all_diseases(self.kb, ev, pred, cfg, self._phi_overlay)
            order = np.argsort(-state.P_tilde)
            best_p = float(state.P_tilde[order[0]]) if len(order) else 0.0
            k_eff = int(np.sum(state.P_tilde > 1e-8)) or 1
            if best_p >= cfg.theta or _entropy_stop(state.H, k_eff, cfg):
                ids = [state.disease_ids[int(i)] for i in order[:10]]
                names = [self.kb.diseases[did].name for did in ids if did in self.kb.diseases]
                return ConsultationResult(
                    ranked_disease_ids=ids,
                    ranked_names=names,
                    P_tilde=[float(state.P_tilde[int(i)]) for i in order[:10]],
                    cot_log=log,
                    low_kb_coverage=low_cov,
                    uncertainty=False,
                    tsa_predicates=list(pred),
                    final_entropy=state.H,
                )

            gaps = topk_gap_priority(self.kb, state, ev, cfg, self._phi_overlay)
            if not gaps:
                low_cov = True
                break
            gap_name, _ = gaps[0]
            q = render_question(gap_name)
            if user_turn is None:
                log.append((q, "<no user_turn; stopping>"))
                uncertainty = True
                break
            ans = user_turn(q)
            log.append((q, ans))
            parse_answer(ans, gap_name, ev)

        state = score_all_diseases(self.kb, ev, pred, cfg, self._phi_overlay)
        order = np.argsort(-state.P_tilde)
        ids = [state.disease_ids[int(i)] for i in order[:10]]
        names = [self.kb.diseases[did].name for did in ids if did in self.kb.diseases]
        best_p = float(state.P_tilde[order[0]]) if len(order) else 0.0
        uncertainty = uncertainty or best_p < cfg.theta
        return ConsultationResult(
            ranked_disease_ids=ids,
            ranked_names=names,
            P_tilde=[float(state.P_tilde[int(i)]) for i in order[:10]],
            cot_log=log,
            low_kb_coverage=low_cov,
            uncertainty=uncertainty,
            tsa_predicates=list(pred),
            final_entropy=state.H,
        )
