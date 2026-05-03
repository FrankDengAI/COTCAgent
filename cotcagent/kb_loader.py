"""Symptom--disease KB loader (Chinese JSON schema) + inverse-disease-frequency weights (paper Eq.~\\ref{eq:idf_weight})."""

from __future__ import annotations

import json
import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

logger = logging.getLogger(__name__)


def _extract_phi(s: Dict[str, Any]) -> Optional[float]:
    for key in ("phi", "link_strength", "weight", "ω"):
        if key not in s or s[key] is None:
            continue
        try:
            v = float(s[key])
        except (TypeError, ValueError):
            continue
        if v > 1.0:
            v = min(0.99, v / 10.0)
        return min(0.99, max(0.05, v))
    return None


@dataclass
class SymptomRecord:
    symptom_id: str
    name: str
    idf: float = 0.0


@dataclass
class DiseaseRecord:
    disease_id: str
    name: str
    symptoms: List[Dict[str, Any]] = field(default_factory=list)


class SymptomDiseaseKB:
    """
    Loads 疾病库 / 症状列表 format, builds disease list and per-symptom IDF:
    w_j^IDF = log(|D| / |{d_i in D : s_j in S_{d_i}||) (natural log).
    Symptom rows may include phi / weight / link_strength for Eq.~(diag_score).
    """

    def __init__(self, kb_path: str | Path, max_diseases: Optional[int] = None):
        self.kb_path = Path(kb_path)
        self.diseases: Dict[str, DiseaseRecord] = {}
        self.symptoms: Dict[str, SymptomRecord] = {}
        self._symptom_disease_count: Dict[str, int] = {}
        self._max_diseases = max_diseases
        self._load()

    def _load(self) -> None:
        raw = json.loads(self.kb_path.read_text(encoding="utf-8"))
        lib = raw.get("疾病库") or raw.get("diseases") or []
        if not lib:
            raise ValueError(f"No diseases in KB: {self.kb_path}")

        n_loaded = 0
        for row in lib:
            if self._max_diseases is not None and n_loaded >= self._max_diseases:
                break
            did = row.get("疾病ID") or row.get("id")
            name = row.get("疾病名称") or row.get("name")
            if not did or not name:
                continue
            syms = row.get("症状列表") or row.get("symptoms") or []
            norm_syms: List[Dict[str, Any]] = []
            for s in syms:
                sid = s.get("symptom_id") or s.get("id")
                sname = s.get("symptom_name") or s.get("name")
                if not sid or not sname:
                    continue
                entry: Dict[str, Any] = {"symptom_id": sid, "symptom_name": sname}
                phi_val = _extract_phi(s)
                if phi_val is not None:
                    entry["phi"] = phi_val
                norm_syms.append(entry)
                self._symptom_disease_count[sid] = self._symptom_disease_count.get(sid, 0) + 1
            self.diseases[did] = DiseaseRecord(disease_id=did, name=name, symptoms=norm_syms)
            n_loaded += 1

        n_d = max(1, len(self.diseases))
        sid_to_name: Dict[str, str] = {}
        for d in self.diseases.values():
            for s in d.symptoms:
                sid_to_name.setdefault(s["symptom_id"], s["symptom_name"])
        for sid, cnt in self._symptom_disease_count.items():
            w = math.log(n_d / max(1, cnt))
            sname = sid_to_name.get(sid, sid)
            self.symptoms[sid] = SymptomRecord(symptom_id=sid, name=sname, idf=w)

        logger.info("KB loaded: %d diseases, %d distinct symptoms", len(self.diseases), len(self.symptoms))

    def disease_ids(self) -> Sequence[str]:
        return tuple(self.diseases.keys())

    def req_symptom_ids(self, disease_id: str) -> List[str]:
        d = self.diseases.get(disease_id)
        if not d:
            return []
        return [s["symptom_id"] for s in d.symptoms]

    def symptom_name_by_id(self, sid: str) -> str:
        rec = self.symptoms.get(sid)
        if rec:
            return rec.name
        for d in self.diseases.values():
            for s in d.symptoms:
                if s["symptom_id"] == sid:
                    return s["symptom_name"]
        return sid

    def idf_weight(self, symptom_id: str) -> float:
        rec = self.symptoms.get(symptom_id)
        if rec:
            return max(rec.idf, 1e-6)
        cnt = self._symptom_disease_count.get(symptom_id, 0)
        n_d = max(1, len(self.diseases))
        return max(math.log(n_d / max(1, cnt)), 1e-6)
