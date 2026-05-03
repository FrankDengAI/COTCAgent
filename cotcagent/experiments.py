"""Longitudinal-style batch eval: top-1 disease ID hit vs patient_info.diseases (paper Sec.~\\ref{sec:experiment})."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .config import COTCConfig
from .cotc_agent import COTCAgent


def gold_disease_ids(patient: Dict) -> Set[str]:
    info = patient.get("patient_info") or {}
    rows = info.get("diseases") or info.get("确诊疾病") or []
    out: Set[str] = set()
    for r in rows:
        if isinstance(r, dict):
            did = r.get("疾病ID") or r.get("id")
            if did:
                out.add(str(did))
    return out


def run_longitudinal_folder(
    agent: COTCAgent,
    patient_dir: Path,
    limit: Optional[int] = None,
) -> Tuple[int, int, List[str]]:
    """Returns (hits, total, per-file notes)."""
    hits = 0
    total = 0
    notes: List[str] = []
    paths = sorted(patient_dir.glob("patient_*.json"))
    if limit:
        paths = paths[:limit]
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        gold = gold_disease_ids(data)
        if not gold:
            notes.append(f"{p.name}: no gold labels")
            continue
        total += 1
        res = agent.run_from_patient_file(p, user_turn=None)
        top = res.ranked_disease_ids[0] if res.ranked_disease_ids else ""
        if top in gold:
            hits += 1
            notes.append(f"{p.name}: hit {top}")
        else:
            notes.append(f"{p.name}: pred={top} gold={gold}")
    return hits, total, notes


def main_cli() -> None:
    root = Path(__file__).resolve().parent.parent
    agent = COTCAgent(COTCConfig(kb_path="disease_symptom_database.json", max_diseases=8000), kb_root=root)
    hits, total, notes = run_longitudinal_folder(agent, root / "patient_data", limit=50)
    acc = hits / total if total else 0.0
    print(f"Top-1 accuracy (first 50 patients with labels): {acc:.3f} ({hits}/{total})")
    for line in notes[:12]:
        print(line)


if __name__ == "__main__":
    main_cli()
