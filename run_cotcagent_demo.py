#!/usr/bin/env python3
"""Demo: paper-aligned COTCAgent on one patient JSON (no LLM calls in core path)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cotcagent import COTCAgent, COTCConfig


def main() -> None:
    cfg = COTCConfig()
    agent = COTCAgent(cfg, kb_root=ROOT)
    patient = ROOT / "patient_data" / "patient_0001.json"
    res = agent.run_from_patient_file(patient, user_turn=None)
    print("Top hypotheses:", list(zip(res.ranked_names[:5], [f"{p:.4f}" for p in res.P_tilde[:5]])))
    print("low_kb_coverage:", res.low_kb_coverage, "uncertainty:", res.uncertainty)
    print("TSA predicates (sample):", res.tsa_predicates[:6])
    if res.cot_log:
        print("CoT log:", res.cot_log[:3])


if __name__ == "__main__":
    main()
