"""Command-line entry for demos, patient runs, and small-batch eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import COTCConfig
from .cotc_agent import COTCAgent
from .experiments import run_longitudinal_folder
from .tsa_router import route_tsa_intents


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="COTCAgent (paper-aligned core)")
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent, help="Project root (KB + patient_data)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_patient = sub.add_parser("patient", help="Run on one patient JSON")
    p_patient.add_argument("json_path", type=Path)
    p_patient.add_argument("--max-diseases", type=int, default=None)
    p_patient.add_argument("--phi", type=str, default=None, help="Optional phi overlay JSON path")
    p_patient.add_argument("--repro", action="store_true", help="Print repro metadata JSON")

    p_eval = sub.add_parser("eval", help="Longitudinal top-1 sweep on patient_data/")
    p_eval.add_argument("--limit", type=int, default=30)
    p_eval.add_argument(
        "--max-diseases",
        type=int,
        default=8000,
        help="Cap KB diseases loaded (speed); use 0 for full KB",
    )

    p_route = sub.add_parser("route", help="Print TSA router tags for a query string")
    p_route.add_argument("query", type=str)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root: Path = args.root
    if args.cmd == "route":
        print(json.dumps(route_tsa_intents(args.query), ensure_ascii=False))
        return 0

    cfg_kw: dict = {}
    if args.cmd == "eval":
        md = int(getattr(args, "max_diseases", 8000) or 0)
        cfg_kw["max_diseases"] = None if md <= 0 else md
    if args.cmd == "patient":
        if getattr(args, "max_diseases", None) is not None:
            cfg_kw["max_diseases"] = args.max_diseases
        if getattr(args, "phi", None):
            cfg_kw["phi_path"] = args.phi

    agent = COTCAgent(COTCConfig(**cfg_kw), kb_root=root)

    if args.cmd == "patient":
        if args.repro:
            print(json.dumps(agent.repro_metadata(sha256=False), indent=2, ensure_ascii=False))
        res = agent.run_from_patient_file(args.json_path, user_turn=None)
        top = list(zip(res.ranked_names[:5], [f"{x:.4f}" for x in res.P_tilde[:5]]))
        print(json.dumps({"top": top, "low_kb_coverage": res.low_kb_coverage, "uncertainty": res.uncertainty}, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "eval":
        hits, total, notes = run_longitudinal_folder(agent, root / "patient_data", limit=args.limit)
        acc = hits / total if total else 0.0
        print(f"top1_acc={acc:.4f} hits={hits} total={total}")
        for line in notes[:15]:
            print(line)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
