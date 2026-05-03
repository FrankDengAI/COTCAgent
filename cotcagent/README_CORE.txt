COTCAgent paper-aligned core (cotcagent/)
==========================================

Modules
-------
- config.py       : T, theta, R_max, gamma, tau_h_frac, optional phi_path, max_diseases
- kb_loader.py    : Chinese 疾病库 JSON + IDF; optional per-edge phi/weight fields on symptoms
- phi_table.py    : Optional JSON overlay for phi(d,s)
- cotc_scoring.py : R_i, softmax, H, Top-gaps
- tsa_summary.py  : Patient JSON -> TSATokens + seed evidence
- tsa_router.py   : Clinical query string -> estimator tags (router stub)
- cotc_agent.py   : Algorithm loop + rank_only() + repro_metadata()
- kb_repro.py     : KB counts / optional SHA-256 for logs
- experiments.py: patient_data/ top-1 sweep (prototype)
- cli.py          : python -m cotcagent <subcommand>

CLI examples (from repo root COTCAgent-main/COTCAgent-main/)
--------------------------------------------------------------
  python -m cotcagent route "sudden worsening of creatinine"
  python -m cotcagent patient patient_data/patient_0001.json --repro
  python -m cotcagent eval --limit 20 --max-diseases 8000
  python -m cotcagent eval --limit 5 --max-diseases 0

phi overlay JSON
----------------
Nested: {"D000001": {"S002108_014": 0.92}}
Or list: {"edges": [["D000001","S002108_014",0.92], ...]}
See cotcagent/data/example_phi_overlay.json

Tests
-----
  pip install -r requirements-cotcagent.txt pytest
  pytest tests/test_cotc_scoring.py -v

Note: max_diseases truncates the KB for speed; for publication runs use None / 0 in CLI eval.
