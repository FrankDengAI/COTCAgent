"""Lightweight TSA intent router: maps free-text queries to estimator tags (paper Sec.~3.1)."""

from __future__ import annotations

import re
from typing import List


def route_tsa_intents(query: str) -> List[str]:
    """
    Returns ordered estimator tags to try. Downstream code picks the first compatible tool.
    """
    q = (query or "").lower()
    tags: List[str] = []
    if re.search(r"break|changepoint|abrupt|突变|拐点", q):
        tags.append("change_point_screen")
    if re.search(r"trend|slope|trajectory|轨迹|上升|下降|恶化", q):
        tags.append("mixed_effects_slope")
    if re.search(r"smooth|sparse|few\s*points|稀疏", q):
        tags.append("robust_smoother")
    if re.search(r"cohort|norm|population|对照", q):
        tags.append("cohort_z_score")
    if not tags:
        tags.append("mixed_effects_slope")
    return tags
