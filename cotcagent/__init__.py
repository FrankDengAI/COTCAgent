"""COTCAgent: TSA + COTC + probabilistic CoT completion (paper-aligned core)."""

from .config import COTCConfig
from .cotc_agent import COTCAgent, ConsultationResult
from .kb_repro import kb_digest
from .phi_table import PhiOverlay
from .tsa_router import route_tsa_intents

__all__ = [
    "COTCAgent",
    "COTCConfig",
    "ConsultationResult",
    "PhiOverlay",
    "kb_digest",
    "route_tsa_intents",
]
