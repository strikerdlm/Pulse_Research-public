"""Hüfner CaO2 ratio coupling — composes CGEM time and Pulse O2 posteriors.

See spec docs/superpowers/specs/2026-05-15-phase-7.3-coupling-mc-conformal-interaction-sobol-design.md
§3.2 for the full mathematical statement. The coupling has zero fitted
parameters; all constants come from the published equations ported in
``pulse_research.physiology.anchors`` (Hüfner 1.34 mL O2/g Hb, dissolved
0.003 mL O2/dL/mmHg, PaCO2=40, RQ=0.8).
"""
from pulse_research.coupling.hufner import (
    cao2_at,
    cao2_reference,
    corrected_time,
    mc_propagate,
)

__all__ = [
    "cao2_at",
    "cao2_reference",
    "corrected_time",
    "mc_propagate",
]
