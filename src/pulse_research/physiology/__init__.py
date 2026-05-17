"""Published-equation physiology anchors for validation and Phase 7.3 coupling.

Each function is a Python port of the canonical closed-form equation as
catalogued by the ``aerospace-calculators`` skill against
strikerdlm/HumanPerformanceCalcs (verified 2026-05-15). Primary references
in each function's docstring.
"""
from pulse_research.physiology.anchors import (
    alveolar_po2,
    fio2_to_representative_altitude_m,
    g_loc_time,
    inspired_po2,
    niermeyer_spo2,
    oxygen_content,
)

__all__ = [
    "alveolar_po2",
    "fio2_to_representative_altitude_m",
    "g_loc_time",
    "inspired_po2",
    "niermeyer_spo2",
    "oxygen_content",
]
