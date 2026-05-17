"""Published-equation physiology anchors.

These are closed-form equations from the literature, ported to Python from
the TypeScript implementations in
strikerdlm/HumanPerformanceCalcs/frontend/src/calculators/atmosphere.ts and
spo2Models.ts (verified 2026-05-15 — Hüfner constant 1.34 mL O2/g Hb).

Used in Phase 7.2 as independent validation anchors (orchestrator's
"External anchor comparison" section) and in Phase 7.3 as the analytical
form of the coupling layer (Hüfner CaO2 ratio).
"""
from __future__ import annotations

# ICAO Doc 7488-CD standard atmosphere (troposphere only).
_SEA_LEVEL_PRESSURE_MMHG = 760.0
_WATER_VAPOR_PRESSURE_37C_MMHG = 47.0
_HUFNER_CONSTANT_ML_O2_PER_G_HB = 1.34  # canonical clinical value
_DISSOLVED_O2_PER_MMHG = 0.003  # mL O2 / dL / mmHg PaO2


def _barometric_pressure_mmhg(altitude_m: float) -> float:
    """ICAO standard atmosphere, troposphere only (0 - 11 km)."""
    if altitude_m < 0:
        return _SEA_LEVEL_PRESSURE_MMHG
    if altitude_m > 11000:
        # Stratosphere not modeled here; fall back to 11 km value.
        altitude_m = 11000.0
    # Standard troposphere lapse: T(h) = 288.15 - 0.0065 * h K.
    # P(h) = P0 * (T/T0)^(g M / (R L)) where exponent ~= 5.25588.
    t0 = 288.15
    t = t0 - 0.0065 * altitude_m
    return float(_SEA_LEVEL_PRESSURE_MMHG * (t / t0) ** 5.25588)


def inspired_po2(altitude_m: float, fio2: float = 0.21) -> float:
    """Inspired PO2 in mmHg at given altitude and FiO2 (West 2012).

    PiO2 = FiO2 * (P_bar - P_H2O) where P_H2O = 47 mmHg at body temperature.
    """
    p_bar = _barometric_pressure_mmhg(altitude_m)
    return fio2 * (p_bar - _WATER_VAPOR_PRESSURE_37C_MMHG)


def alveolar_po2(
    altitude_m: float,
    fio2: float = 0.21,
    *,
    paco2: float = 40.0,
    rq: float = 0.8,
) -> float:
    """Alveolar PO2 via the simplified alveolar gas equation (West 2012).

    PAO2 = PiO2 - PaCO2 / RQ.

    Matches the canonical implementation in
    HumanPerformanceCalcs/frontend/src/calculators/atmosphere.ts::alveolarPO2.
    The full Riley form (with the ``FiO2 + (1 - FiO2)/RQ`` correction)
    differs by ~2 mmHg at FiO2 = 0.21 and grows at high FiO2; the source
    uses the simplified form, which we mirror so anchor numbers match.
    """
    if rq <= 0.0:
        raise ValueError(f"rq must be positive; got {rq}")
    pio2 = inspired_po2(altitude_m, fio2)
    return float(pio2 - paco2 / rq)


def oxygen_content(
    *,
    hb_g_dl: float,
    sao2_fraction: float,
    pao2_mmhg: float,
) -> float:
    """Arterial oxygen content (CaO2) in mL O2 / dL (Hufner formula).

    CaO2 = 1.34 * Hb * SaO2 + 0.003 * PaO2.

    ``sao2_fraction`` is in [0, 1] (NOT a percentage).
    """
    if not 0.0 <= sao2_fraction <= 1.0:
        raise ValueError(
            f"sao2_fraction must be in [0, 1]; got {sao2_fraction}"
        )
    if hb_g_dl <= 0.0:
        raise ValueError(f"hb_g_dl must be positive; got {hb_g_dl}")
    return (
        _HUFNER_CONSTANT_ML_O2_PER_G_HB * hb_g_dl * sao2_fraction
        + _DISSOLVED_O2_PER_MMHG * pao2_mmhg
    )


def g_loc_time(gz: float) -> float:
    """G-LOC time-to-incapacitation at sustained +Gz (simplified Stoll curve).

    Matches the canonical implementation in
    HumanPerformanceCalcs/frontend/src/calculators/atmosphere.ts::gLocTime
    (verified 2026-05-15):

        Gz < 5     -> Infinity (most pilots tolerate indefinitely)
        Gz >= 5    -> max(3.0, 150.0 / (Gz - 4.0)^1.5)

    Returns seconds. The 3-second floor reflects the minimum perceptible
    time-to-G-LOC for high-Gz unprotected runs. Stoll original from
    Stoll 1956 / Whinnery 2006.
    """
    if gz < 5.0:
        return float("inf")
    tolerance = 150.0 / (gz - 4.0) ** 1.5
    return float(max(3.0, tolerance))


def niermeyer_spo2(altitude_m: float, sex: str = "male") -> float:
    """SpO2 fraction at altitude (Niermeyer linear, sex-corrected).

    Matches the canonical implementation in
    HumanPerformanceCalcs/frontend/src/calculators/spo2Models.ts::niermeyerSpo2
    (verified 2026-05-15):

        SpO2_pct = 103.3 - 0.0047 * altitude_m + Z
        Z = 0.7 (male) | 1.4 (female)
        clamped to [50, 100] then divided by 100 to return [0.50, 1.00]

    Validated on healthy populations 0-4 018 m; does NOT model acclimatization.
    Reference: Niermeyer et al., Eur. J. Appl. Physiol. (cited in
    Tushaus et al., Physiological Reports, 2019).
    """
    if sex not in ("male", "female"):
        raise ValueError(f"sex must be 'male' or 'female'; got {sex!r}")
    if altitude_m < 0 or altitude_m > 8848:
        raise ValueError(
            f"altitude_m must be in [0, 8848]; got {altitude_m}"
        )
    z = 0.7 if sex == "male" else 1.4
    spo2_pct = 103.3 - 0.0047 * altitude_m + z
    spo2_pct = max(50.0, min(100.0, spo2_pct))
    return spo2_pct / 100.0


# Convenience: altitude (m) <-> representative FiO2 tier per pulse_glue.py.
# FiO2 >= 0.20 -> 0 m (normobaric);
# 0.16 <= FiO2 < 0.20 -> 3000 m (Hypobaric3000m);
# FiO2 < 0.16 -> 4000 m (Hypobaric4000m).
def fio2_to_representative_altitude_m(fio2: float) -> float:
    """Return the altitude implied by the Pulse FiO2 tier mapping."""
    if fio2 >= 0.20:
        return 0.0
    if fio2 >= 0.16:
        return 3000.0
    return 4000.0
