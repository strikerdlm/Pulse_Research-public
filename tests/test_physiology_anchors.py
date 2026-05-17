"""Tests for ported published-equation anchors used in Phase 7.2 / 7.3."""
from __future__ import annotations


def test_inspired_po2_sea_level_room_air() -> None:
    from pulse_research.physiology.anchors import inspired_po2

    # At sea level (0 m), FiO2=0.21, PiO2 should be ~150 mmHg (West 2012).
    pio2 = inspired_po2(0.0, 0.21)
    assert 145.0 <= pio2 <= 155.0


def test_alveolar_po2_sea_level_room_air() -> None:
    from pulse_research.physiology.anchors import alveolar_po2

    # PAO2 at sea level, FiO2=0.21, PaCO2=40, RQ=0.8 should be ~100 mmHg (West 2012).
    pao2 = alveolar_po2(0.0, 0.21, paco2=40.0, rq=0.8)
    assert 95.0 <= pao2 <= 105.0


def test_alveolar_po2_decreases_with_altitude() -> None:
    from pulse_research.physiology.anchors import alveolar_po2

    pao2_sea = alveolar_po2(0.0, 0.21)
    pao2_3000 = alveolar_po2(3000.0, 0.21)
    pao2_4000 = alveolar_po2(4000.0, 0.21)
    assert pao2_sea > pao2_3000 > pao2_4000


def test_oxygen_content_canonical() -> None:
    from pulse_research.physiology.anchors import oxygen_content

    # Hb=15, SaO2=0.97 (97%), PaO2=100 → CaO2 ≈ 1.34*15*0.97 + 0.003*100
    # ≈ 19.50 + 0.30 = 19.80 mL O2/dL.
    cao2 = oxygen_content(hb_g_dl=15.0, sao2_fraction=0.97, pao2_mmhg=100.0)
    assert 19.5 <= cao2 <= 20.1


def test_oxygen_content_drops_with_hypoxia() -> None:
    from pulse_research.physiology.anchors import oxygen_content

    cao2_normo = oxygen_content(hb_g_dl=14.5, sao2_fraction=0.97, pao2_mmhg=100.0)
    cao2_hypoxic = oxygen_content(hb_g_dl=14.5, sao2_fraction=0.80, pao2_mmhg=60.0)
    assert cao2_hypoxic < cao2_normo


def test_g_loc_time_decreases_with_gz() -> None:
    from pulse_research.physiology.anchors import g_loc_time

    # Canonical Stoll curve uses Gz<5 -> inf, then 150/(Gz-4)^1.5 with floor 3 s.
    t_5p5 = g_loc_time(5.5)
    t_7 = g_loc_time(7.0)
    t_9 = g_loc_time(9.0)
    assert t_5p5 > t_7 > t_9
    assert t_9 >= 3.0


def test_g_loc_time_infinite_below_threshold() -> None:
    from pulse_research.physiology.anchors import g_loc_time

    # Canonical threshold: tolerance is Infinity for Gz < 5 (HumanPerformanceCalcs).
    assert g_loc_time(1.5) == float("inf")
    assert g_loc_time(3.0) == float("inf")
    assert g_loc_time(4.99) == float("inf")


def test_g_loc_time_canonical_values() -> None:
    """Snapshot test against the canonical TS implementation."""
    from pulse_research.physiology.anchors import g_loc_time

    # 150 / (6-4)^1.5 = 150 / 2.828 = 53.03 s
    assert abs(g_loc_time(6.0) - 53.033) < 0.01
    # 150 / (9-4)^1.5 = 150 / 11.18 = 13.42 s
    assert abs(g_loc_time(9.0) - 13.416) < 0.01
    # Floor at 3 s: 150 / (50-4)^1.5 = 0.48 -> clipped to 3.0
    assert g_loc_time(50.0) == 3.0


def test_niermeyer_spo2_drops_with_altitude() -> None:
    from pulse_research.physiology.anchors import niermeyer_spo2

    spo2_sea = niermeyer_spo2(0.0)
    spo2_3000 = niermeyer_spo2(3000.0)
    spo2_4500 = niermeyer_spo2(4500.0)
    assert spo2_sea > spo2_3000 > spo2_4500
    assert 0.0 < spo2_4500 < 1.0


def test_niermeyer_spo2_canonical_values() -> None:
    """Snapshot test against the canonical TS implementation
    (SpO2_pct = 103.3 - 0.0047*altitude + Z, clamped to [50,100], /100)."""
    from pulse_research.physiology.anchors import niermeyer_spo2

    # Sea level male: 103.3 + 0.7 = 104.0 -> clamped to 100 -> 1.00
    assert niermeyer_spo2(0.0, "male") == 1.00
    # 3000 m male: 103.3 - 14.1 + 0.7 = 89.9 -> 0.899
    assert abs(niermeyer_spo2(3000.0, "male") - 0.899) < 1e-3
    # 4500 m female: 103.3 - 21.15 + 1.4 = 83.55 -> 0.8355
    assert abs(niermeyer_spo2(4500.0, "female") - 0.8355) < 1e-3


def test_alveolar_po2_canonical_values() -> None:
    """Snapshot test against the canonical simplified PAO2 = PiO2 - PaCO2/RQ."""
    from pulse_research.physiology.anchors import alveolar_po2, inspired_po2

    pio2_sea = inspired_po2(0.0, 0.21)  # ~149.7 mmHg
    pao2 = alveolar_po2(0.0, 0.21, paco2=40.0, rq=0.8)
    # Simplified PAO2 = PiO2 - 40/0.8 = PiO2 - 50
    assert abs(pao2 - (pio2_sea - 50.0)) < 1e-6
