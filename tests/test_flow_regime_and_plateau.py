"""
Unit tests for Flow-Regime detection, Stabilization recognition, and Sequence interpretation.
"""

import numpy as np
import pytest

from src.flow_regime import (
    detect_sustained_flow_regimes,
    detect_stabilization_plateau,
    analyze_regime_sequence
)
from src.derivative import calculate_bourdet_derivative, calculate_log_slopes


def test_stabilization_plateau_detection():
    # Construct a synthetic plateau (constant derivative at late time)
    times = np.geomspace(0.01, 100.0, 100)
    # Derivative rises then levels off to 50 psi
    derv = 50.0 * (1.0 - np.exp(-times / 2.0))
    slopes = calculate_log_slopes(times, derv, L=0.15)
    
    plateau = detect_stabilization_plateau(times, derv, slopes, slope_tolerance=0.10)
    
    assert plateau["is_detected"] is True
    assert plateau["derv_mean"] == pytest.approx(50.0, rel=0.08)
    assert plateau["confidence"] in ["High", "Moderate"]


def test_regime_sequence_analysis():
    class MockRegime:
        def __init__(self, name, t_start=1.0, t_end=10.0):
            self.regime = name
            self.t_start = t_start
            self.t_end = t_end
            
    # Sequence 1: WBS -> Linear flow (hydraulic fracture)
    r_list1 = [MockRegime("Wellbore Storage", 0.01, 0.1), MockRegime("Linear Flow", 0.1, 5.0)]
    seq1 = analyze_regime_sequence(r_list1)
    assert "characteristic of formation linear flow perpendicular to a hydraulic fracture" in seq1
    
    # Sequence 2: Radial -> Linear flow (boundary/channel hypothesis)
    r_list2 = [MockRegime("Pseudoradial Flow", 10.0, 50.0), MockRegime("Linear Flow", 50.0, 200.0)]
    seq2 = analyze_regime_sequence(r_list2)
    assert "parallel sealing boundaries" in seq2 or "channel reservoir" in seq2
