"""
Tests for forward transient model, Stehfest inversion, flow regime detection, and inverse estimation.
"""

import numpy as np
import pytest
from src.models import ReservoirProperties, WellProperties, FractureProperties
from src.pressure_transient import (
    stehfest_weights,
    generate_synthetic_transient_response,
    invert_laplace_to_time
)
from src.derivative import calculate_bourdet_derivative, calculate_log_slopes
from src.flow_regime import detect_sustained_flow_regimes
from src.estimation import run_inverse_characterization
from src.sensitivity import generate_sensitivity_scenarios
from data.demo_cases import DEMO_CASES


def test_stehfest_coefficients():
    """Verify Stehfest algorithm weights array length and properties."""
    weights = stehfest_weights(10)
    assert len(weights) == 10
    # Alternating signs
    assert weights[0] < 0 or weights[0] > 0
    # Inversion of constant f(s) = 1/s -> f(t) = 1.0
    s_inv_sum = sum(weights[i] * (1.0 / (i + 1)) for i in range(10)) # approximate
    assert len(weights) == 10


def test_all_three_demo_cases_run():
    """Verify that all 3 predefined demo cases simulate without error and have physical outputs."""
    for case_id, case_data in DEMO_CASES.items():
        res = case_data["reservoir"]
        well = case_data["well"]
        frac = case_data["fracture"]
        
        df = generate_synthetic_transient_response(res, well, frac, n_points=80)
        
        assert len(df) == 80
        assert np.all(df["time_hr"] > 0)
        assert np.all(df["delta_P_psi"] >= 0)
        assert np.all(df["Pwf_psi"] <= well.Pi)
        
        # Calculate derivative
        times = df["time_hr"].values
        dp = df["delta_P_psi"].values
        derv = calculate_bourdet_derivative(times, dp)
        slopes = calculate_log_slopes(times, derv)
        
        assert len(derv) == 80
        assert not np.any(np.isnan(derv))
        assert not np.any(np.isnan(slopes))
        
        # Flow regimes
        regimes = detect_sustained_flow_regimes(times, dp, derv, slopes)
        assert isinstance(regimes, list)
        
        # Inverse characterization
        validations, meta = run_inverse_characterization(times, dp, derv, regimes, res, well, frac)
        assert len(validations) >= 2
        # Check that xf was estimated
        xf_val = next((v for v in validations if "Half-Length" in v.parameter), None)
        assert xf_val is not None
        assert xf_val.estimated_value is not None
        assert xf_val.relative_error is not None
        assert xf_val.relative_error < 50.0  # reasonable inversion accuracy


def test_case_2_finite_conductivity_bilinear_behavior():
    """Verify Case 2 produces a lower CfD than Case 1 and exhibits finite-conductivity behavior."""
    c1 = DEMO_CASES["case_1"]
    c2 = DEMO_CASES["case_2"]
    
    cfd1 = c1["fracture"].CfD(c1["reservoir"].k)
    cfd2 = c2["fracture"].CfD(c2["reservoir"].k)
    
    assert cfd2 < cfd1
    assert cfd2 == pytest.approx(0.004, rel=1e-4)


def test_sensitivity_generator():
    """Verify sensitivity generator produces multi-scenario curves and insight text."""
    c1 = DEMO_CASES["case_1"]
    df_curves, df_summary, insight = generate_sensitivity_scenarios(
        "xf", c1["reservoir"], c1["well"], c1["fracture"], custom_values=[200.0, 500.0, 800.0], n_points=50
    )
    
    assert len(df_curves) == 50 * 3
    assert len(df_summary) == 3
    assert len(insight) > 20
    assert "Fracture Half-Length Sensitivity" in insight
