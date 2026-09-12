"""
Tests for Bourdet logarithmic pressure derivative and log-log slope calculations.
"""

import numpy as np
import pytest
from src.derivative import calculate_bourdet_derivative, calculate_log_slopes


def test_derivative_unit_slope_wbs():
    """Verify that linear pressure vs time (WBS, Delta P = a*t) gives derivative = a*t with slope 1.0."""
    t = np.logspace(-3, 0, 80)
    a = 150.0
    dp = a * t
    
    derv = calculate_bourdet_derivative(t, dp, L=0.10)
    # Check interior values match a * t
    interior = slice(10, 70)
    rel_err = np.abs(derv[interior] - dp[interior]) / dp[interior]
    assert np.mean(rel_err) < 0.05
    
    # Check slope
    slopes = calculate_log_slopes(t, derv, L=0.15)
    mean_slope = np.mean(slopes[interior])
    assert mean_slope == pytest.approx(1.0, abs=0.08)


def test_derivative_half_slope_linear_flow():
    """Verify that Delta P = a * t^0.5 gives derivative = 0.5 * a * t^0.5 with slope 0.5."""
    t = np.logspace(-2, 2, 100)
    a = 45.0
    dp = a * np.sqrt(t)
    
    derv = calculate_bourdet_derivative(t, dp, L=0.10)
    exact_derv = 0.5 * a * np.sqrt(t)
    
    interior = slice(15, 85)
    rel_err = np.abs(derv[interior] - exact_derv[interior]) / exact_derv[interior]
    assert np.mean(rel_err) < 0.05
    
    slopes = calculate_log_slopes(t, derv, L=0.15)
    mean_slope = np.mean(slopes[interior])
    assert mean_slope == pytest.approx(0.50, abs=0.06)


def test_derivative_quarter_slope_bilinear_flow():
    """Verify that Delta P = a * t^0.25 gives derivative = 0.25 * a * t^0.25 with slope 0.25."""
    t = np.logspace(-2, 2, 100)
    a = 30.0
    dp = a * np.power(t, 0.25)
    
    derv = calculate_bourdet_derivative(t, dp, L=0.10)
    exact_derv = 0.25 * a * np.power(t, 0.25)
    
    interior = slice(15, 85)
    rel_err = np.abs(derv[interior] - exact_derv[interior]) / exact_derv[interior]
    assert np.mean(rel_err) < 0.05
    
    slopes = calculate_log_slopes(t, derv, L=0.15)
    mean_slope = np.mean(slopes[interior])
    assert mean_slope == pytest.approx(0.25, abs=0.06)


def test_derivative_radial_flow_flat():
    """Verify that Delta P = a * ln(t) + b gives constant derivative = a with slope 0.0."""
    t = np.logspace(0, 3, 100)
    a = 25.0
    dp = a * np.log(t) + 100.0
    
    derv = calculate_bourdet_derivative(t, dp, L=0.10)
    
    interior = slice(15, 85)
    assert np.allclose(derv[interior], a, rtol=0.08)
    
    slopes = calculate_log_slopes(t, derv, L=0.15)
    mean_slope = np.mean(slopes[interior])
    assert mean_slope == pytest.approx(0.0, abs=0.08)


def test_no_nan_or_inf():
    """Ensure derivative calculation contains no NaNs or Infs."""
    t = np.logspace(-4, 2, 120)
    dp = 100.0 * (1.0 - np.exp(-t))
    derv = calculate_bourdet_derivative(t, dp)
    slopes = calculate_log_slopes(t, derv)
    
    assert not np.any(np.isnan(derv))
    assert not np.any(np.isinf(derv))
    assert not np.any(np.isnan(slopes))
    assert not np.any(np.isinf(slopes))
