"""
Unit tests for Conventional Property Estimation:
Semilog permeability, Bourdet plateau permeability, Skin factor,
Linear-flow fracture length, and Bilinear-flow conductivity.
"""

import numpy as np
import pytest

from src.property_estimation import (
    estimate_permeability_semilog,
    estimate_permeability_derivative_plateau,
    estimate_skin_factor,
    estimate_fracture_half_length,
    estimate_fracture_conductivity
)


def test_semilog_and_derivative_plateau_permeability():
    # Known properties
    q = 500.0   # STB/d
    Bo = 1.2    # rb/STB
    mu = 1.2    # cp
    h = 50.0    # ft
    k_true = 10.0 # mD
    
    # Analytical slope: m = 162.6 * q * Bo * mu / (k * h)
    m_expected = (162.6 * q * Bo * mu) / (k_true * h) # 234.144 psi/cycle
    b_true = 50.0
    
    times = np.geomspace(10.0, 100.0, 50)
    dp = m_expected * np.log10(times) + b_true
    
    # 1. Test semilog estimation
    res_semi = estimate_permeability_semilog(times, dp, t_start=10.0, t_end=100.0, q=q, Bo=Bo, mu=mu, h=h)
    assert res_semi["is_valid"] is True
    assert res_semi["k_mD"] == pytest.approx(k_true, rel=1e-3)
    assert res_semi["r_squared"] == pytest.approx(1.0, rel=1e-4)
    
    # 2. Test derivative plateau estimation
    # Analytical plateau: Delta P' = 70.6 * q * Bo * mu / (k * h)
    derv_plat = (70.6 * q * Bo * mu) / (k_true * h)
    res_plat = estimate_permeability_derivative_plateau(derv_plat, q=q, Bo=Bo, mu=mu, h=h)
    assert res_plat["is_valid"] is True
    assert res_plat["k_mD"] == pytest.approx(k_true, rel=1e-3)


def test_skin_factor_estimation():
    k = 10.0
    m = 234.144
    b = 100.0  # Delta P at t = 1 hr
    phi = 0.12
    mu = 1.2
    ct = 15e-6
    rw = 0.25
    
    res_skin = estimate_skin_factor(k=k, slope_m=m, intercept_b=b, phi=phi, mu=mu, ct=ct, rw=rw)
    assert res_skin["is_valid"] is True
    assert isinstance(res_skin["skin"], float)
    
    # Missing parameters should return error without crashing
    res_err = estimate_skin_factor(k=None, slope_m=m, intercept_b=b, phi=phi, mu=mu, ct=ct, rw=rw)
    assert res_err["is_valid"] is False
    assert "Missing or invalid" in res_err["error"]


def test_linear_flow_fracture_length():
    q = 500.0
    Bo = 1.2
    mu = 1.2
    h = 50.0
    k = 5.0
    phi = 0.12
    ct = 15e-6
    xf_true = 450.0
    
    # Linear slope: m_LF = [4.064 * q * Bo / (h * xf)] * sqrt(mu / (k * phi * ct))
    diff_group = np.sqrt(mu / (k * phi * ct))
    m_lf = ((4.064 * q * Bo) / (h * xf_true)) * diff_group
    
    times = np.geomspace(0.1, 5.0, 30)
    dp = m_lf * np.sqrt(times) + 10.0
    
    res_xf = estimate_fracture_half_length(times, dp, t_start=0.1, t_end=5.0, q=q, Bo=Bo, mu=mu, h=h, k=k, phi=phi, ct=ct)
    assert res_xf["is_valid"] is True
    assert res_xf["xf_ft"] == pytest.approx(xf_true, rel=1e-2)


def test_bilinear_flow_fracture_conductivity():
    q = 500.0
    Bo = 1.2
    mu = 1.2
    h = 50.0
    k = 5.0
    phi = 0.12
    ct = 15e-6
    Cf_true = 150.0
    
    # Bilinear slope: m_BL = 44.1 * q * mu * Bo / [ h * sqrt(Cf) * (phi * mu * ct * k)^0.25 ]
    matrix_group = (phi * mu * ct * k) ** 0.25
    m_bl = (44.1 * q * mu * Bo) / (h * np.sqrt(Cf_true) * matrix_group)
    
    times = np.geomspace(0.01, 1.0, 30)
    dp = m_bl * np.power(times, 0.25) + 5.0
    
    res_cf = estimate_fracture_conductivity(times, dp, t_start=0.01, t_end=1.0, q=q, Bo=Bo, mu=mu, h=h, k=k, phi=phi, ct=ct)
    assert res_cf["is_valid"] is True
    assert res_cf["Cf_mD_ft"] == pytest.approx(Cf_true, rel=1e-2)
