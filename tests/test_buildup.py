"""
Unit tests for Pressure Buildup (PBU) analysis:
Horner time, Agarwal equivalent time, Horner slope, permeability, P*, and skin factor.
"""

import numpy as np
import pytest
import pandas as pd

from src.buildup import calculate_buildup_variables, estimate_buildup_horner_parameters


def test_buildup_horner_and_agarwal_variables():
    dt = np.array([0.1, 0.5, 1.0, 5.0, 10.0])
    pws = np.array([3700.0, 3750.0, 3800.0, 3900.0, 3950.0])
    tp = 100.0
    pwf_shutin = 3650.0
    
    df = calculate_buildup_variables(dt_hr=dt, pws_psi=pws, tp_hr=tp, pwf_at_shutin=pwf_shutin)
    
    assert len(df) == 5
    assert np.all(df["delta_pws_psi"] == pws - pwf_shutin)
    # Check Horner ratio: (tp + dt) / dt
    expected_hr = (tp + dt) / dt
    assert np.allclose(df["horner_ratio"], expected_hr)
    # Check Agarwal equivalent time: tp * dt / (tp + dt)
    expected_te = (tp * dt) / (tp + dt)
    assert np.allclose(df["agarwal_te_hr"], expected_te)


def test_buildup_horner_estimation():
    q = 500.0
    Bo = 1.2
    mu = 1.2
    h = 50.0
    k_true = 8.0
    tp = 200.0
    P_star_true = 4000.0
    
    m_horner = (162.6 * q * Bo * mu) / (k_true * h)
    
    dt = np.geomspace(1.0, 50.0, 25)
    hr = (tp + dt) / dt
    pws = P_star_true - m_horner * np.log10(hr)
    
    df_buildup = calculate_buildup_variables(dt_hr=dt, pws_psi=pws, tp_hr=tp, pwf_at_shutin=3600.0)
    
    h_res = estimate_buildup_horner_parameters(
        df_buildup=df_buildup,
        tp_hr=tp,
        pwf_at_shutin=3600.0,
        q=q, Bo=Bo, mu=mu, h=h, phi=0.12, ct=15e-6, rw=0.25
    )
    
    assert h_res["is_valid"] is True
    assert h_res["k_mD"] == pytest.approx(k_true, rel=1e-2)
    assert h_res["P_star_psi"] == pytest.approx(P_star_true, rel=1e-2)
    assert h_res["slope_m_horner"] == pytest.approx(m_horner, rel=1e-2)
    assert isinstance(h_res["skin"], float)
