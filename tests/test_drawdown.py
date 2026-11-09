"""
Unit tests for Pressure Drawdown calculations, column auto-detection, and data validation.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_processing import auto_detect_columns, validate_and_preprocess_drawdown
from src.units import convert_si_to_field, convert_field_to_si


def test_auto_detect_columns():
    df = pd.DataFrame({
        "Time (hrs)": [0.1, 0.5, 1.0],
        "FBHP (psia)": [3900.0, 3850.0, 3800.0],
        "Oil Rate (STB/d)": [500, 500, 500],
        "Bo": [1.2, 1.2, 1.2],
        "Viscosity": [1.2, 1.2, 1.2]
    })
    mapping = auto_detect_columns(df, analysis_mode="drawdown")
    assert mapping["time"] == "Time (hrs)"
    assert mapping["pressure"] == "FBHP (psia)"
    assert mapping["q"] == "Oil Rate (STB/d)"
    assert mapping["bo"] == "Bo"
    assert mapping["mu"] == "Viscosity"


def test_drawdown_calculation_and_filtering():
    # Test dataset with negative times, duplicates, and non-monotonic rows
    df_raw = pd.DataFrame({
        "t": [-1.0, 0.0, 1.0, 1.0, 3.0, 2.0, 5.0, np.nan],
        "pwf": [4000.0, 4000.0, 3900.0, 3895.0, 3800.0, 3850.0, 3700.0, 3600.0]
    })
    Pi = 4000.0
    df_proc, report = validate_and_preprocess_drawdown(
        df_raw=df_raw,
        time_col="t",
        pwf_col="pwf",
        Pi=Pi
    )
    
    assert report["is_valid"] is True
    assert report["has_negative_time"] is True
    assert report["has_duplicates"] is True
    assert report["is_monotonic"] is False
    assert report["nans_removed"] == 1
    
    # Processed rows must be monotonic and have positive time
    assert np.all(df_proc["time_hr"] >= 0)
    assert df_proc["time_hr"].is_monotonic_increasing
    
    # Check Delta P = Pi - Pwf
    assert np.allclose(df_proc["delta_p_psi"], Pi - df_proc["pwf_psi"])


def test_unit_conversions():
    si_inputs = {
        "Pi": 275.8,      # bar -> ~4000 psi
        "q": 79.5,        # m3/d -> ~500 STB/d
        "h": 15.24,       # m -> ~50 ft
        "rw": 0.0762,     # m -> ~0.25 ft
        "ct": 2.175e-4,   # 1/bar -> ~1.5e-5 1/psi
        "xf": 152.4       # m -> ~500 ft
    }
    field = convert_si_to_field(si_inputs)
    assert field["Pi"] == pytest.approx(4000.0, rel=0.01)
    assert field["q"] == pytest.approx(500.0, rel=0.01)
    assert field["h"] == pytest.approx(50.0, rel=0.01)
    assert field["rw"] == pytest.approx(0.25, rel=0.01)
    assert field["ct"] == pytest.approx(1.5e-5, rel=0.01)
    assert field["xf"] == pytest.approx(500.0, rel=0.01)
