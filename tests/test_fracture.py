"""
Tests for hydraulic fracture properties, conductivity calculations, and petrophysical relations.
"""

import pytest
from src.models import ReservoirProperties, WellProperties, FractureProperties
from src.fracture import (
    calculate_fracture_conductivity,
    calculate_dimensionless_conductivity,
    classify_fracture_conductivity_type
)
from src.reservoir import calculate_diffusivity, calculate_tD_xf, calculate_CD_xf


def test_fracture_conductivity_calculation():
    """Verify Cf = kf * wf formula."""
    kf = 10000.0  # mD
    wf = 0.02     # ft
    cf = calculate_fracture_conductivity(kf, wf)
    assert cf == pytest.approx(200.0, rel=1e-5)
    
    # Dataclass property check
    frac = FractureProperties(xf=500.0, kf=kf, wf=wf)
    assert frac.Cf == pytest.approx(200.0, rel=1e-5)


def test_dimensionless_fracture_conductivity():
    """Verify CfD = (kf * wf) / (k * xf) formula."""
    kf = 10000.0  # mD
    wf = 0.02     # ft
    k = 5.0       # mD
    xf = 500.0    # ft
    
    cfd = calculate_dimensionless_conductivity(kf, wf, k, xf)
    assert cfd == pytest.approx(200.0 / 2500.0, rel=1e-5)  # 0.08
    
    frac = FractureProperties(xf=xf, kf=kf, wf=wf)
    assert frac.CfD(k) == pytest.approx(0.08, rel=1e-5)


def test_invalid_parameter_validation():
    """Verify validation triggers on non-physical inputs."""
    with pytest.raises(ValueError):
        FractureProperties(xf=-500.0, kf=1000.0, wf=0.01).validate()
        
    with pytest.raises(ValueError):
        FractureProperties(xf=500.0, kf=0.0, wf=0.01).validate()
        
    with pytest.raises(ValueError):
        ReservoirProperties(k=-5.0, phi=0.12, mu=1.2, ct=15e-6, h=50.0, B=1.2).validate()


def test_conductivity_classification():
    """Test Cinco-Ley classification strings."""
    cat1, _ = classify_fracture_conductivity_type(150.0)
    assert "Infinite-Conductivity" in cat1
    
    cat2, _ = classify_fracture_conductivity_type(25.0)
    assert "High Finite-Conductivity" in cat2
    
    cat3, _ = classify_fracture_conductivity_type(2.0)
    assert "Finite-Conductivity" in cat3
    
    cat4, _ = classify_fracture_conductivity_type(0.01)
    assert "Low-Conductivity" in cat4


def test_dimensionless_time_and_storage():
    """Test reservoir dimensionless group functions."""
    tD = calculate_tD_xf(t_hr=10.0, k=5.0, phi=0.12, mu=1.2, ct=15e-6, xf=500.0)
    assert tD > 0.0
    
    CD = calculate_CD_xf(C=0.01, phi=0.12, ct=15e-6, h=50.0, xf=500.0)
    assert CD > 0.0
