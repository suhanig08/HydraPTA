"""
Dimensionless transformations and theoretical type curve definitions for Pressure Transient Analysis.
Provides consistent petroleum engineering dimensionless variables: tD, pD, p'D.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


def calculate_dimensionless_variables(
    times: np.ndarray,
    delta_p: np.ndarray,
    derivative: np.ndarray,
    k: float,
    h: float,
    q: float,
    Bo: float,
    mu: float,
    phi: float,
    ct: float,
    rw: float,
    xf: Optional[float] = None
) -> pd.DataFrame:
    """
    Transform dimensional time, pressure change, and derivative into standard petroleum engineering
    dimensionless quantities (Field Unit convention):
    
        pD = (k * h * Delta P) / (141.2 * q * Bo * mu)
        p'D = (k * h * Delta P') / (141.2 * q * Bo * mu)
        tD_rw = (0.0002637 * k * t) / (phi * mu * ct * rw^2)
        tD_xf = (0.0002637 * k * t) / (phi * mu * ct * xf^2) (if xf provided)
        
    Note: In radial flow, the dimensionless Bourdet derivative stabilizes at exactly p'D = 0.5.
    
    Returns:
        df_dim: DataFrame containing dimensionless columns.
    """
    times = np.asarray(times, dtype=np.float64)
    delta_p = np.asarray(delta_p, dtype=np.float64)
    derivative = np.asarray(derivative, dtype=np.float64)
    
    # Pressure dimensionless factor
    # Delta P = (141.2 * q * Bo * mu / (k * h)) * pD
    # => pD = Delta P / (141.2 * q * Bo * mu / (k * h))
    p_denom = (141.2 * q * Bo * mu) / (k * h)
    pD = delta_p / max(p_denom, 1e-12)
    pD_prime = derivative / max(p_denom, 1e-12)
    
    # Time dimensionless factor based on rw
    tD_const_rw = (0.0002637 * k) / (phi * mu * ct * (rw ** 2))
    tD_rw = times * tD_const_rw
    
    data = {
        "time_hr": times,
        "delta_p_psi": delta_p,
        "derivative_psi": derivative,
        "tD": tD_rw,
        "pD": pD,
        "pD_prime": pD_prime
    }
    
    if xf is not None and xf > 0:
        tD_const_xf = (0.0002637 * k) / (phi * mu * ct * (xf ** 2))
        data["tD_xf"] = times * tD_const_xf
        
    return pd.DataFrame(data)


def generate_theoretical_radial_type_curve(
    tD_array: np.ndarray,
    skin: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the classic infinite-acting radial flow (line-source) theoretical type curve:
        pD(tD) = 0.5 * (ln(tD) + 0.80907) + skin  (for tD > 100)
        p'D(tD) = 0.5 (constant horizontal derivative plateau)
    """
    tD_safe = np.maximum(tD_array, 1e-3)
    pD_theo = 0.5 * (np.log(tD_safe) + 0.80907) + skin
    pD_prime_theo = np.full_like(tD_array, 0.5)
    return pD_theo, pD_prime_theo
