"""
Pressure Buildup (PBU) analysis module for HydraPTA.
Implements Horner analysis, Agarwal equivalent time transformations, Horner semilog plotting,
permeability estimation, extrapolated reservoir pressure P*, and skin factor calculation.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from scipy.stats import linregress
from .derivative import calculate_bourdet_derivative


def calculate_buildup_variables(
    dt_hr: np.ndarray,
    pws_psi: np.ndarray,
    tp_hr: float,
    pwf_at_shutin: Optional[float] = None
) -> pd.DataFrame:
    """
    Calculate core pressure buildup diagnostic variables:
        - Pressure increase Delta Pws = Pws(dt) - Pwf(dt=0)
        - Horner time ratio = (tp + dt) / dt
        - Agarwal equivalent shut-in time = (tp * dt) / (tp + dt)
        
    Parameters:
        dt_hr: Shut-in elapsed time array (hours).
        pws_psi: Measured shut-in bottomhole pressure array (psi).
        tp_hr: Producing time before shut-in (hours).
        pwf_at_shutin: Flowing pressure immediately prior to shut-in (psi).
                       If None, the first recorded pressure or minimum is used.
    Returns:
        df_buildup: DataFrame of buildup time, pressure, and Horner variables.
    """
    dt_arr = np.asarray(dt_hr, dtype=np.float64)
    pws_arr = np.asarray(pws_psi, dtype=np.float64)
    
    # Exclude non-positive time for Horner and derivative calculations
    valid = (dt_arr > 0) & (pws_arr > 0)
    sub_dt = dt_arr[valid]
    sub_pws = pws_arr[valid]
    
    # Determine Pwf at shut-in
    if pwf_at_shutin is None or pwf_at_shutin <= 0:
        pwf_ref = float(sub_pws[0])
    else:
        pwf_ref = float(pwf_at_shutin)
        
    delta_pws = sub_pws - pwf_ref
    horner_ratio = (tp_hr + sub_dt) / sub_dt
    agarwal_te = (tp_hr * sub_dt) / (tp_hr + sub_dt)
    
    # Calculate Bourdet derivative using Agarwal equivalent time:
    # d(Delta Pws) / d(ln te)
    derv_agarwal = calculate_bourdet_derivative(agarwal_te, delta_pws, L=0.15)
    
    df = pd.DataFrame({
        "dt_hr": sub_dt,
        "pws_psi": sub_pws,
        "delta_pws_psi": delta_pws,
        "horner_ratio": horner_ratio,
        "log10_horner": np.log10(horner_ratio),
        "agarwal_te_hr": agarwal_te,
        "derivative_psi": derv_agarwal
    })
    
    return df


def estimate_buildup_horner_parameters(
    df_buildup: pd.DataFrame,
    tp_hr: float,
    pwf_at_shutin: float,
    q: float,
    Bo: float,
    mu: float,
    h: float,
    phi: float,
    ct: float,
    rw: float,
    horner_ratio_min: Optional[float] = None,
    horner_ratio_max: Optional[float] = None
) -> Dict[str, Any]:
    """
    Perform Horner semilog straight-line analysis on pressure buildup data:
    
        Pws = P* - m_Horner * log10((tp + dt) / dt)
        
    Slope m_Horner (psi/cycle) is used to calculate:
        - Reservoir permeability: k = 162.6 * q * Bo * mu / (m_Horner * h)
        - Extrapolated reservoir pressure: P* at (tp + dt)/dt = 1
        - Pressure at 1 hr: P1hr evaluated on the Horner line at (tp + 1)/1
        - Skin factor: S = 1.151 * [ (P1hr - Pwf_shutin) / m_Horner - log10(k / (phi * mu * ct * rw^2)) + 3.23 ]
        
    Returns:
        results: Dict containing k, P*, P1hr, skin, slope_m, r_squared, etc.
    """
    res: Dict[str, Any] = {
        "k_mD": None,
        "P_star_psi": None,
        "P_1hr_psi": None,
        "skin": None,
        "slope_m_horner": None,
        "intercept_b": None,
        "r_squared": None,
        "num_points": 0,
        "is_valid": False,
        "error": None
    }
    
    if df_buildup.empty or len(df_buildup) < 3:
        res["error"] = "Insufficient buildup data points for Horner analysis."
        return res
        
    # Filter interval
    if horner_ratio_min is not None and horner_ratio_max is not None:
        mask = (df_buildup["horner_ratio"] >= horner_ratio_min) & (df_buildup["horner_ratio"] <= horner_ratio_max)
    else:
        # Default: use the late-time region (smallest Horner ratios, e.g. first 40% of points by Horner ratio)
        sorted_hr = df_buildup["horner_ratio"].sort_values()
        threshold_hr = sorted_hr.iloc[min(len(sorted_hr) // 2 + 2, len(sorted_hr) - 1)]
        mask = df_buildup["horner_ratio"] <= threshold_hr
        
    if np.sum(mask) < 3:
        mask = np.ones(len(df_buildup), dtype=bool)
        
    sub_df = df_buildup[mask]
    x_val = sub_df["log10_horner"].values
    y_val = sub_df["pws_psi"].values
    
    reg = linregress(x_val, y_val)
    # Note: Pws = P* - m * log10(HR), so reg.slope is -m_Horner
    slope_raw = float(reg.slope)
    m_horner = abs(slope_raw)
    p_star = float(reg.intercept)  # at log10(HR) = 0, HR = 1
    r2 = float(reg.rvalue ** 2)
    
    res["slope_m_horner"] = round(m_horner, 3)
    res["P_star_psi"] = round(p_star, 2)
    res["intercept_b"] = round(float(reg.intercept), 2)
    res["r_squared"] = round(r2, 4)
    res["num_points"] = len(sub_df)
    
    if m_horner <= 0:
        res["error"] = "Horner slope is non-positive or indeterminate."
        return res
        
    # Calculate permeability
    k_est = (162.6 * q * Bo * mu) / (m_horner * h)
    res["k_mD"] = round(float(k_est), 3)
    
    # Calculate P1hr on Horner straight line at dt = 1 hr
    hr_1hr = (tp_hr + 1.0) / 1.0
    log_hr_1hr = np.log10(hr_1hr)
    p_1hr = reg.intercept + slope_raw * log_hr_1hr
    res["P_1hr_psi"] = round(float(p_1hr), 2)
    
    # Calculate skin factor
    denom = phi * mu * ct * (rw ** 2)
    if denom > 0 and k_est > 0:
        log_term = np.log10(k_est / denom)
        skin_val = 1.151 * (((p_1hr - pwf_at_shutin) / m_horner) - log_term + 3.23)
        res["skin"] = round(float(skin_val), 2)
        
    res["is_valid"] = True
    return res
