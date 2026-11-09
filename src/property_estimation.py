"""
Conventional petroleum engineering property estimation module.
Implements semilog radial flow analysis, Bourdet plateau permeability estimation,
skin factor determination, linear flow fracture half-length, and bilinear flow conductivity.
All formulations strictly adhere to standard Petroleum Field Unit conventions.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from scipy.stats import linregress


def estimate_permeability_semilog(
    times: np.ndarray,
    delta_p: np.ndarray,
    t_start: float,
    t_end: float,
    q: float,
    Bo: float,
    mu: float,
    h: float
) -> Dict[str, Any]:
    """
    Estimate reservoir permeability k from semilog slope m in the pseudoradial flow regime.
    
    Equation:
        Delta P = m * log10(t) + b
        m = 162.6 * q * Bo * mu / (k * h)  [psi / log-cycle]
        => k = 162.6 * q * Bo * mu / (m * h)  [mD]
        
    Parameters:
        times: Test time array (hr).
        delta_p: Pressure drawdown Delta P array (psi).
        t_start, t_end: Time bounds for semilog straight line.
        q: Flow rate (STB/d).
        Bo: Formation volume factor (rb/STB).
        mu: Oil viscosity (cp).
        h: Net pay thickness (ft).
        
    Returns:
        Dict with keys: k_mD, slope_m, intercept_b, r_squared, num_points, t_start, t_end, is_valid, error
    """
    res: Dict[str, Any] = {
        "k_mD": None,
        "slope_m": None,
        "intercept_b": None,
        "r_squared": None,
        "num_points": 0,
        "t_start": t_start,
        "t_end": t_end,
        "is_valid": False,
        "error": None
    }
    
    if any(p <= 0 for p in [q, Bo, mu, h]):
        res["error"] = "Parameters q, Bo, mu, and h must all be positive numbers."
        return res
        
    mask = (times >= t_start) & (times <= t_end) & (times > 0)
    if np.sum(mask) < 3:
        res["error"] = f"Insufficient points ({np.sum(mask)} found, >= 3 required) in selected interval [{t_start:.3g}, {t_end:.3g}] hr."
        return res
        
    sub_t = times[mask]
    sub_dp = delta_p[mask]
    
    log10_t = np.log10(sub_t)
    reg = linregress(log10_t, sub_dp)
    
    slope_m = float(reg.slope)
    intercept_b = float(reg.intercept)
    r2 = float(reg.rvalue ** 2)
    
    res["slope_m"] = round(slope_m, 3)
    res["intercept_b"] = round(intercept_b, 3)
    res["r_squared"] = round(r2, 4)
    res["num_points"] = int(np.sum(mask))
    
    if slope_m <= 0:
        res["error"] = f"Semilog slope is non-positive (m = {slope_m:.3g} psi/cycle). Physical drawdown requires m > 0."
        return res
        
    k_est = (162.6 * q * Bo * mu) / (slope_m * h)
    res["k_mD"] = round(float(k_est), 3)
    res["is_valid"] = True
    return res


def estimate_permeability_derivative_plateau(
    derivative_plateau: float,
    q: float,
    Bo: float,
    mu: float,
    h: float
) -> Dict[str, Any]:
    """
    Estimate reservoir permeability k from the Bourdet derivative horizontal plateau.
    
    Equation:
        Delta P'_plateau = 70.6 * q * Bo * mu / (k * h)
        => k = 70.6 * q * Bo * mu / (h * Delta P'_plateau)  [mD]
        
    Returns:
        Dict with keys: k_mD, derv_plateau_psi, is_valid, error
    """
    res: Dict[str, Any] = {
        "k_mD": None,
        "derv_plateau_psi": derivative_plateau,
        "is_valid": False,
        "error": None
    }
    
    if derivative_plateau is None or derivative_plateau <= 0:
        res["error"] = "Derivative plateau value must be strictly positive."
        return res
        
    if any(p <= 0 for p in [q, Bo, mu, h]):
        res["error"] = "Parameters q, Bo, mu, and h must all be positive numbers."
        return res
        
    k_est = (70.6 * q * Bo * mu) / (h * derivative_plateau)
    res["k_mD"] = round(float(k_est), 3)
    res["is_valid"] = True
    return res


def estimate_skin_factor(
    k: float,
    slope_m: float,
    intercept_b: float,
    phi: float,
    mu: float,
    ct: float,
    rw: float
) -> Dict[str, Any]:
    """
    Estimate well skin factor S from semilog straight line parameters.
    
    Equation:
        Delta P_1hr = intercept_b  (Delta P at t = 1 hr where log10(1) = 0)
        S = 1.151 * [ (Delta P_1hr / m) - log10(k / (phi * mu * ct * rw^2)) + 3.23 ]
        
    Returns:
        Dict with keys: skin, delta_p_1hr, is_valid, confidence, assumptions, error
    """
    res: Dict[str, Any] = {
        "skin": None,
        "delta_p_1hr": intercept_b,
        "is_valid": False,
        "confidence": "Moderate",
        "assumptions": "Infinite-acting homogeneous radial flow around wellbore; intercept evaluated at t = 1 hr.",
        "error": None
    }
    
    missing = []
    if k is None or k <= 0: missing.append("k")
    if slope_m is None or slope_m <= 0: missing.append("semilog slope m")
    if phi is None or phi <= 0: missing.append("porosity phi")
    if mu is None or mu <= 0: missing.append("viscosity mu")
    if ct is None or ct <= 0: missing.append("compressibility ct")
    if rw is None or rw <= 0: missing.append("wellbore radius rw")
    
    if missing:
        res["error"] = f"Cannot calculate skin factor. Missing or invalid parameters: {', '.join(missing)}."
        return res
        
    denom = phi * mu * ct * (rw ** 2)
    if denom <= 0:
        res["error"] = "Invalid petrophysical group phi * mu * ct * rw^2 <= 0."
        return res
        
    log_group = np.log10(k / denom)
    skin_val = 1.151 * ((intercept_b / slope_m) - log_group + 3.23)
    
    res["skin"] = round(float(skin_val), 2)
    res["is_valid"] = True
    res["confidence"] = "High" if abs(skin_val) < 100 else "Low"
    return res


def estimate_fracture_half_length(
    times: np.ndarray,
    delta_p: np.ndarray,
    t_start: float,
    t_end: float,
    q: float,
    Bo: float,
    mu: float,
    h: float,
    k: float,
    phi: float,
    ct: float
) -> Dict[str, Any]:
    """
    Estimate fracture half-length xf from linear flow (half-slope m_LF on Delta P vs sqrt(t)).
    
    Equation:
        Delta P = m_LF * sqrt(t) + b_LF
        m_LF = [4.06 * q * Bo / (h * xf)] * sqrt(mu / (k * phi * ct))
        => xf = [4.06 * q * Bo / (m_LF * h)] * sqrt(mu / (k * phi * ct))  [ft]
        
    Returns:
        Dict with keys: xf_ft, slope_m_lf, intercept_b, r_squared, num_points, is_valid, error
    """
    res: Dict[str, Any] = {
        "xf_ft": None,
        "slope_m_lf": None,
        "intercept_b": None,
        "r_squared": None,
        "num_points": 0,
        "t_start": t_start,
        "t_end": t_end,
        "is_valid": False,
        "error": None
    }
    
    missing = []
    if q is None or q <= 0: missing.append("q")
    if Bo is None or Bo <= 0: missing.append("Bo")
    if mu is None or mu <= 0: missing.append("mu")
    if h is None or h <= 0: missing.append("h")
    if k is None or k <= 0: missing.append("permeability k")
    if phi is None or phi <= 0: missing.append("porosity phi")
    if ct is None or ct <= 0: missing.append("compressibility ct")
    
    if missing:
        res["error"] = f"Cannot calculate fracture half-length. Missing required parameters: {', '.join(missing)}."
        return res
        
    mask = (times >= t_start) & (times <= t_end) & (times > 0)
    if np.sum(mask) < 3:
        res["error"] = f"Insufficient data points ({np.sum(mask)} found, >= 3 required) in linear flow interval [{t_start:.3g}, {t_end:.3g}] hr."
        return res
        
    sub_t = times[mask]
    sub_dp = delta_p[mask]
    
    sqrt_t = np.sqrt(sub_t)
    reg = linregress(sqrt_t, sub_dp)
    
    m_lf = float(reg.slope)
    b_lf = float(reg.intercept)
    r2 = float(reg.rvalue ** 2)
    
    res["slope_m_lf"] = round(m_lf, 3)
    res["intercept_b"] = round(b_lf, 3)
    res["r_squared"] = round(r2, 4)
    res["num_points"] = int(np.sum(mask))
    
    if m_lf <= 0:
        res["error"] = f"Linear flow slope is non-positive (m_LF = {m_lf:.3g}). Physical response requires m_LF > 0."
        return res
        
    diff_group = np.sqrt(mu / (k * phi * ct))
    xf_est = ((4.064 * q * Bo) / (m_lf * h)) * diff_group
    
    res["xf_ft"] = round(float(xf_est), 1)
    res["is_valid"] = True
    return res


def estimate_fracture_conductivity(
    times: np.ndarray,
    delta_p: np.ndarray,
    t_start: float,
    t_end: float,
    q: float,
    Bo: float,
    mu: float,
    h: float,
    k: float,
    phi: float,
    ct: float
) -> Dict[str, Any]:
    """
    Estimate fracture conductivity Cf = kf * wf from bilinear flow (quarter-slope m_BL on Delta P vs t^(1/4)).
    
    Equation:
        Delta P = m_BL * t^(1/4) + b_BL
        m_BL = [44.1 * q * mu * Bo] / [ h * (kf * wf)^(1/2) * (phi * mu * ct * k)^(1/4) ]
        => Cf = kf * wf = [ (44.1 * q * mu * Bo) / (m_BL * h * (phi * mu * ct * k)^(1/4)) ]^2  [mD-ft]
        
    Returns:
        Dict with keys: Cf_mD_ft, slope_m_bl, intercept_b, r_squared, num_points, is_valid, error
    """
    res: Dict[str, Any] = {
        "Cf_mD_ft": None,
        "slope_m_bl": None,
        "intercept_b": None,
        "r_squared": None,
        "num_points": 0,
        "t_start": t_start,
        "t_end": t_end,
        "is_valid": False,
        "error": None
    }
    
    missing = []
    if q is None or q <= 0: missing.append("q")
    if Bo is None or Bo <= 0: missing.append("Bo")
    if mu is None or mu <= 0: missing.append("mu")
    if h is None or h <= 0: missing.append("h")
    if k is None or k <= 0: missing.append("permeability k")
    if phi is None or phi <= 0: missing.append("porosity phi")
    if ct is None or ct <= 0: missing.append("compressibility ct")
    
    if missing:
        res["error"] = f"Cannot calculate fracture conductivity. Missing required parameters: {', '.join(missing)}."
        return res
        
    mask = (times >= t_start) & (times <= t_end) & (times > 0)
    if np.sum(mask) < 3:
        res["error"] = f"Insufficient data points ({np.sum(mask)} found, >= 3 required) in bilinear flow interval [{t_start:.3g}, {t_end:.3g}] hr."
        return res
        
    sub_t = times[mask]
    sub_dp = delta_p[mask]
    
    t_fourth = np.power(sub_t, 0.25)
    reg = linregress(t_fourth, sub_dp)
    
    m_bl = float(reg.slope)
    b_bl = float(reg.intercept)
    r2 = float(reg.rvalue ** 2)
    
    res["slope_m_bl"] = round(m_bl, 3)
    res["intercept_b"] = round(b_bl, 3)
    res["r_squared"] = round(r2, 4)
    res["num_points"] = int(np.sum(mask))
    
    if m_bl <= 0:
        res["error"] = f"Bilinear flow slope is non-positive (m_BL = {m_bl:.3g}). Physical response requires m_BL > 0."
        return res
        
    matrix_group = (phi * mu * ct * k) ** 0.25
    numerator = 44.1 * q * mu * Bo
    cf_sqrt = numerator / (m_bl * h * matrix_group)
    cf_est = cf_sqrt ** 2
    
    res["Cf_mD_ft"] = round(float(cf_est), 1)
    res["is_valid"] = True
    return res


def perform_full_drawdown_estimation(
    times: np.ndarray,
    delta_p: np.ndarray,
    derivative: np.ndarray,
    plateau_info: Dict[str, Any],
    detected_regimes: List[Any],
    reservoir_params: Dict[str, Any],
    override_radial_interval: Optional[Tuple[float, float]] = None,
    override_linear_interval: Optional[Tuple[float, float]] = None,
    override_bilinear_interval: Optional[Tuple[float, float]] = None
) -> Dict[str, Any]:
    """
    Execute complete property estimation workflow for drawdown test.
    
    Computes:
        1. k from semilog method
        2. k from Bourdet derivative plateau
        3. Percentage discrepancy between k estimates
        4. Skin factor S
        5. Fracture half-length xf (if linear flow present)
        6. Fracture conductivity Cf (if bilinear flow present)
    """
    q = reservoir_params.get("q")
    Bo = reservoir_params.get("Bo")
    mu = reservoir_params.get("mu")
    h = reservoir_params.get("h")
    phi = reservoir_params.get("phi")
    ct = reservoir_params.get("ct")
    rw = reservoir_params.get("rw")
    k_prior = reservoir_params.get("k")  # user supplied, if known
    
    summary: Dict[str, Any] = {
        "k_semilog": None,
        "k_plateau": None,
        "k_diff_pct": None,
        "best_k": None,
        "skin": None,
        "xf": None,
        "Cf": None,
        "notes": []
    }
    
    # 1. Semilog Permeability
    rad_start, rad_end = None, None
    if override_radial_interval:
        rad_start, rad_end = override_radial_interval
    elif plateau_info.get("is_detected"):
        rad_start, rad_end = plateau_info["t_start"], plateau_info["t_end"]
    else:
        # Check detected regimes
        rad_reg = next((r for r in detected_regimes if getattr(r, "regime", "") == "Pseudoradial Flow"), None)
        if rad_reg:
            rad_start, rad_end = rad_reg.t_start, rad_reg.t_end
            
    if rad_start is not None and rad_end is not None:
        k_semi_res = estimate_permeability_semilog(times, delta_p, rad_start, rad_end, q, Bo, mu, h)
        summary["k_semilog"] = k_semi_res
    else:
        summary["k_semilog"] = {"is_valid": False, "error": "Pseudoradial flow regime not identified; cannot perform semilog straight-line analysis."}
        
    # 2. Derivative Plateau Permeability
    if plateau_info.get("is_detected") and plateau_info.get("derv_mean"):
        k_plat_res = estimate_permeability_derivative_plateau(plateau_info["derv_mean"], q, Bo, mu, h)
        summary["k_plateau"] = k_plat_res
    else:
        summary["k_plateau"] = {"is_valid": False, "error": "Stable Bourdet derivative plateau not detected."}
        
    # Compare Permeabilities
    k_semi_val = summary["k_semilog"].get("k_mD") if summary["k_semilog"].get("is_valid") else None
    k_plat_val = summary["k_plateau"].get("k_mD") if summary["k_plateau"].get("is_valid") else None
    
    if k_semi_val is not None and k_plat_val is not None:
        pct_diff = abs(k_semi_val - k_plat_val) / max(k_plat_val, 1e-6) * 100.0
        summary["k_diff_pct"] = round(pct_diff, 2)
        summary["best_k"] = k_semi_val
        summary["notes"].append(
            f"Permeability estimates: Semilog = {k_semi_val:g} mD, Bourdet Plateau = {k_plat_val:g} mD "
            f"(Discrepancy: {pct_diff:.1f}%)."
        )
    elif k_semi_val is not None:
        summary["best_k"] = k_semi_val
    elif k_plat_val is not None:
        summary["best_k"] = k_plat_val
    elif k_prior is not None and k_prior > 0:
        summary["best_k"] = k_prior
        
    # 3. Skin Estimation
    active_k = summary["best_k"]
    if summary["k_semilog"].get("is_valid") and active_k is not None:
        m_slope = summary["k_semilog"]["slope_m"]
        b_intercept = summary["k_semilog"]["intercept_b"]
        skin_res = estimate_skin_factor(active_k, m_slope, b_intercept, phi, mu, ct, rw)
        summary["skin"] = skin_res
    else:
        summary["skin"] = {"is_valid": False, "error": "Skin estimation requires confirmed semilog straight-line slope and intercept."}

    # 4. Fracture Half-Length xf
    lin_start, lin_end = None, None
    if override_linear_interval:
        lin_start, lin_end = override_linear_interval
    else:
        lin_reg = next((r for r in detected_regimes if getattr(r, "regime", "") == "Linear Flow"), None)
        if lin_reg:
            lin_start, lin_end = lin_reg.t_start, lin_reg.t_end
            
    if lin_start is not None and lin_end is not None:
        xf_res = estimate_fracture_half_length(times, delta_p, lin_start, lin_end, q, Bo, mu, h, active_k, phi, ct)
        summary["xf"] = xf_res
    else:
        summary["xf"] = {"is_valid": False, "error": "Linear flow regime (half-slope) not detected."}

    # 5. Fracture Conductivity Cf
    bl_start, bl_end = None, None
    if override_bilinear_interval:
        bl_start, bl_end = override_bilinear_interval
    else:
        bl_reg = next((r for r in detected_regimes if getattr(r, "regime", "") == "Bilinear Flow"), None)
        if bl_reg:
            bl_start, bl_end = bl_reg.t_start, bl_reg.t_end
            
    if bl_start is not None and bl_end is not None:
        cf_res = estimate_fracture_conductivity(times, delta_p, bl_start, bl_end, q, Bo, mu, h, active_k, phi, ct)
        summary["Cf"] = cf_res
    else:
        summary["Cf"] = {"is_valid": False, "error": "Bilinear flow regime (quarter-slope) not detected."}

    return summary
