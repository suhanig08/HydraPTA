"""
Validation benchmark engine for HydraPTA.
Simulates synthetic pressure responses using the forward Laplace/Gaver-Stehfest model,
executes the new automated Bourdet derivative and conventional PTA estimation workflow,
and generates validation comparison tables comparing actual vs estimated values with percentage errors.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from .models import ReservoirProperties, WellProperties, FractureProperties
from .pressure_transient import generate_synthetic_transient_response
from .derivative import calculate_bourdet_derivative, calculate_log_slopes
from .flow_regime import detect_sustained_flow_regimes, detect_stabilization_plateau
from .property_estimation import perform_full_drawdown_estimation
from data.demo_cases import get_demo_case


def run_demo_case_validation(case_id: str, add_noise: bool = False, noise_std: float = 0.25) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate the new PTA analysis engine against a predefined synthetic demo case.
    
    Returns:
        df_val_table: Table with columns: Parameter, Actual value, Estimated value, Percentage error, Model Status
        meta: Dictionary of intermediate diagnostic artifacts
    """
    case = get_demo_case(case_id)
    res: ReservoirProperties = case["reservoir"]
    well: WellProperties = case["well"]
    frac: FractureProperties = case["fracture"]
    
    # 1. Forward simulation
    df_sim = generate_synthetic_transient_response(res, well, frac, n_points=160)
    times = df_sim["time_hr"].values
    dp = df_sim["delta_P_psi"].values
    
    if add_noise:
        np.random.seed(42)
        dp = dp + np.random.normal(0, noise_std, len(dp))
        dp = np.maximum(dp, 0.0)
        
    # 2. Bourdet derivative & slopes
    derivative = calculate_bourdet_derivative(times, dp, L=0.15)
    slopes = calculate_log_slopes(times, derivative, L=0.20)
    
    # 3. Flow regimes & plateau
    regimes = detect_sustained_flow_regimes(times, dp, derivative, slopes)
    plateau = detect_stabilization_plateau(times, derivative, slopes)
    
    # 4. Property estimation
    f_params = {
        "q": well.q, "Bo": res.B, "mu": res.mu, "h": res.h,
        "phi": res.phi, "ct": res.ct, "rw": well.rw, "Pi": well.Pi,
        "k": res.k
    }
    
    est = perform_full_drawdown_estimation(
        times=times,
        delta_p=dp,
        derivative=derivative,
        plateau_info=plateau,
        detected_regimes=regimes,
        reservoir_params=f_params
    )
    
    # 5. Build Validation Comparison Table
    val_rows = []
    
    # Permeability
    true_k = res.k
    est_k_semi = est.get("k_semilog", {}).get("k_mD")
    if est_k_semi is not None:
        err_k = abs(est_k_semi - true_k) / true_k * 100.0
        val_rows.append({
            "Parameter": "Permeability (k) — Semilog",
            "Actual value": f"{true_k:g} mD",
            "Estimated value": f"{est_k_semi:g} mD",
            "Percentage error": f"{err_k:.2f}%",
            "Model Status": "✅ High Accuracy" if err_k < 15.0 else "⚠️ Moderate Accuracy"
        })
    else:
        val_rows.append({
            "Parameter": "Permeability (k) — Semilog",
            "Actual value": f"{true_k:g} mD",
            "Estimated value": "Unresolved",
            "Percentage error": "—",
            "Model Status": "⚠️ Radial flow not sustained"
        })
        
    est_k_plat = est.get("k_plateau", {}).get("k_mD")
    if est_k_plat is not None:
        err_kp = abs(est_k_plat - true_k) / true_k * 100.0
        val_rows.append({
            "Parameter": "Permeability (k) — Bourdet Plateau",
            "Actual value": f"{true_k:g} mD",
            "Estimated value": f"{est_k_plat:g} mD",
            "Percentage error": f"{err_kp:.2f}%",
            "Model Status": "✅ High Accuracy" if err_kp < 15.0 else "⚠️ Moderate Accuracy"
        })

    # Fracture Half-Length xf
    true_xf = frac.xf
    est_xf = est.get("xf", {}).get("xf_ft")
    if est_xf is not None:
        err_xf = abs(est_xf - true_xf) / true_xf * 100.0
        val_rows.append({
            "Parameter": "Fracture Half-Length (xf)",
            "Actual value": f"{true_xf:g} ft",
            "Estimated value": f"{est_xf:g} ft",
            "Percentage error": f"{err_xf:.2f}%",
            "Model Status": "✅ High Accuracy" if err_xf < 20.0 else "⚠️ Moderate Accuracy"
        })
    else:
        val_rows.append({
            "Parameter": "Fracture Half-Length (xf)",
            "Actual value": f"{true_xf:g} ft",
            "Estimated value": "Regime Absent",
            "Percentage error": "—",
            "Model Status": "ℹ️ Linear flow not prominent in this case"
        })

    # Fracture Conductivity Cf
    true_cf = frac.Cf
    est_cf = est.get("Cf", {}).get("Cf_mD_ft")
    if est_cf is not None:
        err_cf = abs(est_cf - true_cf) / true_cf * 100.0
        val_rows.append({
            "Parameter": "Fracture Conductivity (Cf = kf·wf)",
            "Actual value": f"{true_cf:g} mD-ft",
            "Estimated value": f"{est_cf:g} mD-ft",
            "Percentage error": f"{err_cf:.2f}%",
            "Model Status": "✅ High Accuracy" if err_cf < 30.0 else "⚠️ Moderate Accuracy"
        })
    else:
        val_rows.append({
            "Parameter": "Fracture Conductivity (Cf = kf·wf)",
            "Actual value": f"{true_cf:g} mD-ft",
            "Estimated value": "Regime Absent",
            "Percentage error": "—",
            "Model Status": "ℹ️ Bilinear flow absent / infinite conductivity"
        })
        
    df_val = pd.DataFrame(val_rows)
    meta = {
        "times": times,
        "delta_p": dp,
        "derivative": derivative,
        "regimes": regimes,
        "plateau": plateau,
        "estimation": est
    }
    return df_val, meta
