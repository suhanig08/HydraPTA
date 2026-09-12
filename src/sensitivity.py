"""
Parameter sensitivity analysis engine for HydraPTA.
Generates multi-scenario transient responses, extracts diagnostic shifts, and compiles engineering insights.
"""

from typing import List, Dict, Any, Tuple, Optional
import copy
import numpy as np
import pandas as pd

from .models import ReservoirProperties, WellProperties, FractureProperties, FlowRegimeRecord
from .pressure_transient import generate_synthetic_transient_response
from .derivative import calculate_bourdet_derivative, calculate_log_slopes
from .flow_regime import detect_sustained_flow_regimes


SENSITIVITY_PARAMETERS = {
    "xf": {
        "label": "Fracture Half-Length (xf)",
        "unit": "ft",
        "default_values": [100.0, 250.0, 500.0, 750.0, 1000.0],
        "category": "fracture"
    },
    "kf": {
        "label": "Fracture Permeability (kf)",
        "unit": "mD",
        "default_values": [500.0, 1000.0, 5000.0, 10000.0, 50000.0],
        "category": "fracture"
    },
    "wf": {
        "label": "Fracture Width (wf)",
        "unit": "ft",
        "default_values": [0.005, 0.01, 0.02, 0.05, 0.10],
        "category": "fracture"
    },
    "Cf": {
        "label": "Fracture Conductivity (Cf = kf * wf)",
        "unit": "mD-ft",
        "default_values": [5.0, 20.0, 100.0, 200.0, 1000.0],
        "category": "fracture"
    },
    "k": {
        "label": "Reservoir Permeability (k)",
        "unit": "mD",
        "default_values": [0.5, 1.0, 5.0, 10.0, 25.0],
        "category": "reservoir"
    },
    "q": {
        "label": "Well Flow Rate (q)",
        "unit": "STB/day",
        "default_values": [100.0, 250.0, 500.0, 750.0, 1000.0],
        "category": "well"
    }
}


def generate_sensitivity_scenarios(
    param_key: str,
    base_reservoir: ReservoirProperties,
    base_well: WellProperties,
    base_fracture: FractureProperties,
    custom_values: Optional[List[float]] = None,
    n_points: int = 120
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Run sensitivity sweeps for a selected parameter across multiple values.
    
    Returns:
        (df_curves, df_summary_table, engineering_insight_text)
    """
    param_meta = SENSITIVITY_PARAMETERS[param_key]
    values = custom_values if custom_values is not None else param_meta["default_values"]
    
    all_curves_records = []
    summary_records = []
    
    for val in values:
        # Clone base objects
        res = copy.deepcopy(base_reservoir)
        well = copy.deepcopy(base_well)
        frac = copy.deepcopy(base_fracture)
        
        # Apply parameter value
        if param_key == "xf":
            frac.xf = float(val)
        elif param_key == "kf":
            frac.kf = float(val)
        elif param_key == "wf":
            frac.wf = float(val)
        elif param_key == "Cf":
            # Adjust kf keeping wf constant
            frac.kf = float(val) / frac.wf
        elif param_key == "k":
            res.k = float(val)
        elif param_key == "q":
            well.q = float(val)
            
        # Run forward simulation
        df_sim = generate_synthetic_transient_response(res, well, frac, n_points=n_points)
        times = df_sim["time_hr"].values
        dp = df_sim["delta_P_psi"].values
        derv = calculate_bourdet_derivative(times, dp, L=0.15)
        slopes = calculate_log_slopes(times, derv, L=0.20)
        
        # Detect regimes for this scenario
        regimes = detect_sustained_flow_regimes(times, dp, derv, slopes)
        
        # Determine dominant regime and key transition time
        dominant_reg = "Transition"
        transition_time_str = "N/A"
        
        if regimes:
            # Longest regime
            longest = max(regimes, key=lambda r: (r.t_end - r.t_start))
            dominant_reg = longest.regime
            transition_time_str = f"{regimes[0].t_end:.3g} hr"
            
        # Diagnostic observation
        cfd_val = frac.CfD(res.k)
        if cfd_val >= 10.0:
            obs = f"High CfD ({cfd_val:.1f}); Linear flow dominates intermediate times."
        elif cfd_val < 0.1:
            obs = f"Choked fracture ({cfd_val:.4f}); Severe bilinear flow depression."
        else:
            obs = f"Finite conductivity ({cfd_val:.2f}); Bilinear-to-linear transition."
            
        summary_records.append({
            f"{param_meta['label']} ({param_meta['unit']})": f"{val:g}",
            "Dimensionless CfD": f"{cfd_val:.3g}",
            "Dominant Flow Regime": dominant_reg,
            "First Transition Time": transition_time_str,
            "Diagnostic Observation": obs
        })
        
        # Collect curves data
        val_label = f"{val:g} {param_meta['unit']}"
        for t, p, d in zip(times, dp, derv):
            all_curves_records.append({
                "scenario": val_label,
                "param_value": val,
                "time_hr": t,
                "delta_P_psi": p,
                "derivative_psi": d
            })
            
    df_curves = pd.DataFrame(all_curves_records)
    df_summary = pd.DataFrame(summary_records)
    
    # Generate automatic engineering insight based on physical parameter analyzed
    insight = _generate_sensitivity_insight(param_key, values, df_summary)
    
    return df_curves, df_summary, insight


def _generate_sensitivity_insight(param_key: str, values: List[float], df_summary: pd.DataFrame) -> str:
    """Generate dynamic physical petroleum engineering interpretation of sensitivity results."""
    min_v, max_v = min(values), max(values)
    
    if param_key == "xf":
        return (
            f"**Fracture Half-Length Sensitivity ({min_v:g} to {max_v:g} ft):**\n\n"
            "• **Pressure Drawdown Impact**: Longer fracture half-lengths significantly increase total contact area with the formation, "
            "reducing the overall reservoir pressure drop (Delta P) for a given production rate.\n"
            "• **Derivative Signature**: Increasing $x_f$ shifts the formation linear flow period (half-slope $m=0.5$) downward and "
            "substantially delays the onset of late-time pseudoradial flow ($t_{Dxf} \\propto 1/x_f^2$)."
        )
    elif param_key in ["kf", "wf", "Cf"]:
        return (
            f"**Fracture Conductivity Sensitivity ({min_v:g} to {max_v:g}):**\n\n"
            "• **Flow Regime Evolution**: At lower conductivity values ($C_{fD} < 10$), the derivative exhibits a prolonged quarter-slope "
            "($m=0.25$) characteristic of bilinear flow due to internal fracture flow resistance.\n"
            "• **High Conductivity Behavior**: As $C_f$ increases ($C_{fD} \\ge 10$), internal fracture resistance vanishes, the bilinear signature "
            "disappears, and clean formation linear flow ($m=0.5$) dominates before transitioning to pseudoradial flow."
        )
    elif param_key == "k":
        return (
            f"**Reservoir Matrix Permeability Sensitivity ({min_v:g} to {max_v:g} mD):**\n\n"
            "• **Transient Velocity**: Higher reservoir permeability accelerates pressure diffusion through the formation, causing all transient "
            "flow regimes (linear, elliptical) to pass much more rapidly and establishing pseudoradial flow earlier.\n"
            "• **Relative Conductivity**: Higher matrix permeability reduces $C_{fD} = k_f w_f / (k x_f)$, making the fracture behave more like a "
            "finite-conductivity choke relative to the fast matrix feeding."
        )
    elif param_key == "q":
        return (
            f"**Flow Rate Sensitivity ({min_v:g} to {max_v:g} STB/day):**\n\n"
            "• **Linear Scaling**: For Darcy single-phase liquid flow, pressure drop $\\Delta P$ and derivative $\\Delta P'$ scale directly "
            "proportional to flow rate $q$.\n"
            "• **Diagnostic Shape Invariance**: The normalized log-log slopes and flow regime transition times remain identical because dimensionless "
            "diffusivity is independent of production rate."
        )
    return "Sensitivity analysis reveals expected physical transient shifts consistent with analytical reservoir dynamics."
