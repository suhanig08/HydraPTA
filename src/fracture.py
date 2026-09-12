"""
Hydraulic fracture characterization, conductivity calculations, and diagnostic criteria.
"""

from typing import Dict, Any, Tuple
from .models import FractureProperties, ReservoirProperties


def calculate_fracture_conductivity(kf: float, wf: float) -> float:
    """
    Calculate fracture conductivity Cf = kf * wf (mD-ft).
    
    Parameters:
        kf: Fracture permeability in mD
        wf: Fracture width in ft
    Returns:
        Cf: Fracture conductivity in mD-ft
    """
    return kf * wf


def calculate_dimensionless_conductivity(kf: float, wf: float, k: float, xf: float) -> float:
    """
    Calculate dimensionless fracture conductivity CfD = (kf * wf) / (k * xf).
    
    Parameters:
        kf: Fracture permeability in mD
        wf: Fracture width in ft
        k: Reservoir matrix permeability in mD
        xf: Fracture half-length in ft
    Returns:
        CfD: Dimensionless fracture conductivity (dimensionless)
    """
    if k <= 0 or xf <= 0:
        raise ValueError("Reservoir permeability k and fracture half-length xf must be positive.")
    return (kf * wf) / (k * xf)


def classify_fracture_conductivity_type(CfD: float) -> Tuple[str, str]:
    """
    Classify the fracture conductivity behavior according to Cinco-Ley & Samaniego criteria.
    
    Returns:
        (category_name, detailed_description)
    """
    if CfD >= 100.0:
        return (
            "Infinite-Conductivity Fracture",
            f"CfD = {CfD:.2f} (>= 100). Negligible pressure drop along the fracture. "
            "Exhibits formation linear flow (slope 1/2) without observable bilinear flow, "
            "transitioning directly to pseudoradial flow at tDxf > 2-3."
        )
    elif CfD >= 10.0:
        return (
            "High Finite-Conductivity Fracture",
            f"CfD = {CfD:.2f} (10 <= CfD < 100). Fracture acts as a very high-permeability conduit. "
            "Dominantly formation linear flow (slope 1/2), with minimal early-time bilinear signature "
            "masked by wellbore storage."
        )
    elif CfD >= 0.1:
        return (
            "Finite-Conductivity Fracture",
            f"CfD = {CfD:.2f} (0.1 <= CfD < 10). Noticeable pressure drop along the fracture channel. "
            "Exhibits characteristic bilinear flow (slope 1/4) where linear flow in matrix and "
            "linear flow in fracture occur simultaneously."
        )
    else:
        return (
            "Low-Conductivity / Choked Fracture",
            f"CfD = {CfD:.4f} (< 0.1). Severe choke effect inside the fracture channel. "
            "Bilinear flow dominates early/intermediate times with significant additional pressure drop."
        )


def theoretical_regime_limits(CfD: float) -> Dict[str, float]:
    """
    Estimate theoretical dimensionless time limits (tDxf) for flow regime transitions
    based on Cinco-Ley & Samaniego (1981) and Agarwal et al. (1979).
    """
    # Bilinear flow end time: tDxf_be approx 0.1 / (CfD^2) for CfD < 50
    tDxf_bilinear_end = 0.1 / (CfD ** 2) if CfD < 50 else 0.0
    # Formation linear flow start and end
    tDxf_linear_start = 10.0 / (CfD ** 2) if CfD < 50 else 0.001
    tDxf_linear_end = 0.0160 if CfD >= 10 else 0.02
    # Pseudoradial flow start time
    tDxf_pseudoradial_start = 2.0 if CfD >= 10 else 3.0
    
    return {
        "tDxf_bilinear_end": tDxf_bilinear_end,
        "tDxf_linear_start": tDxf_linear_start,
        "tDxf_linear_end": tDxf_linear_end,
        "tDxf_pseudoradial_start": tDxf_pseudoradial_start,
    }
