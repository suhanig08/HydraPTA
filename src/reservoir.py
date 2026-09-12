"""
Reservoir engineering utilities, unit conversions, and petrophysical definitions.
"""

from typing import Dict, Any
from .models import ReservoirProperties, WellProperties


def calculate_diffusivity(k: float, phi: float, mu: float, ct: float) -> float:
    """
    Calculate hydraulic diffusivity eta (ft^2/hr) in Oilfield units.
    eta = 0.0002637 * k / (phi * mu * ct)
    """
    return (0.0002637 * k) / (phi * mu * ct)


def calculate_tD_xf(t_hr: float, k: float, phi: float, mu: float, ct: float, xf: float) -> float:
    """
    Calculate dimensionless time based on fracture half-length xf.
    t_Dxf = 0.0002637 * k * t / (phi * mu * ct * xf^2)
    """
    return (0.0002637 * k * t_hr) / (phi * mu * ct * (xf ** 2))


def calculate_t_from_tD_xf(tD_xf: float, k: float, phi: float, mu: float, ct: float, xf: float) -> float:
    """
    Convert dimensionless time tD_xf back to time in hours.
    t = tD_xf * phi * mu * ct * xf^2 / (0.0002637 * k)
    """
    return (tD_xf * phi * mu * ct * (xf ** 2)) / (0.0002637 * k)


def calculate_CD_xf(C: float, phi: float, ct: float, h: float, xf: float) -> float:
    """
    Calculate dimensionless wellbore storage coefficient based on fracture half-length xf.
    C_Dxf = 0.8936 * C / (phi * ct * h * xf^2)
    """
    return (0.8936 * C) / (phi * ct * h * (xf ** 2))


def calculate_pressure_factor(q: float, B: float, mu: float, k: float, h: float) -> float:
    """
    Calculate the conversion factor from dimensionless pressure pD to pressure drop delta_P (psi).
    delta_P = (141.2 * q * B * mu / (k * h)) * pD
    """
    return (141.2 * q * B * mu) / (k * h)
