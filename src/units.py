"""
Unit conversion utilities for petroleum engineering calculations.
Supports Field Units and SI Units with transparent bidirectional conversions.
"""

from typing import Dict, Any


# Standard conversion factors: SI to Field
# Field units:
#   Pressure: psi
#   Time: hr
#   Rate: STB/day
#   Permeability: mD
#   Viscosity: cp
#   Compressibility: 1/psi
#   Thickness: ft
#   Radius: ft
#   FVF: rb/STB

UNIT_DEFINITIONS = {
    "Field": {
        "pressure": "psi",
        "time": "hr",
        "rate": "STB/d",
        "perm": "mD",
        "viscosity": "cp",
        "compressibility": "1/psi",
        "thickness": "ft",
        "radius": "ft",
        "fvf": "rb/STB",
        "half_length": "ft",
        "conductivity": "mD-ft"
    },
    "SI": {
        "pressure": "bar",
        "time": "hr",
        "rate": "m³/d",
        "perm": "mD",
        "viscosity": "mPa·s",
        "compressibility": "1/bar",
        "thickness": "m",
        "radius": "m",
        "fvf": "m³/m³",
        "half_length": "m",
        "conductivity": "mD-m"
    }
}


def convert_si_to_field(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert dictionary of reservoir and well parameters from SI to Field units.
    SI conventions:
        Pressure: bar -> psi (1 bar = 14.50377 psi)
        Rate: m^3/d -> STB/d (1 m^3 = 6.28981 STB)
        Thickness / length: m -> ft (1 m = 3.28084 ft)
        Compressibility: 1/bar -> 1/psi (divide by 14.50377)
        Viscosity: mPa.s -> cp (1:1)
        Permeability: mD (1:1)
        Porosity: fraction (1:1)
        FVF: m^3/m^3 -> rb/STB (1:1 dimensionally)
    """
    field_params = dict(params)
    
    if "Pi" in field_params and field_params["Pi"] is not None:
        field_params["Pi"] = field_params["Pi"] * 14.50377
    if "q" in field_params and field_params["q"] is not None:
        field_params["q"] = field_params["q"] * 6.28981
    if "h" in field_params and field_params["h"] is not None:
        field_params["h"] = field_params["h"] * 3.28084
    if "rw" in field_params and field_params["rw"] is not None:
        field_params["rw"] = field_params["rw"] * 3.28084
    if "ct" in field_params and field_params["ct"] is not None:
        field_params["ct"] = field_params["ct"] / 14.50377
    if "xf" in field_params and field_params["xf"] is not None:
        field_params["xf"] = field_params["xf"] * 3.28084
        
    return field_params


def convert_field_to_si(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert dictionary of parameters from Field to SI units.
    """
    si_params = dict(params)
    
    if "Pi" in si_params and si_params["Pi"] is not None:
        si_params["Pi"] = si_params["Pi"] / 14.50377
    if "q" in si_params and si_params["q"] is not None:
        si_params["q"] = si_params["q"] / 6.28981
    if "h" in si_params and si_params["h"] is not None:
        si_params["h"] = si_params["h"] / 3.28084
    if "rw" in si_params and si_params["rw"] is not None:
        si_params["rw"] = si_params["rw"] / 3.28084
    if "ct" in si_params and si_params["ct"] is not None:
        si_params["ct"] = si_params["ct"] * 14.50377
    if "xf" in si_params and si_params["xf"] is not None:
        si_params["xf"] = si_params["xf"] / 3.28084
        
    return si_params
