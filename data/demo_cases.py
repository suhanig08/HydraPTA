"""
Predefined Educational Demo Cases for HydraPTA.
All cases utilize the unified petroleum-engineering analytical model.
"""

from typing import Dict, Any, List
from src.models import ReservoirProperties, WellProperties, FractureProperties

DEMO_CASES: Dict[str, Dict[str, Any]] = {
    "case_1": {
        "id": "case_1",
        "case_number": "CASE 01",
        "name": "High-Conductivity Fracture",
        "short_description": "Explore the pressure response and diagnostic signature of a highly conductive hydraulic fracture.",
        "demonstrated_parameter": "High Fracture Permeability (kf = 10,000 mD, CfD = 0.08 / high relative conduit)",
        "educational_objective": (
            "Understand the transient signature of a highly conductive fracture, "
            "the emergence of formation linear flow (half-slope), and the transition toward late-time pseudoradial behavior."
        ),
        "reservoir": ReservoirProperties(
            k=5.0,           # mD
            phi=0.12,        # fraction
            mu=1.2,          # cp
            ct=15e-6,        # 1/psi
            h=50.0,          # ft
            B=1.2            # rb/STB
        ),
        "well": WellProperties(
            q=500.0,         # STB/day
            Pi=4000.0,       # psi
            rw=0.25,         # ft
            C=0.01,          # bbl/psi
            duration=300.0   # hr
        ),
        "fracture": FractureProperties(
            xf=500.0,        # ft
            kf=10000.0,      # mD
            wf=0.02          # ft
        )
    },
    "case_2": {
        "id": "case_2",
        "case_number": "CASE 02",
        "name": "Finite-Conductivity Fracture",
        "short_description": "Demonstrate how reduced fracture conductivity creates internal choke resistance and modifies transient flow.",
        "demonstrated_parameter": "Reduced Fracture Permeability (kf = 1,000 mD, wf = 0.01 ft, Cf = 10 mD-ft)",
        "educational_objective": (
            "Understand how finite fracture conductivity restricts fluid flow inside the fracture, "
            "inducing a characteristic bilinear flow regime (quarter-slope) before matrix linear flow."
        ),
        "reservoir": ReservoirProperties(
            k=5.0,           # mD
            phi=0.12,        # fraction
            mu=1.2,          # cp
            ct=15e-6,        # 1/psi
            h=50.0,          # ft
            B=1.2            # rb/STB
        ),
        "well": WellProperties(
            q=500.0,         # STB/day
            Pi=4000.0,       # psi
            rw=0.25,         # ft
            C=0.01,          # bbl/psi
            duration=300.0   # hr
        ),
        "fracture": FractureProperties(
            xf=500.0,        # ft
            kf=1000.0,       # mD
            wf=0.01          # ft
        )
    },
    "case_3": {
        "id": "case_3",
        "case_number": "CASE 03",
        "name": "Long-Fracture Case",
        "short_description": "Demonstrate the profound effect of deeply penetrating fracture half-length on transient duration.",
        "demonstrated_parameter": "Extended Half-Length (xf = 1,000 ft, doubled fracture reach)",
        "educational_objective": (
            "Understand how increasing fracture half-length extends the duration of fracture-dominated "
            "linear flow and shifts the transition to pseudoradial flow to much later times."
        ),
        "reservoir": ReservoirProperties(
            k=5.0,           # mD
            phi=0.12,        # fraction
            mu=1.2,          # cp
            ct=15e-6,        # 1/psi
            h=50.0,          # ft
            B=1.2            # rb/STB
        ),
        "well": WellProperties(
            q=500.0,         # STB/day
            Pi=4000.0,       # psi
            rw=0.25,         # ft
            C=0.01,          # bbl/psi
            duration=600.0   # hr
        ),
        "fracture": FractureProperties(
            xf=1000.0,       # ft
            kf=10000.0,      # mD
            wf=0.02          # ft
        )
    }
}


def get_demo_case(case_id: str) -> Dict[str, Any]:
    """Retrieve demo case definition by ID."""
    if case_id not in DEMO_CASES:
        raise KeyError(f"Unknown demo case ID '{case_id}'. Available: {list(DEMO_CASES.keys())}")
    return DEMO_CASES[case_id]


def list_demo_cases() -> List[Dict[str, Any]]:
    """Return list of all demo case dictionaries."""
    return list(DEMO_CASES.values())
