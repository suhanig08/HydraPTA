"""
Data models and dataclasses for HydraPTA.
Defines structured containers for reservoir, well, fracture, simulation, and diagnostic data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


@dataclass
class ReservoirProperties:
    """Reservoir petrophysical and fluid properties (Petroleum Engineering Units)."""
    k: float        # Permeability (mD)
    phi: float      # Porosity (fraction, 0-1)
    mu: float       # Dynamic Viscosity (cp)
    ct: float       # Total Compressibility (1/psi)
    h: float        # Net Pay Thickness (ft)
    B: float        # Formation Volume Factor (rb/STB)

    def validate(self) -> None:
        """Validate physical range of reservoir properties."""
        if self.k <= 0:
            raise ValueError(f"Permeability k must be positive, got {self.k} mD")
        if not (0 < self.phi <= 1.0):
            raise ValueError(f"Porosity phi must be in (0, 1], got {self.phi}")
        if self.mu <= 0:
            raise ValueError(f"Viscosity mu must be positive, got {self.mu} cp")
        if self.ct <= 0:
            raise ValueError(f"Total compressibility ct must be positive, got {self.ct} psi^-1")
        if self.h <= 0:
            raise ValueError(f"Thickness h must be positive, got {self.h} ft")
        if self.B <= 0:
            raise ValueError(f"Formation volume factor B must be positive, got {self.B} rb/STB")


@dataclass
class WellProperties:
    """Wellbore and test operational parameters (Petroleum Engineering Units)."""
    q: float        # Flow rate (STB/day)
    Pi: float       # Initial reservoir pressure (psi)
    rw: float       # Wellbore radius (ft)
    C: float        # Wellbore storage coefficient (bbl/psi)
    duration: float # Total test duration (hours)

    def validate(self) -> None:
        """Validate physical range of well test parameters."""
        if self.q <= 0:
            raise ValueError(f"Flow rate q must be positive for drawdown, got {self.q} STB/day")
        if self.Pi <= 0:
            raise ValueError(f"Initial pressure Pi must be positive, got {self.Pi} psi")
        if self.rw <= 0:
            raise ValueError(f"Wellbore radius rw must be positive, got {self.rw} ft")
        if self.C < 0:
            raise ValueError(f"Wellbore storage coefficient C cannot be negative, got {self.C} bbl/psi")
        if self.duration <= 0:
            raise ValueError(f"Test duration must be positive, got {self.duration} hr")


@dataclass
class FractureProperties:
    """Hydraulic fracture geometric and flow properties (Petroleum Engineering Units)."""
    xf: float       # Fracture half-length (ft)
    kf: float       # Fracture permeability (mD)
    wf: float       # Fracture width (ft)

    def validate(self) -> None:
        """Validate physical range of fracture parameters."""
        if self.xf <= 0:
            raise ValueError(f"Fracture half-length xf must be positive, got {self.xf} ft")
        if self.kf <= 0:
            raise ValueError(f"Fracture permeability kf must be positive, got {self.kf} mD")
        if self.wf <= 0:
            raise ValueError(f"Fracture width wf must be positive, got {self.wf} ft")

    @property
    def Cf(self) -> float:
        """Fracture conductivity Cf = kf * wf (mD-ft)."""
        return self.kf * self.wf

    def CfD(self, k: float) -> float:
        """Dimensionless fracture conductivity CfD = (kf * wf) / (k * xf)."""
        if k <= 0 or self.xf <= 0:
            return 0.0
        return (self.kf * self.wf) / (k * self.xf)


@dataclass
class FlowRegimeRecord:
    """Record of a detected flow regime from pressure derivative slope analysis."""
    regime: str             # Regime name (e.g., 'Wellbore Storage', 'Linear Flow')
    t_start: float          # Start time (hr)
    t_end: float            # End time (hr)
    observed_slope: float   # Mean observed log-log derivative slope
    expected_slope: float   # Theoretical slope (1.0, 0.5, 0.25, 0.0)
    interpretation: str     # Physical description of flow mechanism
    confidence: str = "High"


@dataclass
class ParameterValidation:
    """True vs Estimated parameter validation record for synthetic data."""
    parameter: str
    unit: str
    true_value: float
    estimated_value: Optional[float]
    absolute_error: Optional[float]
    relative_error: Optional[float]
    is_reliable: bool
    status_message: str


@dataclass
class PTAAnalysisResult:
    """Complete PTA analysis output container."""
    case_name: str
    reservoir: ReservoirProperties
    well: WellProperties
    fracture: FractureProperties
    df_transient: pd.DataFrame
    flow_regimes: List[FlowRegimeRecord]
    validations: List[ParameterValidation]
    engineering_interpretation: str
    dominant_regime: str
    Cf: float
    CfD: float
