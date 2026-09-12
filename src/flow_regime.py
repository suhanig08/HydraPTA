"""
Flow regime detection, segmentation, and diagnostic classification engine.
Identifies sustained transient flow regimes based on local log-log derivative slopes.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from .models import FlowRegimeRecord


REGIME_DEFINITIONS = [
    {
        "name": "Wellbore Storage",
        "expected_slope": 1.0,
        "slope_min": 0.85,
        "slope_max": 1.15,
        "min_points": 4,
        "interpretation": (
            "Wellbore storage effect dominates early transient response as fluid "
            "decompresses/accumulates inside the wellbore, producing a unit-slope (m=1.0) signature."
        )
    },
    {
        "name": "Bilinear Flow",
        "expected_slope": 0.25,
        "slope_min": 0.18,
        "slope_max": 0.32,
        "min_points": 5,
        "interpretation": (
            "Simultaneous linear flow along the finite-conductivity fracture and perpendicular "
            "linear flow from matrix into the fracture, yielding a characteristic quarter-slope (m=0.25)."
        )
    },
    {
        "name": "Linear Flow",
        "expected_slope": 0.50,
        "slope_min": 0.42,
        "slope_max": 0.58,
        "min_points": 5,
        "interpretation": (
            "Formation linear flow perpendicular to fracture faces into a highly conductive fracture, "
            "producing a half-slope (m=0.50) signature."
        )
    },
    {
        "name": "Pseudoradial Flow",
        "expected_slope": 0.0,
        "slope_min": -0.10,
        "slope_max": 0.10,
        "min_points": 5,
        "interpretation": (
            "Late-time radial flow where the pressure disturbance has propagated far beyond the fracture tips, "
            "yielding a horizontal derivative (m=0.0) reflecting reservoir permeability-thickness (k*h)."
        )
    }
]


def detect_sustained_flow_regimes(
    times: np.ndarray,
    delta_p: np.ndarray,
    derivative: np.ndarray,
    slopes: np.ndarray
) -> List[FlowRegimeRecord]:
    """
    Identify and segment sustained flow regimes from observed derivative slopes.
    
    Parameters:
        times: Array of test times in hours
        delta_p: Array of pressure drop values
        derivative: Array of Bourdet pressure derivatives
        slopes: Array of local log-log derivative slopes m = d(log P') / d(log t)
    Returns:
        records: List of detected FlowRegimeRecord objects ordered chronologically
    """
    n = len(times)
    detected_regimes: List[FlowRegimeRecord] = []
    
    # We examine candidate contiguous index intervals matching each regime's criteria
    for regime_def in REGIME_DEFINITIONS:
        name = regime_def["name"]
        exp_s = regime_def["expected_slope"]
        s_min = regime_def["slope_min"]
        s_max = regime_def["slope_max"]
        min_pts = regime_def["min_points"]
        interp = regime_def["interpretation"]
        
        # Boolean mask of matching points
        mask = (slopes >= s_min) & (slopes <= s_max)
        
        # Additional physical filter for Pseudoradial Flow:
        # Must be in the later half of the test (derivative flat at late time, not an inflection peak)
        if name == "Pseudoradial Flow":
            mask = mask & (times > times[len(times) // 3])
            
        # Find contiguous segments of True in mask
        segments = []
        start_idx = None
        
        for i in range(n):
            if mask[i]:
                if start_idx is None:
                    start_idx = i
            else:
                if start_idx is not None:
                    if (i - start_idx) >= min_pts:
                        segments.append((start_idx, i - 1))
                    start_idx = None
                    
        if start_idx is not None and (n - start_idx) >= min_pts:
            segments.append((start_idx, n - 1))
            
        # For each valid sustained segment, record the regime
        for seg_start, seg_end in segments:
            seg_slopes = slopes[seg_start:seg_end + 1]
            mean_observed_slope = float(np.mean(seg_slopes))
            
            rec = FlowRegimeRecord(
                regime=name,
                t_start=float(times[seg_start]),
                t_end=float(times[seg_end]),
                observed_slope=round(mean_observed_slope, 2),
                expected_slope=exp_s,
                interpretation=interp,
                confidence="High" if (seg_end - seg_start) >= (min_pts * 2) else "Moderate"
            )
            detected_regimes.append(rec)
            
    # Sort chronologically by t_start
    detected_regimes.sort(key=lambda r: r.t_start)
    
    # Merge duplicate adjacent entries of the same regime if overlapping
    merged_regimes: List[FlowRegimeRecord] = []
    for r in detected_regimes:
        if not merged_regimes:
            merged_regimes.append(r)
        else:
            prev = merged_regimes[-1]
            if prev.regime == r.regime and r.t_start <= prev.t_end * 1.5:
                # Merge
                merged_regimes[-1] = FlowRegimeRecord(
                    regime=prev.regime,
                    t_start=prev.t_start,
                    t_end=max(prev.t_end, r.t_end),
                    observed_slope=round((prev.observed_slope + r.observed_slope) / 2.0, 2),
                    expected_slope=prev.expected_slope,
                    interpretation=prev.interpretation,
                    confidence=prev.confidence
                )
            else:
                merged_regimes.append(r)
                
    return merged_regimes


def generate_flow_regime_summary_table(regimes: List[FlowRegimeRecord]) -> pd.DataFrame:
    """Format detected flow regimes into a clean presentation DataFrame."""
    if not regimes:
        return pd.DataFrame(columns=[
            "Flow Regime", "Time Range (hr)", "Observed Slope", "Expected Slope", "Interpretation"
        ])
        
    data = []
    for r in regimes:
        t_range_str = f"{r.t_start:.4g} – {r.t_end:.4g} hr"
        obs_slope_str = f"{r.observed_slope:+.2f}"
        exp_slope_str = f"{r.expected_slope:.2f}"
        
        data.append({
            "Flow Regime": r.regime,
            "Time Range (hr)": t_range_str,
            "Observed Slope": obs_slope_str,
            "Expected Slope": exp_slope_str,
            "Interpretation": r.interpretation
        })
        
    return pd.DataFrame(data)
