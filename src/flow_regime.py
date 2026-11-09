"""
Flow regime detection, segmentation, stabilization recognition, and diagnostic classification engine.
Identifies sustained transient flow regimes based on local log-log derivative slopes,
detects derivative stabilization plateaus, evaluates regime sequences, and reports confidence metrics.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from .models import FlowRegimeRecord


DEFAULT_REGIME_DEFINITIONS = [
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
    slopes: np.ndarray,
    slope_tolerances: Optional[Dict[str, float]] = None,
    min_points_override: Optional[int] = None
) -> List[FlowRegimeRecord]:
    """
    Identify and segment sustained flow regimes from observed derivative slopes.
    
    Parameters:
        times: Array of test times in hours
        delta_p: Array of pressure drop values
        derivative: Array of Bourdet pressure derivatives
        slopes: Array of local log-log derivative slopes m = d(log P') / d(log t)
        slope_tolerances: Optional dict overriding default half-width tolerances for regimes.
                          e.g. {"Wellbore Storage": 0.15, "Linear Flow": 0.08, "Pseudoradial Flow": 0.10}
        min_points_override: Optional override for minimum consecutive points required.
    Returns:
        records: List of detected FlowRegimeRecord objects ordered chronologically
    """
    times = np.asarray(times, dtype=np.float64)
    delta_p = np.asarray(delta_p, dtype=np.float64)
    derivative = np.asarray(derivative, dtype=np.float64)
    slopes = np.asarray(slopes, dtype=np.float64)
    n = len(times)
    
    if n < 4:
        return []
        
    detected_regimes: List[FlowRegimeRecord] = []
    
    # We examine candidate contiguous index intervals matching each regime's criteria
    for regime_def in DEFAULT_REGIME_DEFINITIONS:
        name = regime_def["name"]
        exp_s = regime_def["expected_slope"]
        min_pts = min_points_override or regime_def["min_points"]
        interp = regime_def["interpretation"]
        
        # Check custom tolerances if provided
        if slope_tolerances and name in slope_tolerances:
            tol = slope_tolerances[name]
            s_min = exp_s - tol
            s_max = exp_s + tol
        else:
            s_min = regime_def["slope_min"]
            s_max = regime_def["slope_max"]
        
        # Boolean mask of matching points
        mask = (slopes >= s_min) & (slopes <= s_max)
        
        # Additional physical filter for Pseudoradial Flow:
        # Must be in the later half of the test (derivative flat at late time, not an inflection peak)
        if name == "Pseudoradial Flow":
            mask = mask & (times > times[len(times) // 3])
            
        # Additional physical filter for Wellbore Storage:
        # Must be in the early portion of the test
        if name == "Wellbore Storage":
            mask = mask & (times <= times[len(times) // 2])
            
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
            seg_times = times[seg_start:seg_end + 1]
            mean_observed_slope = float(np.mean(seg_slopes))
            
            # Confidence criteria based on points and log-cycle span
            log_span = np.log10(seg_times[-1]) - np.log10(max(seg_times[0], 1e-6))
            num_pts = seg_end - seg_start + 1
            if num_pts >= (min_pts * 2) or log_span >= 0.5:
                conf = "High"
            elif num_pts >= min_pts:
                conf = "Moderate"
            else:
                conf = "Low"
            
            rec = FlowRegimeRecord(
                regime=name,
                t_start=float(times[seg_start]),
                t_end=float(times[seg_end]),
                observed_slope=round(mean_observed_slope, 2),
                expected_slope=exp_s,
                interpretation=interp,
                confidence=conf
            )
            detected_regimes.append(rec)
            
    # Sort chronologically by t_start
    detected_regimes.sort(key=lambda r: r.t_start)
    
    # Merge duplicate adjacent entries of the same regime if overlapping or proximate
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


def detect_stabilization_plateau(
    times: np.ndarray,
    derivative: np.ndarray,
    slopes: np.ndarray,
    slope_tolerance: float = 0.10,
    max_relative_std: float = 0.12,
    min_points: int = 5
) -> Dict[str, Any]:
    """
    Dedicated stabilization detection module.
    Identifies the established pseudoradial-flow period where the Bourdet derivative
    stabilizes into an approximately constant horizontal plateau (Delta P' ~ constant).
    
    Distinguishes true derivative plateau from temporary noise fluctuations or inflection points.
    
    Returns:
        plateau_info: Dict with keys:
            - is_detected: bool
            - t_start: float (hr)
            - t_end: float (hr)
            - derv_mean: float (psi)
            - derv_std: float (psi)
            - rel_std: float (std / mean)
            - num_points: int
            - log_cycles: float
            - confidence: str ('High', 'Moderate', 'Low', 'None')
            - notes: str
    """
    n = len(times)
    info: Dict[str, Any] = {
        "is_detected": False,
        "t_start": None,
        "t_end": None,
        "derv_mean": None,
        "derv_std": None,
        "rel_std": None,
        "num_points": 0,
        "log_cycles": 0.0,
        "confidence": "None",
        "notes": "No stable derivative plateau identified."
    }
    
    if n < min_points:
        info["notes"] = "Insufficient data points for stabilization analysis."
        return info
        
    # Search primarily in the intermediate-to-late region (t > t[n//4])
    start_search_idx = n // 4
    mask = (np.abs(slopes) <= slope_tolerance) & (derivative > 0)
    mask[:start_search_idx] = False
    
    # Find longest continuous candidate segment
    best_start = None
    best_end = None
    curr_start = None
    max_len = 0
    
    for i in range(start_search_idx, n):
        if mask[i]:
            if curr_start is None:
                curr_start = i
        else:
            if curr_start is not None:
                seg_len = i - curr_start
                if seg_len > max_len:
                    max_len = seg_len
                    best_start = curr_start
                    best_end = i - 1
                curr_start = None
                
    if curr_start is not None:
        seg_len = n - curr_start
        if seg_len > max_len:
            max_len = seg_len
            best_start = curr_start
            best_end = n - 1
            
    if best_start is None or max_len < min_points:
        info["notes"] = "Derivative slope did not sustain within tolerance (|slope| <= " + f"{slope_tolerance:.2f}) for at least {min_points} points."
        return info
        
    sub_times = times[best_start:best_end + 1]
    sub_derv = derivative[best_start:best_end + 1]
    
    d_mean = float(np.mean(sub_derv))
    d_std = float(np.std(sub_derv))
    rel_std = d_std / max(d_mean, 1e-9)
    log_cycles = float(np.log10(sub_times[-1]) - np.log10(max(sub_times[0], 1e-6)))
    
    # Assess variability threshold to avoid noisy temporary flat regions
    if rel_std > max_relative_std:
        info["is_detected"] = False
        info["t_start"] = float(sub_times[0])
        info["t_end"] = float(sub_times[-1])
        info["derv_mean"] = round(d_mean, 2)
        info["derv_std"] = round(d_std, 2)
        info["rel_std"] = round(rel_std * 100, 2)
        info["confidence"] = "Low"
        info["notes"] = (
            f"Candidate flat region detected ({sub_times[0]:.2g} – {sub_times[-1]:.2g} hr), "
            f"but derivative variability is high (std/mean = {rel_std*100:.1f}% > {max_relative_std*100:.1f}%), "
            f"indicating probable numerical noise rather than true pseudoradial stabilization."
        )
        return info
        
    # High confidence if spanning >= 0.5 log cycles and low variation
    if log_cycles >= 0.5 and rel_std <= 0.06:
        conf = "High"
    elif log_cycles >= 0.25 and rel_std <= 0.10:
        conf = "Moderate"
    else:
        conf = "Low"
        
    info["is_detected"] = True
    info["t_start"] = float(sub_times[0])
    info["t_end"] = float(sub_times[-1])
    info["derv_mean"] = round(d_mean, 3)
    info["derv_std"] = round(d_std, 3)
    info["rel_std"] = round(rel_std * 100, 2)
    info["num_points"] = max_len
    info["log_cycles"] = round(log_cycles, 2)
    info["confidence"] = conf
    info["notes"] = (
        f"Stable horizontal Bourdet derivative plateau confirmed between {sub_times[0]:.3g} hr and {sub_times[-1]:.3g} hr "
        f"({max_len} observations over {log_cycles:.2f} log cycles, mean ΔP' = {d_mean:.2f} psi, rel std = {rel_std*100:.1f}%)."
    )
    return info


def identify_flow_transitions(
    times: np.ndarray,
    regimes: List[FlowRegimeRecord]
) -> List[Dict[str, Any]]:
    """
    Identify intermediate transition intervals between distinct detected flow regimes.
    Reports transitions rather than misclassifying them as specific flow regimes.
    """
    if len(regimes) < 2:
        return []
        
    transitions = []
    for i in range(len(regimes) - 1):
        r_prev = regimes[i]
        r_next = regimes[i + 1]
        
        # Check gap between end of previous and start of next
        if r_next.t_start > r_prev.t_end * 1.05:
            transitions.append({
                "type": "Transition Period",
                "from_regime": r_prev.regime,
                "to_regime": r_next.regime,
                "t_start": r_prev.t_end,
                "t_end": r_next.t_start,
                "description": f"Transition from {r_prev.regime} toward {r_next.regime}."
            })
            
    return transitions


def analyze_regime_sequence(regimes: List[FlowRegimeRecord]) -> str:
    """
    Investigate the sequence of identified regimes and return geological/diagnostic interpretations.
    Distinguishes early-time fracture linear flow from late-time boundary linear flow.
    """
    if not regimes:
        return "No sustained flow regimes were identified from derivative slope diagnostics."
        
    regime_names = [r.regime for r in regimes]
    notes = []
    
    # Check linear flow placement
    if "Linear Flow" in regime_names:
        idx_lf = regime_names.index("Linear Flow")
        if idx_lf == 0 or (idx_lf == 1 and regime_names[0] == "Wellbore Storage"):
            notes.append(
                "Linear flow (half-slope) observed immediately following early-time storage. "
                "This sequence is highly characteristic of formation linear flow perpendicular to a hydraulic fracture."
            )
        elif "Pseudoradial Flow" in regime_names:
            idx_rad = regime_names.index("Pseudoradial Flow")
            if idx_lf > idx_rad:
                notes.append(
                    "Linear flow (half-slope) observed after established pseudoradial flow. "
                    "This late-time transition suggests a change in outer flow geometry, which may be associated with "
                    "parallel sealing boundaries, channel reservoir geometry, or compartmentalization. "
                    "(Note: Geological hypothesis based on transient signature; requires seismic and geological corroboration)."
                )
                
    if "Bilinear Flow" in regime_names and "Linear Flow" in regime_names:
        notes.append(
            "Both bilinear (quarter-slope) and linear (half-slope) flow regimes were identified in sequence, "
            "confirming finite-conductivity fracture transient diffusion transitioning into matrix linear flow."
        )
    elif "Bilinear Flow" in regime_names:
        notes.append(
            "Bilinear flow identified without subsequent formation linear flow, indicating a low-conductivity fracture choke."
        )
        
    if "Pseudoradial Flow" in regime_names:
        notes.append("Established pseudoradial flow detected, confirming that the radius of investigation has expanded beyond the near-wellbore fracture region.")
    else:
        notes.append("No pseudoradial flow plateau detected; test duration may be insufficient to reach outer radial flow.")
        
    return " ".join(notes)


def generate_flow_regime_summary_table(regimes: List[FlowRegimeRecord]) -> pd.DataFrame:
    """Format detected flow regimes into a clean presentation DataFrame."""
    if not regimes:
        return pd.DataFrame(columns=[
            "Flow Regime", "Time Range (hr)", "Observed Slope", "Expected Slope", "Confidence", "Interpretation"
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
            "Confidence": getattr(r, "confidence", "Moderate"),
            "Interpretation": r.interpretation
        })
        
    return pd.DataFrame(data)
