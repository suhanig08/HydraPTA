"""
Numerically robust Bourdet logarithmic pressure derivative calculation and slope diagnostics.
Implements the 3-point Bourdet algorithm with an adjustable L parameter, handles nonuniform sampling,
excludes t=0 from logarithmic differentiation, and supports multi-L smoothing comparisons.
"""

from typing import Tuple, Optional, Dict, List
import numpy as np
import pandas as pd


def calculate_bourdet_derivative(
    times: np.ndarray,
    delta_p: np.ndarray,
    L: float = 0.15
) -> np.ndarray:
    """
    Calculate the Bourdet logarithmic pressure derivative d(Delta P) / d(ln t) = t * d(Delta P) / dt.
    Uses the 3-point central logarithmic difference algorithm with search window L.
    
    Parameters:
        times: Array of test time values (hours). Points with t <= 0 are excluded from log differentiation.
        delta_p: Array of pressure drop values Delta P (psi).
        L: Bourdet differentiation distance (in ln(time) units, default 0.15).
    Returns:
        derivative: Array of Bourdet logarithmic pressure derivatives (psi), matching length of times.
    """
    times = np.asarray(times, dtype=np.float64)
    delta_p = np.asarray(delta_p, dtype=np.float64)
    n = len(times)
    
    if n < 3:
        raise ValueError("At least 3 points are required to compute pressure derivative.")
        
    derivative = np.zeros(n, dtype=np.float64)
    
    # Filter points where time > 0
    valid_mask = times > 0
    if np.sum(valid_mask) < 3:
        return derivative
        
    valid_indices = np.where(valid_mask)[0]
    sub_t = times[valid_indices]
    sub_dp = delta_p[valid_indices]
    sub_n = len(sub_t)
    
    log_t = np.log(sub_t)
    sub_derv = np.zeros(sub_n, dtype=np.float64)
    
    for i in range(sub_n):
        # Find left point L where log_t[i] - log_t[left] >= L
        left = i - 1
        while left > 0 and (log_t[i] - log_t[left]) < L:
            left -= 1
            
        # Find right point R where log_t[right] - log_t[i] >= L
        right = i + 1
        while right < (sub_n - 1) and (log_t[right] - log_t[i]) < L:
            right += 1
            
        # Handle left boundary
        if left < 0 or left == i:
            if right < sub_n and right != i:
                dt_r = log_t[right] - log_t[i]
                dp_r = sub_dp[right] - sub_dp[i]
                sub_derv[i] = dp_r / max(dt_r, 1e-12)
            else:
                sub_derv[i] = 0.0
            continue
            
        # Handle right boundary
        if right >= sub_n or right == i:
            dt_l = log_t[i] - log_t[left]
            dp_l = sub_dp[i] - sub_dp[left]
            sub_derv[i] = dp_l / max(dt_l, 1e-12)
            continue
            
        # General interior 3-point Bourdet formula
        dt_l = log_t[i] - log_t[left]
        dp_l = sub_dp[i] - sub_dp[left]
        d_left = dp_l / max(dt_l, 1e-12)
        
        dt_r = log_t[right] - log_t[i]
        dp_r = sub_dp[right] - sub_dp[i]
        d_right = dp_r / max(dt_r, 1e-12)
        
        dt_total = log_t[right] - log_t[left]
        if dt_total > 0:
            sub_derv[i] = (d_left * dt_r + d_right * dt_l) / dt_total
        else:
            sub_derv[i] = (d_left + d_right) / 2.0
            
    # Clean any unphysical negative derivatives at very early numerical onset
    sub_derv = np.maximum(sub_derv, 0.0)
    
    # Place into full array
    derivative[valid_indices] = sub_derv
    return derivative


def calculate_log_slopes(
    times: np.ndarray,
    values: np.ndarray,
    L: float = 0.20
) -> np.ndarray:
    """
    Calculate the local log-log slope m = d(log10 y) / d(log10 x) = d(ln y) / d(ln x).
    
    Parameters:
        times: Positive array of x values (time)
        values: Positive array of y values (e.g. pressure derivative Delta P')
        L: Differentiation distance in ln(time)
    Returns:
        slopes: Array of local log-log slopes
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    n = len(times)
    slopes = np.zeros(n, dtype=np.float64)
    
    # Filter valid positive values
    valid_mask = (times > 0) & (values > 0)
    if np.sum(valid_mask) < 3:
        return slopes
        
    valid_indices = np.where(valid_mask)[0]
    sub_t = times[valid_indices]
    sub_v = values[valid_indices]
    sub_n = len(sub_t)
    
    log_t = np.log(sub_t)
    log_v = np.log(np.maximum(sub_v, 1e-8))
    sub_slopes = np.zeros(sub_n, dtype=np.float64)
    
    for i in range(sub_n):
        left = i - 1
        while left > 0 and (log_t[i] - log_t[left]) < L:
            left -= 1
            
        right = i + 1
        while right < (sub_n - 1) and (log_t[right] - log_t[i]) < L:
            right += 1
            
        if left < 0 or left == i:
            if right < sub_n and right != i:
                sub_slopes[i] = (log_v[right] - log_v[i]) / max(log_t[right] - log_t[i], 1e-12)
            else:
                sub_slopes[i] = 0.0
            continue
            
        if right >= sub_n or right == i:
            sub_slopes[i] = (log_v[i] - log_v[left]) / max(log_t[i] - log_t[left], 1e-12)
            continue
            
        dt_l = log_t[i] - log_t[left]
        dv_l = log_v[i] - log_v[left]
        s_left = dv_l / max(dt_l, 1e-12)
        
        dt_r = log_t[right] - log_t[i]
        dv_r = log_v[right] - log_v[i]
        s_right = dv_r / max(dt_r, 1e-12)
        
        dt_total = log_t[right] - log_t[left]
        if dt_total > 0:
            sub_slopes[i] = (s_left * dt_r + s_right * dt_l) / dt_total
        else:
            sub_slopes[i] = (s_left + s_right) / 2.0
            
    slopes[valid_indices] = sub_slopes
    return slopes


def compare_bourdet_derivatives(
    times: np.ndarray,
    delta_p: np.ndarray,
    L_values: Optional[List[float]] = None
) -> Dict[float, np.ndarray]:
    """
    Compute Bourdet pressure derivatives at multiple differentiation distances L
    to investigate the effect of derivative smoothing.
    
    Parameters:
        times: Array of test time values (hours).
        delta_p: Array of pressure drop values (psi).
        L_values: List of L parameter values (defaults to [0.05, 0.15, 0.30]).
    Returns:
        derivatives_by_L: Dict mapping L float to corresponding derivative array.
    """
    if L_values is None:
        L_values = [0.05, 0.15, 0.30]
        
    results: Dict[float, np.ndarray] = {}
    for L_val in L_values:
        results[L_val] = calculate_bourdet_derivative(times, delta_p, L=L_val)
    return results
