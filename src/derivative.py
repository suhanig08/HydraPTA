"""
Numerically robust Bourdet logarithmic pressure derivative calculation and slope diagnostics.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd


def calculate_bourdet_derivative(
    times: np.ndarray,
    delta_p: np.ndarray,
    L: float = 0.15
) -> np.ndarray:
    """
    Calculate the Bourdet logarithmic pressure derivative d(Delta P) / d(ln t) = t * d(Delta P) / dt.
    
    Parameters:
        times: Array of test time values (hours), strictly positive and monotonic.
        delta_p: Array of pressure drop values Delta P (psi).
        L: Bourdet differentiation distance (in ln(time) units, default 0.15).
    Returns:
        derivative: Array of Bourdet logarithmic pressure derivatives (psi).
    """
    n = len(times)
    if n < 3:
        raise ValueError("At least 3 points are required to compute pressure derivative.")
        
    log_t = np.log(times)
    derivative = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        # Find left point L where log_t[i] - log_t[left] >= L
        left = i - 1
        while left > 0 and (log_t[i] - log_t[left]) < L:
            left -= 1
            
        # Find right point R where log_t[right] - log_t[i] >= L
        right = i + 1
        while right < (n - 1) and (log_t[right] - log_t[i]) < L:
            right += 1
            
        # Handle left boundary
        if left < 0 or left == i:
            if right < n and right != i:
                dt_r = log_t[right] - log_t[i]
                dp_r = delta_p[right] - delta_p[i]
                derivative[i] = dp_r / max(dt_r, 1e-12)
            else:
                derivative[i] = 0.0
            continue
            
        # Handle right boundary
        if right >= n or right == i:
            dt_l = log_t[i] - log_t[left]
            dp_l = delta_p[i] - delta_p[left]
            derivative[i] = dp_l / max(dt_l, 1e-12)
            continue
            
        # General interior Bourdet formula
        dt_l = log_t[i] - log_t[left]
        dp_l = delta_p[i] - delta_p[left]
        d_left = dp_l / max(dt_l, 1e-12)
        
        dt_r = log_t[right] - log_t[i]
        dp_r = delta_p[right] - delta_p[i]
        d_right = dp_r / max(dt_r, 1e-12)
        
        dt_total = log_t[right] - log_t[left]
        if dt_total > 0:
            derivative[i] = (d_left * dt_r + d_right * dt_l) / dt_total
        else:
            derivative[i] = (d_left + d_right) / 2.0
            
    # Clean any unphysical negative derivatives at very early numerical onset
    derivative = np.maximum(derivative, 0.0)
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
    n = len(times)
    slopes = np.zeros(n, dtype=np.float64)
    
    # Avoid log of zero or negative
    safe_values = np.maximum(values, 1e-8)
    log_t = np.log(times)
    log_v = np.log(safe_values)
    
    for i in range(n):
        left = i - 1
        while left > 0 and (log_t[i] - log_t[left]) < L:
            left -= 1
            
        right = i + 1
        while right < (n - 1) and (log_t[right] - log_t[i]) < L:
            right += 1
            
        if left < 0 or left == i:
            if right < n and right != i:
                slopes[i] = (log_v[right] - log_v[i]) / max(log_t[right] - log_t[i], 1e-12)
            else:
                slopes[i] = 0.0
            continue
            
        if right >= n or right == i:
            slopes[i] = (log_v[i] - log_v[left]) / max(log_t[i] - log_t[left], 1e-12)
            continue
            
        dt_l = log_t[i] - log_t[left]
        dv_l = log_v[i] - log_v[left]
        s_left = dv_l / max(dt_l, 1e-12)
        
        dt_r = log_t[right] - log_t[i]
        dv_r = log_v[right] - log_v[i]
        s_right = dv_r / max(dt_r, 1e-12)
        
        dt_total = log_t[right] - log_t[left]
        if dt_total > 0:
            slopes[i] = (s_left * dt_r + s_right * dt_l) / dt_total
        else:
            slopes[i] = (s_left + s_right) / 2.0
            
    return slopes
