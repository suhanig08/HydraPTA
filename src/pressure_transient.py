"""
Hydraulic fracture pressure-transient forward analytical modeling engine.
Implements Laplace-domain reservoir-fracture solutions and the Gaver-Stehfest numerical inversion.
"""

import math
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.special import k0, kv
from scipy.integrate import quad

from .models import ReservoirProperties, WellProperties, FractureProperties
from .reservoir import calculate_tD_xf, calculate_CD_xf, calculate_pressure_factor


def stehfest_weights(N: int = 10) -> np.ndarray:
    """
    Calculate Gaver-Stehfest inversion coefficients for an even integer N.
    
    Parameters:
        N: Stehfest algorithm parameter (typically 8, 10, or 12).
    Returns:
        V: Array of N coefficients (1-indexed conceptually, 0-indexed in NumPy).
    """
    if N % 2 != 0:
        raise ValueError(f"Stehfest parameter N must be even, got {N}")
    
    half_N = N // 2
    V = np.zeros(N, dtype=np.float64)
    
    for i in range(1, N + 1):
        k_min = (i + 1) // 2
        k_max = min(i, half_N)
        sum_k = 0.0
        
        for k in range(k_min, k_max + 1):
            num = (k ** half_N) * math.factorial(2 * k)
            den = (
                math.factorial(half_N - k)
                * math.factorial(k)
                * math.factorial(k - 1)
                * math.factorial(i - k)
                * math.factorial(2 * k - i)
            )
            sum_k += num / den
            
        sign = (-1) ** (half_N + i)
        V[i - 1] = sign * sum_k
        
    return V


def _infinite_conductivity_integral(s: float, xD: float = 0.732) -> float:
    """
    Compute the infinite-conductivity vertical fracture Laplace solution:
    pD_bar_inf(s) = 1 / (2 * s * sqrt(s)) * [ int_0^{sqrt(s)*(1+xD)} K0(u) du + int_0^{sqrt(s)*(1-xD)} K0(u) du ]
    Uses change of variables u = v^2 to eliminate the logarithmic singularity of K0 at 0.
    """
    sqrt_s = math.sqrt(s)
    z1 = sqrt_s * (1.0 + xD)
    z2 = sqrt_s * (1.0 - xD)
    
    def int_k0_eval(z: float) -> float:
        if z <= 0.0:
            return 0.0
        # For very large z, integral approaches pi/2
        if z > 30.0:
            return math.pi / 2.0
        val, _ = quad(lambda v: 2.0 * v * k0(v ** 2), 0.0, math.sqrt(z), limit=50, epsabs=1e-10, epsrel=1e-8)
        return val
    
    val1 = int_k0_eval(z1)
    val2 = int_k0_eval(z2)
    
    return (val1 + val2) / (2.0 * s * sqrt_s)


def laplace_fractured_well_pressure(
    s: float,
    CfD: float,
    CD_xf: float,
    Sf: float = 0.0,
    eta_fD: float = 1e5
) -> float:
    """
    Evaluate dimensionless wellbore pressure in Laplace domain for a fractured well
    with finite/infinite conductivity and wellbore storage.
    
    Parameters:
        s: Laplace parameter (corresponding to tD_xf)
        CfD: Dimensionless fracture conductivity
        CD_xf: Dimensionless wellbore storage coefficient based on xf
        Sf: Fracture face skin factor (default 0.0)
        eta_fD: Dimensionless fracture storativity ratio (default 1e5)
    Returns:
        p_wD_bar: Dimensionless pressure in Laplace space
    """
    if s <= 0:
        return 0.0
    
    sqrt_s = math.sqrt(s)
    
    # 1. Base infinite-conductivity reservoir response
    pD_inf = _infinite_conductivity_integral(s, xD=0.732)
    
    # 2. Finite conductivity choke / internal impedance
    # Beta eigenvalue in fracture
    beta_sq = (s / eta_fD) + (2.0 * sqrt_s / max(CfD, 1e-6))
    beta = math.sqrt(max(beta_sq, 1e-12))
    
    # Stable evaluation of (1/tanh(beta) - 1/beta):
    if beta < 1e-3:
        # Taylor expansion for small beta to eliminate numerical cancellation
        langevin_term = (beta / 3.0) - ((beta ** 3) / 45.0) + ((2.0 * (beta ** 5)) / 945.0)
    elif beta > 25.0:
        langevin_term = 1.0 - (1.0 / beta)
    else:
        langevin_term = (1.0 / math.tanh(beta)) - (1.0 / beta)
        
    delta_pD_finite = (math.pi / (CfD * s * beta)) * langevin_term
    delta_pD_finite = max(delta_pD_finite, 0.0)
    
    # Total fracture dimensionless pressure (without WBS)
    pD_total = pD_inf + delta_pD_finite + (Sf / s)
    
    # 3. Include wellbore storage CD_xf
    if CD_xf > 0.0:
        p_wD_bar = pD_total / (1.0 + CD_xf * (s ** 2) * pD_total)
    else:
        p_wD_bar = pD_total
        
    return p_wD_bar


def invert_laplace_to_time(
    tD_xf: float,
    CfD: float,
    CD_xf: float,
    Sf: float = 0.0,
    N: int = 10,
    V_weights: Optional[np.ndarray] = None
) -> float:
    """
    Perform Stehfest numerical inversion from Laplace domain to time domain for a single tD_xf.
    
    Parameters:
        tD_xf: Dimensionless time
        CfD: Dimensionless fracture conductivity
        CD_xf: Dimensionless wellbore storage coefficient
        Sf: Fracture skin factor
        N: Stehfest parameter (even integer, default 10)
        V_weights: Precalculated Stehfest weights (optional for performance)
    Returns:
        pD: Dimensionless pressure at tD_xf
    """
    if tD_xf <= 0:
        return 0.0
    
    if V_weights is None or len(V_weights) != N:
        V_weights = stehfest_weights(N)
        
    ln2_over_t = math.log(2.0) / tD_xf
    pD = 0.0
    
    for i in range(1, N + 1):
        s_i = i * ln2_over_t
        p_bar = laplace_fractured_well_pressure(s_i, CfD, CD_xf, Sf)
        pD += V_weights[i - 1] * p_bar
        
    pD = pD * ln2_over_t
    return max(pD, 0.0)


def generate_synthetic_transient_response(
    reservoir: ReservoirProperties,
    well: WellProperties,
    fracture: FractureProperties,
    n_points: int = 160,
    t_min: float = 1e-4,
    noise_std_psi: float = 0.0,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic pressure-transient response across early, intermediate, and late times.
    
    Parameters:
        reservoir: ReservoirProperties
        well: WellProperties
        fracture: FractureProperties
        n_points: Number of logarithmically spaced time samples
        t_min: Earliest transient time (hours, default 1e-4)
        noise_std_psi: Standard deviation of optional Gaussian noise in psi (default 0.0 for clean demo)
        random_seed: Seed for reproducible noise
    Returns:
        df: DataFrame with columns ['time_hr', 'tD_xf', 'pD', 'delta_P_psi', 'Pwf_psi']
    """
    reservoir.validate()
    well.validate()
    fracture.validate()
    
    t_max = well.duration
    times = np.logspace(np.log10(t_min), np.log10(t_max), n_points)
    
    CfD = fracture.CfD(reservoir.k)
    CD_xf = calculate_CD_xf(well.C, reservoir.phi, reservoir.ct, reservoir.h, fracture.xf)
    p_factor = calculate_pressure_factor(well.q, reservoir.B, reservoir.mu, reservoir.k, reservoir.h)
    
    V_weights = stehfest_weights(10)
    
    pD_list = []
    tD_list = []
    
    for t_hr in times:
        tD = calculate_tD_xf(t_hr, reservoir.k, reservoir.phi, reservoir.mu, reservoir.ct, fracture.xf)
        tD_list.append(tD)
        pD_val = invert_laplace_to_time(tD, CfD, CD_xf, Sf=0.0, N=10, V_weights=V_weights)
        pD_list.append(pD_val)
        
    pD_arr = np.array(pD_list)
    tD_arr = np.array(tD_list)
    delta_p_arr = p_factor * pD_arr
    
    # Add optional noise if requested
    if noise_std_psi > 0.0:
        rng = np.random.default_rng(random_seed)
        noise = rng.normal(0, noise_std_psi, size=len(delta_p_arr))
        delta_p_arr = np.maximum(delta_p_arr + noise, 0.0)
        
    pwf_arr = well.Pi - delta_p_arr
    
    df = pd.DataFrame({
        "time_hr": times,
        "tD_xf": tD_arr,
        "pD": pD_arr,
        "delta_P_psi": delta_p_arr,
        "Pwf_psi": pwf_arr
    })
    
    return df
