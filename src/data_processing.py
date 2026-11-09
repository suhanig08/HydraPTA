"""
Data ingestion, automatic column recognition, validation, and drawdown preprocessing module.
Preserves original uploaded datasets while validating and standardizing test data for PTA.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import io
import re
import numpy as np
import pandas as pd


# Column name heuristics (lowercase patterns)
TIME_PATTERNS = [
    r"^t$", r"^time", r"^time_hr", r"^time\s*\(?hrs?\)?" , r"^time\s*\(?hours?\)?" ,
    r"^elapsed\s*time", r"^delta[_\s]*t", r"^dt$", r"^t_hr"
]

PWF_PATTERNS = [
    r"^pwf", r"^p_wf", r"^fbhp", r"^fbhp\s*\(?psia?\)?", r"^flowing\s*pressure",
    r"^pressure\s*\(?psia?\)?", r"^pressure", r"^p\s*\(?psi\)?", r"^p$",
    r"^bottom\s*hole\s*pressure", r"^bhp"
]

PWS_PATTERNS = [
    r"^pws", r"^p_ws", r"^sbhp", r"^shut[-_\s]*in\s*pressure", r"^ws_pressure",
    r"^pressure\s*\(?psia?\)?", r"^pressure", r"^p\s*\(?psi\)?", r"^p$", r"^bhp"
]

RATE_PATTERNS = [r"^q$", r"^rate", r"^flow\s*rate", r"^oil\s*rate", r"^production\s*rate"]
BO_PATTERNS = [r"^bo$", r"^b_o$", r"^b$", r"^fvf"]
MU_PATTERNS = [r"^mu$", r"^visc", r"^viscosity", r"^oil\s*viscosity"]


def load_dataset(file_source: Union[str, io.BytesIO, bytes], filename: str = "") -> pd.DataFrame:
    """
    Load dataset from CSV or Excel file.
    
    Parameters:
        file_source: File path, BytesIO stream, or raw bytes.
        filename: Optional filename for extension detection.
    Returns:
        df: Pandas DataFrame of raw file contents.
    """
    if isinstance(file_source, bytes):
        file_source = io.BytesIO(file_source)

    name_lower = filename.lower()
    if name_lower.endswith(".xlsx") or name_lower.endswith(".xls"):
        return pd.read_excel(file_source)
    elif name_lower.endswith(".csv"):
        return pd.read_csv(file_source)
    else:
        # Try CSV first, then Excel fallback
        try:
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            return pd.read_csv(file_source)
        except Exception:
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            return pd.read_excel(file_source)


def auto_detect_columns(
    df: pd.DataFrame,
    analysis_mode: str = "drawdown"
) -> Dict[str, Optional[str]]:
    """
    Automatically identify column names for Time, Pressure (Pwf or Pws), Rate, Bo, and Mu.
    
    Parameters:
        df: Input DataFrame.
        analysis_mode: 'drawdown' or 'buildup'.
    Returns:
        mapping: Dict mapping standard keys ('time', 'pressure', 'q', 'bo', 'mu')
                 to actual DataFrame column names, or None if unmatched.
    """
    cols = list(df.columns)
    mapping: Dict[str, Optional[str]] = {
        "time": None,
        "pressure": None,
        "q": None,
        "bo": None,
        "mu": None
    }
    
    cleaned = {c: str(c).strip().lower() for c in cols}
    
    # 1. Match Time
    for col, clean_name in cleaned.items():
        for pat in TIME_PATTERNS:
            if re.search(pat, clean_name):
                mapping["time"] = col
                break
        if mapping["time"]:
            break
            
    # 2. Match Pressure
    patterns_p = PWF_PATTERNS if analysis_mode == "drawdown" else PWS_PATTERNS
    for col, clean_name in cleaned.items():
        if col == mapping["time"]:
            continue
        for pat in patterns_p:
            if re.search(pat, clean_name):
                mapping["pressure"] = col
                break
        if mapping["pressure"]:
            break
            
    # 3. Optional Rate q
    for col, clean_name in cleaned.items():
        if col in (mapping["time"], mapping["pressure"]):
            continue
        for pat in RATE_PATTERNS:
            if re.search(pat, clean_name):
                mapping["q"] = col
                break
        if mapping["q"]:
            break
            
    # 4. Optional Bo
    for col, clean_name in cleaned.items():
        if col in (mapping["time"], mapping["pressure"], mapping["q"]):
            continue
        for pat in BO_PATTERNS:
            if re.search(pat, clean_name):
                mapping["bo"] = col
                break
        if mapping["bo"]:
            break

    # 5. Optional Mu
    for col, clean_name in cleaned.items():
        if col in (mapping["time"], mapping["pressure"], mapping["q"], mapping["bo"]):
            continue
        for pat in MU_PATTERNS:
            if re.search(pat, clean_name):
                mapping["mu"] = col
                break
        if mapping["mu"]:
            break

    return mapping


def validate_and_preprocess_drawdown(
    df_raw: pd.DataFrame,
    time_col: str,
    pwf_col: str,
    Pi: float,
    q_col: Optional[str] = None,
    bo_col: Optional[str] = None,
    mu_col: Optional[str] = None,
    unit_system: str = "Field"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate raw drawdown input data and compute pressure drawdown Delta P = Pi - Pwf.
    Preserves original uploaded measurements while filtering unphysical entries for analysis.
    
    Returns:
        (df_processed, validation_report)
    """
    report: Dict[str, Any] = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "row_count_original": len(df_raw),
        "row_count_processed": 0,
        "time_min": None,
        "time_max": None,
        "pwf_min": None,
        "pwf_max": None,
        "delta_p_max": None,
        "has_negative_time": False,
        "has_negative_pwf": False,
        "has_duplicates": False,
        "is_monotonic": True,
        "nans_removed": 0
    }

    if df_raw.empty:
        report["is_valid"] = False
        report["errors"].append("The uploaded dataset contains zero rows.")
        return pd.DataFrame(), report

    if time_col not in df_raw.columns:
        report["is_valid"] = False
        report["errors"].append(f"Specified time column '{time_col}' was not found in dataset.")
        return pd.DataFrame(), report

    if pwf_col not in df_raw.columns:
        report["is_valid"] = False
        report["errors"].append(f"Specified pressure column '{pwf_col}' was not found in dataset.")
        return pd.DataFrame(), report

    # Extract required series with numeric coercion
    t_series = pd.to_numeric(df_raw[time_col], errors="coerce")
    p_series = pd.to_numeric(df_raw[pwf_col], errors="coerce")
    
    nan_mask = t_series.isna() | p_series.isna()
    n_nans = int(nan_mask.sum())
    if n_nans > 0:
        report["nans_removed"] = n_nans
        report["warnings"].append(f"Found {n_nans} rows with missing or non-numeric values; these were excluded from analysis.")

    # Convert SI to Field internally if user uploaded SI
    # SI: pressure in bar -> convert to psi by * 14.50377
    p_mult = 14.50377 if unit_system == "SI" else 1.0
    
    # Working subset
    df_clean = pd.DataFrame({
        "time_hr": t_series[~nan_mask].values,
        "pwf_psi": p_series[~nan_mask].values * p_mult
    })

    if len(df_clean) < 3:
        report["is_valid"] = False
        report["errors"].append("At least 3 valid numeric time-pressure points are required for PTA.")
        return pd.DataFrame(), report

    # Check negative time
    if (df_clean["time_hr"] < 0).any():
        report["has_negative_time"] = True
        report["warnings"].append("Found negative time values; points with t < 0 were excluded.")
        df_clean = df_clean[df_clean["time_hr"] >= 0].copy()

    # Check negative pressure
    if (df_clean["pwf_psi"] < 0).any():
        report["has_negative_pwf"] = True
        report["warnings"].append("Found negative bottomhole pressure values; points with Pwf < 0 were excluded.")
        df_clean = df_clean[df_clean["pwf_psi"] >= 0].copy()

    # Check duplicate timestamps
    dup_count = int(df_clean.duplicated(subset=["time_hr"]).sum())
    if dup_count > 0:
        report["has_duplicates"] = True
        report["warnings"].append(f"Found {dup_count} duplicate timestamp records; retaining the first recorded pressure for each time.")
        df_clean = df_clean.drop_duplicates(subset=["time_hr"], keep="first")

    # Check monotonic time
    if not df_clean["time_hr"].is_monotonic_increasing:
        report["is_monotonic"] = False
        report["warnings"].append("Time sequence was not strictly monotonic; sorting chronological observations by time.")
        df_clean = df_clean.sort_values(by="time_hr").reset_index(drop=True)

    # Re-check length
    if len(df_clean) < 3:
        report["is_valid"] = False
        report["errors"].append("Insufficient observations remaining after data cleaning.")
        return pd.DataFrame(), report

    # Calculate Drawdown Delta P(t) = Pi - Pwf(t)
    df_clean["delta_p_psi"] = Pi - df_clean["pwf_psi"]

    # Check if Pwf > Pi (producing negative Delta P)
    neg_dp_count = int((df_clean["delta_p_psi"] < 0).sum())
    if neg_dp_count > 0:
        report["warnings"].append(
            f"Observed Pwf > Pi in {neg_dp_count} measurements (negative drawdown). "
            f"Please verify initial reservoir pressure Pi = {Pi:g} psi."
        )

    # Optional column carry-over
    if q_col and q_col in df_raw.columns:
        q_series = pd.to_numeric(df_raw[q_col], errors="coerce")
        df_clean["q"] = q_series.iloc[df_clean.index].values
    if bo_col and bo_col in df_raw.columns:
        bo_series = pd.to_numeric(df_raw[bo_col], errors="coerce")
        df_clean["Bo"] = bo_series.iloc[df_clean.index].values
    if mu_col and mu_col in df_raw.columns:
        mu_series = pd.to_numeric(df_raw[mu_col], errors="coerce")
        df_clean["mu"] = mu_series.iloc[df_clean.index].values

    # Statistics
    report["row_count_processed"] = len(df_clean)
    report["time_min"] = float(df_clean["time_hr"].min())
    report["time_max"] = float(df_clean["time_hr"].max())
    report["pwf_min"] = float(df_clean["pwf_psi"].min())
    report["pwf_max"] = float(df_clean["pwf_psi"].max())
    report["delta_p_max"] = float(df_clean["delta_p_psi"].max())

    return df_clean, report
