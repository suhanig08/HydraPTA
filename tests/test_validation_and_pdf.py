"""
Unit tests for the synthetic demo validation engine and PDF report generation.
"""

import pytest
from src.validation_engine import run_demo_case_validation
from src.pdf_report import generate_pta_pdf_report
from data.demo_cases import DEMO_CASES


def test_demo_case_validation_all_three_cases():
    """Verify that all 3 demo cases run through the validation engine and produce comparison tables."""
    for case_id in DEMO_CASES.keys():
        df_val, meta = run_demo_case_validation(case_id, add_noise=False)
        assert len(df_val) >= 3
        assert "Parameter" in df_val.columns
        assert "Actual value" in df_val.columns
        assert "Estimated value" in df_val.columns
        assert "Percentage error" in df_val.columns
        assert "Model Status" in df_val.columns
        
        # Meta dictionary checks
        assert "times" in meta
        assert "delta_p" in meta
        assert "derivative" in meta
        assert "regimes" in meta
        assert "plateau" in meta
        assert "estimation" in meta


def test_pdf_report_generation():
    """Verify that the PDF report generates non-empty bytes."""
    props = {
        "Pi": 4000.0, "q": 500.0, "Bo": 1.2, "mu": 1.2,
        "h": 50.0, "rw": 0.25, "phi": 0.12, "ct": 1.5e-5
    }
    est = {
        "k_semilog": {"is_valid": True, "k_mD": 5.0, "r_squared": 0.99},
        "k_plateau": {"is_valid": True, "k_mD": 4.95},
        "skin": {"is_valid": True, "skin": 0.05},
        "xf": {"is_valid": True, "xf_ft": 480.0, "r_squared": 0.98},
        "Cf": {"is_valid": True, "Cf_mD_ft": 190.0, "r_squared": 0.97}
    }
    pdf_bytes = generate_pta_pdf_report(
        analysis_mode="Drawdown Analysis",
        case_name="Automated Test",
        properties=props,
        estimation_summary=est,
        detected_regimes=[],
        verdict_text="Automated test validation successful."
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
