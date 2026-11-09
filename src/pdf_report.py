"""
PDF Engineering Report Generator for HydraPTA using ReportLab.
Generates publication-quality, professional petroleum engineering analysis reports.
"""

from typing import Dict, Any, List, Optional
import io
import datetime
import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable


COLOR_PRIMARY = colors.HexColor("#E8752A")     # HydraPTA Orange
COLOR_CHARCOAL = colors.HexColor("#252525")    # Dark Charcoal
COLOR_LIGHT_BG = colors.HexColor("#F8F9FA")    # Off-white table header
COLOR_BORDER = colors.HexColor("#E5E7EB")      # Subtle border


def generate_pta_pdf_report(
    analysis_mode: str,
    case_name: str,
    properties: Dict[str, Any],
    estimation_summary: Dict[str, Any],
    detected_regimes: List[Any],
    plateau_info: Optional[Dict[str, Any]] = None,
    verdict_text: str = "",
    unit_system: str = "Field"
) -> bytes:
    """
    Generate a complete PDF engineering analysis report as bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=COLOR_CHARCOAL
    )
    
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=COLOR_PRIMARY
    )
    
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=COLOR_CHARCOAL,
        spaceBefore=10,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=COLOR_CHARCOAL
    )

    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=COLOR_CHARCOAL
    )

    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=COLOR_CHARCOAL
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("HydraPTA — Well Test Interpretation Report", title_style))
    report_date = datetime.datetime.now().strftime("%B %d, %Y - %H:%M UTC")
    story.append(Paragraph(f"AUTOMATED PRESSURE TRANSIENT ANALYSIS | {analysis_mode.upper()} MODE | {report_date}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=COLOR_PRIMARY, spaceBefore=6, spaceAfter=12))

    # 2. Executive Summary / Verdict
    story.append(Paragraph("1. Executive Engineering Verdict", section_heading))
    if not verdict_text:
        verdict_text = "Analysis completed. Transient signatures evaluated via Bourdet differentiation and conventional petroleum engineering solutions."
    story.append(Paragraph(verdict_text, body_style))
    story.append(Spacer(1, 10))

    # 3. Reservoir, Fluid & Well Properties Table
    story.append(Paragraph(f"2. Input Petrophysical & Well Parameters ({unit_system} Units)", section_heading))
    p_data = [
        [Paragraph("Parameter", table_header), Paragraph("Symbol", table_header), Paragraph("Value", table_header), Paragraph("Unit", table_header)]
    ]
    
    unit_p = "psi" if unit_system == "Field" else "bar"
    unit_q = "STB/d" if unit_system == "Field" else "m³/d"
    unit_h = "ft" if unit_system == "Field" else "m"
    unit_rw = "ft" if unit_system == "Field" else "m"
    unit_ct = "1/psi" if unit_system == "Field" else "1/bar"
    
    prop_rows = [
        ("Initial Reservoir Pressure", "Pi", f"{properties.get('Pi', 'N/A')}", unit_p),
        ("Production Flow Rate", "q", f"{properties.get('q', 'N/A')}", unit_q),
        ("Formation Volume Factor", "Bo", f"{properties.get('Bo', 'N/A')}", "rb/STB"),
        ("Oil Dynamic Viscosity", "μ", f"{properties.get('mu', 'N/A')}", "cp"),
        ("Net Pay Thickness", "h", f"{properties.get('h', 'N/A')}", unit_h),
        ("Wellbore Radius", "rw", f"{properties.get('rw', 'N/A')}", unit_rw),
        ("Formation Porosity", "φ", f"{properties.get('phi', 'N/A')}", "fraction"),
        ("Total Compressibility", "ct", f"{properties.get('ct', 'N/A')}", unit_ct),
    ]
    for row in prop_rows:
        p_data.append([Paragraph(row[0], table_cell), Paragraph(row[1], table_cell), Paragraph(str(row[2]), table_cell), Paragraph(row[3], table_cell)])

    t_prop = Table(p_data, colWidths=[180, 60, 150, 150])
    t_prop.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_LIGHT_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_prop)
    story.append(Spacer(1, 10))

    # 4. Detected Flow Regimes Table
    story.append(Paragraph("3. Flow-Regime Identification & Slope Diagnostics", section_heading))
    r_data = [
        [Paragraph("Identified Regime", table_header), Paragraph("Interval (hr)", table_header), Paragraph("Observed Slope", table_header), Paragraph("Expected Slope", table_header), Paragraph("Confidence", table_header)]
    ]
    if detected_regimes:
        for r in detected_regimes:
            t_span = f"{r.t_start:.3g} – {r.t_end:.3g}"
            r_data.append([
                Paragraph(r.regime, table_cell),
                Paragraph(t_span, table_cell),
                Paragraph(f"{r.observed_slope:+.2f}", table_cell),
                Paragraph(f"{r.expected_slope:.2f}", table_cell),
                Paragraph(getattr(r, "confidence", "Moderate"), table_cell)
            ])
    else:
        r_data.append([Paragraph("No distinct flow regime sustained within standard slope tolerances.", table_cell), Paragraph("—", table_cell), Paragraph("—", table_cell), Paragraph("—", table_cell), Paragraph("—", table_cell)])

    t_regimes = Table(r_data, colWidths=[160, 100, 90, 90, 100])
    t_regimes.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_LIGHT_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_regimes)
    story.append(Spacer(1, 10))

    # 5. Estimated Reservoir and Fracture Properties
    story.append(Paragraph("4. Reservoir & Fracture Characterization Results", section_heading))
    e_data = [
        [Paragraph("Property", table_header), Paragraph("Estimation Method", table_header), Paragraph("Estimated Value", table_header), Paragraph("Units", table_header), Paragraph("Status", table_header)]
    ]

    # Semilog Permeability
    k_semi = estimation_summary.get("k_semilog", {})
    if k_semi.get("is_valid"):
        e_data.append([
            Paragraph("Permeability (k)", table_cell),
            Paragraph("Semilog Radial Straight Line", table_cell),
            Paragraph(f"{k_semi['k_mD']:g}", table_cell),
            Paragraph("mD", table_cell),
            Paragraph(f"Resolved (R²={k_semi.get('r_squared', '—')})", table_cell)
        ])
    else:
        e_data.append([
            Paragraph("Permeability (k)", table_cell),
            Paragraph("Semilog Radial Straight Line", table_cell),
            Paragraph("Not resolved", table_cell),
            Paragraph("mD", table_cell),
            Paragraph("Regime absent / unconstrained", table_cell)
        ])

    # Derivative Plateau Permeability
    k_plat = estimation_summary.get("k_plateau", {})
    if k_plat.get("is_valid"):
        e_data.append([
            Paragraph("Permeability (k)", table_cell),
            Paragraph("Bourdet Derivative Plateau", table_cell),
            Paragraph(f"{k_plat['k_mD']:g}", table_cell),
            Paragraph("mD", table_cell),
            Paragraph("Resolved from plateau", table_cell)
        ])

    # Skin Factor
    skin_info = estimation_summary.get("skin", {})
    if skin_info.get("is_valid"):
        e_data.append([
            Paragraph("Wellbore Skin (S)", table_cell),
            Paragraph("Semilog Intercept at 1 hr", table_cell),
            Paragraph(f"{skin_info['skin']:+.2f}", table_cell),
            Paragraph("dimensionless", table_cell),
            Paragraph("Resolved", table_cell)
        ])

    # Fracture Half-Length xf
    xf_info = estimation_summary.get("xf", {})
    if xf_info.get("is_valid"):
        e_data.append([
            Paragraph("Fracture Half-Length (xf)", table_cell),
            Paragraph("Linear Flow Slope (ΔP vs √t)", table_cell),
            Paragraph(f"{xf_info['xf_ft']:g}", table_cell),
            Paragraph("ft", table_cell),
            Paragraph(f"Resolved (R²={xf_info.get('r_squared', '—')})", table_cell)
        ])

    # Fracture Conductivity Cf
    cf_info = estimation_summary.get("Cf", {})
    if cf_info.get("is_valid"):
        e_data.append([
            Paragraph("Fracture Conductivity (Cf)", table_cell),
            Paragraph("Bilinear Flow Slope (ΔP vs t^¼)", table_cell),
            Paragraph(f"{cf_info['Cf_mD_ft']:g}", table_cell),
            Paragraph("mD-ft", table_cell),
            Paragraph(f"Resolved (R²={cf_info.get('r_squared', '—')})", table_cell)
        ])

    t_est = Table(e_data, colWidths=[140, 150, 90, 60, 100])
    t_est.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_LIGHT_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_est)
    story.append(Spacer(1, 15))

    # Notes & Disclaimer
    disclaimer = (
        "<b>Notice & Engineering Methodology</b>: HydraPTA calculations are derived from classical petroleum engineering "
        "drawdown and buildup analytical solutions. Parameter estimations require active presence of the respective "
        "flow regimes. Generated automatically by HydraPTA platform."
    )
    story.append(Paragraph(disclaimer, table_cell))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
