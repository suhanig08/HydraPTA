"""
Interactive Plotly visualization module for HydraPTA.
Implements publication-quality petroleum engineering charts with subtle orange-and-charcoal styling.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from .models import FlowRegimeRecord


# Design Tokens & Palette
COLOR_PRIMARY_ORANGE = "#E8752A"
COLOR_DARK_CHARCOAL = "#252525"
COLOR_LIGHT_GRAY = "#F8F9FA"
COLOR_BORDER_GRAY = "#E5E7EB"
COLOR_SECONDARY_BLUE = "#2B6CB0"
COLOR_MUTED_SLATE = "#718096"
COLOR_GRID = "#E2E8F0"

PLOTLY_TEMPLATE = "plotly_white"


def create_log_log_diagnostic_plot(
    times: np.ndarray,
    delta_p: np.ndarray,
    derivative: np.ndarray,
    detected_regimes: List[FlowRegimeRecord],
    case_name: str = "Hydraulically Fractured Well"
) -> go.Figure:
    """
    Generate the central log-log diagnostic plot with Delta P, Bourdet derivative,
    subtle reference slopes (1, 1/2, 1/4, 0), and highlighted flow regime annotations.
    """
    fig = go.Figure()

    # 1. Delta P curve
    fig.add_trace(go.Scatter(
        x=times,
        y=delta_p,
        mode="lines+markers",
        name="Pressure Change ΔP (psi)",
        line=dict(color=COLOR_SECONDARY_BLUE, width=2.5),
        marker=dict(size=4, color=COLOR_SECONDARY_BLUE),
        hovertemplate="<b>Time</b>: %{x:.4g} hr<br><b>ΔP</b>: %{y:.2f} psi<extra></extra>"
    ))

    # 2. Pressure derivative curve
    fig.add_trace(go.Scatter(
        x=times,
        y=derivative,
        mode="lines+markers",
        name="Pressure Derivative ΔP' (psi)",
        line=dict(color=COLOR_PRIMARY_ORANGE, width=2.5),
        marker=dict(size=5, color=COLOR_PRIMARY_ORANGE, symbol="circle"),
        hovertemplate="<b>Time</b>: %{x:.4g} hr<br><b>ΔP'</b>: %{y:.2f} psi<extra></extra>"
    ))

    # 3. Add Reference Slopes
    # Find anchoring points for clean, subtle reference lines
    t_min, t_max = np.min(times), np.max(times)
    dp_min = max(np.min(derivative[derivative > 0]), 1.0)
    
    # Reference Slope 1 (Wellbore storage)
    wbs_regime = next((r for r in detected_regimes if r.regime == "Wellbore Storage"), None)
    t_ref_wbs = wbs_regime.t_start if wbs_regime else times[2]
    idx_wbs = np.argmin(np.abs(times - t_ref_wbs))
    y_ref_wbs = max(derivative[idx_wbs], 5.0)
    t_wbs_line = np.array([t_ref_wbs * 0.4, t_ref_wbs * 2.5])
    y_wbs_line = y_ref_wbs * (t_wbs_line / t_ref_wbs) ** 1.0
    
    fig.add_trace(go.Scatter(
        x=t_wbs_line,
        y=y_wbs_line,
        mode="lines",
        name="Slope = 1 (WBS)",
        line=dict(color="#A0AEC0", width=1.5, dash="dash"),
        hoverinfo="skip"
    ))

    # Reference Slope 1/4 (Bilinear flow)
    bl_regime = next((r for r in detected_regimes if r.regime == "Bilinear Flow"), None)
    if bl_regime:
        t_ref_bl = (bl_regime.t_start + bl_regime.t_end) / 2.0
        idx_bl = np.argmin(np.abs(times - t_ref_bl))
        y_ref_bl = derivative[idx_bl]
        t_bl_line = np.array([bl_regime.t_start * 0.7, bl_regime.t_end * 1.3])
        y_bl_line = y_ref_bl * (t_bl_line / t_ref_bl) ** 0.25
        
        fig.add_trace(go.Scatter(
            x=t_bl_line,
            y=y_bl_line,
            mode="lines",
            name="Slope = 1/4 (Bilinear)",
            line=dict(color="#4A5568", width=1.5, dash="dashdot"),
            hoverinfo="skip"
        ))

    # Reference Slope 1/2 (Linear flow)
    lf_regime = next((r for r in detected_regimes if r.regime == "Linear Flow"), None)
    if lf_regime:
        t_ref_lf = (lf_regime.t_start + lf_regime.t_end) / 2.0
        idx_lf = np.argmin(np.abs(times - t_ref_lf))
        y_ref_lf = derivative[idx_lf]
        t_lf_line = np.array([lf_regime.t_start * 0.7, lf_regime.t_end * 1.3])
        y_lf_line = y_ref_lf * (t_lf_line / t_ref_lf) ** 0.50
        
        fig.add_trace(go.Scatter(
            x=t_lf_line,
            y=y_lf_line,
            mode="lines",
            name="Slope = 1/2 (Linear)",
            line=dict(color="#2D3748", width=1.5, dash="dot"),
            hoverinfo="skip"
        ))

    # Reference Slope 0 (Pseudoradial flow)
    rad_regime = next((r for r in detected_regimes if r.regime == "Pseudoradial Flow"), None)
    if rad_regime:
        t_ref_rad = (rad_regime.t_start + rad_regime.t_end) / 2.0
        idx_rad = np.argmin(np.abs(times - t_ref_rad))
        y_ref_rad = derivative[idx_rad]
        t_rad_line = np.array([rad_regime.t_start * 0.8, t_max])
        y_rad_line = np.array([y_ref_rad, y_ref_rad])
        
        fig.add_trace(go.Scatter(
            x=t_rad_line,
            y=y_rad_line,
            mode="lines",
            name="Slope = 0 (Radial)",
            line=dict(color="#718096", width=1.5, dash="dash"),
            hoverinfo="skip"
        ))

    # 4. Highlight Detected Flow Regime Intervals
    # Color mapping for regime bands
    regime_colors = {
        "Wellbore Storage": "rgba(226, 232, 240, 0.4)",
        "Bilinear Flow": "rgba(254, 235, 226, 0.5)",
        "Linear Flow": "rgba(254, 215, 165, 0.45)",
        "Pseudoradial Flow": "rgba(227, 242, 253, 0.4)"
    }
    
    for r in detected_regimes:
        bg_col = regime_colors.get(r.regime, "rgba(240, 240, 240, 0.3)")
        fig.add_vrect(
            x0=r.t_start,
            x1=r.t_end,
            fillcolor=bg_col,
            layer="below",
            line_width=0,
        )
        
        # Add informative annotation above derivative
        t_mid = 10.0 ** ((np.log10(r.t_start) + np.log10(r.t_end)) / 2.0)
        idx_mid = np.argmin(np.abs(times - t_mid))
        y_annot = derivative[idx_mid] * 1.5
        
        fig.add_annotation(
            x=np.log10(t_mid),
            y=np.log10(max(y_annot, 1.0)),
            xref="x",
            yref="y",
            text=f"<b>{r.regime}</b><br>Slope: {r.observed_slope:+.2f} (Exp: {r.expected_slope:.2f})",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.2,
            arrowcolor=COLOR_PRIMARY_ORANGE if r.regime in ["Linear Flow", "Bilinear Flow"] else COLOR_DARK_CHARCOAL,
            ax=0,
            ay=-35,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor=COLOR_PRIMARY_ORANGE if r.regime in ["Linear Flow", "Bilinear Flow"] else COLOR_BORDER_GRAY,
            borderwidth=1,
            borderpad=4,
            font=dict(size=10, color=COLOR_DARK_CHARCOAL)
        )

    # Layout configuration
    fig.update_xaxes(
        type="log",
        title_text="<b>Elapsed Test Time, t (hours)</b>",
        gridcolor=COLOR_GRID,
        linecolor=COLOR_DARK_CHARCOAL,
        tickfont=dict(color=COLOR_DARK_CHARCOAL),
        showline=True
    )
    
    fig.update_yaxes(
        type="log",
        title_text="<b>Pressure Change ΔP & Derivative ΔP' (psi)</b>",
        gridcolor=COLOR_GRID,
        linecolor=COLOR_DARK_CHARCOAL,
        tickfont=dict(color=COLOR_DARK_CHARCOAL),
        showline=True
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text=f"<b>Log-Log PTA Diagnostic Plot — {case_name}</b>",
            font=dict(size=16, color=COLOR_DARK_CHARCOAL)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor=COLOR_BORDER_GRAY,
            borderwidth=1
        ),
        margin=dict(l=60, r=40, t=70, b=60),
        hovermode="closest",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF"
    )

    return fig


def create_pressure_history_plot(
    times: np.ndarray,
    pwf: np.ndarray,
    Pi: float,
    case_name: str = "Hydraulically Fractured Well"
) -> go.Figure:
    """Generate Bottomhole Flowing Pressure Pwf vs Time history plot."""
    fig = go.Figure()

    # Initial Pressure line
    fig.add_hline(
        y=Pi,
        line_dash="dash",
        line_color="#A0AEC0",
        annotation_text=f"Initial Pressure Pi = {Pi:g} psi",
        annotation_position="top right"
    )

    # Bottomhole pressure curve
    fig.add_trace(go.Scatter(
        x=times,
        y=pwf,
        mode="lines",
        name="Flowing Bottomhole Pressure Pwf (psi)",
        line=dict(color=COLOR_PRIMARY_ORANGE, width=2.5),
        hovertemplate="<b>Time</b>: %{x:.4g} hr<br><b>Pwf</b>: %{y:.2f} psi<extra></extra>"
    ))

    fig.update_xaxes(
        type="log",
        title_text="<b>Time, t (hours)</b>",
        gridcolor=COLOR_GRID,
        linecolor=COLOR_DARK_CHARCOAL,
        showline=True
    )
    
    fig.update_yaxes(
        title_text="<b>Bottomhole Pressure, Pwf (psi)</b>",
        gridcolor=COLOR_GRID,
        linecolor=COLOR_DARK_CHARCOAL,
        showline=True
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text=f"<b>Bottomhole Pressure History — {case_name}</b>",
            font=dict(size=15, color=COLOR_DARK_CHARCOAL)
        ),
        margin=dict(l=60, r=40, t=60, b=50),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF"
    )

    return fig


def create_delta_p_plot(
    times: np.ndarray,
    delta_p: np.ndarray,
    case_name: str = "Hydraulically Fractured Well"
) -> go.Figure:
    """Generate Pressure Change Delta P vs Time plot."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=times,
        y=delta_p,
        mode="lines",
        name="Pressure Change ΔP (psi)",
        line=dict(color=COLOR_SECONDARY_BLUE, width=2.5),
        hovertemplate="<b>Time</b>: %{x:.4g} hr<br><b>ΔP</b>: %{y:.2f} psi<extra></extra>"
    ))

    fig.update_xaxes(
        type="log",
        title_text="<b>Time, t (hours)</b>",
        gridcolor=COLOR_GRID,
        linecolor=COLOR_DARK_CHARCOAL,
        showline=True
    )
    
    fig.update_yaxes(
        title_text="<b>Pressure Drawdown, ΔP (psi)</b>",
        gridcolor=COLOR_GRID,
        linecolor=COLOR_DARK_CHARCOAL,
        showline=True
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text=f"<b>Pressure Drawdown Response — {case_name}</b>",
            font=dict(size=15, color=COLOR_DARK_CHARCOAL)
        ),
        margin=dict(l=60, r=40, t=60, b=50),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF"
    )

    return fig


def create_sensitivity_comparison_plot(
    df_curves: pd.DataFrame,
    param_label: str,
    y_variable: str = "delta_P_psi"
) -> go.Figure:
    """
    Generate interactive multi-curve sensitivity plot for Delta P or Bourdet Derivative.
    
    Parameters:
        df_curves: DataFrame with columns ['scenario', 'time_hr', 'delta_P_psi', 'derivative_psi']
        param_label: Parameter display name
        y_variable: 'delta_P_psi' or 'derivative_psi'
    """
    fig = go.Figure()
    scenarios = df_curves["scenario"].unique()
    
    # Palette with orange to deep slate gradient
    colors = px.colors.sample_colorscale("Plasma", [i / max(len(scenarios) - 1, 1) for i in range(len(scenarios))])
    
    is_derv = (y_variable == "derivative_psi")
    y_title = "<b>Pressure Derivative, ΔP' (psi)</b>" if is_derv else "<b>Pressure Drawdown, ΔP (psi)</b>"
    chart_title = f"<b>Sensitivity: {param_label} vs Pressure Derivative ΔP'</b>" if is_derv else f"<b>Sensitivity: {param_label} vs Pressure Drawdown ΔP</b>"

    for idx, scen in enumerate(scenarios):
        df_scen = df_curves[df_curves["scenario"] == scen]
        fig.add_trace(go.Scatter(
            x=df_scen["time_hr"],
            y=df_scen[y_variable],
            mode="lines",
            name=str(scen),
            line=dict(width=2.2, color=colors[idx]),
            hovertemplate="<b>Scenario</b>: " + str(scen) + "<br><b>Time</b>: %{x:.4g} hr<br><b>Value</b>: %{y:.2f} psi<extra></extra>"
        ))

    fig.update_xaxes(
        type="log",
        title_text="<b>Test Time, t (hours)</b>",
        gridcolor=COLOR_GRID,
        linecolor=COLOR_DARK_CHARCOAL,
        showline=True
    )
    
    fig.update_yaxes(
        type="log" if is_derv else "linear",
        title_text=y_title,
        gridcolor=COLOR_GRID,
        linecolor=COLOR_DARK_CHARCOAL,
        showline=True
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=chart_title, font=dict(size=15, color=COLOR_DARK_CHARCOAL)),
        legend=dict(
            title=dict(text=f"<b>{param_label}</b>"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor=COLOR_BORDER_GRAY,
            borderwidth=1
        ),
        margin=dict(l=60, r=40, t=60, b=50),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF"
    )

    return fig
