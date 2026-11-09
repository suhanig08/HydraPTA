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



def create_semilog_analysis_plot(
    times: np.ndarray,
    delta_p: np.ndarray,
    t_start: float,
    t_end: float,
    slope_m: Optional[float] = None,
    intercept_b: Optional[float] = None,
    r_squared: Optional[float] = None,
    case_name: str = "Well Test Analysis"
) -> go.Figure:
    """
    Generate semilog pressure plot (Delta P vs log10(t)) with highlighted regression interval,
    fitted radial flow straight line, and regression diagnostics.
    """
    fig = go.Figure()
    
    times = np.asarray(times)
    delta_p = np.asarray(delta_p)
    valid = times > 0
    t_plot = times[valid]
    dp_plot = delta_p[valid]

    # 1. All raw data points
    fig.add_trace(go.Scatter(
        x=t_plot,
        y=dp_plot,
        mode="markers",
        name="All Measurements",
        marker=dict(size=4, color="#CBD5E0", symbol="circle"),
        hovertemplate="<b>Time</b>: %{x:.4g} hr<br><b>ΔP</b>: %{y:.2f} psi<extra></extra>"
    ))

    # 2. Selected interval points
    mask_reg = (t_plot >= t_start) & (t_plot <= t_end)
    if np.sum(mask_reg) > 0:
        fig.add_trace(go.Scatter(
            x=t_plot[mask_reg],
            y=dp_plot[mask_reg],
            mode="markers",
            name="Selected Interval",
            marker=dict(size=6, color=COLOR_PRIMARY_ORANGE, symbol="circle"),
            hovertemplate="<b>Selected Time</b>: %{x:.4g} hr<br><b>ΔP</b>: %{y:.2f} psi<extra></extra>"
        ))

    # 3. Straight line fit
    if slope_m is not None and intercept_b is not None:
        t_line = np.geomspace(max(t_start * 0.5, np.min(t_plot)), min(t_end * 2.0, np.max(t_plot)), 50)
        dp_line = slope_m * np.log10(t_line) + intercept_b
        
        stat_label = f"m = {slope_m:.2f} psi/cycle"
        if r_squared is not None:
            stat_label += f", R² = {r_squared:.4f}"
            
        fig.add_trace(go.Scatter(
            x=t_line,
            y=dp_line,
            mode="lines",
            name=f"Radial Semilog Fit ({stat_label})",
            line=dict(color=COLOR_SECONDARY_BLUE, width=2.5, dash="solid"),
            hovertemplate="<b>Fitted Line</b><br>Time: %{x:.4g} hr<br>ΔP: %{y:.2f} psi<extra></extra>"
        ))

    fig.update_xaxes(
        type="log",
        title_text="<b>Test Time, t (hours) — Logarithmic Scale</b>",
        gridcolor=COLOR_GRID,
        linecolor=COLOR_DARK_CHARCOAL,
        showline=True
    )
    
    fig.update_yaxes(
        title_text="<b>Pressure Drawdown, ΔP (psi) — Linear Scale</b>",
        gridcolor=COLOR_GRID,
        linecolor=COLOR_DARK_CHARCOAL,
        showline=True
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text=f"<b>Semilog Radial-Flow Analysis Plot — {case_name}</b>",
            font=dict(size=15, color=COLOR_DARK_CHARCOAL)
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
        margin=dict(l=60, r=40, t=60, b=50),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF"
    )

    return fig


def create_dimensionless_diagnostic_plot(
    df_dim: pd.DataFrame,
    case_name: str = "Well Test Analysis",
    include_theoretical: bool = True
) -> go.Figure:
    """
    Generate Dimensionless Pressure pD and Dimensionless Derivative p'D log-log diagnostic plot.
    Overlays the theoretical radial infinite-acting line source type curve (p'D = 0.5 plateau).
    """
    fig = go.Figure()
    
    tD = df_dim["tD"].values
    pD = df_dim["pD"].values
    pD_prime = df_dim["pD_prime"].values

    # 1. Dimensionless Pressure pD
    fig.add_trace(go.Scatter(
        x=tD,
        y=pD,
        mode="lines+markers",
        name="Dimensionless Pressure (pD)",
        line=dict(color=COLOR_SECONDARY_BLUE, width=2.2),
        marker=dict(size=4, color=COLOR_SECONDARY_BLUE),
        hovertemplate="<b>tD</b>: %{x:.3e}<br><b>pD</b>: %{y:.3f}<extra></extra>"
    ))

    # 2. Dimensionless Derivative p'D
    fig.add_trace(go.Scatter(
        x=tD,
        y=pD_prime,
        mode="lines+markers",
        name="Dimensionless Derivative (p'D)",
        line=dict(color=COLOR_PRIMARY_ORANGE, width=2.2),
        marker=dict(size=4, color=COLOR_PRIMARY_ORANGE),
        hovertemplate="<b>tD</b>: %{x:.3e}<br><b>p'D</b>: %{y:.3f}<extra></extra>"
    ))

    # 3. Theoretical Radial Flow Horizontal Derivative p'D = 0.5
    if include_theoretical and len(tD) > 0:
        tD_min = np.min(tD[tD > 0])
        tD_max = np.max(tD)
        
        fig.add_trace(go.Scatter(
            x=[tD_min, tD_max],
            y=[0.5, 0.5],
            mode="lines",
            name="Theoretical Radial Plateau (p'D = 0.5)",
            line=dict(color="#4A5568", width=1.8, dash="dash"),
            hoverinfo="skip"
        ))

    fig.update_xaxes(
        type="log",
        title_text="<b>Dimensionless Time, tD</b>",
        gridcolor=COLOR_GRID,
        linecolor=COLOR_DARK_CHARCOAL,
        showline=True
    )
    
    fig.update_yaxes(
        type="log",
        title_text="<b>Dimensionless Variables, pD & p'D</b>",
        gridcolor=COLOR_GRID,
        linecolor=COLOR_DARK_CHARCOAL,
        showline=True
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text=f"<b>Dimensionless Diagnostic Plot — {case_name}</b>",
            font=dict(size=15, color=COLOR_DARK_CHARCOAL)
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
        margin=dict(l=60, r=40, t=60, b=50),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF"
    )

    return fig


def create_smoothing_comparison_plot(
    times: np.ndarray,
    derivatives_by_L: Dict[float, np.ndarray],
    case_name: str = "Derivative Smoothing Sensitivity"
) -> go.Figure:
    """
    Generate comparison plot showing Bourdet pressure derivatives at multiple L distances.
    """
    fig = go.Figure()
    palette = [COLOR_PRIMARY_ORANGE, COLOR_SECONDARY_BLUE, "#2D3748", "#38A169"]

    for idx, (L_val, derv) in enumerate(derivatives_by_L.items()):
        col = palette[idx % len(palette)]
        fig.add_trace(go.Scatter(
            x=times,
            y=derv,
            mode="lines",
            name=f"L = {L_val:.2f}",
            line=dict(color=col, width=2.0),
            hovertemplate=f"<b>L = {L_val:.2f}</b><br>Time: %{{x:.4g}} hr<br>ΔP': %{{y:.2f}} psi<extra></extra>"
        ))

    fig.update_xaxes(
        type="log",
        title_text="<b>Test Time, t (hours)</b>",
        gridcolor=COLOR_GRID,
        linecolor=COLOR_DARK_CHARCOAL,
        showline=True
    )
    
    fig.update_yaxes(
        type="log",
        title_text="<b>Bourdet Pressure Derivative, ΔP' (psi)</b>",
        gridcolor=COLOR_GRID,
        linecolor=COLOR_DARK_CHARCOAL,
        showline=True
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text=f"<b>Bourdet Smoothing Sensitivity (Different L Values) — {case_name}</b>",
            font=dict(size=15, color=COLOR_DARK_CHARCOAL)
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
        margin=dict(l=60, r=40, t=60, b=50),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF"
    )

    return fig


def create_horner_plot(
    df_buildup: pd.DataFrame,
    slope_m_horner: Optional[float] = None,
    intercept_b: Optional[float] = None,
    P_star: Optional[float] = None,
    P_1hr: Optional[float] = None,
    tp_hr: float = 100.0,
    case_name: str = "Buildup Test Analysis"
) -> go.Figure:
    """
    Generate classic petroleum engineering Horner semilog plot:
    Pws vs log10((tp + dt) / dt) with inverted X-axis (decreasing from right to left).
    """
    fig = go.Figure()

    hr = df_buildup["horner_ratio"].values
    pws = df_buildup["pws_psi"].values

    # 1. Measured buildup pressures
    fig.add_trace(go.Scatter(
        x=hr,
        y=pws,
        mode="markers",
        name="Shut-In Pressure Pws",
        marker=dict(size=5, color=COLOR_SECONDARY_BLUE),
        hovertemplate="<b>Horner Ratio</b>: %{x:.2f}<br><b>Pws</b>: %{y:.2f} psi<extra></extra>"
    ))

    # 2. Fitted Horner straight line extrapolated to HR = 1 (P*)
    if slope_m_horner is not None and intercept_b is not None:
        hr_line = np.geomspace(1.0, np.max(hr) * 1.2, 50)
        pws_line = intercept_b - slope_m_horner * np.log10(hr_line)
        
        label_str = f"Horner Straight Line (m = {slope_m_horner:.2f} psi/cycle)"
        if P_star is not None:
            label_str += f", P* = {P_star:.1f} psi"
            
        fig.add_trace(go.Scatter(
            x=hr_line,
            y=pws_line,
            mode="lines",
            name=label_str,
            line=dict(color=COLOR_PRIMARY_ORANGE, width=2.5),
            hovertemplate="<b>Horner Line</b><br>Horner Ratio: %{x:.2f}<br>Pws: %{y:.2f} psi<extra></extra>"
        ))
        
        # Mark P* at HR = 1
        if P_star is not None:
            fig.add_trace(go.Scatter(
                x=[1.0],
                y=[P_star],
                mode="markers+text",
                name=f"Extrapolated P* = {P_star:.1f} psi",
                marker=dict(size=9, color="#E53E3E", symbol="diamond"),
                text=[f"P* = {P_star:.1f} psi"],
                textposition="top right"
            ))

    # Invert X-axis as standard in petroleum engineering Horner plots
    fig.update_xaxes(
        type="log",
        autorange="reversed",
        title_text="<b>Horner Time Ratio, (tp + Δt) / Δt — Inverted Scale</b>",
        gridcolor=COLOR_GRID,
        linecolor=COLOR_DARK_CHARCOAL,
        showline=True
    )
    
    fig.update_yaxes(
        title_text="<b>Shut-In Bottomhole Pressure, Pws (psi)</b>",
        gridcolor=COLOR_GRID,
        linecolor=COLOR_DARK_CHARCOAL,
        showline=True
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text=f"<b>Horner Semilog Plot — {case_name}</b>",
            font=dict(size=15, color=COLOR_DARK_CHARCOAL)
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
        margin=dict(l=60, r=40, t=60, b=50),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF"
    )

    return fig


def create_buildup_diagnostic_plot(
    df_buildup: pd.DataFrame,
    case_name: str = "Buildup Test Analysis"
) -> go.Figure:
    """
    Generate log-log diagnostic plot for pressure buildup using Agarwal equivalent time:
    Delta Pws and Agarwal Bourdet derivative Delta P'ws vs Agarwal equivalent time te.
    """
    fig = go.Figure()
    
    te = df_buildup["agarwal_te_hr"].values
    dp = df_buildup["delta_pws_psi"].values
    derv = df_buildup["derivative_psi"].values

    fig.add_trace(go.Scatter(
        x=te,
        y=dp,
        mode="lines+markers",
        name="Pressure Increase ΔPws (psi)",
        line=dict(color=COLOR_SECONDARY_BLUE, width=2.5),
        marker=dict(size=4, color=COLOR_SECONDARY_BLUE),
        hovertemplate="<b>te</b>: %{x:.4g} hr<br><b>ΔPws</b>: %{y:.2f} psi<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=te,
        y=derv,
        mode="lines+markers",
        name="Agarwal Bourdet Derivative ΔP' (psi)",
        line=dict(color=COLOR_PRIMARY_ORANGE, width=2.5),
        marker=dict(size=5, color=COLOR_PRIMARY_ORANGE),
        hovertemplate="<b>te</b>: %{x:.4g} hr<br><b>ΔP'</b>: %{y:.2f} psi<extra></extra>"
    ))

    fig.update_xaxes(
        type="log",
        title_text="<b>Agarwal Equivalent Shut-In Time, te (hours)</b>",
        gridcolor=COLOR_GRID,
        linecolor=COLOR_DARK_CHARCOAL,
        showline=True
    )
    
    fig.update_yaxes(
        type="log",
        title_text="<b>ΔPws & Bourdet Derivative ΔP' (psi)</b>",
        gridcolor=COLOR_GRID,
        linecolor=COLOR_DARK_CHARCOAL,
        showline=True
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text=f"<b>Buildup Diagnostic Plot (Agarwal Equivalent Time) — {case_name}</b>",
            font=dict(size=15, color=COLOR_DARK_CHARCOAL)
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
        margin=dict(l=60, r=40, t=60, b=50),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF"
    )

    return fig
