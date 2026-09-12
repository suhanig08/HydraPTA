"""
HydraPTA: Hydraulic Fracture Pressure Transient Analyzer
Streamlit Web Application & Engineering Dashboard.
"""

import time
import streamlit as st
import numpy as np
import pandas as pd

from src.models import (
    ReservoirProperties,
    WellProperties,
    FractureProperties,
    PTAAnalysisResult
)
from src.fracture import classify_fracture_conductivity_type
from src.pressure_transient import generate_synthetic_transient_response
from src.derivative import calculate_bourdet_derivative, calculate_log_slopes
from src.flow_regime import detect_sustained_flow_regimes, generate_flow_regime_summary_table
from src.estimation import run_inverse_characterization
from src.sensitivity import (
    SENSITIVITY_PARAMETERS,
    generate_sensitivity_scenarios
)
from src.plotting import (
    create_log_log_diagnostic_plot,
    create_pressure_history_plot,
    create_delta_p_plot,
    create_sensitivity_comparison_plot
)
from data.demo_cases import DEMO_CASES, get_demo_case, list_demo_cases


# --- Page Configuration ---
st.set_page_config(
    page_title="HydraPTA — Hydraulic Fracture Pressure Transient Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS Styling (Subtle Orange & Charcoal Palette) ---
st.markdown("""
<style>
    /* Global styles */
    .main {
        background-color: #FFFFFF;
        color: #252525;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #252525;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Header Accent */
    .brand-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #252525;
        margin-bottom: 0px;
    }
    
    .brand-tagline {
        font-size: 1.1rem;
        font-weight: 500;
        color: #E8752A;
        margin-bottom: 1.2rem;
    }
    
    /* Cards */
    .pta-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: all 0.2s ease-in-out;
    }
    
    .pta-card:hover {
        border-color: #CBD5E0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .pta-card-header {
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #E8752A;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    
    .metric-badge {
        display: inline-block;
        background-color: #F8F9FA;
        border: 1px solid #E2E8F0;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #4A5568;
    }
    
    .badge-locked {
        background-color: #FEF3C7;
        color: #92400E;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 12px;
        border: 1px solid #FDE68A;
    }
    
    .badge-soon {
        background-color: #E2E8F0;
        color: #4A5568;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 12px;
    }
    
    /* Summary bar */
    .summary-bar {
        background-color: #F8F9FA;
        border: 1px solid #E5E7EB;
        border-left: 4px solid #E8752A;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.5rem;
    }
    
    /* Buttons */
    div.stButton > button {
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.15s ease-in-out;
    }
    
    div.stButton > button:hover {
        border-color: #E8752A;
        color: #E8752A;
    }
</style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "home"
if "selected_case_id" not in st.session_state:
    st.session_state["selected_case_id"] = "case_1"
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None


# --- Navigation Functions ---
def navigate_to(page: str, case_id: str = None):
    st.session_state["current_page"] = page
    if case_id:
        st.session_state["selected_case_id"] = case_id
    st.rerun()


# --- Cached PTA Execution Engine ---
@st.cache_data(show_spinner=False)
def execute_pta_pipeline(case_id: str) -> PTAAnalysisResult:
    """Execute complete forward model, derivative, regime detection, and estimation workflow."""
    case_data = get_demo_case(case_id)
    res = case_data["reservoir"]
    well = case_data["well"]
    frac = case_data["fracture"]
    
    # 1. Forward simulation
    df = generate_synthetic_transient_response(res, well, frac, n_points=160)
    times = df["time_hr"].values
    dp = df["delta_P_psi"].values
    
    # 2. Derivative & Slopes
    derv = calculate_bourdet_derivative(times, dp, L=0.15)
    slopes = calculate_log_slopes(times, derv, L=0.20)
    df["derivative_psi"] = derv
    df["log_slope"] = slopes
    
    # 3. Flow Regime Detection
    regimes = detect_sustained_flow_regimes(times, dp, derv, slopes)
    
    # 4. Inverse Estimation & Validation
    validations, meta = run_inverse_characterization(times, dp, derv, regimes, res, well, frac)
    
    # 5. Dominant Regime & Narrative Interpretation
    Cf = frac.Cf
    CfD = frac.CfD(res.k)
    _, cond_desc = classify_fracture_conductivity_type(CfD)
    
    if regimes:
        # Pick dominant non-WBS regime if available
        non_wbs = [r for r in regimes if r.regime != "Wellbore Storage"]
        dominant_reg = non_wbs[0].regime if non_wbs else regimes[0].regime
    else:
        dominant_reg = "Transient Transition"
        
    interpretation_text = (
        f"The simulated pressure transient exhibits distinct signature transitions for a fractured well with "
        f"k = {res.k:g} mD and half-length xf = {frac.xf:g} ft. Dimensionless fracture conductivity is CfD = {CfD:.3g}. "
        f"{cond_desc} The Bourdet derivative reflects {dominant_reg.lower()} during the intermediate test duration, "
        f"confirming theoretical pressure diffusion characteristics."
    )
    
    return PTAAnalysisResult(
        case_name=case_data["name"],
        reservoir=res,
        well=well,
        fracture=frac,
        df_transient=df,
        flow_regimes=regimes,
        validations=validations,
        engineering_interpretation=interpretation_text,
        dominant_regime=dominant_reg,
        Cf=Cf,
        CfD=CfD
    )


# ==============================================================================
# PAGE 1: LANDING PAGE
# ==============================================================================
def render_landing_page():
    # Header
    st.markdown('<div class="brand-title">HydraPTA</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-tagline">Hydraulic Fracture Pressure Transient Analyzer</div>', unsafe_allow_html=True)
    
    st.markdown("""
    > **From pressure response to fracture insight.**
    
    Explore how hydraulic-fracture properties influence pressure-transient behavior, diagnostic signatures, 
    flow regimes, and fracture characterization. Built with analytical petroleum engineering formulations.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="pta-card">
            <div class="pta-card-header">MODE 01 — FULLY FUNCTIONAL</div>
            <h3 style="margin-top:0.3rem; margin-bottom:0.6rem;">Demo Cases</h3>
            <p style="color:#4A5568; font-size:0.95rem; line-height:1.5;">
                Explore predefined fractured-well scenarios and understand how fracture properties, 
                conductivity, and length influence pressure-transient signatures and flow regimes.
            </p>
            <ul style="color:#4A5568; font-size:0.9rem; padding-left:1.2rem;">
                <li>High-Conductivity Fracture (Linear Flow Focus)</li>
                <li>Finite-Conductivity Fracture (Bilinear Flow Focus)</li>
                <li>Long-Fracture Penetration (Extended Linear Transient)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Demo Cases →", type="primary", use_container_width=True):
            navigate_to("case_selection")
            
    with col2:
        st.markdown("""
        <div class="pta-card" style="background-color: #FAFAFA;">
            <div class="pta-card-header" style="color:#718096;">
                MODE 02 — <span class="badge-soon">COMING SOON</span>
            </div>
            <h3 style="margin-top:0.3rem; margin-bottom:0.6rem; color:#718096;">User Input</h3>
            <p style="color:#718096; font-size:0.95rem; line-height:1.5;">
                Analyze your own pressure-transient data, custom reservoir configurations, 
                and upload well-test gauge CSV measurements for automated diagnostic processing.
            </p>
            <ul style="color:#A0AEC0; font-size:0.9rem; padding-left:1.2rem;">
                <li>Custom petrophysical & fracture property input</li>
                <li>Well test pressure gauge CSV upload & filtering</li>
                <li>Automated type-curve matching & inverse characterization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.button("User Input (Coming Soon)", disabled=True, use_container_width=True)
        
    st.markdown("---")
    st.caption("HydraPTA is an educational analytical tool designed for petroleum engineering analysis and technical interviews.")


# ==============================================================================
# PAGE 2: DEMO CASE SELECTION
# ==============================================================================
def render_case_selection_page():
    col_back, col_title = st.columns([1, 6])
    with col_back:
        if st.button("← Home"):
            navigate_to("home")
    with col_title:
        st.subheader("Select Educational Demo Scenario")
        
    st.markdown(
        "Select one of the predefined fractured-well test cases below to inspect its parameters, "
        "simulate the pressure-transient response, analyze diagnostic slopes, and perform inverse fracture characterization."
    )
    
    st.info("ℹ️ **Note**: Illustrative synthetic values for educational purposes. Not field data.")
    
    cases = list_demo_cases()
    cols = st.columns(len(cases))
    
    for idx, c in enumerate(cases):
        with cols[idx]:
            st.markdown(f"""
            <div class="pta-card" style="height: 380px; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div class="pta-card-header">{c['case_number']}</div>
                    <h4 style="margin-top:0.2rem; margin-bottom:0.5rem;">{c['name']}</h4>
                    <p style="color:#4A5568; font-size:0.88rem; line-height:1.4;">
                        {c['short_description']}
                    </p>
                    <div style="margin-top:0.8rem; margin-bottom:0.8rem;">
                        <span style="font-size:0.78rem; font-weight:700; color:#718096;">KEY PARAMETER:</span><br>
                        <span class="metric-badge" style="font-size:0.82rem; margin-top:2px;">{c['demonstrated_parameter']}</span>
                    </div>
                    <div>
                        <span style="font-size:0.78rem; font-weight:700; color:#718096;">EDUCATIONAL OBJECTIVE:</span>
                        <p style="color:#4A5568; font-size:0.82rem; line-height:1.35; margin-top:2px;">
                            {c['educational_objective']}
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Select {c['case_number']} →", key=f"btn_{c['id']}", use_container_width=True, type="primary" if idx==0 else "secondary"):
                navigate_to("case_review", c["id"])


# ==============================================================================
# PAGE 3: INPUT REVIEW & RUN ANALYSIS
# ==============================================================================
def render_case_review_page():
    case_id = st.session_state["selected_case_id"]
    case_data = get_demo_case(case_id)
    res: ReservoirProperties = case_data["reservoir"]
    well: WellProperties = case_data["well"]
    frac: FractureProperties = case_data["fracture"]
    
    col_nav1, col_nav2 = st.columns([1, 6])
    with col_nav1:
        if st.button("← Cases"):
            navigate_to("case_selection")
    with col_nav2:
        st.subheader(f"{case_data['case_number']} — {case_data['name']}")
        
    st.markdown(f"*{case_data['educational_objective']}*")
    
    st.markdown('<span class="badge-locked">🔒 Predefined Educational Case — Inputs Locked</span>', unsafe_allow_html=True)
    st.caption("Illustrative synthetic values for educational purposes. Not field data.")
    
    st.markdown("---")
    
    # 3-Column Parameter Grouping
    col_res, col_well, col_frac = st.columns(3)
    
    with col_res:
        st.markdown("##### 🪨 Reservoir Properties")
        df_res = pd.DataFrame([
            {"Parameter": "Permeability (k)", "Value": f"{res.k:g}", "Unit": "mD"},
            {"Parameter": "Porosity (φ)", "Value": f"{res.phi:g}", "Unit": "fraction"},
            {"Parameter": "Viscosity (μ)", "Value": f"{res.mu:g}", "Unit": "cp"},
            {"Parameter": "Total Compressibility (ct)", "Value": f"{res.ct:e}", "Unit": "psi⁻¹"},
            {"Parameter": "Net Thickness (h)", "Value": f"{res.h:g}", "Unit": "ft"},
            {"Parameter": "FVF (B)", "Value": f"{res.B:g}", "Unit": "rb/STB"},
        ])
        st.dataframe(df_res, hide_index=True, use_container_width=True)
        
    with col_well:
        st.markdown("##### ⛽ Well & Test Properties")
        df_well = pd.DataFrame([
            {"Parameter": "Flow Rate (q)", "Value": f"{well.q:g}", "Unit": "STB/day"},
            {"Parameter": "Initial Pressure (Pi)", "Value": f"{well.Pi:g}", "Unit": "psi"},
            {"Parameter": "Wellbore Radius (rw)", "Value": f"{well.rw:g}", "Unit": "ft"},
            {"Parameter": "Wellbore Storage (C)", "Value": f"{well.C:g}", "Unit": "bbl/psi"},
            {"Parameter": "Test Duration", "Value": f"{well.duration:g}", "Unit": "hr"},
        ])
        st.dataframe(df_well, hide_index=True, use_container_width=True)
        
    with col_frac:
        st.markdown("##### ⚡ Fracture Properties")
        df_frac = pd.DataFrame([
            {"Parameter": "Half-Length (xf)", "Value": f"{frac.xf:g}", "Unit": "ft"},
            {"Parameter": "Fracture Permeability (kf)", "Value": f"{frac.kf:g}", "Unit": "mD"},
            {"Parameter": "Fracture Width (wf)", "Value": f"{frac.wf:g}", "Unit": "ft"},
        ])
        st.dataframe(df_frac, hide_index=True, use_container_width=True)
        
    st.markdown("---")
    
    # Automatically Calculated Fracture Metrics
    st.markdown("##### 📐 Derived Fracture Metrics")
    Cf = frac.Cf
    CfD = frac.CfD(res.k)
    
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric(
            label="Fracture Conductivity (Cf = kf × wf)",
            value=f"{Cf:g} mD-ft",
            help="Product of fracture permeability and fracture width."
        )
    with m_col2:
        st.metric(
            label="Dimensionless Conductivity (CfD)",
            value=f"{CfD:.4g}",
            help="CfD = (kf * wf) / (k * xf). Determines whether flow is bilinear or formation linear."
        )
    with m_col3:
        cat_name, _ = classify_fracture_conductivity_type(CfD)
        st.markdown(f"**Conductivity Classification:**")
        st.markdown(f'<span class="metric-badge" style="color:#E8752A;">{cat_name}</span>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Execution Button
    if st.button("🚀 Run PTA Analysis", type="primary", use_container_width=True):
        progress_placeholder = st.empty()
        status_text = st.empty()
        
        stages = [
            "Initializing reservoir model",
            "Generating pressure response",
            "Calculating pressure derivative",
            "Analyzing diagnostic signature",
            "Identifying flow regimes",
            "Estimating fracture properties",
            "Preparing sensitivity analysis"
        ]
        
        for i, stage in enumerate(stages):
            pct = int(((i + 1) / len(stages)) * 100)
            progress_placeholder.progress(pct)
            status_text.text(f"Status: {stage}...")
            time.sleep(0.06)
            
        progress_placeholder.empty()
        status_text.empty()
        
        # Store result in state
        result = execute_pta_pipeline(case_id)
        st.session_state["analysis_result"] = result
        navigate_to("results")


# ==============================================================================
# PAGE 4: RESULTS DASHBOARD
# ==============================================================================
def render_results_page():
    result: PTAAnalysisResult = st.session_state["analysis_result"]
    if result is None:
        case_id = st.session_state.get("selected_case_id", "case_1")
        result = execute_pta_pipeline(case_id)
        st.session_state["analysis_result"] = result
        
    case_id = st.session_state["selected_case_id"]
    
    # Navigation & Header
    col_back, col_title = st.columns([1, 6])
    with col_back:
        if st.button("← Cases"):
            navigate_to("case_selection")
    with col_title:
        st.markdown(f"### PTA Results & Fracture Diagnostics — {result.case_name}")
        
    # Top Compact Summary Bar
    st.markdown(f"""
    <div class="summary-bar">
        <div style="display:flex; flex-wrap:wrap; gap:1.8rem; align-items:center;">
            <div>
                <span style="font-size:0.75rem; color:#718096; text-transform:uppercase; font-weight:700;">Case</span><br>
                <strong style="color:#252525; font-size:1.05rem;">{result.case_name}</strong>
            </div>
            <div>
                <span style="font-size:0.75rem; color:#718096; text-transform:uppercase; font-weight:700;">Reservoir Perm (k)</span><br>
                <strong style="color:#252525; font-size:1.05rem;">{result.reservoir.k:g} mD</strong>
            </div>
            <div>
                <span style="font-size:0.75rem; color:#718096; text-transform:uppercase; font-weight:700;">Half-Length (xf)</span><br>
                <strong style="color:#252525; font-size:1.05rem;">{result.fracture.xf:g} ft</strong>
            </div>
            <div>
                <span style="font-size:0.75rem; color:#718096; text-transform:uppercase; font-weight:700;">Fracture Cf</span><br>
                <strong style="color:#252525; font-size:1.05rem;">{result.Cf:g} mD-ft</strong>
            </div>
            <div>
                <span style="font-size:0.75rem; color:#718096; text-transform:uppercase; font-weight:700;">Dimensionless CfD</span><br>
                <strong style="color:#252525; font-size:1.05rem;">{result.CfD:.3g}</strong>
            </div>
            <div>
                <span style="font-size:0.75rem; color:#718096; text-transform:uppercase; font-weight:700;">Dominant Flow Regime</span><br>
                <strong style="color:#E8752A; font-size:1.05rem;">{result.dominant_regime}</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # -------------------------------------------------------------------------
    # 1. OVERVIEW & ENGINEERING INTERPRETATION
    # -------------------------------------------------------------------------
    st.markdown("#### 1. Overview & Engineering Interpretation")
    st.info(f"📋 **Analysis Summary**: {result.engineering_interpretation}")
    
    st.markdown("---")
    
    # -------------------------------------------------------------------------
    # 2. PTA DIAGNOSTICS & PRESSURE CURVES
    # -------------------------------------------------------------------------
    st.markdown("#### 2. PTA Diagnostic Plots")
    
    times = result.df_transient["time_hr"].values
    dp = result.df_transient["delta_P_psi"].values
    derv = result.df_transient["derivative_psi"].values
    pwf = result.df_transient["Pwf_psi"].values
    
    # Central Diagnostic Plot
    fig_diag = create_log_log_diagnostic_plot(
        times=times,
        delta_p=dp,
        derivative=derv,
        detected_regimes=result.flow_regimes,
        case_name=result.case_name
    )
    st.plotly_chart(fig_diag, use_container_width=True)
    
    # Secondary Pressure Curves (Tabs)
    tab1, tab2 = st.tabs(["📉 Bottomhole Pressure History Pwf(t)", "📊 Pressure Drawdown ΔP(t)"])
    with tab1:
        fig_pwf = create_pressure_history_plot(times, pwf, result.well.Pi, result.case_name)
        st.plotly_chart(fig_pwf, use_container_width=True)
    with tab2:
        fig_dp = create_delta_p_plot(times, dp, result.case_name)
        st.plotly_chart(fig_dp, use_container_width=True)
        
    st.markdown("---")
    
    # -------------------------------------------------------------------------
    # 3. FLOW REGIME IDENTIFICATION
    # -------------------------------------------------------------------------
    st.markdown("#### 3. Flow-Regime Identification & Slope Diagnostics")
    st.markdown(
        "Local log-log derivative slopes $m = \\frac{d\\log \\Delta P'}{d\\log t}$ were continuously evaluated. "
        "The following sustained flow regimes were identified from diagnostic slope signatures:"
    )
    
    df_regimes = generate_flow_regime_summary_table(result.flow_regimes)
    st.dataframe(df_regimes, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # -------------------------------------------------------------------------
    # 4. FRACTURE CHARACTERIZATION & SYNTHETIC VALIDATION
    # -------------------------------------------------------------------------
    st.markdown("#### 4. Fracture Characterization & Synthetic-Data Validation")
    st.markdown("""
    The synthetic transient response was analyzed through specialized linear/bilinear flow regressions and 
    non-linear derivative matching **without reading the original input parameters**. The recovered properties 
    are validated against true values below:
    """)
    
    val_rows = []
    for v in result.validations:
        est_str = f"{v.estimated_value:g} {v.unit}" if v.estimated_value is not None else "Not reliably estimated"
        abs_err_str = f"{v.absolute_error:g} {v.unit}" if v.absolute_error is not None else "—"
        rel_err_str = f"{v.relative_error:.2f}%" if v.relative_error is not None else "—"
        status_icon = "✅ Reliable" if v.is_reliable else "⚠️ Limitation"
        
        val_rows.append({
            "Parameter": v.parameter,
            "True Value": f"{v.true_value:g} {v.unit}",
            "Estimated Value": est_str,
            "Absolute Error": abs_err_str,
            "Relative Error": rel_err_str,
            "Model Status": status_icon,
            "Diagnostic Notes": v.status_message
        })
        
    df_val = pd.DataFrame(val_rows)
    st.dataframe(df_val, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # -------------------------------------------------------------------------
    # 5. PARAMETER SENSITIVITY ANALYSIS
    # -------------------------------------------------------------------------
    st.markdown("#### 5. Parameter Sensitivity Analysis")
    st.markdown(
        "Investigate how varying individual reservoir, well, or fracture properties dynamically alters the "
        "drawdown curves and derivative flow-regime signatures."
    )
    
    param_options = {k: v["label"] for k, v in SENSITIVITY_PARAMETERS.items()}
    selected_param_key = st.selectbox(
        "Select Sensitivity Parameter:",
        options=list(param_options.keys()),
        format_func=lambda x: param_options[x],
        index=0
    )
    
    param_meta = SENSITIVITY_PARAMETERS[selected_param_key]
    
    # Run sensitivity simulation
    df_sens_curves, df_sens_summary, sens_insight = generate_sensitivity_scenarios(
        param_key=selected_param_key,
        base_reservoir=result.reservoir,
        base_well=result.well,
        base_fracture=result.fracture
    )
    
    col_sens1, col_sens2 = st.columns(2)
    with col_sens1:
        fig_sens_dp = create_sensitivity_comparison_plot(
            df_sens_curves, param_meta["label"], y_variable="delta_P_psi"
        )
        st.plotly_chart(fig_sens_dp, use_container_width=True)
    with col_sens2:
        fig_sens_derv = create_sensitivity_comparison_plot(
            df_sens_curves, param_meta["label"], y_variable="derivative_psi"
        )
        st.plotly_chart(fig_sens_derv, use_container_width=True)
        
    st.markdown("##### Sensitivity Summary Table")
    st.dataframe(df_sens_summary, hide_index=True, use_container_width=True)
    
    st.markdown(f"##### 💡 Engineering Interpretation\n{sens_insight}")
    
    st.markdown("---")
    
    # -------------------------------------------------------------------------
    # 6. EDUCATIONAL EXPANDERS
    # -------------------------------------------------------------------------
    st.markdown("#### 6. Educational Guide & PTA Theory")
    
    with st.expander("❓ What am I looking at?"):
        st.markdown("""
        You are looking at a **Log-Log Pressure Transient Diagnostic Plot** (also called a Bourdet Diagnostic Plot). 
        The top curve represents the total reservoir pressure drawdown $\\Delta P = P_i - P_{wf}(t)$, and the bottom 
        curve represents the logarithmic pressure derivative $\\Delta P' = \\frac{d\\Delta P}{d\\ln t} = t \\frac{d\\Delta P}{dt}$.
        The slope of the derivative curve acts as a fingerprint for specific reservoir and fracture flow regimes.
        """)
        
    with st.expander("❓ Why does the pressure derivative matter?"):
        st.markdown("""
        Pressure drawdown $\\Delta P$ is an integrated cumulative response that smooths out subtle flow changes. 
        The logarithmic derivative $\\Delta P'$ isolates instantaneous rate of pressure change with log-time, 
        amplifying distinct geometric flow regimes (wellbore storage, bilinear, linear, radial) into characteristic straight lines.
        """)
        
    with st.expander("❓ What does a 1/2 slope indicate?"):
        st.markdown("""
        A **half-slope ($m = 0.5$)** indicates **Formation Linear Flow**. Fluid moves linearly through the formation 
        perpendicular to the planar fracture faces. This regime is prominent in high-conductivity fractures ($C_{fD} \\ge 10$) 
        and allows direct determination of fracture half-length $x_f$.
        """)
        
    with st.expander("❓ What is bilinear flow (1/4 slope)?"):
        st.markdown("""
        **Bilinear flow ($m = 0.25$)** occurs in finite-conductivity fractures ($C_{fD} < 10$). Two simultaneous linear flow 
        processes occur: linear flow within the fracture toward the wellbore, and linear flow from the formation into 
        the fracture. The slope is inversely proportional to $\\sqrt{k_f w_f}$.
        """)
        
    with st.expander("❓ What is dimensionless fracture conductivity (CfD)?"):
        st.markdown("""
        $$C_{fD} = \\frac{k_f w_f}{k x_f}$$
        It measures the relative flow capacity of the fracture compared to the reservoir matrix feeding it. 
        - $C_{fD} \\ge 100$: Infinite conductivity (matrix flow controls production).
        - $0.1 \\le C_{fD} < 10$: Finite conductivity (fracture choke restricts transient flow).
        - $C_{fD} < 0.1$: Severe fracture choking.
        """)
        
    with st.expander("❓ What is pseudoradial flow (0 slope)?"):
        st.markdown("""
        At late times, the radius of investigation expands far beyond the fracture half-length $x_f$. 
        The fracture then appears as an expanded wellbore with effective radius $r_w' = x_f / 2$. 
        The derivative becomes flat ($m = 0$), allowing estimation of formation permeability $k$.
        """)
        
    st.markdown("---")
    
    # -------------------------------------------------------------------------
    # 7. EXPORT DATA
    # -------------------------------------------------------------------------
    st.markdown("#### 7. Export Analysis Data")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        csv_transient = result.df_transient.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Pressure Transient Data (CSV)",
            data=csv_transient,
            file_name=f"HydraPTA_{case_id}_transient.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_exp2:
        csv_sens = df_sens_curves.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download Sensitivity Data [{param_meta['label']}] (CSV)",
            data=csv_sens,
            file_name=f"HydraPTA_sensitivity_{selected_param_key}.csv",
            mime="text/csv",
            use_container_width=True
        )


# --- Application Main Router ---
def main():
    page = st.session_state["current_page"]
    if page == "home":
        render_landing_page()
    elif page == "case_selection":
        render_case_selection_page()
    elif page == "case_review":
        render_case_review_page()
    elif page == "results":
        render_results_page()
    else:
        render_landing_page()


if __name__ == "__main__":
    main()
