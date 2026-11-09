"""
User Input Mode (Mode 2) for HydraPTA.
Provides complete interactive workflows for Drawdown Analysis and Buildup Analysis,
including file upload, column mapping, property inputs (Field & SI), diagnostic plotting,
automatic flow-regime & stabilization recognition, parameter estimation, and PDF report export.
"""

from typing import Dict, Any, List, Optional, Tuple
import io
import numpy as np
import pandas as pd
import streamlit as pd_st
import streamlit as st

from .units import UNIT_DEFINITIONS, convert_si_to_field, convert_field_to_si
from .data_processing import (
    load_dataset,
    auto_detect_columns,
    validate_and_preprocess_drawdown
)
from .derivative import (
    calculate_bourdet_derivative,
    calculate_log_slopes,
    compare_bourdet_derivatives
)
from .flow_regime import (
    detect_sustained_flow_regimes,
    detect_stabilization_plateau,
    identify_flow_transitions,
    analyze_regime_sequence,
    generate_flow_regime_summary_table
)
from .property_estimation import (
    estimate_permeability_semilog,
    estimate_permeability_derivative_plateau,
    estimate_skin_factor,
    estimate_fracture_half_length,
    estimate_fracture_conductivity,
    perform_full_drawdown_estimation
)
from .dimensionless import (
    calculate_dimensionless_variables,
    generate_theoretical_radial_type_curve
)
from .buildup import (
    calculate_buildup_variables,
    estimate_buildup_horner_parameters
)
from .plotting import (
    create_log_log_diagnostic_plot,
    create_pressure_history_plot,
    create_delta_p_plot,
    create_semilog_analysis_plot,
    create_dimensionless_diagnostic_plot,
    create_smoothing_comparison_plot,
    create_horner_plot,
    create_buildup_diagnostic_plot
)
from .pdf_report import generate_pta_pdf_report


def render_drawdown_mode():
    """Render complete Drawdown Analysis section."""
    st.markdown("### 📉 Pressure Drawdown Analysis")
    st.markdown(
        "Upload flowing bottom-hole pressure gauge measurements, configure reservoir and fluid parameters, "
        "and perform automated pressure derivative diagnostics and property estimation."
    )
    
    # -------------------------------------------------------------------------
    # SECTION 1: DATA INGESTION & COLUMN MAPPING
    # -------------------------------------------------------------------------
    st.markdown("#### Section 1: Upload Pressure-Time Dataset")
    
    col_up1, col_up2 = st.columns([3, 2])
    with col_up1:
        uploaded_file = st.file_uploader(
            "Upload Well-Test Data (CSV or Excel .xlsx)",
            type=["csv", "xlsx", "xls"],
            key="drawdown_file_uploader",
            help="File must contain at least time and flowing pressure (Pwf / FBHP)."
        )
    with col_up2:
        st.markdown("**Quick Test with Demo Data:**")
        if st.button("📥 Load Sample Drawdown Dataset", use_container_width=True):
            try:
                st.session_state["raw_df_drawdown"] = pd.read_csv("data/sample_drawdown.csv")
                st.session_state["drawdown_filename"] = "sample_drawdown.csv"
                st.success("Loaded sample drawdown gauge dataset (120 pts).")
            except Exception as e:
                st.error(f"Could not load sample file: {e}")
                
    if uploaded_file is not None:
        try:
            df_uploaded = load_dataset(uploaded_file, uploaded_file.name)
            st.session_state["raw_df_drawdown"] = df_uploaded
            st.session_state["drawdown_filename"] = uploaded_file.name
        except Exception as e:
            st.error(f"Error parsing file: {e}")
            return

    raw_df = st.session_state.get("raw_df_drawdown", None)
    if raw_df is None:
        st.info("👆 Please upload an Excel or CSV dataset, or click 'Load Sample Drawdown Dataset' to proceed.")
        return

    # Automatic Column Recognition with Manual Fallback
    auto_cols = auto_detect_columns(raw_df, analysis_mode="drawdown")
    col_list = list(raw_df.columns)
    
    st.markdown("##### Column Mapping")
    col_map1, col_map2, col_map3, col_map4, col_map5 = st.columns(5)
    
    with col_map1:
        default_t_idx = col_list.index(auto_cols["time"]) if auto_cols["time"] in col_list else 0
        time_col = st.selectbox("Time Column *", options=col_list, index=default_t_idx, key="dd_t_col")
    with col_map2:
        default_p_idx = col_list.index(auto_cols["pressure"]) if auto_cols["pressure"] in col_list else min(1, len(col_list)-1)
        pwf_col = st.selectbox("Flowing Pressure (Pwf) *", options=col_list, index=default_p_idx, key="dd_p_col")
    with col_map3:
        q_options = ["None (Use Panel Input)"] + col_list
        default_q_idx = q_options.index(auto_cols["q"]) if auto_cols["q"] in q_options else 0
        q_col_sel = st.selectbox("Rate (q) [Optional]", options=q_options, index=default_q_idx, key="dd_q_col")
    with col_map4:
        bo_options = ["None (Use Panel Input)"] + col_list
        default_bo_idx = bo_options.index(auto_cols["bo"]) if auto_cols["bo"] in bo_options else 0
        bo_col_sel = st.selectbox("Bo [Optional]", options=bo_options, index=default_bo_idx, key="dd_bo_col")
    with col_map5:
        mu_options = ["None (Use Panel Input)"] + col_list
        default_mu_idx = mu_options.index(auto_cols["mu"]) if auto_cols["mu"] in mu_options else 0
        mu_col_sel = st.selectbox("Viscosity (μ) [Optional]", options=mu_options, index=default_mu_idx, key="dd_mu_col")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION 2: RESERVOIR AND FLUID PROPERTIES PANEL
    # -------------------------------------------------------------------------
    st.markdown("#### Section 2: Reservoir, Fluid & Well Parameters")
    
    unit_system = st.radio("Unit System:", options=["Field", "SI"], horizontal=True, key="dd_unit_sys")
    unit_dict = UNIT_DEFINITIONS[unit_system]
    
    c_p1, c_p2, c_p3, c_p4 = st.columns(4)
    with c_p1:
        pi_val = st.number_input(f"Initial Pressure Pi ({unit_dict['pressure']}) *", value=4000.0 if unit_system=="Field" else 275.8, min_value=1.0, step=50.0)
        q_val = st.number_input(f"Production Rate q ({unit_dict['rate']}) *", value=500.0 if unit_system=="Field" else 79.5, min_value=0.1, step=25.0)
    with c_p2:
        bo_val = st.number_input(f"Formation Volume Factor Bo ({unit_dict['fvf']}) *", value=1.20, min_value=0.5, max_value=3.0, step=0.05)
        mu_val = st.number_input(f"Oil Viscosity μ ({unit_dict['viscosity']}) *", value=1.20, min_value=0.05, max_value=100.0, step=0.1)
    with c_p3:
        h_val = st.number_input(f"Net Pay Thickness h ({unit_dict['thickness']}) *", value=50.0 if unit_system=="Field" else 15.24, min_value=1.0, step=5.0)
        phi_val = st.number_input("Porosity φ (fraction) *", value=0.12, min_value=0.01, max_value=0.45, step=0.01)
    with c_p4:
        ct_val = st.number_input(f"Total Compressibility ct ({unit_dict['compressibility']}) *", value=1.5e-5 if unit_system=="Field" else 2.17e-4, format="%.2e", min_value=1e-8)
        rw_val = st.number_input(f"Wellbore Radius rw ({unit_dict['radius']}) *", value=0.25 if unit_system=="Field" else 0.0762, min_value=0.05, step=0.05)

    # Convert to Field internally if SI was selected
    raw_params = {
        "Pi": pi_val, "q": q_val, "Bo": bo_val, "mu": mu_val,
        "h": h_val, "phi": phi_val, "ct": ct_val, "rw": rw_val
    }
    field_params = convert_si_to_field(raw_params) if unit_system == "SI" else raw_params

    # Optional known prior properties (e.g. k, S, xf)
    with st.expander("⚙️ Optional Prior Estimates (leave blank to let HydraPTA estimate)"):
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            k_prior_input = st.text_input("Prior Permeability k (mD) [Optional]", value="", help="Leave empty if unknown.")
        with col_opt2:
            s_prior_input = st.text_input("Prior Skin S [Optional]", value="", help="Leave empty if unknown.")
        with col_opt3:
            xf_prior_input = st.text_input("Prior Half-Length xf (ft) [Optional]", value="", help="Leave empty if unknown.")
            
        k_prior = float(k_prior_input) if k_prior_input.strip() else None
        s_prior = float(s_prior_input) if s_prior_input.strip() else None
        xf_prior = float(xf_prior_input) if xf_prior_input.strip() else None
        if k_prior: field_params["k"] = k_prior
        if s_prior: field_params["S"] = s_prior
        if xf_prior: field_params["xf"] = xf_prior

    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION 3: PREVIEW & DATA VALIDATION
    # -------------------------------------------------------------------------
    st.markdown("#### Section 3: Data Validation & Processed Drawdown Preview")
    
    q_col_name = q_col_sel if q_col_sel != "None (Use Panel Input)" else None
    bo_col_name = bo_col_sel if bo_col_sel != "None (Use Panel Input)" else None
    mu_col_name = mu_col_sel if mu_col_sel != "None (Use Panel Input)" else None

    df_processed, report = validate_and_preprocess_drawdown(
        df_raw=raw_df,
        time_col=time_col,
        pwf_col=pwf_col,
        Pi=field_params["Pi"],
        q_col=q_col_name,
        bo_col=bo_col_name,
        mu_col=mu_col_name,
        unit_system=unit_system
    )

    # Validation feedback badges
    if not report["is_valid"]:
        for err in report["errors"]:
            st.error(f"❌ {err}")
        return
        
    v_col1, v_col2, v_col3, v_col4 = st.columns(4)
    with v_col1:
        st.metric("Total Rows Processed", f"{report['row_count_processed']} / {report['row_count_original']}")
    with v_col2:
        st.metric("Time Span", f"{report['time_min']:.3g} – {report['time_max']:.3g} hr")
    with v_col3:
        st.metric("Pwf Range", f"{report['pwf_min']:.1f} – {report['pwf_max']:.1f} psi")
    with v_col4:
        st.metric("Max Drawdown ΔP", f"{report['delta_p_max']:.1f} psi")

    for warn in report["warnings"]:
        st.warning(f"⚠️ {warn}")

    with st.expander("👁️ View Processed Drawdown Table (Time, Pwf, ΔP)", expanded=False):
        st.dataframe(df_processed.head(100), use_container_width=True)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION 4: DERIVATIVE SETTINGS & MULTI-L INVESTIGATION
    # -------------------------------------------------------------------------
    st.markdown("#### Section 4: Bourdet Pressure Derivative Settings")
    c_d1, c_d2 = st.columns([2, 3])
    with c_d1:
        L_param = st.slider(
            "Bourdet Differentiation Distance (L):",
            min_value=0.05,
            max_value=0.50,
            value=0.15,
            step=0.05,
            help="Higher L values smooth high-frequency gauge noise; lower L values preserve subtle slope inflections."
        )
    with c_d2:
        enable_multi_L = st.checkbox("Enable Multi-L Smoothing Comparison Plot", value=False)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION 5: RUN PTA ANALYSIS BUTTON
    # -------------------------------------------------------------------------
    if st.button("🚀 Run Pressure Drawdown Analysis", type="primary", use_container_width=True):
        times = df_processed["time_hr"].values
        dp = df_processed["delta_p_psi"].values
        pwf = df_processed["pwf_psi"].values
        
        # 1. Bourdet derivative & local slopes
        derivative = calculate_bourdet_derivative(times, dp, L=L_param)
        slopes = calculate_log_slopes(times, derivative, L=L_param + 0.05)
        df_processed["derivative_psi"] = derivative
        df_processed["log_slope"] = slopes
        
        # 2. Flow regime detection
        regimes = detect_sustained_flow_regimes(times, dp, derivative, slopes)
        
        # 3. Stabilization plateau detection
        plateau = detect_stabilization_plateau(times, derivative, slopes)
        
        # 4. Property estimation
        est_results = perform_full_drawdown_estimation(
            times=times,
            delta_p=dp,
            derivative=derivative,
            plateau_info=plateau,
            detected_regimes=regimes,
            reservoir_params=field_params
        )
        
        # 5. Dimensionless variables
        k_for_dim = est_results["best_k"] or field_params.get("k", 5.0)
        xf_for_dim = est_results["xf"]["xf_ft"] if est_results.get("xf", {}).get("is_valid") else field_params.get("xf", 300.0)
        
        df_dim = calculate_dimensionless_variables(
            times=times,
            delta_p=dp,
            derivative=derivative,
            k=k_for_dim,
            h=field_params["h"],
            q=field_params["q"],
            Bo=field_params["Bo"],
            mu=field_params["mu"],
            phi=field_params["phi"],
            ct=field_params["ct"],
            rw=field_params["rw"],
            xf=xf_for_dim
        )
        
        # Save to session state
        st.session_state["dd_results"] = {
            "df_processed": df_processed,
            "df_dim": df_dim,
            "regimes": regimes,
            "plateau": plateau,
            "est_results": est_results,
            "field_params": field_params,
            "L_param": L_param,
            "enable_multi_L": enable_multi_L,
            "unit_system": unit_system
        }
        st.success("PTA Analysis complete! Review results below.")

    # -------------------------------------------------------------------------
    # SECTION 6 & 7 & 8: RESULTS DASHBOARD
    # -------------------------------------------------------------------------
    if "dd_results" in st.session_state:
        res = st.session_state["dd_results"]
        df_p = res["df_processed"]
        df_dim = res["df_dim"]
        regimes = res["regimes"]
        plateau = res["plateau"]
        est = res["est_results"]
        f_params = res["field_params"]
        times = df_p["time_hr"].values
        dp = df_p["delta_p_psi"].values
        derv = df_p["derivative_psi"].values

        st.markdown("### 📊 PTA Diagnostic Results & Interpretations")

        # Top Metric Cards
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            k_semi = est.get("k_semilog", {})
            k_val_str = f"{k_semi['k_mD']:g} mD" if k_semi.get("is_valid") else "Unresolved"
            st.metric("Permeability (Semilog)", k_val_str, help="Estimated from Delta P vs log10(t) slope during pseudoradial flow.")
        with m_c2:
            k_plat = est.get("k_plateau", {})
            k_plat_str = f"{k_plat['k_mD']:g} mD" if k_plat.get("is_valid") else "Unresolved"
            st.metric("Permeability (Plateau)", k_plat_str, help="Estimated from Delta P'_plateau = 70.6 q Bo mu / (k h).")
        with m_c3:
            skin_res = est.get("skin", {})
            skin_val_str = f"{skin_res['skin']:+.2f}" if skin_res.get("is_valid") else "Unresolved"
            st.metric("Skin Factor (S)", skin_val_str, help="Wellbore skin factor calculated from semilog intercept.")
        with m_c4:
            xf_res = est.get("xf", {})
            cf_res = est.get("Cf", {})
            if xf_res.get("is_valid"):
                st.metric("Half-Length (xf)", f"{xf_res['xf_ft']:g} ft", help="From linear flow slope m_LF.")
            elif cf_res.get("is_valid"):
                st.metric("Conductivity (Cf)", f"{cf_res['Cf_mD_ft']:g} mD-ft", help="From bilinear flow slope m_BL.")
            else:
                st.metric("Fracture Parameter", "Unconstrained", help="Neither linear nor bilinear flow was clearly detected.")

        if est.get("k_diff_pct") is not None:
            st.info(f"💡 **Permeability Discrepancy**: Semilog k = {est['k_semilog']['k_mD']} mD vs Derivative Plateau k = {est['k_plateau']['k_mD']} mD (Difference: **{est['k_diff_pct']:.1f}%**).")

        st.markdown("---")

        # ---------------------------------------------------------------------
        # DIAGNOSTIC PLOTS
        # ---------------------------------------------------------------------
        st.markdown("#### Diagnostic Plots")
        tab_loglog, tab_dim, tab_semilog, tab_smoothing = st.tabs([
            "📈 Log-Log Diagnostic Plot",
            "📐 Dimensionless Plot (pD & p'D)",
            "📏 Semilog Radial Analysis",
            "🔍 Multi-L Smoothing Sensitivity"
        ])

        with tab_loglog:
            fig_diag = create_log_log_diagnostic_plot(
                times=times,
                delta_p=dp,
                derivative=derv,
                detected_regimes=regimes,
                case_name="Uploaded Well-Test Drawdown"
            )
            st.plotly_chart(fig_diag, use_container_width=True)

        with tab_dim:
            fig_dim = create_dimensionless_diagnostic_plot(
                df_dim=df_dim,
                case_name="Uploaded Well-Test Drawdown",
                include_theoretical=True
            )
            st.plotly_chart(fig_dim, use_container_width=True)

        with tab_semilog:
            # Interactive manual interval override controls for semilog fit
            st.markdown("##### Semilog Regression Interval Selection")
            rad_reg = next((r for r in regimes if r.regime == "Pseudoradial Flow"), None)
            default_t1 = rad_reg.t_start if rad_reg else float(times[len(times)//2])
            default_t2 = rad_reg.t_end if rad_reg else float(times[-1])
            
            c_sl1, c_sl2 = st.columns(2)
            with c_sl1:
                manual_t1 = st.number_input("Regression Start Time (hr):", value=float(default_t1), min_value=float(times[0]), max_value=float(times[-1]), step=1.0)
            with c_sl2:
                manual_t2 = st.number_input("Regression End Time (hr):", value=float(default_t2), min_value=float(times[0]), max_value=float(times[-1]), step=1.0)
                
            # Re-evaluate semilog fit on user-selected range
            re_semi = estimate_permeability_semilog(
                times, dp, manual_t1, manual_t2, f_params["q"], f_params["Bo"], f_params["mu"], f_params["h"]
            )
            
            fig_semi = create_semilog_analysis_plot(
                times=times,
                delta_p=dp,
                t_start=manual_t1,
                t_end=manual_t2,
                slope_m=re_semi.get("slope_m"),
                intercept_b=re_semi.get("intercept_b"),
                r_squared=re_semi.get("r_squared"),
                case_name="Uploaded Well-Test Drawdown"
            )
            st.plotly_chart(fig_semi, use_container_width=True)
            
            if re_semi.get("is_valid"):
                st.caption(f"Fitted Semilog Slope: m = {re_semi['slope_m']} psi/cycle, Intercept = {re_semi['intercept_b']} psi, R² = {re_semi['r_squared']}. Implied k = {re_semi['k_mD']} mD.")

        with tab_smoothing:
            derivatives_multi = compare_bourdet_derivatives(times, dp, L_values=[0.05, 0.15, 0.30])
            fig_smooth = create_smoothing_comparison_plot(times, derivatives_multi, case_name="Smoothing Sensitivity")
            st.plotly_chart(fig_smooth, use_container_width=True)

        st.markdown("---")

        # ---------------------------------------------------------------------
        # FLOW REGIME SUMMARY & STABILIZATION RECOGNITION
        # ---------------------------------------------------------------------
        st.markdown("#### Flow-Regime Detection & Stabilization Recognition")
        
        col_reg_tbl, col_plat_info = st.columns([3, 2])
        with col_reg_tbl:
            st.markdown("##### Detected Sustained Flow Regimes")
            df_reg_tbl = generate_flow_regime_summary_table(regimes)
            st.dataframe(df_reg_tbl, hide_index=True, use_container_width=True)
            
            # Sequence narrative
            seq_text = analyze_regime_sequence(regimes)
            st.info(f"🧭 **Diagnostic Sequence Analysis**: {seq_text}")

        with col_plat_info:
            st.markdown("##### Derivative Stabilization Plateau")
            if plateau.get("is_detected"):
                st.success(f"✅ **Stabilization Confirmed**: Plateau detected between {plateau['t_start']:.3g} and {plateau['t_end']:.3g} hr.")
                st.markdown(f"- **Mean Plateau Derivative ΔP'**: `{plateau['derv_mean']} psi`")
                st.markdown(f"- **Plateau Standard Deviation**: `{plateau['derv_std']} psi` (Variation: `{plateau['rel_std']}%`)")
                st.markdown(f"- **Logarithmic Duration**: `{plateau['log_cycles']} log cycles` ({plateau['num_points']} observations)")
                st.markdown(f"- **Confidence Level**: **{plateau['confidence']}**")
            else:
                st.warning(f"⚠️ {plateau.get('notes', 'No derivative plateau detected.')}")

        st.markdown("---")

        # ---------------------------------------------------------------------
        # ENGINEERING VERDICT & EXPORT
        # ---------------------------------------------------------------------
        st.markdown("#### Engineering Verdict & Technical Summary")
        
        verdict_lines = []
        if regimes:
            reg_names = [r.regime for r in regimes]
            verdict_lines.append(f"Identified {len(regimes)} distinct transient regime(s): {', '.join(reg_names)}.")
        else:
            verdict_lines.append("No sustained flow regime met strict slope tolerance criteria.")
            
        if est.get("best_k"):
            verdict_lines.append(f"Formation permeability is estimated at approximately {est['best_k']} mD.")
        else:
            verdict_lines.append("Insufficient data to establish formation permeability reliably.")
            
        if est.get("skin", {}).get("is_valid"):
            verdict_lines.append(f"Estimated wellbore skin factor is S = {est['skin']['skin']:+.2f}.")
            
        if est.get("xf", {}).get("is_valid"):
            verdict_lines.append(f"Formation linear flow is consistent with a hydraulic fracture half-length of xf = {est['xf']['xf_ft']} ft.")
            
        if est.get("Cf", {}).get("is_valid"):
            verdict_lines.append(f"Bilinear flow indicates finite fracture conductivity Cf = {est['Cf']['Cf_mD_ft']} mD-ft.")
            
        verdict_text = " ".join(verdict_lines)
        st.info(f"📋 **Automated Interpretation**: {verdict_text}")

        # Download Buttons: CSV & PDF Report
        st.markdown("##### Export Results")
        col_down1, col_down2 = st.columns(2)
        with col_down1:
            csv_export = df_p.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Processed PTA Data (CSV)",
                data=csv_export,
                file_name="HydraPTA_drawdown_processed.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_down2:
            pdf_bytes = generate_pta_pdf_report(
                analysis_mode="Drawdown Analysis",
                case_name="Uploaded Well-Test Drawdown",
                properties=f_params,
                estimation_summary=est,
                detected_regimes=regimes,
                plateau_info=plateau,
                verdict_text=verdict_text,
                unit_system=res["unit_system"]
            )
            st.download_button(
                "📄 Download Engineering Report (PDF)",
                data=pdf_bytes,
                file_name="HydraPTA_Drawdown_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )


def render_buildup_mode():
    """Render complete Buildup Analysis section."""
    st.markdown("### 📈 Pressure Buildup (PBU) Analysis")
    st.markdown(
        "Analyze shut-in pressure gauge data using Horner semilog analysis, Agarwal equivalent time transformations, "
        "extrapolated reservoir pressure P*, permeability, and wellbore skin factor estimation."
    )
    
    # Ingestion
    st.markdown("#### Section 1: Upload Shut-In Pressure Dataset")
    col_up1, col_up2 = st.columns([3, 2])
    with col_up1:
        uploaded_file = st.file_uploader(
            "Upload Buildup Test Data (CSV or Excel .xlsx)",
            type=["csv", "xlsx", "xls"],
            key="buildup_file_uploader",
            help="File must contain shut-in time (dt) and shut-in pressure (Pws)."
        )
    with col_up2:
        st.markdown("**Quick Test with Demo Buildup Data:**")
        if st.button("📥 Load Sample Buildup Dataset", use_container_width=True):
            try:
                st.session_state["raw_df_buildup"] = pd.read_csv("data/sample_buildup.csv")
                st.session_state["buildup_filename"] = "sample_buildup.csv"
                st.success("Loaded sample buildup dataset.")
            except Exception as e:
                st.error(f"Could not load sample buildup: {e}")

    if uploaded_file is not None:
        try:
            df_uploaded = load_dataset(uploaded_file, uploaded_file.name)
            st.session_state["raw_df_buildup"] = df_uploaded
            st.session_state["buildup_filename"] = uploaded_file.name
        except Exception as e:
            st.error(f"Error parsing file: {e}")
            return

    raw_df = st.session_state.get("raw_df_buildup", None)
    if raw_df is None:
        st.info("👆 Please upload an Excel/CSV dataset or click 'Load Sample Buildup Dataset' to begin.")
        return

    # Auto column recognition
    auto_cols = auto_detect_columns(raw_df, analysis_mode="buildup")
    col_list = list(raw_df.columns)
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        default_dt_idx = col_list.index(auto_cols["time"]) if auto_cols["time"] in col_list else 0
        dt_col = st.selectbox("Shut-In Time Column (Δt) *", options=col_list, index=default_dt_idx, key="bu_dt_col")
    with col_m2:
        default_pws_idx = col_list.index(auto_cols["pressure"]) if auto_cols["pressure"] in col_list else min(1, len(col_list)-1)
        pws_col = st.selectbox("Shut-In Pressure (Pws) *", options=col_list, index=default_pws_idx, key="bu_pws_col")

    st.markdown("---")

    # Reservoir and Operational Properties
    st.markdown("#### Section 2: Production History & Reservoir Parameters")
    unit_system = st.radio("Unit System:", options=["Field", "SI"], horizontal=True, key="bu_unit_sys")
    unit_dict = UNIT_DEFINITIONS[unit_system]

    c_b1, c_b2, c_b3, c_b4 = st.columns(4)
    with c_b1:
        tp_val = st.number_input("Producing Time Before Shut-In tp (hr) *", value=200.0, min_value=1.0, step=10.0)
        q_val = st.number_input(f"Rate Prior to Shut-In q ({unit_dict['rate']}) *", value=500.0 if unit_system=="Field" else 79.5, min_value=1.0, step=25.0)
    with c_b2:
        pwf_shutin = st.number_input(f"Pwf at Shut-In Moment ({unit_dict['pressure']})", value=3600.0 if unit_system=="Field" else 248.0, min_value=0.0)
        bo_val = st.number_input(f"Bo ({unit_dict['fvf']}) *", value=1.20, min_value=0.5, step=0.05)
    with c_b3:
        mu_val = st.number_input(f"Viscosity μ ({unit_dict['viscosity']}) *", value=1.20, min_value=0.05, step=0.1)
        h_val = st.number_input(f"Net Pay h ({unit_dict['thickness']}) *", value=50.0 if unit_system=="Field" else 15.24, min_value=1.0, step=5.0)
    with c_b4:
        phi_val = st.number_input("Porosity φ (fraction) *", value=0.12, min_value=0.01, step=0.01)
        ct_val = st.number_input(f"Compressibility ct ({unit_dict['compressibility']}) *", value=1.5e-5 if unit_system=="Field" else 2.17e-4, format="%.2e")
        rw_val = st.number_input(f"Radius rw ({unit_dict['radius']}) *", value=0.25 if unit_system=="Field" else 0.0762, min_value=0.05)

    bu_params = {
        "tp": tp_val, "q": q_val, "pwf_shutin": pwf_shutin, "Bo": bo_val,
        "mu": mu_val, "h": h_val, "phi": phi_val, "ct": ct_val, "rw": rw_val
    }
    field_bu_params = convert_si_to_field(bu_params) if unit_system == "SI" else bu_params

    st.markdown("---")

    # Run Buildup Analysis Button
    if st.button("🚀 Run Pressure Buildup Analysis", type="primary", use_container_width=True):
        dt_arr = pd.to_numeric(raw_df[dt_col], errors="coerce").dropna().values
        pws_arr = pd.to_numeric(raw_df[pws_col], errors="coerce").dropna().values
        
        df_buildup = calculate_buildup_variables(
            dt_hr=dt_arr,
            pws_psi=pws_arr,
            tp_hr=field_bu_params["tp"],
            pwf_at_shutin=field_bu_params["pwf_shutin"]
        )
        
        horner_res = estimate_buildup_horner_parameters(
            df_buildup=df_buildup,
            tp_hr=field_bu_params["tp"],
            pwf_at_shutin=field_bu_params["pwf_shutin"],
            q=field_bu_params["q"],
            Bo=field_bu_params["Bo"],
            mu=field_bu_params["mu"],
            h=field_bu_params["h"],
            phi=field_bu_params["phi"],
            ct=field_bu_params["ct"],
            rw=field_bu_params["rw"]
        )
        
        st.session_state["bu_results"] = {
            "df_buildup": df_buildup,
            "horner_res": horner_res,
            "field_params": field_bu_params,
            "unit_system": unit_system
        }
        st.success("Buildup Analysis complete! Review Horner plots and parameters below.")

    if "bu_results" in st.session_state:
        b_res = st.session_state["bu_results"]
        df_b = b_res["df_buildup"]
        h_res = b_res["horner_res"]
        
        st.markdown("### 📊 Buildup Results Dashboard")
        
        # Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Permeability (k)", f"{h_res.get('k_mD', '—')} mD", help="From Horner straight line slope.")
        with c2:
            st.metric("Extrapolated Pressure (P*)", f"{h_res.get('P_star_psi', '—')} psi", help="Extrapolated to (tp+dt)/dt = 1.")
        with c3:
            st.metric("Pressure at 1 hr (P1hr)", f"{h_res.get('P_1hr_psi', '—')} psi")
        with c4:
            st.metric("Well Skin Factor (S)", f"{h_res.get('skin', '—')}", help="Calculated using P1hr and Pwf at shut-in.")

        st.markdown("---")

        # Plots
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig_horner = create_horner_plot(
                df_buildup=df_b,
                slope_m_horner=h_res.get("slope_m_horner"),
                intercept_b=h_res.get("intercept_b"),
                P_star=h_res.get("P_star_psi"),
                P_1hr=h_res.get("P_1hr_psi"),
                tp_hr=b_res["field_params"]["tp"],
                case_name="Uploaded Buildup Test"
            )
            st.plotly_chart(fig_horner, use_container_width=True)
            
        with col_p2:
            fig_bu_diag = create_buildup_diagnostic_plot(
                df_buildup=df_b,
                case_name="Uploaded Buildup Test"
            )
            st.plotly_chart(fig_bu_diag, use_container_width=True)

        st.markdown("---")

        # Export Buttons
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            csv_b = df_b.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Buildup Data (CSV)",
                data=csv_b,
                file_name="HydraPTA_buildup_processed.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_d2:
            pdf_b = generate_pta_pdf_report(
                analysis_mode="Buildup Analysis",
                case_name="Uploaded Buildup Test",
                properties=b_res["field_params"],
                estimation_summary={"k_semilog": h_res, "skin": {"is_valid": True, "skin": h_res.get("skin")}},
                detected_regimes=[],
                verdict_text=f"Horner analysis yields formation permeability k = {h_res.get('k_mD')} mD, extrapolated P* = {h_res.get('P_star_psi')} psi, and skin S = {h_res.get('skin')}.",
                unit_system=b_res["unit_system"]
            )
            st.download_button(
                "📄 Download Buildup Report (PDF)",
                data=pdf_b,
                file_name="HydraPTA_Buildup_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )


def render_user_input_page():
    """Main routing wrapper for Mode 2: User Input Analysis."""
    col_back, col_title = st.columns([1, 6])
    with col_back:
        if st.button("← Home", key="btn_back_home_user"):
            st.session_state["current_page"] = "home"
            st.rerun()
    with col_title:
        st.subheader("Mode 02 — User Input Well-Test Analysis")
        
    analysis_tab1, analysis_tab2 = st.tabs(["📉 1. Drawdown Analysis", "📈 2. Buildup Analysis"])
    with analysis_tab1:
        render_drawdown_mode()
    with analysis_tab2:
        render_buildup_mode()
