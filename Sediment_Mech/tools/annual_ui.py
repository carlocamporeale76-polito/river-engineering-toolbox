"""
Sediment_Mech.tools.annual_ui

Dedicated UI for annual bedload computation using a Flow Duration Curve (FDC).
This module focuses only on annual integration (no per-class drill-down).
"""

from __future__ import annotations
import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Ensure repo root on path (important for Streamlit Cloud)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Sediment_Mech.core.io import read_psd_surface_csv, read_fdc_csv
from Sediment_Mech.core.annual import compute_annual_bedload


def render() -> None:

    st.header("Annual Bedload Integration")
    st.write(
        "This tool integrates fractional bedload transport over a flow duration curve "
        "to compute annual sediment yield per unit channel width."
    )

    # ----------------------------
    # Sidebar Inputs
    # ----------------------------
    st.sidebar.header("Input files")
    psd_file = st.sidebar.file_uploader("Upload surface PSD CSV (D_i_m, f_i)", type=["csv"])
    fdc_file = st.sidebar.file_uploader("Upload FDC CSV (Q_m3s + p_time)", type=["csv"])

    st.sidebar.header("Physical parameters")
    rho = st.sidebar.number_input("Fluid density ρ (kg/m³)", value=1000.0, format="%.1f")
    rho_s = st.sidebar.number_input("Sediment density ρ_s (kg/m³)", value=2650.0, format="%.1f")
    g = st.sidebar.number_input("Gravity g (m/s²)", value=9.81, format="%.3f")

    st.sidebar.header("Hydraulics")
    S = st.sidebar.number_input("Energy slope S (-)", value=0.005, format="%.5f")
    width = st.sidebar.number_input("Channel width b (m)", value=20.0, format="%.2f")
    n_manning = st.sidebar.number_input("Manning n (-)", value=0.035, format="%.4f")

    st.sidebar.header("Transport model")
    model_ui = st.sidebar.selectbox(
        "Bedload relation",
        ["Parker (1990)", "Ashida & Michiue (1972)", "Wilcock & Crowe (2003)"],
    )

    D_sand = st.sidebar.number_input("Sand threshold (m)", value=0.002, format="%.4f")

    run_button = st.sidebar.button("Run annual computation")

    if psd_file is None or fdc_file is None:
        st.info("Upload both PSD and FDC files to proceed.")
        return

    # ----------------------------
    # Read input data
    # ----------------------------
    try:
        df_psd = read_psd_surface_csv(psd_file)
        df_fdc = read_fdc_csv(fdc_file)
    except Exception as e:
        st.error(f"Input parsing error: {e}")
        return

    st.subheader("Input diagnostics")
    st.write("Surface PSD:")
    st.dataframe(df_psd)

    st.write("Flow Duration Curve:")
    st.dataframe(df_fdc)

    if not run_button:
        return

    # ----------------------------
    # Compute Annual Bedload
    # ----------------------------
    model_key = (
        "parker" if "Parker" in model_ui
        else "ashida" if "Ashida" in model_ui
        else "wilcock"
    )

    try:
        out = compute_annual_bedload(
            df_psd=df_psd,
            df_fdc=df_fdc,
            model=model_key,
            rho=rho,
            rho_s=rho_s,
            g=g,
            S=S,
            width=width,
            n_manning=n_manning,
            D_sand=D_sand,
        )
    except Exception as e:
        st.error(f"Computation error: {e}")
        return

    df_results = out["results_df"]
    annual_total = out["annual_bedload_m3_per_m"]

    # ----------------------------
    # Results
    # ----------------------------
    st.subheader("Annual Results")

    st.metric(
        "Annual bedload (per unit width)",
        f"{annual_total:.4e} m³/m/year"
    )

    st.dataframe(
        df_results.style.format({
            "Q_m3s": "{:.3f}",
            "h_m": "{:.3f}",
            "tau_Pa": "{:.3f}",
            "q_b_tot_m2s": "{:.4e}",
            "p_time": "{:.4f}",
            "annual_contribution_m3_per_m": "{:.4e}",
        })
    )

    # ----------------------------
    # Plots
    # ----------------------------
    st.subheader("Diagnostics")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        ax1.plot(df_results["Q_m3s"], df_results["q_b_tot_m2s"], marker="o")
        ax1.set_xlabel("Discharge Q (m³/s)")
        ax1.set_ylabel("Total bedload q_b,tot (m²/s)")
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.bar(df_results["Q_m3s"], df_results["annual_contribution_m3_per_m"])
        ax2.set_xlabel("Discharge class Q (m³/s)")
        ax2.set_ylabel("Annual contribution (m³/m/year)")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

    # ----------------------------
    # Download
    # ----------------------------
    st.download_button(
        "Download annual results (CSV)",
        data=df_results.to_csv(index=False),
        file_name="annual_bedload_results.csv",
        mime="text/csv",
    )

    st.success("Annual computation completed successfully.")
