"""
Sediment_Mech.tools.bedload_ui
Streamlit UI for the Surface-based bedload transport tool (chapter hub).
Provides sidebar inputs, uploads for PSD and FDC, model selection and run.
Relies on core computational functions from Sediment_Mech.core.*
The heavy numerical work should be implemented in core functions (compute_annual_bedload, compute_parker1990, etc).
"""

from __future__ import annotations
import os
import sys
from typing import Dict

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Ensure repository root is importable when Streamlit runs page from webapp
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Core functions (these must exist in Sediment_Mech.core)
try:
    from Sediment_Mech.core.io import read_psd_surface_csv, read_fdc_csv
    from Sediment_Mech.core.annual import compute_annual_bedload
    from Sediment_Mech.core.active_layer import geometric_mean_Dsg, sorting_sigma_s_log2
    from Sediment_Mech.core.ashida_michiue import compute_ashida_michiue
    from Sediment_Mech.core.parker1990 import compute_parker1990
    from Sediment_Mech.core.wilcock_crowe2003 import compute_wilcock_crowe2003
except Exception:
    # If imports fail, we still allow the UI to load and surface an error on Run.
    read_psd_surface_csv = read_fdc_csv = compute_annual_bedload = None
    geometric_mean_Dsg = sorting_sigma_s_log2 = None
    compute_ashida_michiue = compute_parker1990 = compute_wilcock_crowe2003 = None

EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")
EXAMPLE_PSD = os.path.join(EXAMPLES_DIR, "psd_surface_example.csv")
EXAMPLE_FDC = os.path.join(EXAMPLES_DIR, "fdc_example.csv")


def model_key(model_ui: str) -> str:
    if model_ui.startswith("Parker"):
        return "parker"
    if model_ui.startswith("Ashida"):
        return "ashida"
    return "wilcock"


def _plot_psd_bar(df_psd: pd.DataFrame, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig = ax.figure
    widths = df_psd["D_i_m"].astype(float).values * 0.12
    ax.bar(df_psd["D_i_m"], df_psd["f_i"], width=widths, align="center")
    ax.set_xscale("log")
    ax.set_xlabel("Grain size D (m)")
    ax.set_ylabel("Surface fraction f_i")
    ax.grid(True, which="both", alpha=0.25)
    return fig


def render() -> None:
    st.header("Surface-based bedload transport (full tool)")
    st.write("This tool computes fractional bedload using Ashida, Parker or Wilcock & Crowe relations and integrates over a Flow Duration Curve to estimate annual load.")

    # Sidebar inputs
    st.sidebar.header("Inputs — files")
    psd_file = st.sidebar.file_uploader("Upload surface PSD CSV (D_i_m, f_i)", type=["csv"])
    fdc_file = st.sidebar.file_uploader("Upload flow duration curve CSV (Q_m3s, p_time)", type=["csv"])

    if st.sidebar.button("Load example inputs"):
        if os.path.exists(EXAMPLE_PSD):
            psd_file = open(EXAMPLE_PSD, "rb")
        if os.path.exists(EXAMPLE_FDC):
            fdc_file = open(EXAMPLE_FDC, "rb")

    st.sidebar.header("Physical parameters")
    rho = st.sidebar.number_input("Fluid density ρ (kg/m³)", value=1000.0, format="%.1f")
    rho_s = st.sidebar.number_input("Sediment density ρ_s (kg/m³)", value=2650.0, format="%.1f")
    g = st.sidebar.number_input("Gravity g (m/s²)", value=9.81, format="%.3f")
    Delta = (rho_s - rho) / rho
    st.sidebar.write(f"Submerged specific density Δ = {Delta:.3f}")

    st.sidebar.header("Hydraulics")
    S = st.sidebar.number_input("Energy slope S (-)", value=0.005, format="%.5f")
    width = st.sidebar.number_input("Channel width b (m)", value=20.0, format="%.2f")
    n_manning = st.sidebar.number_input("Manning n (-)", value=0.035, format="%.4f")

    st.sidebar.header("Model")
    model_ui = st.sidebar.selectbox(
        "Bedload relation",
        ["Parker (1990)", "Ashida & Michiue (1972)", "Wilcock & Crowe (2003)"],
    )

    D_sand = st.sidebar.number_input("Sand threshold (m)", value=0.002, format="%.4f")
    extended_low_f = st.sidebar.checkbox("Wilcock–Crowe: include extra low-f branch (as in slides)", value=True)
    remove_sand_parker = st.sidebar.checkbox("Parker: remove sand and renormalize (recommended)", value=True)
    run_button = st.sidebar.button("Run calculation")

    st.subheader("Inputs diagnostics")
    if psd_file is None or fdc_file is None:
        st.info("Upload both PSD CSV and FDC CSV (or load examples) to proceed.")
        return

    # Parse inputs
    try:
        if read_psd_surface_csv is None or read_fdc_csv is None:
            raise ImportError("Core I/O functions not available. Check Sediment_Mech.core.io.")
        df_psd = read_psd_surface_csv(psd_file)
        df_fdc = read_fdc_csv(fdc_file)
    except Exception as e:
        st.error(f"Error reading input files: {e}")
        return

    # PSD diagnostics
    try:
        D_sg_all = geometric_mean_Dsg(df_psd) if geometric_mean_Dsg is not None else None
        sigma_s_all = sorting_sigma_s_log2(df_psd) if sorting_sigma_s_log2 is not None else None
    except Exception:
        D_sg_all = sigma_s_all = None

    Fs = float(df_psd.loc[df_psd["D_i_m"] < D_sand, "f_i"].sum())

    colA, colB, colC = st.columns(3)
    colA.metric("D_sg (m)", f"{D_sg_all:.4e}" if D_sg_all is not None else "n/a")
    colB.metric("σ_s (psi std)", f"{sigma_s_all:.3f}" if sigma_s_all is not None else "n/a")
    colC.metric("Surface sand fraction F_s", f"{Fs:.3f}")

    st.subheader("Surface PSD (active layer fractions)")
    st.dataframe(df_psd)
    fig_psd = _plot_psd_bar(df_psd)
    st.pyplot(fig_psd, clear_figure=True)

    if not run_button:
        st.info("Adjust parameters and press 'Run calculation' in the sidebar.")
        return

    # Run calculation
    if compute_annual_bedload is None:
        st.error("Core computation `compute_annual_bedload` not available. Ensure Sediment_Mech.core.annual is implemented and importable.")
        return

    mkey = model_key(model_ui)
    try:
        out_annual = compute_annual_bedload(
            df_psd=df_psd,
            df_fdc=df_fdc,
            model=mkey,
            rho=rho,
            rho_s=rho_s,
            g=g,
            S=S,
            width=width,
            n_manning=n_manning,
            D_sand=D_sand,
            remove_sand_parker=remove_sand_parker,
            extended_low_f=extended_low_f,
        )
    except Exception as e:
        st.error(f"Computation failed: {e}")
        return

    # Extract results
    df_ann = out_annual.get("results_df", pd.DataFrame())
    annual_total = out_annual.get("annual_bedload_m3_per_m", None)

    st.subheader("Annual results")
    st.metric("Annual bedload (per unit width)", f"{annual_total:.4e} m³/m/year" if annual_total is not None else "n/a")

    if not df_ann.empty:
        st.dataframe(df_ann.style.format({
            "Q_m3s": "{:.3f}",
            "h_m": "{:.3f}",
            "tau_Pa": "{:.3f}",
            "q_b_tot_m2s": "{:.4e}",
            "p_time": "{:.4f}",
            "annual_contribution_m3_per_m": "{:.4e}",
        }))

    # plots
    st.subheader("Flow-duration integration diagnostics")
    if not df_ann.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig1, ax1 = plt.subplots()
            ax1.plot(df_ann["Q_m3s"], df_ann["q_b_tot_m2s"], marker="o")
            ax1.set_xlabel("Discharge Q (m³/s)")
            ax1.set_ylabel("Total bedload q_b,tot (m²/s)")
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1, clear_figure=True)

        with c2:
            fig2, ax2 = plt.subplots()
            ax2.bar(df_ann["Q_m3s"], df_ann["annual_contribution_m3_per_m"])
            ax2.set_xlabel("Discharge class Q (m³/s)")
            ax2.set_ylabel("Annual contribution (m³/m/year)")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2, clear_figure=True)

        st.subheader("Annual contribution share by discharge class")
        fig3, ax3 = plt.subplots()
        share = df_ann["annual_contribution_m3_per_m"] / df_ann["annual_contribution_m3_per_m"].sum()
        ax3.bar(df_ann["Q_m3s"], share)
        ax3.set_xlabel("Discharge class Q (m³/s)")
        ax3.set_ylabel("Fraction of annual load (-)")
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3, clear_figure=True)

        # Download results
        st.download_button(
            "Download annual table (CSV)",
            data=df_ann.to_csv(index=False),
            file_name="annual_bedload_results.csv",
            mime="text/csv",
        )

    # Drill-down per-class at selected Q
    st.subheader("Single-discharge drill-down (per-class transport)")
    if not df_ann.empty:
        Q_options = sorted(df_ann["Q_m3s"].tolist())
        Q_sel = float(st.select_slider("Select discharge Q (m³/s)", options=Q_options))
        row = df_ann.loc[df_ann["Q_m3s"] == Q_sel].iloc[0]
        h_sel = float(row["h_m"])
        tau_sel = float(row["tau_Pa"])
        st.write(f"Selected Q = {Q_sel:.3f} m³/s → h = {h_sel:.3f} m, τ = {tau_sel:.3f} Pa")

        # call model-specific per-class diagnostics (if provided by core)
        if mkey == "ashida" and compute_ashida_michiue is not None:
            tau_star_i = {idx: tau_sel / ((rho_s - rho) * g * float(r["D_i_m"])) for idx, r in df_psd.iterrows()}
            res_df, qtot = compute_ashida_michiue(df_psd, tau_star_i, g, (rho_s - rho) / rho)
            perclass = res_df.copy()
            qtot_label = qtot
        elif mkey == "parker" and compute_parker1990 is not None:
            D_sg = geometric_mean_Dsg(df_psd) if geometric_mean_Dsg is not None else None
            tau_star_sg = tau_sel / ((rho_s - rho) * g * D_sg) if D_sg is not None else None
            out = compute_parker1990(
                df_psd=df_psd,
                tau_star_sg=tau_star_sg,
                tau_star_i=None,
                g=g,
                Delta=(rho_s - rho) / rho,
                remove_sand=remove_sand_parker,
                D_sand=D_sand,
            )
            perclass = out["results_df"].copy()
            qtot_label = out["q_b_tot"]
            st.caption(f"Parker diagnostics: ω = {out.get('omega', float('nan')):.4f}, σ_s = {out.get('sigma_s', float('nan')):.3f}, sand removed = {out.get('sand_fraction_removed', float('nan')):.3f}")
        elif mkey == "wilcock" and compute_wilcock_crowe2003 is not None:
            out = compute_wilcock_crowe2003(
                df_psd=df_psd,
                tau=tau_sel,
                rho=rho,
                rho_s=rho_s,
                g=g,
                D_sand=D_sand,
                extended_low_f=extended_low_f,
            )
            perclass = out["results_df"].copy()
            qtot_label = out["q_b_tot"]
            st.caption(f"Wilcock–Crowe diagnostics: F_s = {out.get('Fs', float('nan')):.3f}, τ*rm = {out.get('tau_star_rm', float('nan')):.4f}")
        else:
            perclass = pd.DataFrame()
            qtot_label = None

        if qtot_label is not None:
            st.metric("q_b,tot at selected Q", f"{qtot_label:.4e} m²/s")
        if not perclass.empty:
            fig4, ax4 = plt.subplots()
            ax4.plot(perclass["D_i_m"], perclass["q_bi_m2s"], marker="o")
            ax4.set_xlabel("Grain size D_i (m)")
            ax4.set_ylabel("Class transport q_bi (m²/s)")
            ax4.set_xscale("log")
            ax4.grid(True, which="both", alpha=0.3)
            st.pyplot(fig4, clear_figure=True)

            # show phi-like diagnostic if present
            phi_col = "Phi_i" if "Phi_i" in perclass.columns else ("Wstar_i" if "Wstar_i" in perclass.columns else None)
            if phi_col is not None:
                fig5, ax5 = plt.subplots()
                ax5.plot(perclass["D_i_m"], perclass[phi_col], marker="o")
                ax5.set_xlabel("Grain size D_i (m)")
                ax5.set_ylabel(f"{phi_col} (-)")
                ax5.set_xscale("log")
                ax5.grid(True, which="both", alpha=0.3)
                st.pyplot(fig5, clear_figure=True)

            st.dataframe(perclass)

            st.download_button(
                "Download per-class table at selected Q (CSV)",
                data=perclass.to_csv(index=False),
                file_name=f"perclass_Q_{Q_sel:.3f}.csv".replace(".", "p"),
                mime="text/csv",
            )

    st.success("Computation completed.")
