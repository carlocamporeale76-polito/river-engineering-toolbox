from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from Sediment_Mech.core.io import read_psd_surface_csv, read_fdc_csv
from Sediment_Mech.core.annual import compute_annual_bedload
from Sediment_Mech.core.active_layer import geometric_mean_Dsg, sorting_sigma_s_log2
from Sediment_Mech.core.ashida_michiue import compute_ashida_michiue
from Sediment_Mech.core.parker1990 import compute_parker1990
from Sediment_Mech.core.wilcock_crowe2003 import compute_wilcock_crowe2003


st.set_page_config(page_title="Ch.2 — Sediment Mechanics", layout="wide")
st.title("Chapter 2 — Sediment Mechanics")
st.subheader("Surface-based bedload transport for mixtures (full)")

# -----------------------------
# Sidebar: inputs
# -----------------------------
st.sidebar.header("Inputs")
psd_file = st.sidebar.file_uploader("Upload surface PSD CSV (D_i_m, f_i)", type=["csv"])
fdc_file = st.sidebar.file_uploader("Upload flow duration curve CSV (Q_m3s + p_time or days_per_year)", type=["csv"])

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

# -----------------------------
# Helpers
# -----------------------------
def logx_plot(ax):
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)

def model_key(model_ui: str) -> str:
    if model_ui.startswith("Parker"):
        return "parker"
    if model_ui.startswith("Ashida"):
        return "ashida"
    return "wilcock"

# -----------------------------
# Main: load data
# -----------------------------
if psd_file is None or fdc_file is None:
    st.info("Upload both PSD CSV and FDC CSV to proceed.")
    st.stop()

df_psd = read_psd_surface_csv(psd_file)
df_fdc = read_fdc_csv(fdc_file)

# PSD diagnostics
D_sg_all = geometric_mean_Dsg(df_psd)
sigma_s_all = sorting_sigma_s_log2(df_psd)
Fs = float(df_psd.loc[df_psd["D_i_m"] < D_sand, "f_i"].sum())

colA, colB, colC = st.columns(3)
colA.metric("D_sg (m)", f"{D_sg_all:.4e}")
colB.metric("σ_s (psi std)", f"{sigma_s_all:.3f}")
colC.metric("Surface sand fraction F_s", f"{Fs:.3f}")

# PSD plot
st.subheader("Surface PSD (active layer fractions)")
fig_psd, ax = plt.subplots()
ax.bar(df_psd["D_i_m"].astype(float), df_psd["f_i"].astype(float), width=df_psd["D_i_m"].astype(float) * 0.15)
ax.set_xlabel("Grain size D_i (m)")
ax.set_ylabel("Surface fraction f_i (-)")
logx_plot(ax)
st.pyplot(fig_psd, clear_figure=True)

if not run_button:
    st.stop()

mkey = model_key(model_ui)

# Annual computation
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
)

df_ann = out_annual["results_df"].copy()
annual_total = out_annual["annual_bedload_m3_per_m"]

st.subheader("Annual results")
st.metric("Annual bedload (per unit width)", f"{annual_total:.4e} m³/m/year")

st.dataframe(
    df_ann.style.format({
        "Q_m3s": "{:.3f}",
        "h_m": "{:.3f}",
        "tau_Pa": "{:.3f}",
        "q_b_tot_m2s": "{:.4e}",
        "p_time": "{:.4f}",
        "annual_contribution_m3_per_m": "{:.4e}",
    })
)

# Annual plots
st.subheader("Flow-duration integration diagnostics")
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

st.download_button(
    "Download annual table (CSV)",
    data=df_ann.to_csv(index=False),
    file_name="annual_bedload_results.csv",
    mime="text/csv",
)

# Drill-down
st.subheader("Single-discharge drill-down (per-class transport)")
Q_sel = float(st.select_slider("Select discharge Q (m³/s)", options=sorted(df_ann["Q_m3s"].tolist())))
row = df_ann.loc[df_ann["Q_m3s"] == Q_sel].iloc[0]
h_sel = float(row["h_m"])
tau_sel = float(row["tau_Pa"])

st.write(f"Selected Q = {Q_sel:.3f} m³/s → h = {h_sel:.3f} m, τ = {tau_sel:.3f} Pa")

if mkey == "ashida":
    tau_star_i = {idx: tau_sel / ((rho_s - rho) * g * float(r["D_i_m"])) for idx, r in df_psd.iterrows()}
    res_df, qtot = compute_ashida_michiue(df_psd, tau_star_i, g, Delta)
    perclass = res_df.copy()
    qtot_label = qtot

elif mkey == "parker":
    D_sg = geometric_mean_Dsg(df_psd)
    tau_star_sg = tau_sel / ((rho_s - rho) * g * D_sg)
    out = compute_parker1990(
        df_psd=df_psd,
        tau_star_sg=tau_star_sg,
        tau_star_i=None,
        g=g,
        Delta=Delta,
        remove_sand=remove_sand_parker,
        D_sand=D_sand,
    )
    perclass = out["results_df"].copy()
    qtot_label = out["q_b_tot"]
    st.caption(f"Parker diagnostics: ω = {out['omega']:.4f}, σ_s = {out['sigma_s']:.3f}, sand removed = {out['sand_fraction_removed']:.3f}")

else:  # wilcock
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
    st.caption(f"Wilcock–Crowe diagnostics: F_s = {out['Fs']:.3f}, τ*rm = {out['tau_star_rm']:.4f}")

st.metric("q_b,tot at selected Q", f"{qtot_label:.4e} m²/s")

fig4, ax4 = plt.subplots()
ax4.plot(perclass["D_i_m"], perclass["q_bi_m2s"], marker="o")
ax4.set_xlabel("Grain size D_i (m)")
ax4.set_ylabel("Class transport q_bi (m²/s)")
logx_plot(ax4)
st.pyplot(fig4, clear_figure=True)

phi_col = "Phi_i" if "Phi_i" in perclass.columns else ("Wstar_i" if "Wstar_i" in perclass.columns else None)
if phi_col is not None:
    fig5, ax5 = plt.subplots()
    ax5.plot(perclass["D_i_m"], perclass[phi_col], marker="o")
    ax5.set_xlabel("Grain size D_i (m)")
    ax5.set_ylabel(f"{phi_col} (-)")
    logx_plot(ax5)
    st.pyplot(fig5, clear_figure=True)

st.dataframe(perclass)

st.download_button(
    "Download per-class table at selected Q (CSV)",
    data=perclass.to_csv(index=False),
    file_name=f"perclass_Q_{Q_sel:.3f}.csv".replace(".", "p"),
    mime="text/csv",
)
