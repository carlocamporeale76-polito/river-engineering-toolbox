# Sediment_Mech/tools/gsd_ui.py

from __future__ import annotations
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from Sediment_Mech.Core import gsd_io


def _plot_distribution(df: pd.DataFrame):
    D_mm = df["D_i_m"] * 1000.0
    perc = df["f_i"] * 100.0

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(D_mm, perc, width=D_mm * 0.15)
    ax.set_xscale("log")
    ax.set_xlabel("Diameter (mm)")
    ax.set_ylabel("Percent (%)")
    ax.set_title("Grain Size Distribution")
    ax.grid(True, which="both", linestyle=":")
    st.pyplot(fig)


def _plot_cumulative(df: pd.DataFrame):
    dfc = gsd_io.cumulative_from_gsd(df)

    D_mm = dfc["D_i_m"] * 1000.0
    cum = dfc["cum_percent"]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(D_mm, cum, marker="o")
    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Diameter (mm)")
    ax.set_ylabel("Cumulative (%)")
    ax.set_title("Cumulative Curve")
    ax.grid(True, which="both", linestyle=":")
    st.pyplot(fig)


def _plot_phi_hist(df: pd.DataFrame):
    df_phi = df.copy()
    df_phi["phi"] = df_phi["D_i_m"].apply(gsd_io.phi_from_d)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(df_phi["phi"], weights=df_phi["f_i"], bins=8)
    ax.set_xlabel("Phi")
    ax.set_ylabel("Weighted fraction")
    ax.set_title("Phi Histogram")
    ax.grid(True, linestyle=":")
    st.pyplot(fig)


def render():
    st.header("GSD Calculator")

    uploaded = st.file_uploader("Upload GSD Excel (.xlsx)", type=["xlsx"])

    if uploaded is None:
        st.info("Upload a valid GSD Excel file to enable calculations.")
        return

    try:
        df = gsd_io.read_gsd_xlsx(uploaded)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    st.subheader("Parsed distribution")
    st.dataframe(df)

    if st.button("Compute statistics"):
        results = gsd_io.compute_all_stats(df)

        stats = results["stats"]
        percentiles = results["percentiles_m"]
        table = results["table"]

        st.subheader("Key Percentiles")
        cols = st.columns(3)
        cols[0].metric("D16 (m)", f"{stats['D16_m']:.6f}")
        cols[1].metric("D50 (m)", f"{stats['D50_m']:.6f}")
        cols[2].metric("D84 (m)", f"{stats['D84_m']:.6f}")

        st.subheader("Other Statistics")
        st.json({
            "phi_mean": stats["phi_mean"],
            "phi_std": stats["phi_std"],
            "geometric_mean_m": stats["geometric_mean_m"],
            "folk_ward_sort": stats["folk_ward_sort"]
        })

        st.subheader("Percentiles (m)")
        st.table(pd.Series(percentiles).rename("D_p_m"))

        st.subheader("Summary Table")
        st.dataframe(table)

        st.subheader("Plots")
        _plot_distribution(df)
        _plot_cumulative(df)
        _plot_phi_hist(df)

        # download results
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            table.to_excel(writer, sheet_name="Summary", index=False)
            pd.DataFrame([stats]).to_excel(writer, sheet_name="Stats", index=False)

        buffer.seek(0)

        st.download_button(
            "Download Results (.xlsx)",
            buffer,
            file_name="GSD_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
