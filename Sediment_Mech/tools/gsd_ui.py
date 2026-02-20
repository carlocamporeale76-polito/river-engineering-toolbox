# Sediment_Mech/tools/gsd_ui.py
st.write("DEBUG: new GSD UI version loaded")
from __future__ import annotations
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from Sediment_Mech.Core import gsd_io

st.set_option("deprecation.showPyplotGlobalUse", False)

def _make_template_excel() -> bytes:
    """Return a small XLSX bytes object usable as template (D_mm, percent)."""
    df = pd.DataFrame({
        "D_mm": [0.5, 1, 2, 4, 8, 16],
        "percent": [5, 10, 20, 30, 25, 10]
    })
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="GSD")
    return bio.getvalue()

def _plot_size_distribution(df: pd.DataFrame):
    """Bar plot of class percent vs diameter (mm)."""
    D_mm = df["D_i_m"].values * 1000.0
    perc = df["f_i"].values * 100.0
    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.bar(D_mm.astype(float), perc, width=np.diff(np.append(D_mm, D_mm[-1]*1.2)), align='center')
    ax.set_xscale('log')
    ax.set_xlabel("Diameter (mm)")
    ax.set_ylabel("Percent (%)")
    ax.set_title("Grain size distribution (classes)")
    ax.grid(True, which="both", ls=":", lw=0.5)
    st.pyplot(fig)

def _plot_cumulative(df: pd.DataFrame):
    """Cumulative percent vs diameter."""
    dfc = gsd_io.cumulative_from_gsd(df)
    D_mm = dfc["D_i_m"].values * 1000.0
    cum = dfc["cum_percent"].values
    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.plot(D_mm, cum, marker='o', linestyle='-')
    ax.set_xscale('log')
    ax.set_ylim(0, 100)
    ax.set_xlabel("Diameter (mm)")
    ax.set_ylabel("Cumulative percent (%)")
    ax.set_title("Cumulative grain size curve")
    ax.grid(True, which="both", ls=":", lw=0.5)
    st.pyplot(fig)

def _plot_phi_histogram(df: pd.DataFrame):
    """Histogram of phi distribution using class centers weighted by f_i."""
    df_phi = df.copy()
    df_phi["phi"] = df_phi["D_i_m"].apply(gsd_io.phi_from_d)
    # expand approximate sample for plotting weighting (or use weighted histogram)
    phi_vals = df_phi["phi"].values
    weights = df_phi["f_i"].values
    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.hist(phi_vals, bins=8, weights=weights, rwidth=0.8)
    ax.set_xlabel("Phi")
    ax.set_ylabel("Weighted fraction")
    ax.set_title("Phi histogram (weighted by f_i)")
    ax.grid(True, ls=":", lw=0.5)
    st.pyplot(fig)

def render():
    st.header("GSD Calculator")
    st.write("Carica un file Excel (.xlsx) con la distribuzione granulometrica (foglio 'GSD' o 'Calculator').")
    col1, col2 = st.columns([1,1])

    with col1:
        uploaded = st.file_uploader("Upload GSD .xlsx", type=["xlsx"])
        btn_template = st.button("Download template (.xlsx)")

    with col2:
        st.markdown("**Opzioni**")
        normalize_on_load = st.checkbox("Normalizza frazioni all'upload (default ON)", value=True)
        show_plots_immediately = st.checkbox("Mostra plot automaticamente dopo il calcolo", value=True)

    if btn_template:
        data = _make_template_excel()
        st.download_button("Download GSD template", data, file_name="GSD_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    df = None
    if uploaded is not None:
        # read using gsd_io; handle errors gracefully
        try:
            df = gsd_io.read_gsd_xlsx(uploaded)
            st.success("File caricato e parsato correttamente.")
            st.write(f"Classi lette: {len(df)}")
            st.dataframe(df.head(20))
        except Exception as e:
            st.error(f"Errore nel parsing del file: {e}")
            st.stop()

    # Provide compute button if file present
    if df is not None:
        compute = st.button("Compute statistics and plots")
        # allow auto compute if requested
        if compute or show_plots_immediately:
            try:
                out = gsd_io.compute_all_stats(df)
                stats = out["stats"]
                percentiles = out["percentiles_m"]
                table = out["table"]

                # Display stats
                st.subheader("Key statistics")
                stat_cols = st.columns(3)
                stat_cols[0].metric("D50 (m)", f"{stats['D50_m']:.6f}" if not pd.isna(stats['D50_m']) else "n/a")
                stat_cols[1].metric("D16 (m)", f"{stats['D16_m']:.6f}" if not pd.isna(stats['D16_m']) else "n/a")
                stat_cols[2].metric("D84 (m)", f"{stats['D84_m']:.6f}" if not pd.isna(stats['D84_m']) else "n/a")

                st.write("Other statistics")
                st.json({
                    "phi_mean": stats.get("phi_mean"),
                    "phi_std": stats.get("phi_std"),
                    "geometric_mean_m": stats.get("geometric_mean_m"),
                    "folk_ward_sort": stats.get("folk_ward_sort")
                })

                st.subheader("Percentiles (meters)")
                st.table(pd.Series(percentiles).rename("D_p_m").to_frame())

                st.subheader("Summary table (classes)")
                st.dataframe(table)

                # plots
                st.subheader("Plots")
                _plot_size_distribution(df)
                _plot_cumulative(df)
                _plot_phi_histogram(df)

                # allow download of processed table
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    table.to_excel(writer, index=False, sheet_name="Summary")
                    pd.DataFrame([stats]).to_excel(writer, index=False, sheet_name="Stats")
                buf.seek(0)
                st.download_button("Download processed results (.xlsx)", buf.read(), file_name="GSD_results.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as e:
                st.error(f"Errore nel calcolo delle statistiche: {e}")
                st.stop()
    else:
        st.info("Carica un file .xlsx per abilitare il calcolo.")
