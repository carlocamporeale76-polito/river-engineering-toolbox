"""
Sediment_Mech.tools.gsd_ui
Streamlit UI for the GSD Calculator tool.

Expected to be called as `render()` from the chapter hub page.
Relies on `Sediment_Mech.core.io.read_gsd_xlsx` (or equivalent) to parse uploaded Excel.
"""

from __future__ import annotations
import io
import os
from typing import Optional

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

try:
    from Sediment_Mech.core.io import read_gsd_xlsx
except Exception:
    # fallback: try import from gsd_io (if user placed function there)
    try:
        from Sediment_Mech.core.gsd_io import read_gsd_xlsx  # type: ignore
    except Exception:
        read_gsd_xlsx = None  # type: ignore

EXAMPLES_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "examples")
EXAMPLE_XLSX = os.path.join(EXAMPLES_DIR, "GSDCalculator_example.xlsx")


def _plot_psd(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    # bar width proportional to size (log scale)
    widths = df["D_i_m"].astype(float).values * 0.12
    ax.bar(df["D_i_m"].astype(float), df["f_i"].astype(float), width=widths, align="center")
    ax.set_xscale("log")
    ax.set_xlabel("Grain size $D_i$ (m)")
    ax.set_ylabel("Surface fraction $f_i$")
    ax.grid(True, which="both", alpha=0.25)
    return fig


def render() -> None:
    st.header("GSD Calculator")
    st.write(
        "Upload the original Excel file exported from the GSD tool "
        "or use the synthetic example shipped with the toolbox."
    )

    col1, col2 = st.columns([2, 1])
    uploaded = col1.file_uploader("Upload GSD Excel (.xlsx, .xls)", type=["xlsx", "xls"])
    use_example = col2.button("Load example")

    if use_example:
        if os.path.exists(EXAMPLE_XLSX):
            with open(EXAMPLE_XLSX, "rb") as f:
                uploaded = io.BytesIO(f.read())
        else:
            st.error("Example file not found in examples/ directory.")
            return

    if uploaded is None:
        st.info("Please upload a GSD Excel file or press 'Load example' to try the sample.")
        return

    if read_gsd_xlsx is None:
        st.error("GSD reader function `read_gsd_xlsx` not found. Please ensure `Sediment_Mech.core.io` exposes it.")
        return

    # Parse the file
    try:
        df_psd = read_gsd_xlsx(uploaded)
    except Exception as e:
        st.error(f"Error parsing GSD file: {e}")
        return

    # Basic validation
    if not {"D_i_m", "f_i"}.issubset(df_psd.columns):
        st.error("Parsed file does not contain required columns 'D_i_m' and 'f_i'.")
        st.write("Columns found:", list(df_psd.columns))
        return

    # Normalize fractions if needed
    s = float(df_psd["f_i"].sum())
    if abs(s - 1.0) > 1e-6:
        st.warning(f"Surface fractions sum to {s:.6f}. They will be normalized to sum = 1 for calculations.")
        df_psd = df_psd.copy()
        df_psd["f_i"] = df_psd["f_i"] / s

    st.subheader("Parsed surface PSD")
    st.dataframe(df_psd.reset_index(drop=True))

    # Plot
    fig = _plot_psd(df_psd)
    st.pyplot(fig, clear_figure=True)

    # Export options
    csv_bytes = df_psd.to_csv(index=False).encode("utf-8")
    st.download_button("Download PSD CSV", data=csv_bytes, file_name="psd_surface_from_gsd.csv", mime="text/csv")

    # Excel export: write PSD to a single-sheet workbook
    try:
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_psd.to_excel(writer, sheet_name="surface_psd", index=False)
        output.seek(0)
        st.download_button(
            "Download PSD Excel",
            data=output.read(),
            file_name="psd_surface_from_gsd.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception:
        # non-fatal: excel writer not available
        pass

    st.markdown(
        "Usage notes:\n\n"
        "- The reader accepts columns named either `D_i_m` (meters) or `D_i_mm` (millimetres) "
        "and `f_i` (fraction). If `D_i_mm` is present it will be converted to meters.\n"
        "- Fractions are normalized automatically if they do not sum to 1.\n"
        "- The exported CSV is compatible with the Surface-based bedload tool in this chapter."
    )
