# Sediment_Mech/tools/gsd_ui.py
from typing import Optional
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from Sediment_Mech.core.io import read_gsd_xlsx  # function descritta prima

def render():
    st.header("GSD Calculator")
    st.write("Upload the original GSDCalculator.xlsx or use the example.")
    uploaded = st.file_uploader("Upload GSD excel", type=["xlsx","xls"])
    if st.button("Load example"):
        uploaded = open("examples/GSDCalculator_example.xlsx", "rb")
    if uploaded is None:
        st.info("Upload file or load example to proceed.")
        return

    try:
        df = read_gsd_xlsx(uploaded)
    except Exception as e:
        st.error(f"Failed to parse GSD file: {e}")
        return

    st.subheader("Parsed PSD (surface)")
    st.dataframe(df)
    fig, ax = plt.subplots()
    ax.bar(df["D_i_m"], df["f_i"], width=df["D_i_m"]*0.12)
    ax.set_xscale("log")
    ax.set_xlabel("D (m)")
    ax.set_ylabel("f_i")
    ax.grid(True, which="both", alpha=0.3)
    st.pyplot(fig, clear_figure=True)

    # allow CSV/Excel download
    st.download_button("Download PSD CSV", data=df.to_csv(index=False), file_name="psd_from_gsd.csv", mime="text/csv")
