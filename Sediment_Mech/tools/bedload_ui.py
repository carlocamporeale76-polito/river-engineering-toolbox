# Sediment_Mech/tools/bedload_ui.py
import streamlit as st
import matplotlib.pyplot as plt
from Sediment_Mech.core.io import read_psd_surface_csv, read_fdc_csv
from Sediment_Mech.core.active_layer import geometric_mean_Dsg
from Sediment_Mech.core.parker1990 import compute_parker1990
from Sediment_Mech.core.wilcock_crowe2003 import compute_wilcock_crowe2003
from Sediment_Mech.core.ashida_michiue import compute_ashida_michiue

def render():
    st.header("Surface-based bedload transport (full)")
    st.sidebar.markdown("Upload inputs for bedload tools")
    psd_file = st.sidebar.file_uploader("PSD CSV (D_i_m, f_i)", type=["csv"])
    fdc_file = st.sidebar.file_uploader("FDC CSV (Q_m3s, p_time)", type=["csv"])
    # (rest of your existing interactive logic goes here)
    # For clarity, keep the heavy logic in Sediment_Mech.core.* and call from here.
    st.info("Use the sidebar to upload PSD and FDC, then run calculations.")
