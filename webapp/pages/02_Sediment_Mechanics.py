# webapp/pages/02_Sediment_Mechanics.py
from __future__ import annotations
import os, sys
# ensure repo root is on path (same fix as other pages)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import streamlit as st
from Sediment_Mech.tools.gsd_ui import render as render_gsd
from Sediment_Mech.tools.bedload_ui import render as render_bedload
from Sediment_Mech.tools.annual_ui import render as render_annual  # optional

st.set_page_config(page_title="Ch.2 — Sediment Mechanics", layout="wide")
st.title("Chapter 2 — Sediment Mechanics")
st.write("Select a tool or section from the panel below.")

# Navigation for subpages/tools
PAGES = {
    "Overview & Theory": None,
    "GSD Calculator": render_gsd,
    "Surface-based bedload (full tool)": render_bedload,    
    "Annual bedload integration": render_annual,
}

choice = st.sidebar.selectbox("Choose subpage", list(PAGES.keys()))

# Render overview (static text + link to QR/DOI)
if choice == "Overview & Theory":
    st.header("GSD Calculator — overview")
    st.markdown(
        "..."
    )    
    st.header("Surface-based bedload transport — overview")
    st.markdown(
        "This chapter implements theory and provides tools for: "
        "- surface-based bedload relations (Ashida, Parker, Wilcock & Crowe), "
        "GSD processing, and annual integration. Use the menu at left to open a tool."
    )    
    st.header("Annual bedload integration — overview")
    st.markdown(
        "..."
    )
else:
    render_fn = PAGES[choice]
    if render_fn is None:
        st.error("Page not yet implemented.")
    else:
        render_fn()
