# webapp/Home.py

import streamlit as st

st.set_page_config(page_title="River Engineering Toolbox", layout="wide")

st.title("River Engineering Toolbox")
st.write(
    """
This web companion provides engineering calculators associated with the textbook.
Use the left sidebar to select a chapter/tool.
"""
)

st.markdown("## Citation (DOI)")
st.markdown("River Engineering Toolbox (v1.0.0), Zenodo.")
st.markdown("https://doi.org/10.5281/zenodo.18702576")

st.markdown("## Available tools")
st.markdown(
    """
- **Chapter 2 – Sediment Mechanics**
  - Surface-based bedload transport for mixtures (Ashida–Michiue, Parker, Wilcock–Crowe)
"""
)

st.markdown("## How to use")
st.markdown(
    """
1. Go to the chapter page from the sidebar (e.g., *02_Sediment_Mechanics*).
2. Upload:
   - a surface PSD CSV (`D_i_m`, `f_i`)
   - a flow duration curve CSV (`Q_m3s` + `p_time` or `days_per_year`)
3. Select the transport relation and hydraulic parameters.
4. Run and download results (CSV and Excel).
"""
)

st.markdown("## Input templates")

st.markdown("### Surface PSD (`psd_surface.csv`)")
st.code(
    "class_id,D_i_m,f_i\n"
    "1,0.032,0.15\n"
    "2,0.016,0.35\n"
    "3,0.008,0.30\n"
    "4,0.004,0.20\n",
    language="text",
)

st.markdown("### Flow duration curve (`fdc.csv`) — option A (recommended: fractions)")
st.code(
    "Q_m3s,p_time\n"
    "5,0.30\n"
    "10,0.25\n"
    "20,0.20\n"
    "40,0.15\n"
    "80,0.10\n",
    language="text",
)

st.markdown("### Flow duration curve (`fdc.csv`) — option B (days per year)")
st.code(
    "Q_m3s,days_per_year\n"
    "5,110\n"
    "10,91\n"
    "20,73\n"
    "40,55\n"
    "80,36\n",
    language="text",
)
