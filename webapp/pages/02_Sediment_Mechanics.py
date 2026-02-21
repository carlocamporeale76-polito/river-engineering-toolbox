# webapp/pages/02_Sediment_Mechanics.py
from __future__ import annotations

import os
import sys
import streamlit as st

# Ensure repo root is on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Sediment_Mech.tools.gsd_ui import render as render_gsd
from Sediment_Mech.tools.bedload_ui import render as render_bedload
from Sediment_Mech.tools.annual_ui import render as render_annual
from Sediment_Mech.tools.settling_velocity_ui import render as render_settling


st.set_page_config(page_title="Ch.2 — Sediment Mechanics", layout="wide")
st.title("Chapter 2 — Sediment Mechanics")
st.write("Select a tool or section from the panel below.")


PAGES = {
    "Overview & Theory": None,
    "GSD Calculator": render_gsd,
    "Settling velocity": render_settling,
    "Surface-based bedload (full tool)": render_bedload,
    "Annual bedload integration": render_annual,
}

choice = st.sidebar.selectbox("Choose subpage", list(PAGES.keys()))


def _image_path(*parts: str) -> str:
    # Look for images under webapp/Figures first, then repo-root/Figures
    here = os.path.abspath(os.path.dirname(__file__))  # webapp/pages
    webapp_dir = os.path.abspath(os.path.join(here, ".."))  # webapp
    cand1 = os.path.join(webapp_dir, *parts)
    if os.path.isfile(cand1):
        return cand1
    cand2 = os.path.join(REPO_ROOT, *parts)
    return cand2


if choice == "Overview & Theory":

    st.header("Surface-based bedload transport — overview")
    st.markdown(
        """
        This chapter provides tools for:
        - surface-based bedload relations (Ashida–Michiue, Parker, Wilcock & Crowe),
        - GSD processing,
        - settling velocity,
        - annual integration.
        Use the menu at left to open a tool.
        """
    )

    st.markdown(
        "Interactive web calculator: "
        "[Streamlit app](https://river-engineering-toolbox-8bvdo2ohn6rc5sikkgujd3.streamlit.app/)."
    )

    qr = _image_path("Figures", "qr_toolbox_streamlit.png")
    if os.path.isfile(qr):
        st.image(qr, width=160)

    st.divider()

    # ---------------- Settling velocity overview with complete tables ----------------
    st.header("Settling velocity — theory (Section 1.7.2)")

    st.markdown(
        """
        The settling velocity tool implements the generalized drag formulation (Eq. 1.39) and the
        resulting explicit expression for the terminal fall velocity (Eq. 1.40), together with the
        parameter sets in Table 1.3 and the alternative formulas in Table 1.4.
        """
    )

    st.subheader("Equations")

    st.markdown("Generalized drag coefficient (Eq. 1.39):")
    st.latex(r"C_D=\left[\left(\frac{P}{Re}\right)^{1/m}+Q^{1/m}\right]^m")

    st.markdown("Particle Reynolds number and nominal diameter:")
    st.latex(r"Re=\frac{w_s d_n}{\nu},\qquad d_n\approx \frac{d}{0.9}")

    st.markdown("Terminal settling velocity (Eq. 1.40):")
    st.latex(r"""
    w_s=\frac{P}{Q}\frac{\nu}{d_n}
    \left[
        \left(\frac{1}{4}+\frac{4Q}{3P^2}D_*^3\right)^{1/2}
        -\frac{1}{2}
    \right]^m
    """)

    st.markdown("Dimensionless particle parameter:")
    st.latex(r"D_* = d_n\left(\frac{(s-1)g}{\nu^2}\right)^{1/3}")

    st.subheader("Table 1.3 — Values of P, Q, and m (complete)")

    st.markdown(
        """
        | Reference | P | Q | m |
        |---|---:|---:|---:|
        | Rubey (1933) | 24 (for dn ≤ 1 mm) and 0 (for dn > 1 mm) | 2.1 | 1 |
        | Zhang (1961) | 34 | 1.2 | 1 |
        | Zanke (1977) | 24 (for dn ≤ 1 mm) and 0 (for dn > 1 mm) | 1.1 | 1 |
        | Raudkivi (1990) | 32 | 1.2 | 1 |
        | Fredsøe and Deigaard (1992) | 36 | 1.4 | 1 |
        | Julien (1998) | 24 | 1.5 | 1 |
        | Cheng (1997) | 32 | 1 | 1.5 |
        | Soulsby (1997) | 26.4 | 1.27 | 1 |
        | She et al. (2005) | 35 | 1.56 | 1 |
        | Wu and Wang (2006) | 53.5 exp(–0.65 Sp) | 5.65 exp(–2.5 Sp) | 0.7 + 0.9 Sp |
        | Camenen (2007) | 24.6 | 0.96 | 1.53 |
        """
    )

    st.subheader("Table 1.4 — ws(D*) formulas (complete)")

    st.markdown(
        r"""
        | Reference | Formula | Range of D* |
        |---|---|---|
        | Hallermeier (1981) | \( w_s = \frac{\nu}{d_n}\frac{D_*^3}{18} \) | \(D_* \le 3.42\) |
        | Hallermeier (1981) | \( w_s = \frac{\nu}{d_n}\frac{D_*^{2.1}}{6} \) | \(3.42 < D_* \le 21.54\) |
        | Hallermeier (1981) | \( w_s = 1.05\,\frac{\nu}{d_n}D_*^{1.5} \) | \(D_* > 21.54\) |
        | Chang and Liou (2001) | \( w_s = 1.68\,\frac{\nu}{d_n}\frac{D_*^{1.389}}{1+30.22D_*^{1.611}} \) | – |
        | Guo (2002) | \( w_s = \frac{\nu}{d_n}\frac{D_*^3}{24+0.866D_*^{1.5}} \) | – |
        """
    )

    st.subheader("Hindered settling (high concentrations) — Eqs. 1.44–1.46")

    st.markdown("Richardson & Zaki (1954), Eq. 1.44:")
    st.latex(r"w_{sc}=w_s(1-C)^n")

    st.markdown("Oliver (1961), Eq. 1.45:")
    st.latex(r"w_{sc}=w_s(1-2.15C)(1-0.75C^{0.33})")

    st.markdown("Sha (1965), Eq. 1.46:")
    st.latex(r"w_{sc}=w_s\left[1-\frac{C}{2d_{50}^{0.5}}\right]^3")

    st.markdown(
        """
        The tool computes clear-water settling velocity using multiple formulas (as in Example 1.4),
        displays the distribution of predicted values (histogram + mean/std), and can:
        - apply hindered-settling corrections for high concentrations,
        - compute settling velocity for an entire GSD, returning box-plots per class.
        """
    )

else:
    render_fn = PAGES[choice]
    if render_fn is None:
        st.error("Page not yet implemented.")
    else:
        render_fn()
