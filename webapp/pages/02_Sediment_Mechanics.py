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
from Sediment_Mech.tools.settling_velocity_ui import render as render_settling

st.set_page_config(page_title="Ch.2 — Sediment Mechanics", layout="wide")
st.title("Chapter 2 — Sediment Mechanics")
st.write("Select a tool or section from the panel below.")

# Navigation for subpages/tools
PAGES = {
    "Overview & Theory": None,
    "GSD Calculator": render_gsd,
    "Settling velocity": render_settling,
    "Surface-based bedload (full tool)": render_bedload,    
    "Annual bedload integration": render_annual,
}

choice = st.sidebar.selectbox("Choose subpage", list(PAGES.keys()))

# Render overview (static text + link to QR/DOI)
if choice == "Overview & Theory":

    st.header("GSD Calculator — overview")
    st.markdown(
        """
        The GSD tool computes granulometric statistics from a discrete grain-size distribution.
        It evaluates cumulative curves, percentiles (D16, D50, D84, etc.), phi-transforms,
        geometric mean diameter and sorting metrics.
        """
    )

    # ------------------------------------------------------------------
        st.header("Settling velocity — theory (Section 1.7.2)")

    st.markdown(
        """
        This tool implements the settling velocity framework of Section 1.7.2.

        The drag coefficient is generalized (Eq. 1.39):
        """
    )
    st.latex(r"C_D=\left[\left(\frac{P}{Re}\right)^{1/m}+Q^{1/m}\right]^m")

    st.markdown(
        r"""
        The particle Reynolds number is computed using the **nominal diameter** \(d_n\):
        \[
        Re=\frac{w_s d_n}{\nu},
        \qquad d_n \approx \frac{d}{0.9},
        \]
        where \(d\) is the median sieve diameter.  
        """
    )
    st.latex(r"Re=\frac{w_s d_n}{\nu},\qquad d_n\approx \frac{d}{0.9}")

    st.markdown("Using Eq. (1.39), the terminal fall velocity is obtained as (Eq. 1.40):")
    st.latex(r"""
    w_s=\frac{P}{Q}\frac{\nu}{d_n}
    \left[
        \left(\frac{1}{4}+\frac{4Q}{3P^2}D_*^3\right)^{1/2}
        -\frac{1}{2}
    \right]^m
    """)

    st.markdown("with the dimensionless particle parameter:")
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
        """
        | Reference | Formula | Range of D* |
        |---|---|---|
        | Hallermeier (1981) | \\( w_s = \\frac{\\nu}{d_n}\\frac{D_*^3}{18} \\) | \\(D_* \\le 3.42\\) |
        | Hallermeier (1981) | \\( w_s = \\frac{\\nu}{d_n}\\frac{D_*^{2.1}}{6} \\) | \\(3.42 < D_* \\le 21.54\\) |
        | Hallermeier (1981) | \\( w_s = 1.05\\,\\frac{\\nu}{d_n}D_*^{1.5} \\) | \\(D_* > 21.54\\) |
        | Chang and Liou (2001) | \\( w_s = 1.68\\,\\frac{\\nu}{d_n}\\frac{D_*^{1.389}}{1+30.22D_*^{1.611}} \\) | – |
        | Guo (2002) | \\( w_s = \\frac{\\nu}{d_n}\\frac{D_*^3}{24+0.866D_*^{1.5}} \\) | – |
        """
    )

    st.markdown(
        """
        The tool also includes the additional fits introduced immediately after these tables
        (e.g., Dietrich 1982 polynomial fit, and the Jiménez & Madsen 2003 formulation),
        and reproduces Example 1.4 workflow when a single diameter is provided.
        """
    )

    st.subheader("Hindered settling (high concentrations) — Eqs. 1.44–1.46")
    st.markdown("For high concentrations, the clear-water settling velocity is reduced:")

    st.markdown("**Richardson & Zaki (1954), Eq. 1.44**")
    st.latex(r"w_{sc}=w_s(1-C)^n")

    st.markdown("**Oliver (1961), Eq. 1.45**")
    st.latex(r"w_{sc}=w_s(1-2.15C)(1-0.75C^{0.33})")

    st.markdown("**Sha (1965), Eq. 1.46**")
    st.latex(r"w_{sc}=w_s\left[1-\frac{C}{2d_{50}^{0.5}}\right]^3")

    st.markdown(
        """
        The tool plots the correction factors vs concentration \(C\) and applies them to the
        computed clear-water \(w_s\).
        """
    )

    # ------------------------------------------------------------------
    st.subheader("Hindered settling (high concentrations)")

    st.markdown(
        """
        For high sediment concentrations the clear-water velocity is reduced.
        The tool implements the following correction models:
        """
    )

    st.latex(r"""
    w_{sc} = w_s (1 - C)^n
    """)

    st.markdown("Richardson & Zaki (1954), Eq. 1.44")

    st.latex(r"""
    w_{sc} = w_s (1 - 2.15C)(1 - 0.75C^{0.33})
    """)

    st.markdown("Oliver (1961), Eq. 1.45")

    st.latex(r"""
    w_{sc} = w_s \left[ 1 - \frac{C}{2 d_{50}^{0.5}} \right]^3
    """)

    st.markdown("Sha (1965), Eq. 1.46")

    st.markdown(
        """
        Soulsby (1997) proposed an alternative approach by modifying
        the coefficients \(P\) and \(Q\):
        """
    )

    st.latex(r"""
    P = \frac{26}{(1-C)^{4.7}}, \quad
    Q = \frac{1.3}{(1-C)^{4.7}}
    """)

    st.markdown(
        """
        which are then inserted back into Eq. (1.40).

        The tool allows:
        1) Clear-water computation for a single diameter (as in Example 1.4)  
        2) Ensemble evaluation across multiple formulas  
        3) Computation for an entire GSD with box-plots per class  
        4) Concentration-dependent correction curves
        """
    )

    # ------------------------------------------------------------------
    st.header("Surface-based bedload transport — overview")

    st.markdown(
        """
        The bedload transport module implements surface-based relations
        (Ashida–Michiue, Parker, Wilcock & Crowe) and links them with
        the processed GSD and settling computations.
        """
    )

    st.header("Annual bedload integration — overview")

    st.markdown(
        """
        The annual integration tool combines bedload predictors with
        flow-duration curves to estimate long-term sediment yield.
        """
    )
else:
    render_fn = PAGES[choice]
    if render_fn is None:
        st.error("Page not yet implemented.")
    else:
        render_fn()
