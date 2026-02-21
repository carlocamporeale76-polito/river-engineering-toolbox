# Sediment_Mech/tools/settling_velocity_ui.py
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from Sediment_Mech.Core import gsd_io  # usa il tuo reader GSD esistente
from Sediment_Mech.Core import settling_velocity as sv


def _hist_stats_plot(values: np.ndarray, title: str, xlabel: str):
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.hist(values, bins=10, rwidth=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, ls=":", lw=0.5)
    st.pyplot(fig, clear_figure=True)

    st.write(
        {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "n": int(len(values)),
        }
    )


def _boxplot_by_class(df_long: pd.DataFrame, title: str):
    # df_long columns: class_id, D_i_m, method, ws_m_s
    classes = sorted(df_long["class_id"].unique())
    data = [df_long.loc[df_long["class_id"] == c, "ws_m_s"].values for c in classes]
    labels = [str(c) for c in classes]

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_title(title)
    ax.set_xlabel("Class id")
    ax.set_ylabel("Settling velocity (m/s)")
    ax.grid(True, ls=":", lw=0.5)
    st.pyplot(fig, clear_figure=True)


def render() -> None:
    st.header("Settling velocity tool")
    st.caption(
        "Implements Section 1.7.2 (Eqs. 1.39–1.43, Tables 1.3–1.4) and hindered settling (Eqs. 1.44–1.46; optional Eq. 1.47)."
    )

    tab1, tab2 = st.tabs(["1) Clear water", "2) Hindered settling"])

    # -------------------------
    # TAB 1: CLEAR WATER
    # -------------------------
    with tab1:
        mode = st.radio("Input mode", ["Single diameter (Example 1.4 style)", "GSD distribution (box-plots per class)"], horizontal=True)

        st.subheader("Common parameters")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            s = st.number_input("Relative density s = ρs/ρ", value=2.65, min_value=1.01, step=0.01)
        with c2:
            nu = st.number_input("Kinematic viscosity ν (m²/s)", value=1.0e-6, format="%.3e")
        with c3:
            g = st.number_input("g (m/s²)", value=9.81, format="%.3f")
        with c4:
            Sp = st.number_input("Shape factor Sp", value=0.7, min_value=0.1, max_value=1.0, step=0.05)

        st.subheader("Formula selection")
        table13_methods = list(sv.TABLE_13.keys()) + ["Wu & Wang (2006)"]
        sel13 = st.multiselect(
            "Eq. (1.40) + Table 1.3 methods",
            options=table13_methods,
            default=["Cheng (1997)", "Soulsby (1997)", "Wu & Wang (2006)"],
        )
        sel14 = st.multiselect(
            "Table 1.4 methods",
            options=["Hallermeier (1981)", "Chang & Liou (2001)", "Guo (2002)"],
            default=["Hallermeier (1981)", "Chang & Liou (2001)", "Guo (2002)"],
        )
        include_dietrich = st.checkbox("Include Dietrich (1982) fit (Eq. 1.41)", value=True)
        include_jm = st.checkbox("Include Jiménez & Madsen (2003) (Eq. 1.43)", value=True)

        if mode.startswith("Single"):
            st.subheader("Single diameter (as in Example 1.4)")
            cA, cB = st.columns([1, 1])
            with cA:
                diameter_kind = st.selectbox("Diameter provided as", ["Nominal diameter dn", "Median sieve diameter d50 (convert dn=d50/0.9)"])
                d_mm = st.number_input("Diameter value (mm)", value=0.5, min_value=0.0001, format="%.4f")
            with cB:
                st.markdown("**Notes**")
                st.markdown("- dn ≈ d/0.9 in the book text.")
                st.markdown("- D* computed as in Example 1.4.")

            dn_m = (d_mm * 1e-3) if diameter_kind.startswith("Nominal") else sv.nominal_diameter_from_median_sieve(d_mm * 1e-3)
            Dst = sv.D_star(dn_m, s, g, nu)

            st.write(
                {
                    "dn (m)": dn_m,
                    "D*": Dst,
                }
            )

            if st.button("Compute ws (suite of formulas)"):
                df = sv.compute_ws_suite(
                    dn_m=dn_m,
                    s=s,
                    g=g,
                    nu_m2s=nu,
                    Sp=Sp,
                    methods_table13=sel13,
                    methods_table14=sel14,
                    include_dietrich=include_dietrich,
                    include_jm=include_jm,
                )
                st.dataframe(df)

                vals = df["ws_m_s"].values.astype(float)
                _hist_stats_plot(vals, "Distribution of ws across selected formulas", "ws (m/s)")

                # allow download
                out = io.BytesIO()
                df.to_csv(out, index=False)
                st.download_button("Download results (CSV)", out.getvalue(), file_name="ws_single_diameter.csv", mime="text/csv")

        else:
            st.subheader("GSD distribution -> box-plots per class")
            uploaded = st.file_uploader("Upload GSD Excel (.xlsx)", type=["xlsx"])
            if uploaded is None:
                st.info("Upload a GSD file to compute ws for each class (your existing GSD format).")
            else:
                try:
                    df_gsd = gsd_io.read_gsd_xlsx(uploaded)
                except Exception as e:
                    st.error(f"Error reading GSD: {e}")
                    return

                st.write("Parsed GSD:", df_gsd.head(20))

                # For each class diameter, compute suite => distribution across methods => boxplot
                rows = []
                for _, r in df_gsd.iterrows():
                    class_id = int(r["class_id"])
                    Di = float(r["D_i_m"])
                    # treat class diameter as dn directly (user can decide; consistent for class centers)
                    dn_m = Di
                    df_suite = sv.compute_ws_suite(
                        dn_m=dn_m,
                        s=s,
                        g=g,
                        nu_m2s=nu,
                        Sp=Sp,
                        methods_table13=sel13,
                        methods_table14=sel14,
                        include_dietrich=include_dietrich,
                        include_jm=include_jm,
                    )
                    for _, rr in df_suite.iterrows():
                        rows.append(
                            {
                                "class_id": class_id,
                                "D_i_m": Di,
                                "family": rr["family"],
                                "method": rr["method"],
                                "ws_m_s": float(rr["ws_m_s"]),
                            }
                        )

                df_long = pd.DataFrame(rows)
                st.dataframe(df_long.head(50))

                _boxplot_by_class(df_long, "ws distributions per class (across selected formulas)")

                # summary stats per class (mean/std across formulas)
                st.subheader("Per-class statistics (across formulas)")
                agg = df_long.groupby(["class_id", "D_i_m"])["ws_m_s"].agg(["mean", "std", "min", "max", "count"]).reset_index()
                st.dataframe(agg)

                # download
                out = io.BytesIO()
                df_long.to_csv(out, index=False)
                st.download_button("Download all class/formula results (CSV)", out.getvalue(), file_name="ws_gsd_all_methods.csv", mime="text/csv")

    # -------------------------
    # TAB 2: HINDERED SETTLING
    # -------------------------
    with tab2:
        st.subheader("Clear-water baseline (ws)")
        st.write("First compute a clear-water ws, then apply correction factors vs concentration C.")

        c1, c2 = st.columns([1, 1])
        with c1:
            d_mm = st.number_input("Nominal diameter dn (mm) for hindered calc", value=0.5, min_value=0.0001, format="%.4f")
            dn_m = d_mm * 1e-3
            base_method = st.selectbox("Baseline ws method", ["Cheng (1997)", "Soulsby (1997)", "Wu & Wang (2006)", "Dietrich (1982) fit"])
        with c2:
            C = st.number_input("Volumetric concentration C (-)", value=0.05, min_value=0.0, max_value=0.6, step=0.01)
            C_grid_max = st.number_input("Max C for plots", value=0.4, min_value=0.05, max_value=0.6, step=0.05)

        # compute ws baseline
        if base_method == "Dietrich (1982) fit":
            ws0 = sv.ws_dietrich_1982_fit(dn_m, s, g, nu)
        elif base_method == "Wu & Wang (2006)":
            ws0 = sv.ws_from_table13_method("Wu & Wang (2006)", dn_m, s, g, nu, Sp)
        else:
            ws0 = sv.ws_from_table13_method(base_method, dn_m, s, g, nu, Sp)

        Re0 = sv.reynolds_particle(ws0, dn_m, nu)
        n_rz = sv.n_richardson_zaki(Re0)

        st.write({"ws_clear (m/s)": ws0, "Re": Re0, "n (R-Z)": n_rz})

        st.subheader("Hindered settling at given C")
        include_sha = st.checkbox("Include Sha (1965) Eq. 1.46 (requires d50, fine sediment)", value=False)
        d50_mm_for_sha = None
        if include_sha:
            d50_mm_for_sha = st.number_input("d50 (mm) for Sha formula", value=0.01, min_value=0.0001, format="%.4f")

        include_soulsby_adj = st.checkbox("Include Soulsby adjustment Eq. 1.47 (P,Q change)", value=True)

        df_h = sv.compute_hindered_suite(
            ws_clear=ws0,
            dn_m=dn_m,
            nu_m2s=nu,
            C=C,
            d50_mm_for_sha=d50_mm_for_sha,
            include_soulsby_adjustment=include_soulsby_adj,
        )

        # If Soulsby adjustment included, compute wsc via Eq. 1.40 with adjusted P,Q and m=1
        if include_soulsby_adj:
            mask = df_h["method"].str.contains("Soulsby")
            if mask.any():
                P = float(df_h.loc[mask, "P"].values[0])
                Q = float(df_h.loc[mask, "Q"].values[0])
                wsc_soulsby = sv.ws_eq_140(dn_m, s, g, nu, P=P, Q=Q, m=1.0)
                df_h.loc[mask, "wsc_m_s"] = wsc_soulsby
                df_h.loc[mask, "factor"] = wsc_soulsby / ws0

        st.dataframe(df_h)

        st.subheader("Correction factor vs C (plots)")
        Cgrid = np.linspace(0.0, float(C_grid_max), 80)

        # Richardson-Zaki factor curve (n computed at baseline Re)
        f_rz = (1.0 - Cgrid) ** n_rz
        f_ol = (1.0 - 2.15 * Cgrid) * (1.0 - 0.75 * (Cgrid ** 0.33))

        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        ax.plot(Cgrid, f_rz, label="Richardson & Zaki (Eq. 1.44)")
        ax.plot(Cgrid, f_ol, label="Oliver (Eq. 1.45)")
        if include_sha and d50_mm_for_sha is not None:
            f_sha = (1.0 - (Cgrid / (2.0 * (d50_mm_for_sha ** 0.5)))) ** 3
            ax.plot(Cgrid, f_sha, label="Sha (Eq. 1.46)")
        ax.set_xlabel("C (-)")
        ax.set_ylabel("Correction factor (wsc/ws)")
        ax.set_title("Hindered-settling correction factors")
        ax.grid(True, ls=":", lw=0.5)
        ax.legend()
        st.pyplot(fig, clear_figure=True)

        if include_soulsby_adj:
            # show Soulsby factor curve by recomputing adjusted P,Q and ws for each C
            factors = []
            for c in Cgrid:
                P, Q = sv.soulsby_adjusted_PQ(float(c))
                wsc = sv.ws_eq_140(dn_m, s, g, nu, P=P, Q=Q, m=1.0)
                factors.append(wsc / ws0)

            fig2, ax2 = plt.subplots(figsize=(7.0, 3.3))
            ax2.plot(Cgrid, factors)
            ax2.set_xlabel("C (-)")
            ax2.set_ylabel("wsc/ws (Soulsby Eq. 1.47 + Eq. 1.40)")
            ax2.set_title("Soulsby hindered settling via P,Q adjustment")
            ax2.grid(True, ls=":", lw=0.5)
            st.pyplot(fig2, clear_figure=True)
