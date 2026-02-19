# webapp/app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

st.set_page_config(page_title="River Engineering Toolbox — Chapter Sediment Mechanics", layout="wide")

st.title("River Engineering Toolbox — Surface-based bedload calculator (minimal)")
st.markdown(
    """
Simple calculator to compute class-specific bedload using surface-based relations
(Parker 1990, Ashida & Michiue 1972, Wilcock & Crowe 2003).  
Upload a CSV with columns: `D_i` (m), `f_i` (fraction, sums to 1), optional `tau_star_i` (dimensionless).
If `tau_star_i` is missing, provide a single reference Shields stress `tau_star_ref` that will be applied to all classes.
"""
)

# Sidebar inputs
st.sidebar.header("Model settings")
model = st.sidebar.selectbox("Choose transport relation", ["Parker 1990", "Ashida-Michiue 1972", "Wilcock-Crowe 2003"])
g = 9.81
rho = st.sidebar.number_input("Fluid density ρ (kg/m³)", value=1000.0, format="%.1f")
rho_s = st.sidebar.number_input("Sediment density ρ_s (kg/m³)", value=2650.0, format="%.1f")
Delta = (rho_s - rho) / rho
st.sidebar.write(f"Submerged specific density Δ = {Delta:.3f}")

st.sidebar.markdown("**Shields stress input**")
uploaded = st.sidebar.file_uploader("Upload CSV (columns: D_i, f_i, optional tau_star_i)", type=["csv"])
use_single_tau = st.sidebar.checkbox("Use single tau* for all classes (if checked, ignore column tau_star_i)", True)
tau_star_ref = None
if use_single_tau:
    tau_star_ref = st.sidebar.number_input("Reference Shields stress τ* (applied to all classes)", value=0.05, format="%.4f")

# Example CSV
if st.sidebar.button("Show example CSV"):
    example = "D_i,f_i,tau_star_i\n0.020,0.6,0.05\n0.010,0.3,0.05\n0.002,0.1,0.05\n"
    st.sidebar.code(example)

# Read data
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
else:
    st.info("No CSV uploaded yet — you can test with 'Show example CSV' then copy/paste into a new file and upload.")
    df = pd.DataFrame()

if not df.empty:
    # Validate columns
    if "D_i" not in df.columns or "f_i" not in df.columns:
        st.error("CSV must contain columns: D_i (m), f_i (fraction). Optional: tau_star_i")
        st.stop()

    # Normalize f_i if not sum to 1
    sum_f = df["f_i"].sum()
    if not np.isclose(sum_f, 1.0):
        st.warning(f"Sum of f_i is {sum_f:.4f} — normalizing fractions to sum to 1.")
        df["f_i"] = df["f_i"] / sum_f

    # compute D_sg (geometric mean)
    # D_sg = exp(sum f_i * ln(D_i))
    if (df["D_i"] <= 0).any():
        st.error("All D_i must be positive.")
        st.stop()
    D_sg = float(np.exp((df["f_i"] * np.log(df["D_i"])).sum()))
    st.write(f"Computed surface geometric mean D_sg = {D_sg:.4e} m")

    # Prepare tau_star_i
    if use_single_tau:
        df["tau_star_i"] = tau_star_ref
    else:
        if "tau_star_i" not in df.columns:
            st.error("CSV does not contain tau_star_i column and 'Use single tau*' is unchecked.")
            st.stop()
        else:
            pass

    # Optional Parker omega: let user input sigma_s and override omega
    use_custom_omega = st.sidebar.checkbox("Provide custom ω (mobility scaling) scalar (override Parker ω calc)", False)
    if use_custom_omega:
        omega_scalar = st.sidebar.number_input("ω (scalar)", value=1.0, format="%.4f")
    else:
        # fallback: set ω = 1 for minimal app. (Advanced: implement full sigma0/omega0 lookup)
        omega_scalar = 1.0

    # Compute model-specific Phi_i
    def phi_parker(row, D_sg, omega):
        # parameter phi_i = omega * tau_*,sg / 0.0386 * (D_i/D_sg)^(-0.0951)
        tau_star_sg = row["tau_star_i"]  # minimal app: use tau_star_i as reference
        return omega * (tau_star_sg / 0.0386) * (row["D_i"] / D_sg) ** (-0.0951)

    def Phi_parker(row, phi, tau_star_i):
        # apply Parker piecewise formula
        if phi > 1.59:
            return tau_star_i ** 3 * 1.2513 * (1 - 0.853 / phi) ** 4.5
        elif phi > 1.0:
            return tau_star_i ** 3 * np.exp(14.2 * (phi - 1) - 9.28 * (phi - 1) ** 2)
        else:
            return 0.0

    def Phi_ashida(row, tau_star_i, tau_star_ci):
        # Ashida-Michiue formula
        ts = tau_star_i
        tsc = tau_star_ci
        if ts <= tsc:
            return 0.0
        return 17.0 * (ts - tsc) * (np.sqrt(ts) - np.sqrt(tsc))

    def tau_star_c_egiazaroff(D_i, D_sg):
        ratio = D_i / D_sg
        if ratio < 0.4:
            return 0.0421 * D_sg / D_i
        else:
            return 0.433 / (np.log(19 * ratio) ** 2)

    def Phi_wilcock(row, phi, tau_star_i):
        # minimal implementation using the Wilcock & Crowe piecewise (as in slides)
        if phi < 1.0:
            # fallback small phi relation (rare)
            return tau_star_i ** 3 * 0.00218 * phi ** 14.2
        elif phi < 1.35:
            return tau_star_i ** 3 * 0.002 * phi ** 7.5
        else:
            return tau_star_i ** 3 * 14.0 * (1 - 0.894 / np.sqrt(phi)) ** 4.5

    # compute Phi and q_bi
    results = []
    for _, row in df.iterrows():
        D_i = float(row["D_i"])
        f_i = float(row["f_i"])
        tau_star_i = float(row["tau_star_i"])
        if model == "Parker 1990":
            phi_i = phi_parker(row, D_sg, omega_scalar)
            Phi_i = Phi_parker(row, phi_i, tau_star_i)
        elif model == "Ashida-Michiue 1972":
            tau_star_ci = tau_star_c_egiazaroff(D_i, D_sg)
            Phi_i = Phi_ashida(row, tau_star_i, tau_star_ci)
            phi_i = np.nan
        else:  # Wilcock-Crowe
            # minimal phi: reuse Parker's phi as placeholder; a better app would compute tau*ssrg etc.
            phi_i = phi_parker(row, D_sg, omega_scalar)
            Phi_i = Phi_wilcock(row, phi_i, tau_star_i)

        q_bi = f_i * np.sqrt(g * Delta * D_i ** 3) * Phi_i  # m^2/s (volumetric per unit width)
        results.append({"D_i": D_i, "f_i": f_i, "tau_star_i": tau_star_i,
                        "phi_i": phi_i, "Phi_i": Phi_i, "q_bi_m2s": q_bi})

    res_df = pd.DataFrame(results)
    res_df["q_bi_m2s"] = res_df["q_bi_m2s"].astype(float)
    q_b_tot = res_df["q_bi_m2s"].sum()

    st.subheader("Results")
    st.write(f"Model: {model} — total bedload q_b,tot = {q_b_tot:.6e} m^2/s (volumetric per unit width)")
    st.dataframe(res_df.style.format({"D_i": "{:.6f}", "f_i": "{:.3f}", "tau_star_i": "{:.4f}",
                                      "phi_i": "{:.4f}", "Phi_i": "{:.4e}", "q_bi_m2s": "{:.6e}"}))

    # Download results
    csv = res_df.to_csv(index=False)
    st.download_button("Download results CSV", data=csv, file_name="bedload_results.csv", mime="text/csv")

    st.info("Notes: this minimal app uses a simplified Parker implementation. For full Parker behavior, "
            "implement sigma0/omega0 lookup tables or provide ω directly. Use results for educational/quick checks only.")
