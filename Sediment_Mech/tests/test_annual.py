import pandas as pd
from core.annual import compute_annual_bedload

def test_annual_runs():

    df_psd = pd.DataFrame({
        "D_i_m": [0.02, 0.01, 0.004],
        "f_i": [0.5, 0.3, 0.2]
    })

    df_fdc = pd.DataFrame({
        "Q_m3s": [5, 10, 20],
        "p_time": [0.4, 0.35, 0.25]
    })

    out = compute_annual_bedload(
        df_psd=df_psd,
        df_fdc=df_fdc,
        model="parker",
        rho=1000,
        rho_s=2650,
        g=9.81,
        S=0.01,
        width=10,
        n_manning=0.035
    )

    assert out["annual_bedload_m3_per_m"] >= 0
