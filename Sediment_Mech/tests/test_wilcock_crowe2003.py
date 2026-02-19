import pandas as pd
from Sediment_Mech.core.wilcock_crowe2003 import compute_wilcock_crowe2003

def test_wc2003_runs():
    df = pd.DataFrame({
        "D_i_m": [0.016, 0.008, 0.004, 0.001],  # include sand
        "f_i":   [0.35, 0.30, 0.20, 0.15]
    })
    out = compute_wilcock_crowe2003(
        df_psd=df,
        tau=30.0,          # Pa (example)
        rho=1000.0,
        rho_s=2650.0,
        g=9.81,
        D_sand=0.002,
        extended_low_f=True
    )
    assert out["q_b_tot"] >= 0.0
    assert 0.0 <= out["Fs"] <= 1.0
    assert out["D_sm"] > 0.0
