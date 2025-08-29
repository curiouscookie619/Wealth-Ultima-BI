import math
import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ===================
# Load embedded data
# ===================
DATA_DIR = Path(__file__).parent / "data"

def load_rate_tables():
    charges = pd.read_csv(DATA_DIR / "rates_charges_by_year.csv")  # Year,F,G,K,L,M,N,O,P,Q
    mort    = pd.read_csv(DATA_DIR / "mortality_grid.csv")         # Age,Male_perThousand,Female_perThousand
    funds   = pd.read_csv(DATA_DIR / "rates_fund_fmc.csv")         # Fund,FMC_annual
    windows = json.loads((DATA_DIR / "windows.json").read_text())  # P1_months,Q1_months
    return charges, mort, funds, windows

CHARGES_DF, MORT_DF, FUNDS_DF, WINDOWS = load_rate_tables()

# Global constants from your workbook
SERVICE_TAX = 0.18
COI_GST     = 0.18
COI_SCALER  = 1.0
DEFAULT_CE1 = 0.08
DEFAULT_FMC = 0.0135

def monthly_rate(annual):
    return (1.0 + float(annual))**(1.0/12.0) - 1.0

def attained_age(entry_age, policy_year):
    return entry_age + (policy_year - 1)

def mortality_rate_per_thousand(age, gender):
    age = max(int(age), int(MORT_DF["Age"].min()))
    age = min(age, int(MORT_DF["Age"].max()))
    row = MORT_DF.loc[MORT_DF["Age"] == age].iloc[0]
    return float(row["Female_perThousand"] if gender.lower().startswith("f") else row["Male_perThousand"])

def charges_for_year(policy_year, entry_age, gender):
    row = CHARGES_DF.loc[CHARGES_DF["Year"] == policy_year]
    if row.empty: row = CHARGES_DF.iloc[[-1]]
    row = row.iloc[0]
    age = attained_age(entry_age, policy_year)
    K = mortality_rate_per_thousand(age, gender)
    return dict(
        F=float(row["AllocRate_F"]), G=float(row["AdminParam_G"]), K=float(K),
        M=float(row["Add_M_pctPrem"]), N=float(row["Add_N_pctCumPrem"]),
        O=float(row["Add_O_pctAvgCH_Q1"]), P=float(row["Add_P_pctAvgCH_P1"]), Q=float(row["Add_Q_pctAvgCH_Q1"]),
    )

def sumproduct_fmc(allocation_dict):
    df = FUNDS_DF.copy()
    df["alloc"] = df["Fund"].map(lambda f: allocation_dict.get(f, 0.0))
    return float((df["alloc"] * df["FMC_annual"]).sum())

def run_projection(
    la_age, la_gender, option, lcb_flag,
    annual_premium, sum_assured,
    pt_years, ppt_years, mode,
    annual_growth, fmc_annual,
    *,
    apply_gst_on_alloc_admin=False,   # Excel: False
    include_rider_mortality=False,    # Excel main block: off unless you use PH/WOP leg
    rider_sor_fn=lambda year, month: 0.0,  # if enabled, provide SoR for rider
    rider_per_mille_fn=lambda age, gender: 0.0  # if enabled, rider mortality per-mille
):
    months = pt_years * 12
    results = []
    CK_prev = 0.0
    cum_prem = 0.0
    sum_F_to_date = 0.0
    CH_hist = []

    mode_map = {"Annual": 1, "Semi-Annual": 2, "Quarterly": 4, "Monthly": 12}
    pay_freq = mode_map.get(mode, 1)
    inc_every = int(12 / pay_freq)
    mr = monthly_rate(annual_growth)

    for m in range(1, months + 1):
        B = math.ceil(m / 12)              # Policy Year (Excel: Year)
        C = ((m - 1) % 12) + 1             # Policy Month
        D = 1 if B <= pt_years else 0      # Status

        r = charges_for_year(B, la_age, la_gender)
        F_rate, G = r["F"], r["G"]         # Allocation rate & Admin param
        K_main = r["K"]                    # Main mortality (per-thousand)

        # Premium flow (by frequency)
        F = annual_premium if (B <= ppt_years and ((C - 1) % inc_every == 0)) else 0.0
        sum_F_to_date += F
        cum_prem += F

        # Allocation charge (NO GST in Excel here)
        BS = F_rate * D * F
        BU = F - BS                         # Net allocated
        # Fund value at start = last month CK + new allocation
        BV = (BU + CK_prev) * D

        # Policy Admin (NO GST in Excel here)
        next_G = charges_for_year(B + 1, la_age, la_gender)["G"]
        BW = D * (G * annual_premium + next_G) / 12.0

        # Some legacy illustrations sometimes model GST on admin/alloc
        # but your Excel "YieldCal" block does NOT. Keep behind switch:
        if apply_gst_on_alloc_admin:
            BX = BW * SERVICE_TAX
        else:
            BX = 0.0

        # Fund Value after CHRG (Excel column)
        BY = BV - BW - BX

        # Death Benefit Option logic (match Excel incl. infant clause)
        floor_105 = 1.05 * cum_prem

        # attained age at current year
        att_age = attained_age(la_age, B)

        if str(option) in {"1", "Option 1"}:
            db_core = BY + sum_assured
        elif str(option) in {"2", "Option 2"}:
            if att_age < 1 and m <= 12:
                # Infant-first-year clause
                db_core = floor_105
            else:
                db_core = max(BY, sum_assured)
        else:
            db_core = BY  # fallback: equals FV

        BZ = max(floor_105, db_core) * D   # Death Benefit (Excel BZ)
        CA = max(BZ - BY, 0.0)             # Sum at Risk

        # COI main leg
        CB_main = CA * (K_main / 12000.0) * COI_SCALER

        # Optional second leg (e.g., PH/WOP rider) – default 0 unless enabled
        if include_rider_mortality:
            rider_sor = rider_sor_fn(B, C)                 # your SoR for rider base
            L_rider   = rider_per_mille_fn(att_age, la_gender)  # per-thousand
            CB_rider  = rider_sor * (L_rider / 12000.0)
        else:
            CB_rider = 0.0

        CB = CB_main + CB_rider
        CC = CB * COI_GST                  # GST only on COI (Excel)
        CD = BY - CB - CC                  # Interim post-COI

        # Investment income (Excel CE; mathematically same as CD * mr)
        CE = CD * mr * D

        # FMC + GST (Excel CF & CG)
        CF = (CD + CE) * (fmc_annual / 12.0)
        CG = CF * SERVICE_TAX

        # FV before GA (Excel CH)
        CH = CD + CE - CF - CG
        CH_hist.append(CH)

        # Year-end additions (Excel CI/CJ)
        CI = CJ = 0.0
        if D == 1 and C == 12 and B > 5:
            P1, Q1 = WINDOWS["P1_months"], WINDOWS["Q1_months"]
            avg_P1 = np.mean(CH_hist[-min(P1, len(CH_hist)):]) if CH_hist else 0.0
            avg_Q1 = np.mean(CH_hist[-min(Q1, len(CH_hist)):]) if CH_hist else 0.0

            # IMPORTANT: ensure your CSV stores DECIMALS (e.g., 0.0025 for 0.25%).
            M, N, O, P, Q = r["M"], r["N"], r["O"], r["P"], r["Q"]

            CI = (avg_Q1 * O) + (avg_P1 * P) + (annual_premium * M) + (sum_F_to_date * N)
            if B <= ppt_years:
                CJ = avg_Q1 * Q

        CK = CH + CI + CJ
        CK_prev = CK

        results.append({
            "Year": B, "Month": C,
            "F": round(F, 2), "BS": round(BS, 2), "BU": round(BU, 2),
            "BV": round(BV, 2), "BW": round(BW, 2), "BY": round(BY, 2),
            "BZ": round(BZ, 2), "CA": round(CA, 2), "CB": round(CB, 2), "CC": round(CC, 2),
            "CD": round(CD, 2), "CE": round(CE, 2), "CF": round(CF, 2), "CG": round(CG, 2),
            "CH": round(CH, 2), "CI": round(CI, 2), "CJ": round(CJ, 2), "CK": round(CK, 2)
        })

    return pd.DataFrame(results)


# ===============
# Streamlit UI
# ===============
st.set_page_config(page_title="Wealth Ultima — Standalone BI Calculator", layout="wide")
st.title("Wealth Ultima — Standalone BI Calculator")
st.caption("No Excel needed. Uses embedded rate tables. Produces full monthly YieldCal columns.")

with st.sidebar:
    st.header("Inputs")
    la_age    = st.number_input("Life Assured Age (last birthday)",0,99,34,1)
    la_gender = st.selectbox("Life Assured Gender",["Female","Male"],index=0)
    option    = st.selectbox("Death Benefit Option",["Option 1","Option 2"],index=0)
    lcb_flag  = st.selectbox("Little Champ Benefit",["No","Yes"],index=0)
    annual_premium = st.number_input("Annualised Premium (₹)",0.0,1e9,250000.0,1000.0,format="%.2f")
    sum_assured    = st.number_input("Sum Assured (₹)",0.0,1e9,3250000.0,50000.0,format="%.2f")
    pt_years  = st.number_input("Policy Term (years)",1,100,26,1)
    ppt_years = st.number_input("PPT (years)",1,100,15,1)
    mode      = st.selectbox("Premium Mode",["Annual"],index=0)
    st.divider(); st.subheader("Investment")
    use_fund_mix = st.checkbox("Use fund mix (compute FMC via SUMPRODUCT)",value=False)
    if use_fund_mix:
        st.caption("Enter allocations (must sum to 100%).")
        alloc={}; total=0
        for _,row in FUNDS_DF.iterrows():
            pct=st.number_input(f"{row['Fund']} (%)",0.0,100.0,0.0,1.0,key=row['Fund'])
            alloc[row['Fund']]=pct/100.0; total+=pct
        if abs(total-100)>1e-6: st.warning(f"Total={total:.2f}%. Make it 100%.")
        fmc_annual=sumproduct_fmc(alloc) if abs(total-100)<1e-6 else DEFAULT_FMC
        st.write(f"Effective FMC: **{fmc_annual:.4f}**")
    else:
        fmc_annual=st.number_input("Effective FMC (annual)",0.0,0.1,DEFAULT_FMC,0.0005,format="%.4f")
    annual_growth=st.number_input("Assumed Annual Growth (CE1)",0.0,1.0,DEFAULT_CE1,0.005,format="%.3f")

if st.button("Run Illustration"):
    df=run_projection(la_age,la_gender,option,lcb_flag,annual_premium,sum_assured,
                      pt_years,ppt_years,mode,annual_growth,fmc_annual)
    st.success("Done. Monthly table below.")
    st.dataframe(df,use_container_width=True)
    st.download_button("Download CSV",df.to_csv(index=False),"monthly_yieldcal.csv","text/csv")
else:
    st.info("Enter inputs on the left and click **Run Illustration**.")
