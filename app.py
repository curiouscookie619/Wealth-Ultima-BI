# app.py
import math
import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import datetime as dt

# ===================
# Data & constants
# ===================
DATA_DIR    = Path(__file__).parent / "data"
SERVICE_TAX = 0.18       # GST (18%)
COI_GST     = 0.18       # GST on COI (18%)
COI_SCALER  = 1.0        # (no day-count proration)
CE1_ANNUAL  = 0.08       # FIXED growth rate (8% p.a.)
MR          = (1.0 + CE1_ANNUAL)**(1.0/12.0) - 1.0  # monthly rate

@st.cache_data
def load_rate_tables():
    # Expected files in ./data
    charges = pd.read_csv(DATA_DIR / "rates_charges_by_year.csv")     # Year, AllocRate_F
    mort    = pd.read_csv(DATA_DIR / "mortality_grid.csv")            # Age, Male_perThousand, Female_perThousand
    funds   = pd.read_csv(DATA_DIR / "rates_fund_fmc.csv")            # Fund, FMC_annual (decimal)
    windows = json.loads((DATA_DIR / "windows.json").read_text())     # {"Q1_months": 12, "P1_months": 60}
    return charges, mort, funds, windows

CHARGES_DF, MORT_DF, FUNDS_DF, WINDOWS = load_rate_tables()

# ===================
# Helpers
# ===================
def attained_age(entry_age, policy_year):
    return entry_age + (policy_year - 1)

def mortality_rate_per_thousand(age, gender):
    age = max(int(age), int(MORT_DF["Age"].min()))
    age = min(age,       int(MORT_DF["Age"].max()))
    row = MORT_DF.loc[MORT_DF["Age"] == age].iloc[0]
    return float(row["Female_perThousand"] if gender.lower().startswith("f") else row["Male_perThousand"])

def charges_for_year(policy_year, entry_age, gender):
    row = CHARGES_DF.loc[CHARGES_DF["Year"] == policy_year]
    if row.empty: row = CHARGES_DF.iloc[[-1]]
    row = row.iloc[0]
    F_rate = float(row.get("AllocRate_F", 0.0))        # decimal (e.g., 0.06)
    admin_rate = 0.0165 if policy_year <= 5 else 0.0   # 1.65% p.a. only in Yrs 1–5
    K = mortality_rate_per_thousand(attained_age(entry_age, policy_year), gender)
    return dict(F=F_rate, admin_rate=admin_rate, K=K)

def sumproduct_fmc(allocation_dict):
    # allocation_dict: {FundName: decimal_share}
    df = FUNDS_DF.copy()
    df["alloc"] = df["Fund"].map(lambda f: allocation_dict.get(f, 0.0))
    return float((df["alloc"] * df["FMC_annual"]).sum())

def modal_schedule(mode):
    if mode == "Annual":      return {"N":1,  "months":[1]}
    if mode == "Semi-Annual": return {"N":2,  "months":[1,7]}
    if mode == "Quarterly":   return {"N":4,  "months":[1,4,7,10]}
    if mode == "Monthly":     return {"N":12, "months":list(range(1,13))}
    return {"N":1, "months":[1]}

def year_paid_this_year(rows, year, annual_premium):
    # All due premiums for the year are paid if sum(F) == Annual Premium (±1 rupee)
    paid = sum(r["F"] for r in rows if r["Year"] == year)
    return abs(paid - annual_premium) < 1.0

# ===================
# Core projection
# ===================
def run_projection(
    la_dob, la_gender,
    annual_premium, sum_assured,
    pt_years, ppt_years, mode,
    allocation_dict # {fund: decimal_share}, sum == 1.0
):
    today = dt.date.today()
    la_age = today.year - la_dob.year - ((today.month, today.day) < (la_dob.month, la_dob.day))

    months   = pt_years * 12
    results  = []
    CK_prev  = 0.0
    CH_hist  = []  # store CH each month for rolling averages

    # Premium mode → installment amount & schedule
    spec        = modal_schedule(mode)
    N           = spec["N"]
    scheduled   = spec["months"]
    installment = annual_premium / N if N > 0 else annual_premium

    # FMC always via SUMPRODUCT (must sum to 1.0)
    fmc_effective = sumproduct_fmc(allocation_dict)

    for m in range(1, months+1):
        B = math.ceil(m/12)                # Policy Year
        C = ((m-1)%12)+1                   # Month 1..12
        D = 1 if B <= pt_years else 0      # In-force flag

        r = charges_for_year(B, la_age, la_gender)
        F_rate, admin_rate, K = r["F"], r["admin_rate"], r["K"]
        F_rate = min(max(F_rate, 0.0), 1.0)

        # F: Premium at the beginning of the scheduled months (only while B<=PPT)
        F  = installment if (B <= ppt_years and C in scheduled) else 0.0

        # Allocation charge & GST on it
        BS = F_rate * F * D                            # Premium Allocation Charge
        BT = BS * SERVICE_TAX                          # GST on Allocation Charge (18%)

        # Net premium allocated to fund
        BU = F - BS - BT

        # Fund at start (this month)
        BV = (CK_prev + BU) * D

        # Admin charge (Yrs 1–5) & GST on admin
        BW = ((admin_rate * annual_premium) / 12.0) * D
        BX = BW * SERVICE_TAX

        # Fund Value after CHRG (pre-COI) — anchor
        BY = BV - BW - BX

        # BZ: Death Benefit = higher of (Sum Assured, BY) within term
        BZ = max(sum_assured, BY) if D == 1 else 0.0

        # CA: Sum at Risk
        CA = max(BZ - BY, 0.0)

        # CB: COI (mortality) & CC: GST on COI
        CB = CA * (K / 12000.0) * COI_SCALER * D
        CC = CB * COI_GST

        # CD: Fund after COI
        CD = BY - CB - CC

        # CE: Investment income at fixed 8% p.a. (monthly MR)
        CE = CD * MR * D

        # CF: FMC (monthly from annual) on (CD + CE) & CG: GST on FMC
        CF = (CD + CE) * (fmc_effective / 12.0)
        CG = CF * SERVICE_TAX

        # CH: Fund Value before additions
        CH = CD + CE - CF - CG
        CH_hist.append(CH)

        # Year-end additions: CI (GA + BA) and CJ (Loyalty)
        CI = 0.0
        CJ = 0.0
        if D == 1 and C == 12:
            q1 = int(WINDOWS.get("Q1_months", 12))   # 12-month window
            p1 = int(WINDOWS.get("P1_months", 60))   # 60-month window

            avg_Q1 = np.mean(CH_hist[-min(q1, len(CH_hist)):]) if CH_hist else 0.0
            avg_P1 = np.mean(CH_hist[-min(p1, len(CH_hist)):]) if CH_hist else 0.0

            # GA: 0.25% of last 12-month average, from end of 6th year onwards
            GA = 0.0025 * avg_Q1 if B >= 6 else 0.0

            # BA: Booster — end of every 5th year from year 10
            if B >= 10 and (B % 5 == 0):
                BA = (0.0275 * avg_P1) if B in (10, 15) else (0.0350 * avg_P1)
            else:
                BA = 0.0

            CI = GA + BA

            # CJ: Loyalty — 0.15% of last 12-month average
            # From end of 6th year till end of PPT, provided all due premiums for the year are paid
            if B >= 6:
                if ppt_years == 5:
                    CJ = 0.0
                elif ppt_years == 6:
                    CJ = 0.0015 * avg_Q1 if (B == 6 and year_paid_this_year(results, B, annual_premium)) else 0.0
                else:
                    if (B <= ppt_years) and year_paid_this_year(results, B, annual_premium):
                        CJ = 0.0015 * avg_Q1

        # CK: Fund Value at end
        CK = CH + CI + CJ
        CK_prev = CK

        # Round everything to nearest integer for output
        results.append({
            "Year": int(B),
            "Month": int(C),
            "F": round(F),
            "BS": round(BS), "BT": round(BT),
            "BU": round(BU),
            "BV": round(BV),
            "BW": round(BW), "BX": round(BX),
            "BY": round(BY),
            "BZ": round(BZ), "CA": round(CA),
            "CB": round(CB), "CC": round(CC),
            "CD": round(CD), "CE": round(CE),
            "CF": round(CF), "CG": round(CG),
            "CH": round(CH),
            "CI": round(CI), "CJ": round(CJ),
            "CK": round(CK),
        })

    return pd.DataFrame(results)

# ===================
# UI
# ===================
st.set_page_config(page_title="Wealth Ultima — BI Calculator", layout="wide")
st.title("Wealth Ultima — Standalone BI Calculator")
st.caption("Growth fixed at 8% p.a. | FMC always via fund allocation | All outputs rounded to nearest rupee.")

with st.sidebar:
    st.header("Policy Inputs")

    la_dob    = st.date_input("Life Assured DOB", value=dt.date(1990, 1, 1), min_value=dt.date(1900,1,1))
    la_gender = st.selectbox("Life Assured Gender", ["Female", "Male"], index=0)

    annual_premium = st.number_input("Annualised Premium (₹)", 0.0, 1e12, 250000.0, 1000.0, format="%.2f")
    sum_assured    = st.number_input("Sum Assured (₹)",        0.0, 1e12, 3250000.0, 50000.0, format="%.2f")

    col1, col2 = st.columns(2)
    with col1:
        pt_years  = st.number_input("Policy Term (years)", 1, 100, 26, 1)
    with col2:
        ppt_years = st.number_input("PPT (years)",         1, 100, 15, 1)

    mode = st.selectbox("Premium Mode", ["Annual", "Semi-Annual", "Quarterly", "Monthly"], index=0)

    st.divider()
    st.subheader("Fund Allocation (must total 100%)")

    # Friendlier allocation UI: default equal split + quick reset
    default_split = round(100.0 / max(len(FUNDS_DF), 1), 2)
    if st.button("Reset to equal split"):
        for _, row in FUNDS_DF.iterrows():
            st.session_state[row["Fund"]] = default_split

    alloc = {}
    total_pct = 0.0
    for _, row in FUNDS_DF.iterrows():
        key = row["Fund"]
        default_val = st.session_state.get(key, default_split)
        pct = st.number_input(f"{key} (%)", 0.0, 100.0, float(default_val), 0.5, key=key)
        alloc[key] = pct / 100.0
        total_pct += pct

    st.progress(min(total_pct / 100.0, 1.0))
    st.write(f"**Total: {total_pct:.2f}%**")

    if abs(total_pct - 100.0) < 1e-6:
        eff_fmc = sumproduct_fmc(alloc)
        st.success(f"Effective FMC (annual): **{eff_fmc:.4f}**")
    else:
        st.warning("Allocation must total 100% to run the illustration.")

st.info("Death Benefit option is hidden (DB = max(Sum Assured, Fund Value)). Growth fixed at 8% p.a.")

run = st.button("Run Illustration", type="primary")
if run:
    if abs(total_pct - 100.0) >= 1e-6:
        st.error("Please ensure allocation totals 100% before running.")
    else:
        df = run_projection(
            la_dob, la_gender,
            annual_premium, sum_assured,
            pt_years, ppt_years, mode,
            alloc
        )
        st.success("Done. Monthly table below.")
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), "monthly_yieldcal.csv", "text/csv")
else:
    st.caption("Set inputs and click **Run Illustration**.")
