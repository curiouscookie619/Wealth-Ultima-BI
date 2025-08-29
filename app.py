# app.py
import math
import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ===================
# Data & constants
# ===================
DATA_DIR   = Path(__file__).parent / "data"
SERVICE_TAX = 0.18              # GST (18%)
COI_GST     = 0.18              # GST on COI (18%)
COI_SCALER  = 1.0               # keep 1.0 (no day-count proration)
CE1_ANNUAL  = 0.08              # GROWTH RATE FIXED at 8% p.a. (frozen)

def monthly_rate_from_annual(annual):
    # exact compounding conversion to monthly
    return (1.0 + float(annual))**(1.0/12.0) - 1.0

MR = monthly_rate_from_annual(CE1_ANNUAL)

@st.cache_data
def load_rate_tables():
    # Expect these files inside ./data/
    # rates_charges_by_year.csv must have columns: Year, AllocRate_F (decimal like 0.06)
    # mortality_grid.csv must have columns: Age, Male_perThousand, Female_perThousand
    # rates_fund_fmc.csv must have columns: Fund, FMC_annual (decimal e.g., 0.0135)
    # windows.json must have: {"Q1_months": 12, "P1_months": 60}
    charges = pd.read_csv(DATA_DIR / "rates_charges_by_year.csv")
    mort    = pd.read_csv(DATA_DIR / "mortality_grid.csv")
    funds   = pd.read_csv(DATA_DIR / "rates_fund_fmc.csv")
    windows = json.loads((DATA_DIR / "windows.json").read_text())
    return charges, mort, funds, windows

CHARGES_DF, MORT_DF, FUNDS_DF, WINDOWS = load_rate_tables()

def attained_age(entry_age, policy_year):
    # attained age at start of policy_year
    return entry_age + (policy_year - 1)

def mortality_rate_per_thousand(age, gender):
    # clamp to grid
    age = max(int(age), int(MORT_DF["Age"].min()))
    age = min(age,       int(MORT_DF["Age"].max()))
    row = MORT_DF.loc[MORT_DF["Age"] == age].iloc[0]
    return float(row["Female_perThousand"] if gender.lower().startswith("f") else row["Male_perThousand"])

def charges_for_year(policy_year, entry_age, gender):
    # pick row by Year; if beyond table, use last row
    row = CHARGES_DF.loc[CHARGES_DF["Year"] == policy_year]
    if row.empty:
        row = CHARGES_DF.iloc[[-1]]
    row = row.iloc[0]
    F_rate = float(row.get("AllocRate_F", 0.0))
    # admin we freeze as 1.65% AP p.a. for years 1..5, else 0 (per user)
    admin_rate = 0.0165 if policy_year <= 5 else 0.0
    # mortality by attained age/gender
    K = mortality_rate_per_thousand(attained_age(entry_age, policy_year), gender)
    return dict(F=F_rate, admin_rate=admin_rate, K=K)

def sumproduct_fmc(allocation_dict):
    df = FUNDS_DF.copy()
    df["alloc"] = df["Fund"].map(lambda f: allocation_dict.get(f, 0.0))
    return float((df["alloc"] * df["FMC_annual"]).sum())

def modal_schedule(mode):
    # months within a policy year when instalments are collected (at the beginning)
    if mode == "Annual":
        return {"N":1,  "months":[1]}
    if mode == "Semi-Annual":
        return {"N":2,  "months":[1,7]}
    if mode == "Quarterly":
        return {"N":4,  "months":[1,4,7,10]}
    if mode == "Monthly":
        return {"N":12, "months":list(range(1,13))}
    return {"N":1, "months":[1]}

def year_paid_this_year(rows, year, annual_premium):
    # check: have all due premiums for this policy year been paid?
    # i.e., sum(F) for that year's rows equals AP (± small tolerance)
    paid = sum(r["F"] for r in rows if r["Year"] == year)
    return abs(paid - annual_premium) < 0.05

# ===================
# Core projection
# ===================
def run_projection(
    la_age, la_gender, option, lcb_flag,
    annual_premium, sum_assured,
    pt_years, ppt_years, mode,
    fmc_annual, use_fund_mix=False, alloc=None
):
    months = pt_years * 12
    results = []
    CK_prev = 0.0
    cum_prem = 0.0
    CH_hist  = []  # running month-end pre-additions values (CH)

    # modal instalment & months
    spec = modal_schedule(mode)
    N = spec["N"]
    scheduled = spec["months"]
    installment = annual_premium / N if N > 0 else annual_premium

    # FMC effective (override by SUMPRODUCT if chosen and allocations valid)
    if use_fund_mix and alloc and abs(sum(alloc.values()) - 1.0) < 1e-8:
        fmc_effective = sumproduct_fmc(alloc)
    else:
        fmc_effective = fmc_annual

    for m in range(1, months+1):
        B = math.ceil(m/12)             # policy year
        C = ((m-1)%12)+1                # month (1..12)
        D = 1 if B <= pt_years else 0   # status (in force)

        # Premium F (collected at beginning of scheduled months, only while B<=PPT)
        F = installment if (B <= ppt_years and C in scheduled) else 0.0
        # Allocation charge BS = F_rate × F
        r = charges_for_year(B, la_age, la_gender)
        F_rate, admin_rate, K = r["F"], r["admin_rate"], r["K"]
        F_rate = min(max(F_rate, 0.0), 1.0)  # guard
        BS = F_rate * F * D

        # Net premium allocated BU
        BU = F - BS

        # Fund at start BV
        BV = (CK_prev + BU) * D

        # Admin charge BW (years 1..5 only) and GST BX
        BW = ((admin_rate * annual_premium) / 12.0) * D
        BX = BW * SERVICE_TAX
        # Fund value after CHRG (pre-COI) BY
        BY = BV - BW - BX

        # Death Benefit (user-specified): higher of (SA, BY)
        if D == 1:
            if str(option) in {"1","Option 1"}:
                # Even though Opt1 is typically SA + FV in some specs, user wants SA vs FV
                db_core = max(sum_assured, BY)
            elif str(option) in {"2","Option 2"}:
                db_core = max(sum_assured, BY)
            else:
                db_core = max(sum_assured, BY)
            BZ = db_core
        else:
            BZ = 0.0

        # Sum at risk CA = max(0, BZ - BY)
        CA = max(BZ - BY, 0.0)

        # COI CB = CA * (K per-mille / 12000); CC = GST on COI
        CB = CA * (K / 12000.0) * COI_SCALER * D
        CC = CB * COI_GST

        # Post-COI interim fund CD
        CD = BY - CB - CC

        # Investment income CE at fixed 8% p.a. (monthly MR)
        CE = CD * MR * D

        # FMC & GST on FMC
        CF = (CD + CE) * (fmc_effective / 12.0)
        CG = CF * SERVICE_TAX

        # FV before GA/Loyalty (pre-additions)
        CH = CD + CE - CF - CG
        CH_hist.append(CH)

        # ---- Year-end additions: CI (GA + BA) & CJ (Loyalty) ----
        CI = 0.0
        CJ = 0.0

        if D == 1 and C == 12:
            # Bases for averages
            q1 = int(WINDOWS.get("Q1_months", 12))   # 12
            p1 = int(WINDOWS.get("P1_months", 60))   # 60

            avg_Q1 = np.mean(CH_hist[-min(q1, len(CH_hist)):]) if CH_hist else 0.0
            avg_P1 = np.mean(CH_hist[-min(p1, len(CH_hist)):]) if CH_hist else 0.0

            # ---- Guaranteed Addition (GA) ----
            # 0.25% of avg of last 12 months, from end of 6th year onwards
            if B >= 6:
                GA = 0.0025 * avg_Q1
            else:
                GA = 0.0

            # ---- Booster Addition (BA) ----
            # At end of 10th & 15th: 2.75% of avg last 60 months
            # At end of 20th and every 5th thereafter: 3.50% of avg last 60 months
            if B >= 10 and (B % 5 == 0):
                if B in (10, 15):
                    BA = 0.0275 * avg_P1
                else:
                    BA = 0.0350 * avg_P1
            else:
                BA = 0.0

            CI = GA + BA

            # ---- Loyalty Addition (CJ) ----
            # Added at year-end from end of 6th policy year till end of PPT,
            # provided all premiums due for that year are paid.
            if B >= 6:
                if ppt_years == 5:
                    CJ = 0.0
                elif ppt_years == 6:
                    eligible = (B == 6)
                    # compliance check: all due for year B paid
                    eligible = eligible and year_paid_this_year(results, B, annual_premium)
                    CJ = 0.0015 * avg_Q1 if eligible else 0.0
                else:
                    # PPT >= 7 → years 6..PPT
                    if B <= ppt_years and year_paid_this_year(results, B, annual_premium):
                        CJ = 0.0015 * avg_Q1
                    else:
                        CJ = 0.0

        # Final fund value at end CK
        CK = CH + CI + CJ
        CK_prev = CK

        results.append({
            "Year": B, "Month": C, "Status": D,
            "F": round(F, 2),
            "BS": round(BS, 2), "BU": round(BU, 2),
            "BV": round(BV, 2),
            "BW": round(BW, 2), "BX": round(BX, 2),
            "BY": round(BY, 2),
            "BZ": round(BZ, 2), "CA": round(CA, 2),
            "CB": round(CB, 2), "CC": round(CC, 2),
            "CD": round(CD, 2), "CE": round(CE, 2),
            "CF": round(CF, 2), "CG": round(CG, 2),
            "CH": round(CH, 2),
            "CI": round(CI, 2), "CJ": round(CJ, 2),
            "CK": round(CK, 2),
        })

    return pd.DataFrame(results)

# ===================
# Streamlit UI
# ===================
st.set_page_config(page_title="Wealth Ultima — Standalone BI Calculator", layout="wide")
st.title("Wealth Ultima — Standalone BI Calculator")
st.caption("Uses embedded rate tables. Implements final agreed logics. Growth fixed at 8% p.a.")

with st.sidebar:
    st.header("Inputs")

    la_age    = st.number_input("Life Assured Age (last birthday)", 0, 99, 34, 1)
    la_gender = st.selectbox("Life Assured Gender", ["Female", "Male"], index=0)

    option    = st.selectbox("Death Benefit Option", ["Option 1", "Option 2"], index=1)
    lcb_flag  = st.selectbox("Little Champ Benefit (currently not used)", ["No", "Yes"], index=0)

    annual_premium = st.number_input("Annualised Premium (₹)", 0.0, 1e12, 250000.0, 1000.0, format="%.2f")
    sum_assured    = st.number_input("Sum Assured (₹)",        0.0, 1e12, 3250000.0, 50000.0, format="%.2f")

    pt_years  = st.number_input("Policy Term (years)", 1, 100, 26, 1)
    ppt_years = st.number_input("PPT (years)",         1, 100, 15, 1)

    # Premium Mode (now user-selectable)
    mode      = st.selectbox("Premium Mode", ["Annual", "Semi-Annual", "Quarterly", "Monthly"], index=0)

    st.divider(); st.subheader("FMC")
    use_fund_mix = st.checkbox("Use fund mix (compute FMC via SUMPRODUCT)", value=False)
    if use_fund_mix:
        st.caption("Enter allocations (must sum to 100%).")
        alloc = {}; total = 0.0
        for _, row in FUNDS_DF.iterrows():
            pct = st.number_input(f"{row['Fund']} (%)", 0.0, 100.0, 0.0, 1.0, key=row['Fund'])
            alloc[row['Fund']] = pct/100.0
            total += pct
        if abs(total - 100.0) > 1e-6:
            st.warning(f"Total allocation = {total:.2f}%. It must equal 100%.")
            fmc_annual = 0.0135  # fallback 1.35% p.a.
        else:
            fmc_annual = sumproduct_fmc(alloc)
            st.write(f"Effective FMC (annual): **{fmc_annual:.4f}**")
    else:
        fmc_annual = st.number_input("Effective FMC (annual)", 0.0, 0.10, 0.0135, 0.0005, format="%.4f")
        alloc = None

st.info("Annual growth is fixed at **8% p.a.** per agreed logic (not editable).")

if st.button("Run Illustration"):
    df = run_projection(
        la_age, la_gender, option, lcb_flag,
        annual_premium, sum_assured,
        pt_years, ppt_years, mode,
        fmc_annual, use_fund_mix, alloc
    )
    st.success("Done. Monthly table below.")
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False), "monthly_yieldcal.csv", "text/csv")
else:
    st.caption("Set inputs and click **Run Illustration**.")
