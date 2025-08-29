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
DATA_DIR   = Path(__file__).parent / "data"
SERVICE_TAX = 0.18
COI_GST     = 0.18
COI_SCALER  = 1.0
CE1_ANNUAL  = 0.08   # FIXED growth rate
MR = (1.0 + CE1_ANNUAL)**(1.0/12.0) - 1.0  # monthly rate

@st.cache_data
def load_rate_tables():
    charges = pd.read_csv(DATA_DIR / "rates_charges_by_year.csv")
    mort    = pd.read_csv(DATA_DIR / "mortality_grid.csv")
    funds   = pd.read_csv(DATA_DIR / "rates_fund_fmc.csv")
    windows = json.loads((DATA_DIR / "windows.json").read_text())
    return charges, mort, funds, windows

CHARGES_DF, MORT_DF, FUNDS_DF, WINDOWS = load_rate_tables()

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
    F_rate = float(row.get("AllocRate_F", 0.0))
    admin_rate = 0.0165 if policy_year <= 5 else 0.0
    K = mortality_rate_per_thousand(attained_age(entry_age, policy_year), gender)
    return dict(F=F_rate, admin_rate=admin_rate, K=K)

def sumproduct_fmc(allocation_dict):
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
    paid = sum(r["F"] for r in rows if r["Year"] == year)
    return abs(paid - annual_premium) < 1.0

# ===================
# Core projection
# ===================
def run_projection(
    la_dob, la_gender, option, lcb_flag,
    annual_premium, sum_assured,
    pt_years, ppt_years, mode,
    fmc_annual, use_fund_mix=False, alloc=None
):
    today = dt.date.today()
    la_age = today.year - la_dob.year - ((today.month, today.day) < (la_dob.month, la_dob.day))

    months = pt_years * 12
    results = []
    CK_prev = 0.0
    cum_prem = 0.0
    CH_hist  = []

    spec = modal_schedule(mode)
    N = spec["N"]
    scheduled = spec["months"]
    installment = annual_premium / N if N > 0 else annual_premium

    if use_fund_mix and alloc and abs(sum(alloc.values()) - 1.0) < 1e-8:
        fmc_effective = sumproduct_fmc(alloc)
    else:
        fmc_effective = fmc_annual

    for m in range(1, months+1):
        B = math.ceil(m/12); C = ((m-1)%12)+1; D = 1 if B <= pt_years else 0
        r = charges_for_year(B, la_age, la_gender)
        F_rate, admin_rate, K = r["F"], r["admin_rate"], r["K"]
        F_rate = min(max(F_rate, 0.0), 1.0)

        # Premium (at beginning of scheduled months, only while B <= PPT)
        F  = installment if (B <= ppt_years and C in scheduled) else 0.0

# Allocation charge & GST on allocation
        BS = F_rate * F * D                   # Premium Allocation Charge
        BT = BS * SERVICE_TAX                 # GST on Premium Allocation Charge (18%)

# Net premium allocated to fund
        BU = F - BS - BT

# Fund at start
        BV = (CK_prev + BU) * D


        BW = ((admin_rate * annual_premium) / 12.0) * D
        BX = BW * SERVICE_TAX
        BY = BV - BW - BX

        if D==1:
            db_core = max(sum_assured, BY)
            BZ = db_core
        else:
            BZ = 0
        CA = max(BZ - BY, 0)

        CB = CA * (K/12000.0) * COI_SCALER * D
        CC = CB * COI_GST
        CD = BY - CB - CC

        CE = CD * MR * D
        CF = (CD + CE) * (fmc_effective / 12.0)
        CG = CF * SERVICE_TAX

        CH = CD + CE - CF - CG
        CH_hist.append(CH)

        CI=CJ=0.0
        if D==1 and C==12:
            avg_Q1 = np.mean(CH_hist[-12:]) if CH_hist else 0.0
            avg_P1 = np.mean(CH_hist[-60:]) if CH_hist else 0.0
            GA = 0.0025*avg_Q1 if B>=6 else 0.0
            if B>=10 and B%5==0:
                BA = 0.0275*avg_P1 if B in (10,15) else 0.035*avg_P1
            else: BA=0.0
            CI = GA+BA
            if B>=6:
                if ppt_years==5: CJ=0
                elif ppt_years==6:
                    if B==6 and year_paid_this_year(results,B,annual_premium):
                        CJ=0.0015*avg_Q1
                else:
                    if B<=ppt_years and year_paid_this_year(results,B,annual_premium):
                        CJ=0.0015*avg_Q1
        CK = CH+CI+CJ
        CK_prev=CK

        results.append({
            "Year":B,"Month":C,"F":round(F),"BS":round(BS),"BT": round(BT),"BU":round(BU),"BV":round(BV),
            "BW":round(BW),"BX":round(BX),"BY":round(BY),"BZ":round(BZ),"CA":round(CA),
            "CB":round(CB),"CC":round(CC),"CD":round(CD),"CE":round(CE),"CF":round(CF),
            "CG":round(CG),"CH":round(CH),"CI":round(CI),"CJ":round(CJ),"CK":round(CK)
        })
    return pd.DataFrame(results)

# ===================
# Streamlit UI
# ===================
st.set_page_config(page_title="Wealth Ultima — BI Calculator", layout="wide")
st.title("Wealth Ultima — Standalone BI Calculator")
st.caption("Growth fixed at 8% p.a.; outputs rounded to nearest integer.")

with st.sidebar:
    st.header("Inputs")

    la_dob   = st.date_input("Life Assured DOB", value=dt.date(1990,1,1), min_value=dt.date(1900,1,1))
    la_gender= st.selectbox("Life Assured Gender",["Female","Male"],index=0)

    option    = st.selectbox("Death Benefit Option",["Option 1","Option 2"],index=1)
    lcb_flag  = st.selectbox("Little Champ Benefit (not used)",["No","Yes"],index=0)

    annual_premium = st.number_input("Annualised Premium (₹)",0.0,1e12,250000.0,1000.0,format="%.2f")
    sum_assured    = st.number_input("Sum Assured (₹)",0.0,1e12,3250000.0,50000.0,format="%.2f")

    pt_years  = st.number_input("Policy Term (years)",1,100,26,1)
    ppt_years = st.number_input("PPT (years)",1,100,15,1)

    mode      = st.selectbox("Premium Mode",["Annual","Semi-Annual","Quarterly","Monthly"],index=0)

    st.divider(); st.subheader("FMC")
    use_fund_mix = st.checkbox("Use fund mix (compute FMC via SUMPRODUCT)",value=False)
    if use_fund_mix:
        alloc={}; total=0
        for _,row in FUNDS_DF.iterrows():
            pct=st.number_input(f"{row['Fund']} (%)",0.0,100.0,0.0,1.0,key=row['Fund'])
            alloc[row['Fund']]=pct/100.0; total+=pct
        if abs(total-100)>1e-6: st.warning(f"Total={total:.2f}%. Make it 100%.")
        fmc_annual=sumproduct_fmc(alloc) if abs(total-100)<1e-6 else 0.0135
        st.write(f"Effective FMC: **{fmc_annual:.4f}**")
    else:
        fmc_annual=st.number_input("Effective FMC (annual)",0.0,0.1,0.0135,0.0005,format="%.4f")
        alloc=None

st.info("Annual growth is fixed at **8% p.a.** per agreed logic (not editable).")

if st.button("Run Illustration"):
    df=run_projection(la_dob,la_gender,option,lcb_flag,
                      annual_premium,sum_assured,
                      pt_years,ppt_years,mode,
                      fmc_annual,use_fund_mix,alloc)
    st.success("Done. Monthly table below.")
    st.dataframe(df,use_container_width=True)
    st.download_button("Download CSV",df.to_csv(index=False),"monthly_yieldcal.csv","text/csv")
