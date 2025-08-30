# app.py
import math
import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import datetime as dt
import re

def format_in_indian_system(num):
    """
    Format an integer with Indian-style digit grouping:
    1234567 -> '12,34,567'
    123456  -> '1,23,456'
    """
    try:
        num = int(num)
    except Exception:
        return num

    s = str(num)
    # First group (last 3 digits)
    last3 = s[-3:]
    rest = s[:-3]
    if not rest:
        return last3
    # Add commas every 2 digits in the rest
    rest = re.sub(r'(\d)(?=(\d{2})+$)', r'\1,', rest)
    return rest + ',' + last3


# ===================
# Data & constants
# ===================
DATA_DIR    = Path(__file__).parent / "data"
SERVICE_TAX = 0.18       # GST (18%)
COI_GST     = 0.18       # GST on COI (18%)
COI_SCALER  = 1.0        # (no day-count proration)

@st.cache_data
def load_rate_tables():
    # Expected files in ./data
    # - rates_charges_by_year.csv: Year, AllocRate_F (decimal, e.g. 0.06)
    # - mortality_grid.csv: Age, Male_perThousand, Female_perThousand
    # - rates_fund_fmc.csv: Fund, FMC_annual (decimal, e.g. 0.0135)
    # - windows.json: {"Q1_months": 12, "P1_months": 60}
    charges = pd.read_csv(DATA_DIR / "rates_charges_by_year.csv")
    mort    = pd.read_csv(DATA_DIR / "mortality_grid.csv")
    funds   = pd.read_csv(DATA_DIR / "rates_fund_fmc.csv")
    windows = json.loads((DATA_DIR / "windows.json").read_text())
    return charges, mort, funds, windows

CHARGES_DF, MORT_DF, FUNDS_DF, WINDOWS = load_rate_tables()

# ===================
# Helpers
# ===================
def monthly_rate_from_annual(annual):
    return (1.0 + float(annual))**(1.0/12.0) - 1.0

def attained_age(entry_age, policy_year):
    return entry_age + (policy_year - 1)

def mortality_rate_per_thousand(age, gender):
    age = max(int(age), int(MORT_DF["Age"].min()))
    age = min(age,       int(MORT_DF["Age"].max()))
    row = MORT_DF.loc[MORT_DF["Age"] == age].iloc[0]
    return float(row["Female_perThousand"] if str(gender).lower().startswith("f") else row["Male_perThousand"])

def charges_for_year(policy_year, entry_age, gender):
    row = CHARGES_DF.loc[CHARGES_DF["Year"] == policy_year]
    if row.empty: row = CHARGES_DF.iloc[[-1]]
    row = row.iloc[0]
    F_rate = float(row.get("AllocRate_F", 0.0))        # decimal (e.g., 0.06)
    admin_rate = 0.0165 if policy_year <= 5 else 0.0   # 1.65% p.a. only in Yrs 1–5
    K = mortality_rate_per_thousand(attained_age(entry_age, policy_year), gender)
    return dict(F=F_rate, admin_rate=admin_rate, K=K)

def filter_investable_funds(df):
    # Remove "Discontinuance Fund"
    mask = ~df["Fund"].str.contains("discontinuance", case=False, na=False)
    return df.loc[mask].reset_index(drop=True)

def parse_percent_input(s: str) -> float:
    """
    Parse user-typed percent like '25', '25%', ' 25.5 '.
    Returns a float percent in [0, 100]. Non-numeric => 0.
    """
    if s is None:
        return 0.0
    s = str(s).strip().replace("%", "")
    try:
        val = float(s)
    except Exception:
        return 0.0
    return max(0.0, min(val, 100.0))

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
    paid = sum(r["F_raw"] for r in rows if r["Year"] == year)
    return abs(paid - annual_premium) < 1.0

# ===================
# Core projection (monthly engine)
# ===================
def run_projection(
    la_dob, la_gender,
    annual_premium, sum_assured,
    pt_years, ppt_years, mode,
    allocation_dict,                 # {fund: decimal_share}, sum == 1.0
    growth_annual                    # 0.04 or 0.08
):
    MR = monthly_rate_from_annual(growth_annual)

    # compute entry age from DOB at run time
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

        # CE: Investment income at growth_annual (monthly MR)
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

        # store raw (float) values; rounding is applied in yearly summary presentation
        results.append({
            "Year": B, "Month": C,
            "F_raw": F,
            "BS_raw": BS, "BT_raw": BT, "BU_raw": BU,
            "BV_raw": BV,
            "BW_raw": BW, "BX_raw": BX,
            "BY_raw": BY,
            "BZ_raw": BZ, "CA_raw": CA,
            "CB_raw": CB, "CC_raw": CC,
            "CD_raw": CD, "CE_raw": CE,
            "CF_raw": CF, "CG_raw": CG,
            "CH_raw": CH,
            "CI_raw": CI, "CJ_raw": CJ,
            "CK_raw": CK,
        })

    return pd.DataFrame(results)

# ===================
# Yearly summary builder
# ===================
def make_yearly_summary(df_monthly):
    """
    Build the yearly table with columns:
    Policy Year | Annualised Premium | Mortality Charges | Other Charges | GST |
    Fund value at End of Year | Surrender Value | Death Benefit
    """
    # sums within year
    grp = df_monthly.groupby("Year", as_index=False).agg(
        Annualised_Premium=("F_raw", "sum"),
        Mortality_Charges=("CB_raw", "sum"),
        Other_Charges=("BS_raw", "sum")  # start with allocation charge (excl GST)
    )

    # add Admin & FMC into 'Other Charges' (exclude any GSTs & exclude COI)
    admin_sum = df_monthly.groupby("Year")["BW_raw"].sum().reset_index(name="BW_sum")
    fmc_sum   = df_monthly.groupby("Year")["CF_raw"].sum().reset_index(name="CF_sum")
    grp = grp.merge(admin_sum, on="Year", how="left").merge(fmc_sum, on="Year", how="left")
    grp["Other_Charges"] = grp["Other_Charges"] + grp["BW_sum"].fillna(0.0) + grp["CF_sum"].fillna(0.0)
    grp.drop(columns=["BW_sum","CF_sum"], inplace=True)

    # total GST (BT + BX + CC + CG)
    gst_sum = (
        df_monthly.groupby("Year")[["BT_raw","BX_raw","CC_raw","CG_raw"]]
        .sum()
        .reset_index()
    )
    gst_sum["GST"] = gst_sum[["BT_raw","BX_raw","CC_raw","CG_raw"]].sum(axis=1)
    grp = grp.merge(gst_sum[["Year","GST"]], on="Year", how="left")

    # Fund value at end of year (CK of month 12), Death Benefit at end of year (BZ of month 12)
    eoy = df_monthly[df_monthly["Month"] == 12].loc[:, ["Year","CK_raw","BZ_raw"]].rename(
        columns={"CK_raw":"Fund_at_End_of_Year","BZ_raw":"Death_Benefit"}
    )
    grp = grp.merge(eoy, on="Year", how="left")

    # Surrender Value: 0 till 5th year, else equal to Fund value at end of year
    grp["Surrender_Value"] = np.where(grp["Year"] <= 5, 0.0, grp["Fund_at_End_of_Year"])

    # Round to nearest rupee for presentation
    for col in ["Annualised_Premium","Mortality_Charges","Other_Charges","GST","Fund_at_End_of_Year","Surrender_Value","Death_Benefit"]:
        grp[col] = grp[col].round(0).astype(int)

    grp = grp.sort_values("Year").reset_index(drop=True)
    return grp

# ===================
# UI
# ===================
st.set_page_config(page_title="Wealth Ultima — BI Calculator", layout="wide")
st.title("Wealth Ultima — BI Calculator")
st.caption("Dual-case yearly summaries at 4% and 8% p.a. | FMC via fund allocation | Type allocations freely (no steppers).")

with st.sidebar:
    st.header("Policy Inputs")

    la_dob    = st.date_input("Life Assured DOB", value=dt.date(1990, 1, 1), min_value=dt.date(1900,1,1))
    la_gender = st.selectbox("Life Assured Gender", ["Female", "Male"], index=0)

    annual_premium = st.number_input("Annualised Premium (₹)", 0.0, 1e12, 250000.0, 1000.0, format="%.2f")
    sum_assured    = st.number_input("Sum Assured (₹)",        0.0, 1e12, 4000000.0, 50000.0, format="%.2f")

    col1, col2 = st.columns(2)
    with col1:
        pt_years  = st.number_input("Policy Term (years)", 1, 100, 26, 1)
    with col2:
        ppt_years = st.number_input("PPT (years)",         1, 100, 15, 1)

    mode = st.selectbox("Premium Mode", ["Annual", "Semi-Annual", "Quarterly", "Monthly"], index=0)

    st.divider()
    st.subheader("Fund Allocation (must total 100%)")
    FUNDS_INVESTABLE = filter_investable_funds(FUNDS_DF)
    st.caption("Type % for each fund (e.g., 25 or 25%). Total must equal 100%.")

    alloc = {}
    total_pct = 0.0
    for _, row in FUNDS_INVESTABLE.iterrows():
        key = f"alloc_{row['Fund']}"
        default_txt = st.session_state.get(key, "")
        txt = st.text_input(row["Fund"], value=default_txt, key=key, placeholder="e.g., 25")
        pct = parse_percent_input(txt)
        alloc[row["Fund"]] = pct / 100.0
        total_pct += pct

    st.progress(min(total_pct / 100.0, 1.0))
    st.write(f"**Total: {total_pct:.2f}%**")

st.info("Death Benefit = max(Sum Assured, Fund Value after charges). Discontinuance Fund is excluded from allocations.")

# --- Bottom of app.py: action + output ---
st.info("Death Benefit = max(Sum Assured, Fund Value after charges). Discontinuance Fund is excluded from allocations.")

run = st.button("Generate Yearly Summaries", type="primary")
if run:
    if abs(total_pct - 100.0) >= 1e-6:
        st.error("Please ensure allocation totals 100% before running.")
    else:
        # Run monthly engine twice: 4% and 8%
        df_4 = run_projection(
            la_dob, la_gender,
            annual_premium, sum_assured,
            pt_years, ppt_years, mode,
            alloc, growth_annual=0.04
        )
        df_8 = run_projection(
            la_dob, la_gender,
            annual_premium, sum_assured,
            pt_years, ppt_years, mode,
            alloc, growth_annual=0.08
        )

        # Build yearly summaries
        yr4 = make_yearly_summary(df_4)
        yr8 = make_yearly_summary(df_8)

        # Helper: Indian formatting for display
        import re
        def format_in_indian_system(num):
            try:
                num = int(num)
            except Exception:
                return num
            s = str(num)
            last3 = s[-3:]
            rest = s[:-3]
            if not rest:
                return last3
            rest = re.sub(r'(\d)(?=(\d{2})+$)', r'\1,', rest)
            return rest + ',' + last3

        def format_summary(df):
            df_fmt = df.copy()
            for col in [
                "Annualised_Premium",
                "Mortality_Charges",
                "Other_Charges",
                "GST",
                "Fund_at_End_of_Year",
                "Surrender_Value",
                "Death_Benefit"
            ]:
                df_fmt[col] = df_fmt[col].apply(format_in_indian_system)
            return df_fmt

        # Display side-by-side with Indian formatting
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("At 4% p.a. Gross Investment Return")
            st.dataframe(
                format_summary(yr4).rename(columns={
                    "Year":"Policy Year",
                    "Annualised_Premium":"Annualised Premium",
                    "Mortality_Charges":"Mortality, Morbidity Charges",
                    "Other_Charges":"Other Charges*",
                    "GST":"GST",
                    "Fund_at_End_of_Year":"Fund at End of Year",
                    "Surrender_Value":"Surrender Value",
                    "Death_Benefit":"Death Benefit"
                }),
                use_container_width=True
            )
        with c2:
            st.subheader("At 8% p.a. Gross Investment Return")
            st.dataframe(
                format_summary(yr8).rename(columns={
                    "Year":"Policy Year",
                    "Annualised_Premium":"Annualised Premium",
                    "Mortality_Charges":"Mortality, Morbidity Charges",
                    "Other_Charges":"Other Charges*",
                    "GST":"GST",
                    "Fund_at_End_of_Year":"Fund at End of Year",
                    "Surrender_Value":"Surrender Value",
                    "Death_Benefit":"Death Benefit"
                }),
                use_container_width=True
            )

        # CSV downloads (keep raw integers for machine-readability)
        st.download_button(
            "Download Yearly Summary (4%) CSV",
            yr4.to_csv(index=False),
            "yearly_summary_4pct.csv",
            "text/csv"
        )
        st.download_button(
            "Download Yearly Summary (8%) CSV",
            yr8.to_csv(index=False),
            "yearly_summary_8pct.csv",
            "text/csv"
        )
else:
    st.caption("Set inputs and click **Generate Yearly Summaries**.")
