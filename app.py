# app_inforce.py
import math
import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import datetime as dt
import re

# ===================
# Data & constants
# ===================
DATA_DIR    = Path(__file__).parent / "data"
SERVICE_TAX = 0.18       # GST (18%)
COI_GST     = 0.18       # GST on COI (18%)
COI_SCALER  = 1.0
DF_ANNUAL   = 0.05       # Discontinuance Fund growth (p.a.) in lock-in
CONT_ANNUAL = 0.08       # Continue growth (p.a.)

@st.cache_data
def load_rate_tables():
    # Required files in ./data
    # - rates_charges_by_year.csv: Year, AllocRate_F (decimal)
    # - mortality_grid.csv: Age, Male_perThousand, Female_perThousand
    # - rates_fund_fmc.csv: Fund, FMC_annual (decimal)
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
def monthly_rate_from_annual(annual: float) -> float:
    return (1.0 + float(annual))**(1.0/12.0) - 1.0

def attained_age(entry_age, policy_year):  # (kept for future)
    return entry_age + (policy_year - 1)

def mortality_rate_per_thousand(age, gender):
    age = max(int(age), int(MORT_DF["Age"].min()))
    age = min(age,       int(MORT_DF["Age"].max()))
    row = MORT_DF.loc[MORT_DF["Age"] == age].iloc[0]
    return float(row["Female_perThousand"] if str(gender).lower().startswith("f") else row["Male_perThousand"])

def charges_for_year(policy_year, la_age_today, gender):
    row = CHARGES_DF.loc[CHARGES_DF["Year"] == policy_year]
    if row.empty: row = CHARGES_DF.iloc[[-1]]
    row = row.iloc[0]
    F_rate = float(row.get("AllocRate_F", 0.0))
    admin_rate = 0.0165 if policy_year <= 5 else 0.0  # 1.65% p.a. only in Yrs 1–5
    K = mortality_rate_per_thousand(la_age_today + (policy_year-1), gender)
    return dict(F=F_rate, admin_rate=admin_rate, K=K)

def filter_investable_funds(df):
    return df.loc[~df["Fund"].str.contains("discontinuance", case=False, na=False)].reset_index(drop=True)

def parse_percent_input(s: str) -> float:
    if s is None: return 0.0
    s = str(s).strip().replace("%", "")
    try:
        v = float(s)
    except Exception:
        return 0.0
    return max(0.0, min(v, 100.0))

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
    paid = sum(r["F_raw"] for r in rows if r["Year"] == year)
    return abs(paid - annual_premium) < 1.0

def format_in_indian_system(num):
    try:
        num = int(round(num))
    except Exception:
        return num
    s = str(num)
    last3 = s[-3:]; rest = s[:-3]
    if not rest: return last3
    rest = re.sub(r'(\d)(?=(\d{2})+$)', r'\1,', rest)
    return rest + ',' + last3

# ---------- Date helpers & Status logic ----------
def add_years(d: dt.date, years: int) -> dt.date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(month=2, day=28, year=d.year + years)

def months_between(d0: dt.date, d1: dt.date) -> int:
    sign = 1 if d1 >= d0 else -1
    a, b = (d0, d1) if sign == 1 else (d1, d0)
    m = (b.year - a.year) * 12 + (b.month - a.month)
    if b.day < a.day: m -= 1
    return sign * m

def policy_year_today(issue_date: dt.date, valuation_date: dt.date, pt_years: int) -> int:
    elapsed_months = max(0, months_between(issue_date, valuation_date))
    py = 1 + (elapsed_months // 12)
    return max(1, min(py, pt_years))

def lockin_end_date(issue_date: dt.date) -> dt.date:
    return add_years(issue_date, 5)

def ppt_end_date(issue_date: dt.date, ppt_years: int) -> dt.date:
    return add_years(issue_date, ppt_years)

def determine_policy_status(issue_date: dt.date, ptd_date: dt.date, valuation_date: dt.date, ppt_years: int):
    lk_end = lockin_end_date(issue_date)
    ppt_end = ppt_end_date(issue_date, ppt_years)
    in_lockin = valuation_date <= lk_end
    if ptd_date >= valuation_date:
        return "PREMIUM_PAYING", lk_end, ppt_end, in_lockin
    if valuation_date > ppt_end:
        return "FULLY_PAID_UP", lk_end, ppt_end, in_lockin
    if in_lockin:
        return "DISCONTINUED", lk_end, ppt_end, True
    return "RPU", lk_end, ppt_end, False

# ===================
# Core projection (monthly engine)
# ===================
def run_projection_inforce(
    issue_date, valuation_date, ptd_date,
    la_dob, la_gender,
    annual_premium, sum_assured,
    pt_years, ppt_years, mode,
    allocation_dict,
    fv0_seed,
):
    """Returns: df_monthly, status, lk_end, ppt_end, in_lockin"""
    status, lk_end, ppt_end, in_lockin = determine_policy_status(issue_date, ptd_date, valuation_date, ppt_years)

    la_age_today = valuation_date.year - la_dob.year - ((valuation_date.month, valuation_date.day) < (la_dob.month, la_dob.day))
    months   = pt_years * 12
    results  = []
    CK_prev  = float(fv0_seed)  # seed at valuation
    CH_hist  = []

    spec        = modal_schedule(mode)
    N           = spec["N"]
    scheduled   = spec["months"]
    installment = annual_premium / N if N > 0 else annual_premium
    fmc_effective = sumproduct_fmc(allocation_dict)

    # Start projecting from the next month after valuation
    start_month_index = months_between(issue_date, valuation_date) + 1
    start_month_index = max(1, start_month_index)

    for m in range(start_month_index, months+1):
        B = math.ceil(m/12)                # Policy Year
        C = ((m-1)%12)+1                   # Month 1..12
        D = 1 if B <= pt_years else 0

        # Annual growth for this month
        if status == "DISCONTINUED" and (B <= 5):
            annual_r = DF_ANNUAL
        else:
            annual_r = CONT_ANNUAL
        MR = monthly_rate_from_annual(annual_r)

        # Charges parameters
        r = charges_for_year(B, la_age_today, la_gender)
        F_rate, admin_rate, K = r["F"], r["admin_rate"], r["K"]
        F_rate = min(max(F_rate, 0.0), 1.0)

        # Premiums by status
        if status == "PREMIUM_PAYING":
            F = installment if (B <= ppt_years and C in scheduled) else 0.0
        else:
            F = 0.0

        # Allocation + GST
        BS = F_rate * F * D
        BT = BS * SERVICE_TAX
        BU = F - BS - BT

        # Fund at start
        BV = (CK_prev + BU) * D

        # Admin + GST
        BW = ((admin_rate * annual_premium) / 12.0) * D
        BX = BW * SERVICE_TAX

        # Pre-COI
        BY = BV - BW - BX

        # DB anchor for SAR math (not displayed)
        BZ = max(sum_assured, BY) if D == 1 else 0.0
        CA = max(BZ - BY, 0.0)

        # Mortality + GST
        CB = CA * (K / 12000.0) * COI_SCALER * D
        CC = CB * COI_GST

        # Post-COI
        CD = BY - CB - CC

        # Growth
        CE = CD * MR * D

        # FMC + GST
        CF = (CD + CE) * (fmc_effective / 12.0)
        CG = CF * SERVICE_TAX

        # Pre-additions
        CH = CD + CE - CF - CG
        CH_hist.append(CH)

        # Year-end additions
        CI = 0.0
        CJ = 0.0
        if D == 1 and C == 12:
            q1 = int(WINDOWS.get("Q1_months", 12))
            p1 = int(WINDOWS.get("P1_months", 60))
            avg_Q1 = np.mean(CH_hist[-min(q1, len(CH_hist)):]) if CH_hist else 0.0
            avg_P1 = np.mean(CH_hist[-min(p1, len(CH_hist)):]) if CH_hist else 0.0

            # GA (from Y6)
            GA = 0.0025 * avg_Q1 if B >= 6 else 0.0

            # Booster (Y10,15=2.75%; else 3.5% on 5y cadence)
            if B >= 10 and (B % 5 == 0):
                BA = (0.0275 * avg_P1) if B in (10, 15) else (0.0350 * avg_P1)
            else:
                BA = 0.0

            CI = GA + BA

            # Loyalty (only when premium-paying & year fully paid)
            if status == "PREMIUM_PAYING" and B >= 6:
                if ppt_years == 5:
                    pass
                elif ppt_years == 6:
                    CJ = 0.0015 * avg_Q1 if (B == 6 and year_paid_this_year(results, B, annual_premium)) else 0.0
                else:
                    if (B <= ppt_years) and year_paid_this_year(results, B, annual_premium):
                        CJ = 0.0015 * avg_Q1

        CK = CH + CI + CJ
        CK_prev = CK

        results.append({
            "Year": B, "Month": C,
            "F_raw": F,
            "BS_raw": BS, "BT_raw": BT, "BU_raw": BU,
            "BW_raw": BW, "BX_raw": BX,
            "BY_raw": BY,
            "CB_raw": CB, "CC_raw": CC,
            "CF_raw": CF, "CG_raw": CG,
            "CH_raw": CH,
            "CI_raw": CI, "CJ_raw": CJ,
            "CK_raw": CK,
            "annual_r": annual_r,
        })

    return pd.DataFrame(results), status, lk_end, ppt_end, in_lockin

# ===================
# Output builders
# ===================
def yearly_fv_and_chargepct(df_monthly):
    """Year | FV_EOY | Charges% (Allocation+Admin+Mortality+FMC+all GSTs) / FV_EOY"""
    grp = df_monthly.groupby("Year", as_index=False).agg(
        F=("F_raw","sum"),
        BS=("BS_raw","sum"), BT=("BT_raw","sum"),
        BW=("BW_raw","sum"), BX=("BX_raw","sum"),
        CB=("CB_raw","sum"), CC=("CC_raw","sum"),
        CF=("CF_raw","sum"), CG=("CG_raw","sum"),
    )
    eoy = df_monthly[df_monthly["Month"] == 12].loc[:, ["Year","CK_raw"]].rename(columns={"CK_raw":"FV_EOY"})
    out = grp.merge(eoy, on="Year", how="left")
    out["Total_Charges"] = out[["BS","BT","BW","BX","CB","CC","CF","CG"]].sum(axis=1)
    out["Charges_pct_of_FV"] = np.where(out["FV_EOY"]>0, (out["Total_Charges"]/out["FV_EOY"])*100.0, 0.0)
    tbl = out.loc[:, ["Year","FV_EOY","Charges_pct_of_FV"]].copy()
    tbl["FV_EOY"] = tbl["FV_EOY"].round(0).astype("Int64")
    tbl["Charges_pct_of_FV"] = tbl["Charges_pct_of_FV"].round(2)
    return tbl

def fv_at_year(df_monthly, year_n: int):
    row = df_monthly[(df_monthly["Month"]==12) & (df_monthly["Year"]==year_n)]
    if row.empty: return None
    return float(row["CK_raw"].iloc[0])

def discontinuance_y5_value(fv0_seed, months_to_y5_end):
    """Grow FV0 monthly at DF_ANNUAL until end of 5th policy year and return closing value."""
    if months_to_y5_end <= 0:
        return float(fv0_seed)
    mr = monthly_rate_from_annual(DF_ANNUAL)
    fv = float(fv0_seed)
    for _ in range(months_to_y5_end):
        fv *= (1.0 + mr)
    return fv

def partial_withdrawal_availability(df_monthly, pt_years):
    """EOY basis: PY>=6; FV_after >= 105% of total premiums paid (incl. that year). Min ₹500."""
    prem_by_year = df_monthly.groupby("Year")["F_raw"].sum().reindex(range(1, pt_years+1), fill_value=0.0)
    cum_prem = prem_by_year.cumsum()
    fv_eoy = df_monthly[df_monthly["Month"]==12].set_index("Year")["CK_raw"].reindex(range(1, pt_years+1)).fillna(0.0)

    rows = []
    for y in range(1, pt_years+1):
        fv = float(fv_eoy.get(y, 0.0))
        th = 1.05 * float(cum_prem.get(y, 0.0))
        if y < 6:
            rows.append(dict(Policy_Year=y, Eligible="No", Max_Available=0, Notes="Not allowed before 6th year"))
            continue
        max_avail = max(0.0, fv - th)
        if max_avail < 500.0:
            rows.append(dict(Policy_Year=y, Eligible="No", Max_Available=0, Notes="Would breach 105% rule / < ₹500"))
        else:
            rows.append(dict(Policy_Year=y, Eligible="Yes", Max_Available=int(round(max_avail)), Notes="OK"))
    return pd.DataFrame(rows)

# ===================
# UI
# ===================
st.set_page_config(page_title="Wealth Ultima — In-Force Illustration (IFI)", layout="wide")
st.title("Wealth Ultima — In-Force Illustration")
st.caption("Status auto-computed from PTD / Lock-in. Growth: 8% p.a. (continue) and 5% p.a. in Discontinuance Fund (lock-in).")

with st.sidebar:
    st.header("Policy Inputs")

    # Required: dates + FV today
    issue_date     = st.date_input("Issue Date", value=dt.date(2018, 1, 1), min_value=dt.date(1900,1,1))
    ptd_date       = st.date_input("Paid-to-Date (PTD)", value=dt.date(2025, 1, 1), min_value=dt.date(1900,1,1))
    valuation_date = st.date_input("Valuation Date", value=dt.date.today(), min_value=dt.date(1900,1,1))
    fv0_seed       = st.number_input("Fund Value Today (₹)", 0.0, 1e12, 250000.0, 1000.0, format="%.2f")

    # LA + SA
    la_dob    = st.date_input("Life Assured DOB", value=dt.date(1990, 1, 1), min_value=dt.date(1900,1,1))
    la_gender = st.selectbox("Life Assured Gender", ["Female", "Male"], index=0)
    sum_assured    = st.number_input("Sum Assured (₹)", 0.0, 1e12, 4000000.0, 50000.0, format="%.2f")

    # Premium & terming
    annual_premium = st.number_input("Annualised Premium (₹)", 0.0, 1e12, 250000.0, 1000.0, format="%.2f")
    col1, col2, col3 = st.columns(3)
    with col1:
        pt_years  = st.number_input("Policy Term (years)", 1, 100, 26, 1)
    with col2:
        ppt_years = st.number_input("PPT (years)",         1, 100, 15, 1)
    with col3:
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

run = st.button("Generate In-Force Outputs", type="primary")

if run:
    if abs(total_pct - 100.0) >= 1e-6:
        st.error("Please ensure allocation totals 100% before running.")
    else:
        # Run monthly engine (seeded with FV0; status auto-computed from dates)
        df_m, status, lk_end, ppt_end, in_lockin = run_projection_inforce(
            issue_date, valuation_date, ptd_date,
            la_dob, la_gender,
            annual_premium=annual_premium,
            sum_assured=sum_assured,
            pt_years=pt_years, ppt_years=ppt_years, mode=mode,
            allocation_dict=alloc,
            fv0_seed=fv0_seed
        )

        # Snapshot
        st.subheader("Snapshot")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write(f"**Issue Date:** {issue_date.strftime('%d-%b-%Y')}")
            st.write(f"**Valuation Date:** {valuation_date.strftime('%d-%b-%Y')}")
            st.write(f"**Lock-in End:** {lk_end.strftime('%d-%b-%Y')}")
        with c2:
            st.write(f"**Status (computed):** {status}")
            st.write(f"**PPT End:** {ppt_end.strftime('%d-%b-%Y')}")
            st.write(f"**Mode:** {mode}")
        with c3:
            st.write(f"**Fund Value Today:** ₹{format_in_indian_system(fv0_seed)}")
            st.write(f"**Annual Premium:** ₹{format_in_indian_system(annual_premium)}")
            st.write(f"**PT/PPT:** {pt_years} / {ppt_years}")

        # Projection Summary KPIs (8% continue; DF 5% only for lock-in discontinuance)
        st.subheader("Projection Summary (8% p.a.; DF at 5% p.a. in lock-in)")
        # Map “in 2 years” and “in 5 years” to the policy years at those future dates
        py_in_2 = policy_year_today(issue_date, add_years(valuation_date, 2), pt_years)
        py_in_5 = policy_year_today(issue_date, add_years(valuation_date, 5), pt_years)
        fv_2y = fv_at_year(df_m, py_in_2)
        fv_5y = fv_at_year(df_m, py_in_5)

        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Fund Value in 2 Years (Continue @8%)", f"₹{format_in_indian_system(fv_2y) if fv_2y else '—'}")
        with colB:
            st.metric("Fund Value in 5 Years (Continue @8%)", f"₹{format_in_indian_system(fv_5y) if fv_5y else '—'}")

        # If currently Discontinued within lock-in → DF payout at end of Y5 @5% and comparison vs Continue @8%
        if status == "DISCONTINUED":
            end_y5 = add_years(issue_date, 5)
            months_to_y5_end = max(0, months_between(valuation_date, end_y5))
            df_y5 = discontinuance_y5_value(fv0_seed, months_to_y5_end)
            with colC:
                st.metric("DF Payout at end of 5th Year (@5%)", f"₹{format_in_indian_system(df_y5)}")
            # Compare vs Continue path FV at 5 years from now
            if fv_5y is not None:
                delta = fv_5y - df_y5
                st.info(
                    f"**Comparison:** Continue (@8%, with premiums) in 5 years = ₹{format_in_indian_system(fv_5y)} "
                    f"vs DF payout at end of 5th year = ₹{format_in_indian_system(df_y5)} "
                    f"→ **Δ = ₹{format_in_indian_system(delta)}**"
                )

        # Yearly Projection (EOY FV + Charges% of FV)
        st.subheader("Yearly Projection (End-of-Year)")
        yr_tbl = yearly_fv_and_chargepct(df_m)
        yr_tbl_fmt = yr_tbl.copy()
        yr_tbl_fmt["FV_EOY"] = yr_tbl_fmt["FV_EOY"].apply(lambda x: format_in_indian_system(x) if pd.notnull(x) else "—")
        st.dataframe(yr_tbl_fmt.rename(columns={
            "Year":"Policy Year",
            "FV_EOY":"Fund Value at Year End",
            "Charges_pct_of_FV":"Charges as % of FV"
        }), use_container_width=True)

        # Partial Withdrawals (Year-wise availability)
        st.subheader("Partial Withdrawal — Eligibility & Maximum Available (EOY basis)")
        st.caption("Rules: Allowed from 6th policy year and LA age ≥18; min ₹500; FV after withdrawal ≥ 105% of total premiums paid. (Top-up precedence to be applied when history is connected.)")
        pw_tbl = partial_withdrawal_availability(df_m, pt_years=pt_years)
        pw_fmt = pw_tbl.copy()
        pw_fmt["Max_Available"] = pw_fmt["Max_Available"].apply(lambda x: f"₹{format_in_indian_system(x)}" if x else "₹0")
        st.dataframe(pw_fmt.rename(columns={
            "Policy_Year":"Policy Year",
            "Eligible":"Eligible?",
            "Max_Available":"Max Available",
            "Notes":"Notes"
        }), use_container_width=True)

        # Tax & switching notes
        st.subheader("Tax & Switching (Quick Notes)")
        st.markdown(
            "- **Maturity proceeds tax-free** under Section 10(10D) (pre-2021 policies with annual premium ≤ 10% of SA).\n"
            "- **Death benefit** is always tax-free.\n"
            "- **Fund switches** inside ULIP are unlimited & **not taxable**; MF switches are taxable events.\n"
            "- Buying fresh insurance later is typically **more expensive** (higher age, medicals, GST)."
        )

else:
    st.caption("Fill inputs and click **Generate In-Force Outputs**.")
