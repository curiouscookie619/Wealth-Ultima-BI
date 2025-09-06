# app_inforce_pdf.py
# Streamlit Cloud–ready: parses POS PDF with pdfplumber (text PDFs),
# asks only PTD + FV Today, then generates In-Force outputs per your spec.

import io, re, math, json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import datetime as dt

# ---- Optional date helper (robust human dates)
from dateutil import parser as dparser

# ---- PDF text extraction
import pdfplumber

# ===================
# Data & constants
# ===================
DATA_DIR    = Path(__file__).parent / "data"
SERVICE_TAX = 0.18       # GST (18%)
COI_GST     = 0.18       # GST on COI (18%)
COI_SCALER  = 1.0
DF_ANNUAL   = 0.05       # Discontinuance Fund growth (p.a.) within lock-in
CONT_ANNUAL = 0.08       # Continue path growth (p.a.)

@st.cache_data
def load_rate_tables():
    # In ./data:
    # - rates_charges_by_year.csv: Year, AllocRate_F
    # - mortality_grid.csv: Age, Male_perThousand, Female_perThousand
    # - rates_fund_fmc.csv: Fund, FMC_annual
    # - windows.json: {"Q1_months":12,"P1_months":60}
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
    admin_rate = 0.0165 if policy_year <= 5 else 0.0
    K = mortality_rate_per_thousand(la_age_today + (policy_year-1), gender)
    return dict(F=F_rate, admin_rate=admin_rate, K=K)

def filter_investable_funds(df):
    return df.loc[~df["Fund"].str.contains("discontinuance", case=False, na=False)].reset_index(drop=True)

def sumproduct_fmc(allocation_dict):
    df = FUNDS_DF.copy()
    df["alloc"] = df["Fund"].map(lambda f: allocation_dict.get(f, 0.0))
    return float((df["alloc"] * df["FMC_annual"]).sum())

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
    import re as _re
    rest = _re.sub(r'(\d)(?=(\d{2})+$)', r'\1,', rest)
    return rest + ',' + last3

# ---- Dates & status
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
# POS PDF parser (text PDFs via pdfplumber)
# ===================
DATE_PATTERNS = [
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
    r"\b(\d{1,2}\s*[A-Za-z]{3,9}\s*\d{2,4})\b",
    r"\b(\d{1,2}-[A-Za-z]{3}-\d{2,4})\b",
]
CURRENCY = r"₹?\s?([\d,]+(?:\.\d{1,2})?)"
PCT = r"(\d{1,3}(?:\.\d+)?)[ ]*%"

def _first_match(patterns, text: str):
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            return m.group(1).strip()
    return None

def _to_number(s: str):
    if not s: return None
    s = s.replace(",", "").replace("₹","").strip()
    try:
        return float(s)
    except:
        return None

def _parse_human_date(s: str) -> dt.date | None:
    if not s: return None
    try:
        return dparser.parse(s, dayfirst=True, fuzzy=True).date()
    except Exception:
        # try common manual formats
        for fmt in ("%d-%m-%Y","%d/%m/%Y","%d-%b-%Y","%d %b %Y","%d %B %Y","%d/%m/%y","%d-%m-%y"):
            try:
                return dt.datetime.strptime(s, fmt).date()
            except:
                continue
    return None

def _extract_issue_date_anchor(page) -> str | None:
    try:
        words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
    except Exception:
        return None
    for w in words or []:
        if w['text'].strip().lower() == 'date':
            y0, y1 = w['top'], w['bottom']
            x_right = w['x1']
            right_words = [
                rw for rw in words
                if rw['x0'] > x_right and not (rw['bottom'] < y0 or rw['top'] > y1)
            ]
            if right_words:
                right_words.sort(key=lambda rw: rw['x0'])
                val = right_words[0]['text'].strip().rstrip(",.;")
                hit = _first_match(DATE_PATTERNS, val)
                if hit: return hit
    return None

def parse_pos_pdf(file_bytes: bytes):
    out = {
        "issue_date": None, "pt_years": None, "ppt_years": None, "mode": None,
        "annual_premium": None, "sum_assured": None, "allocations": [], "raw_text": ""
    }
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        all_text = []
        issue_date_found = None
        for pidx, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            all_text.append(txt)
            if pidx == 0 and not issue_date_found:
                issue_date_found = _extract_issue_date_anchor(page)
        full = "\n".join(all_text)
        out["raw_text"] = full

        # Issue date
        out["issue_date"] = issue_date_found or _first_match(DATE_PATTERNS, full)

        # PT / PPT
        m_pt = re.search(r"(Policy\s*Term|PT)\s*[:\-]?\s*(\d{1,3})\s*(years|yrs)?", full, re.I)
        if m_pt: out["pt_years"] = int(m_pt.group(2))
        m_ppt = re.search(r"(Premium\s*Payment\s*Term|PPT)\s*[:\-]?\s*(\d{1,3})\s*(years|yrs)?", full, re.I)
        if m_ppt: out["ppt_years"] = int(m_ppt.group(2))

        # Mode
        m_mode = re.search(r"(Premium\s*Mode|Mode)\s*[:\-]?\s*(Annual|Semi-Annual|Quarterly|Monthly)", full, re.I)
        if m_mode: out["mode"] = m_mode.group(2).title()

        # Annual Premium
        m_ap = re.search(r"(Annual(?:ised)?\s*Premium)[^₹\d]*" + CURRENCY, full, re.I)
        if m_ap: out["annual_premium"] = _to_number(m_ap.group(1))

        # Sum Assured
        m_sa = re.search(r"(Sum\s*Assured)[^₹\d]*" + CURRENCY, full, re.I)
        if m_sa: out["sum_assured"] = _to_number(m_sa.group(1))

        # Fund allocations (simple “Fund … 25%” lines)
        allocations = []
        for line in full.splitlines():
            m = re.search(r"([A-Za-z][A-Za-z0-9 \-&]+?)\s+"+PCT+r"$", line.strip())
            if m:
                fund = m.group(1).strip()
                pct = float(m.group(2))
                if 0.0 <= pct <= 100.0:
                    allocations.append((fund, pct))
        out["allocations"] = allocations

    return out

# ===================
# In-Force projection engine
# ===================
def run_projection_inforce(
    issue_date, valuation_date, ptd_date,
    la_dob, la_gender,
    annual_premium, sum_assured,
    pt_years, ppt_years, mode,
    allocation_dict,
    fv0_seed,
):
    status, lk_end, ppt_end, in_lockin = determine_policy_status(issue_date, ptd_date, valuation_date, ppt_years)

    la_age_today = valuation_date.year - la_dob.year - ((valuation_date.month, valuation_date.day) < (la_dob.month, la_dob.day))
    months   = pt_years * 12
    results  = []
    CK_prev  = float(fv0_seed)  # seed with today's FV
    CH_hist  = []

    # schedule
    if mode == "Annual":      N, scheduled = 1,  [1]
    elif mode == "Semi-Annual": N, scheduled = 2, [1,7]
    elif mode == "Quarterly":   N, scheduled = 4, [1,4,7,10]
    else:                       N, scheduled = 12, list(range(1,13))
    installment = annual_premium / N if N > 0 else annual_premium

    fmc_effective = sumproduct_fmc(allocation_dict)

    # Start from next policy month after valuation
    start_month_index = months_between(issue_date, valuation_date) + 1
    start_month_index = max(1, start_month_index)

    for m in range(start_month_index, months+1):
        B = math.ceil(m/12)                # Policy Year
        C = ((m-1)%12)+1                   # Month 1..12
        D = 1 if B <= pt_years else 0

        # Growth regime
        annual_r = DF_ANNUAL if (status == "DISCONTINUED" and B <= 5) else CONT_ANNUAL
        MR = monthly_rate_from_annual(annual_r)

        r = charges_for_year(B, la_age_today, la_gender)
        F_rate, admin_rate, K = r["F"], r["admin_rate"], r["K"]
        F_rate = min(max(F_rate, 0.0), 1.0)

        # Premiums only in Premium-Paying
        F = installment if (status == "PREMIUM_PAYING" and B <= ppt_years and C in scheduled) else 0.0

        # Allocation & GST
        BS = F_rate * F * D
        BT = BS * SERVICE_TAX
        BU = F - BS - BT

        # Fund at start
        BV = (CK_prev + BU) * D

        # Admin & GST
        BW = ((admin_rate * annual_premium) / 12.0) * D
        BX = BW * SERVICE_TAX

        # Pre-COI
        BY = BV - BW - BX

        # DB anchor for SAR math (not displayed)
        BZ = max(sum_assured, BY) if D == 1 else 0.0
        CA = max(BZ - BY, 0.0)

        # Mortality & GST
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

        # Additions:
        CI = 0.0
        CJ = 0.0
        if D == 1 and C == 12:
            q1 = int(WINDOWS.get("Q1_months", 12))
            p1 = int(WINDOWS.get("P1_months", 60))
            avg_Q1 = np.mean(CH_hist[-min(q1, len(CH_hist)):]) if CH_hist else 0.0
            avg_P1 = np.mean(CH_hist[-min(p1, len(CH_hist)):]) if CH_hist else 0.0

            GA = 0.0025 * avg_Q1 if B >= 6 else 0.0
            if B >= 10 and (B % 5 == 0):
                BA = (0.0275 * avg_P1) if B in (10, 15) else (0.0350 * avg_P1)
            else:
                BA = 0.0

            if determine_policy_status(issue_date, ptd_date, valuation_date, ppt_years)[0] == "DISCONTINUED":
                CI, CJ = 0.0, 0.0
            elif determine_policy_status(issue_date, ptd_date, valuation_date, ppt_years)[0] in ("RPU","FULLY_PAID_UP"):
                CI, CJ = GA + BA, 0.0
            else:  # PREMIUM_PAYING
                CI = GA + BA
                if B >= 6:
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

    return pd.DataFrame(results), status, lockin_end_date(issue_date), ppt_end_date(issue_date, ppt_years)

# ===================
# Output builders
# ===================
def yearly_fv_and_chargepct(df_monthly):
    grp = df_monthly.groupby("Year", as_index=False).agg(
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

def fv_at_year_eoy(df_monthly, year_n: int):
    row = df_monthly[(df_monthly["Month"]==12) & (df_monthly["Year"]==year_n)]
    if row.empty: return None
    return float(row["CK_raw"].iloc[0])

def discontinuance_y5_value_from_today(fv0_seed, issue_date, valuation_date):
    end_y5 = add_years(issue_date, 5)
    months_to_y5_end = max(0, months_between(valuation_date, end_y5))
    mr = monthly_rate_from_annual(DF_ANNUAL)
    fv = float(fv0_seed)
    for _ in range(months_to_y5_end):
        fv *= (1.0 + mr)
    return fv

def partial_withdrawal_availability(df_monthly, pt_years):
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
st.set_page_config(page_title="Wealth Ultima — In-Force Illustration (PDF-powered)", layout="wide")
st.title("Wealth Ultima — In-Force Illustration (PDF-powered)")
st.caption("Upload POS PDF → auto-read details. Enter only PTD & Current FV. Growth: 8% continue; 5% in DF during lock-in discontinuance.")

with st.sidebar:
    st.header("Step 1 — Upload POS PDF")
    pdf_file = st.file_uploader("Upload POS PDF", type=["pdf"])
    parsed = {}
    if pdf_file is not None:
        parsed = parse_pos_pdf(pdf_file.read())

    st.header("Step 2 — Provide only these")
    ptd_date = st.date_input("Paid-to-Date (PTD)")
    fv0_seed = st.number_input("Fund Value Today (₹)", 0.0, 1e12, 0.0, 1000.0, format="%.2f")

# Parsed → defaults (preview)
st.subheader("Auto-read from PDF (preview)")
colA, colB, colC = st.columns(3)

_issue_date_raw = parsed.get("issue_date")
issue_date = _parse_human_date(_issue_date_raw) if _issue_date_raw else dt.date(2018,1,1)
pt_years  = int(parsed.get("pt_years") or 26)
ppt_years = int(parsed.get("ppt_years") or 15)
mode      = (parsed.get("mode") or "Annual")
annual_premium = float(parsed.get("annual_premium") or 250000.0)
sum_assured    = float(parsed.get("sum_assured") or 4000000.0)
alloc_from_pdf = parsed.get("allocations", [])

with colA:
    st.write(f"**Issue Date:** {issue_date.strftime('%d-%b-%Y')}")
    st.write(f"**PT / PPT:** {pt_years} / {ppt_years}")
with colB:
    st.write(f"**Mode:** {mode}")
    st.write(f"**Annual Premium:** ₹{format_in_indian_system(annual_premium)}")
with colC:
    st.write(f"**Sum Assured:** ₹{format_in_indian_system(sum_assured)}")

if alloc_from_pdf:
    st.markdown("**Fund Allocation (from PDF):**")
    total_pct = 0.0
    for fund, pct in alloc_from_pdf:
        st.write(f"- {fund}: {pct:.2f}%")
        total_pct += pct
    st.write(f"**Total:** {total_pct:.2f}%")
else:
    st.warning("No allocations detected in PDF. Using equal-weight across investable funds.")
    inv = filter_investable_funds(FUNDS_DF)
    eq = 1.0 / max(1, len(inv))
    alloc_from_pdf = [(f, 100.0*eq) for f in inv["Fund"].tolist()]

# Optional overrides
with st.expander("Override parsed values (optional)"):
    issue_date = st.date_input("Issue Date (override)", value=issue_date)
    pt_years   = st.number_input("Policy Term (years)", 1, 100, pt_years, 1)
    ppt_years  = st.number_input("PPT (years)", 1, 100, ppt_years, 1)
    mode       = st.selectbox("Premium Mode", ["Annual","Semi-Annual","Quarterly","Monthly"],
                              index=["Annual","Semi-Annual","Quarterly","Monthly"].index(mode))
    annual_premium = st.number_input("Annualised Premium (₹)", 0.0, 1e12, annual_premium, 1000.0)
    sum_assured    = st.number_input("Sum Assured (₹)", 0.0, 1e12, sum_assured, 50000.0)
    st.markdown("**Fund Allocation (edit if needed):**")
    alloc_dict = {}
    total = 0.0
    for fund, pct in alloc_from_pdf:
        v = st.number_input(fund, 0.0, 100.0, float(pct), 0.5, key=f"alloc_{fund}")
        alloc_dict[fund] = v/100.0
        total += v
    st.write(f"Total: {total:.2f}%")
else:
    alloc_dict = {fund: pct/100.0 for fund, pct in alloc_from_pdf}

# LA (kept in an expander; override if the POS PDF doesn’t have it)
with st.expander("Life Assured details (only if needed)"):
    la_dob    = st.date_input("Life Assured DOB", value=dt.date(1990,1,1))
    la_gender = st.selectbox("Life Assured Gender", ["Female","Male"], index=0)

run = st.button("Generate In-Force Tables", type="primary")

if run:
    if abs(sum(alloc_dict.values()) - 1.0) > 1e-6:
        st.error("Fund allocation must total 100%. Please adjust overrides.")
        st.stop()

    valuation_date = dt.date.today()
    df_m, status, lk_end, ppt_end = run_projection_inforce(
        issue_date, valuation_date, ptd_date,
        la_dob, la_gender,
        annual_premium=annual_premium, sum_assured=sum_assured,
        pt_years=pt_years, ppt_years=ppt_years, mode=mode,
        allocation_dict=alloc_dict,
        fv0_seed=fv0_seed
    )

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

    # KPIs: EOY of policy years that are +2y and +5y from today
    st.subheader("Projection KPIs (EOY, Continue @8%; DF @5% if discontinued in lock-in)")
    py_in_2 = policy_year_today(issue_date, add_years(valuation_date, 2), pt_years)
    py_in_5 = policy_year_today(issue_date, add_years(valuation_date, 5), pt_years)
    fv_2y = fv_at_year_eoy(df_m, py_in_2)
    fv_5y = fv_at_year_eoy(df_m, py_in_5)

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Fund Value in 2 Years (EOY)", f"₹{format_in_indian_system(fv_2y) if fv_2y else '—'}")
    with k2:
        st.metric("Fund Value in 5 Years (EOY)", f"₹{format_in_indian_system(fv_5y) if fv_5y else '—'}")

    if status == "DISCONTINUED":
        df_y5 = discontinuance_y5_value_from_today(fv0_seed, issue_date, valuation_date)
        with k3:
            st.metric("DF Payout at end of 5th Year (@5%)", f"₹{format_in_indian_system(df_y5)}")
        if fv_5y is not None:
            delta = fv_5y - df_y5
            st.info(
                f"**Comparison:** Continue (@8%, with premiums) in 5 years = ₹{format_in_indian_system(fv_5y)} "
                f"vs DF payout at end of 5th year = ₹{format_in_indian_system(df_y5)} "
                f"→ **Δ = ₹{format_in_indian_system(delta)}**"
            )

    st.subheader("Yearly Projection (End-of-Year)")
    yr_tbl = yearly_fv_and_chargepct(df_m)
    yr_fmt = yr_tbl.copy()
    yr_fmt["FV_EOY"] = yr_fmt["FV_EOY"].apply(lambda x: format_in_indian_system(x) if pd.notnull(x) else "—")
    st.dataframe(
        yr_fmt.rename(columns={"Year":"Policy Year","FV_EOY":"Fund Value at Year End","Charges_pct_of_FV":"Charges as % of FV"}),
        use_container_width=True
    )

    st.subheader("Partial Withdrawal — Eligibility & Maximum Available (EOY)")
    st.caption("Rules: from 6th policy year; min ₹500; FV after withdrawal ≥ 105% of total premiums paid.")
    pw_tbl = partial_withdrawal_availability(df_m, pt_years=pt_years)
    pw_fmt = pw_tbl.copy()
    pw_fmt["Max_Available"] = pw_fmt["Max_Available"].apply(lambda x: f"₹{format_in_indian_system(x)}" if x else "₹0")
    st.dataframe(
        pw_fmt.rename(columns={"Policy_Year":"Policy Year","Eligible":"Eligible?","Max_Available":"Max Available","Notes":"Notes"}),
        use_container_width=True
    )

else:
    st.caption("Upload the POS PDF, enter PTD & Current FV, then click **Generate In-Force Tables**.")
