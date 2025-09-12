# app.py — Streamlit Cloud–ready In-Force Illustration (PDF-powered)
# Upload POS PDF -> auto-read Issue Date, PT/PPT, Mode, Premium, SA, Allocations, LA Name
# Enter only PTD + Current FV. Projections: 8% continue; 5% DF in lock-in if discontinued.
# Retention Pitch: bullet points + dynamic highlights of strongest hooks.
# UPDATED: Issue Date parsing — prioritize KV fields; fallback to signature line; then safe scan:
#          • skip parentheses (e.g., fund codes "(ULIF00118/08/11...)")
#          • skip lines with "ULIF"
#          • accept only years >= 2015

import io, re, math, json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import datetime as dt

from dateutil import parser as dparser
import pdfplumber

# ===================
# Data & constants
# ===================
DATA_DIR    = Path(__file__).parent / "data"
SERVICE_TAX = 0.18
COI_GST     = 0.18
COI_SCALER  = 1.0
DF_ANNUAL   = 0.05   # Discontinuance fund growth (lock-in)
CONT_ANNUAL = 0.08   # Continue growth
ULIP_TAX_CHANGE_DATE = dt.date(2021, 2, 1)  # Budget 2021 ULIP change reference date

@st.cache_data
def load_rate_tables():
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

def add_years(d: dt.date, years: int) -> dt.date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(month=2, day=28, year=d.year + years)

def add_months(d: dt.date, months: int) -> dt.date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    day = min(d.day, [31,29 if y%4==0 and (y%100!=0 or y%400==0) else 28,31,30,31,30,31,31,30,31,30,31][m-1])
    return dt.date(y, m, day)

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

# Estimate total premium paid till PTD (assumes premiums were paid as scheduled until PTD)
def premiums_paid_till_ptd(issue_date: dt.date, ptd_date: dt.date, ppt_years: int, annual_premium: float, mode: str) -> float:
    if ptd_date is None: return 0.0
    if mode == "Annual":      scheduled = [1]
    elif mode == "Semi-Annual": scheduled = [1,7]
    elif mode == "Quarterly":   scheduled = [1,4,7,10]
    else:                       scheduled = list(range(1,13))
    N = len(scheduled)
    installment = annual_premium / N if N else annual_premium

    count = 0
    for y in range(1, ppt_years+1):
        for s in scheduled:
            m_index = (y-1)*12 + s  # policy-month index (1-based)
            due_date = add_months(issue_date, m_index-1)
            if due_date <= min(ptd_date, dt.date.today()):
                count += 1
    return count * installment

# ===================
# POS PDF parser
# ===================
PCT = r"(\d{1,3}(?:\.\d+)?)[ ]*%"

def _to_number(s: str):
    if not s: return None
    s = s.replace(",", "").replace("₹","").strip()
    try: return float(s)
    except: return None

def _parse_human_date(s: str) -> dt.date:
    if not s: return None
    s = s.strip()
    try:
        d = dparser.parse(s, dayfirst=True, fuzzy=True).date()
        if d.year < 100: d = dt.date(2000 + d.year, d.month, d.day)
        return d
    except Exception:
        pass
    # Fallbacks
    m = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{2})$", s)
    if m:
        dd, mm, yy = map(int, m.groups())
        return dt.date(2000+yy, mm, dd)
    m = re.match(r"^(\d{1,2})[ -/]([A-Za-z]{3,9})[ -/](\d{2})$", s)
    if m:
        dd = int(m.group(1))
        mm = dt.datetime.strptime(m.group(2)[:3], "%b").month
        yy = int(m.group(3))
        return dt.date(2000+yy, mm, dd)
    m = re.match(r"^(\d{1,2})[-/](\d{2,4})$", s)
    if m:
        mm, yy = int(m.group(1)), int(m.group(2))
        if yy < 100: yy += 2000
        return dt.date(yy, mm, 1)
    m = re.match(r"^([A-Za-z]{3,9})[-/](\d{2,4})$", s)
    if m:
        mm = dt.datetime.strptime(m.group(1)[:3], "%b").month
        yy = int(m.group(2))
        if yy < 100: yy += 2000
        return dt.date(yy, mm, 1)
    return None

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+"," ", s.lower()).strip()

# --- NEW: year filter (>= 2015) ---
def _valid_year(y: int) -> bool:
    return y >= 2015

def _parse_human_date_safe(s: str):
    d = _parse_human_date(s)
    if d and _valid_year(d.year):
        return d
    return None

def parse_pos_pdf(file_bytes: bytes):
    out = {
        "issue_date": None, "pt_years": None, "ppt_years": None, "mode": None,
        "annual_premium": None, "sum_assured": None, "allocations": [], "raw_text": "",
        "la_name": None
    }

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        all_text = []
        kv = {}
        allocations = []

        for page in pdf.pages:
            txt = page.extract_text() or ""
            all_text.append(txt)

            # Tables: KV & Allocations
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []

            for tbl in tables:
                rows = []
                for r in tbl or []:
                    if not r: continue
                    rows.append([(c or "").strip() for c in r])

                # Key-Value grid
                for r in rows:
                    if len(r) >= 2 and r[0] and r[1]:
                        lab = _norm(r[0]); val = r[1].strip()
                        if len(lab) > 3 and val:
                            kv[lab] = val

                # Allocation table
                hdr_idx = None
                for i, r in enumerate(rows):
                    joined = " | ".join(r).lower()
                    if "fund" in joined and "allocation" in joined:
                        hdr_idx = i; break
                if hdr_idx is not None:
                    for r in rows[hdr_idx+1:]:
                        pct_cell = next((c for c in r if re.search(PCT, c or "")), None)
                        if pct_cell:
                            m = re.search(PCT, pct_cell)
                            pct = float(m.group(1))
                            fund_name = next((c for c in r if isinstance(c, str) and c.strip()), "").strip()
                            if fund_name and 0 <= pct <= 100:
                                allocations.append((fund_name, pct))

        full = "\n".join(all_text)
        out["raw_text"] = full

    # Map KV labels to fields
    def get_kv(*aliases):
        for a in aliases:
            v = kv.get(_norm(a))
            if v: return v
        return None

    # LA name
    out["la_name"] = get_kv("Name of the Life Assured", "Name of Life Assured", "Life Assured", "LA Name")

    # PT / PPT
    pt_val  = get_kv("Policy Term (in Years)", "Policy Term in Years", "Policy Term")
    ppt_val = get_kv("Premium Payment Term (in Years)", "Premium Payment Term in Years", "PPT")
    try: out["pt_years"]  = int(re.search(r"\d{1,3}", pt_val).group(0)) if pt_val else None
    except: pass
    try: out["ppt_years"] = int(re.search(r"\d{1,3}", ppt_val).group(0)) if ppt_val else None
    except: pass

    # Mode
    mode_val = get_kv("Mode of payment of premium", "Premium Mode", "Mode")
    if mode_val:
        mv = mode_val.strip().title()
        if mv in ("Annual","Semi-Annual","Quarterly","Monthly"):
            out["mode"] = mv

    # AP / SA
    ap_val = (get_kv("Annualized Premium (in Rupees)", "Annualised Premium (in Rupees)")
              or get_kv("Annualized Premium in Rupees", "Annualised Premium in Rupees")
              or get_kv("Amount of Instalment Premium (in Rupees)"))
    out["annual_premium"] = _to_number(ap_val)

    sa_val = get_kv("Sum Assured (in Rupees)", "Sum Assured in Rupees", "Sum Assured")
    out["sum_assured"] = _to_number(sa_val)

    # --- NEW: Issue/Commencement Date extraction (safe & prioritized) ---
    issue_val = get_kv("Date of Commencement", "Policy Commencement Date", "Issue Date", "Policy Start Date")
    if issue_val:
        d = _parse_human_date_safe(issue_val)
        if d:
            out["issue_date"] = d.strftime("%d-%b-%Y")

    # Fallback 1: signature line (Place ... Date ...)
    if not out["issue_date"]:
        m = re.search(
            r"(?:Place[^\\n]*?Date\s*[:\-]?\s*)(\d{1,2}[ -/][A-Za-z]{3,9}[ -/]\d{2,4}|\d{1,2}[ -/]\d{1,2}[ -/]\d{2,4})",
            out["raw_text"], re.IGNORECASE
        )
        if m:
            d = _parse_human_date_safe(m.group(1))
            if d:
                out["issue_date"] = d.strftime("%d-%b-%Y")

    # Fallback 2: scan text lines, but skip parentheses & ULIF and enforce year >= 2015
    if not out["issue_date"]:
        for line in out["raw_text"].splitlines():
            if "ULIF" in line.upper():
                continue
            # Remove bracketed substrings entirely
            line_no_paren = re.sub(r"\([^)]*\)", "", line)
            m = re.search(
                r"(\d{1,2}[ -/][A-Za-z]{3,9}[ -/]\d{2,4}|\d{1,2}[ -/]\d{1,2}[ -/]\d{2,4})",
                line_no_paren
            )
            if m:
                d = _parse_human_date_safe(m.group(1))
                if d:
                    out["issue_date"] = d.strftime("%d-%b-%Y")
                    break

    out["allocations"] = allocations
    if not out["allocations"]:
        for line in out["raw_text"].splitlines():
            line = line.strip()
            m = re.search(r"([A-Za-z][A-Za-z0-9 \-&/]+?)\s+"+PCT+r"$", line)
            if m:
                fund = m.group(1).strip()
                pct  = float(m.group(2))
                if 0 <= pct <= 100:
                    out["allocations"].append((fund, pct))

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
    CK_prev  = float(fv0_seed)
    CH_hist  = []

    if mode == "Annual":      N, scheduled = 1,  [1]
    elif mode == "Semi-Annual": N, scheduled = 2, [1,7]
    elif mode == "Quarterly":   N, scheduled = 4, [1,4,7,10]
    else:                       N, scheduled = 12, list(range(1,13))
    installment = annual_premium / N if N > 0 else annual_premium

    fmc_effective = sumproduct_fmc(allocation_dict)

    start_month_index = months_between(issue_date, valuation_date) + 1
    start_month_index = max(1, start_month_index)

    for m in range(start_month_index, months+1):
        B = math.ceil(m/12)
        C = ((m-1)%12)+1
        D = 1 if B <= pt_years else 0

        annual_r = DF_ANNUAL if (status == "DISCONTINUED" and B <= 5) else CONT_ANNUAL
        MR = monthly_rate_from_annual(annual_r)

        r = charges_for_year(B, la_age_today, la_gender)
        F_rate, admin_rate, K = r["F"], r["admin_rate"], r["K"]
        F_rate = min(max(F_rate, 0.0), 1.0)

        F = installment if (status == "PREMIUM_PAYING" and B <= ppt_years and C in scheduled) else 0.0

        BS = F_rate * F * D
        BT = BS * SERVICE_TAX
        BU = F - BS - BT

        BV = (CK_prev + BU) * D

        BW = ((admin_rate * annual_premium) / 12.0) * D
        BX = BW * SERVICE_TAX

        BY = BV - BW - BX

        BZ = max(sum_assured, BY) if D == 1 else 0.0
        CA = max(BZ - BY, 0.0)

        CB = CA * (K / 12000.0) * COI_SCALER * D
        CC = CB * COI_GST

        CD = BY - CB - CC

        CE = CD * MR * D

        CF = (CD + CE) * (fmc_effective / 12.0)
        CG = CF * SERVICE_TAX

        CH = CD + CE - CF - CG
        CH_hist.append(CH)

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

            if status == "DISCONTINUED":
                CI, CJ = 0.0, 0.0
            elif status in ("RPU","FULLY_PAID_UP"):
                CI, CJ = GA + BA, 0.0
            else:
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
def yearly_fv_series(df_monthly, pt_years):
    eoy = (
        df_monthly[df_monthly["Month"] == 12]
        .loc[:, ["Year","CK_raw"]]
        .rename(columns={"CK_raw":"FV_EOY"})
        .drop_duplicates(subset=["Year"])
        .sort_values("Year")
    )
    idx = pd.Index(range(1, pt_years+1), name="Year")
    eoy = eoy.set_index("Year").reindex(idx).reset_index()
    return eoy

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

# Preview (with debug expander)
st.subheader("Auto-read from PDF (preview)")
with st.expander("Show parsed fields (debug)"):
    st.json({
        "life_assured_name": parsed.get("la_name"),
        "issue_date_raw": parsed.get("issue_date"),
        "pt_years": parsed.get("pt_years"),
        "ppt_years": parsed.get("ppt_years"),
        "mode": parsed.get("mode"),
        "annual_premium_raw": parsed.get("annual_premium"),
        "sum_assured_raw": parsed.get("sum_assured"),
        "allocations": parsed.get("allocations", [])[:10],
    })

colA, colB, colC = st.columns(3)

_issue_date_raw = parsed.get("issue_date")
# parsed["issue_date"] is already a normalized string if found via safe paths; else we will parse it:
issue_date = None
if _issue_date_raw:
    try:
        issue_date = _parse_human_date(_issue_date_raw)
    except:
        issue_date = None
if not issue_date:
    issue_date = dt.date(2018,1,1)  # conservative default if nothing found

pt_years  = int(parsed.get("pt_years") or 26)
ppt_years = int(parsed.get("ppt_years") or 15)
mode      = (parsed.get("mode") or "Annual")
annual_premium = float(parsed.get("annual_premium") or 250000.0)
sum_assured    = float(parsed.get("sum_assured") or 4000000.0)
alloc_from_pdf = parsed.get("allocations", [])
la_name        = parsed.get("la_name") or "—"

with colA:
    st.write(f"**Life Assured:** {la_name}")
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

override = st.checkbox("Override parsed values (optional)", value=False)
if override:
    with st.expander("Override parsed values (optional)", expanded=True):
        issue_date = st.date_input("Issue Date (override)", value=issue_date)
        pt_years   = st.number_input("Policy Term (years)", 1, 100, pt_years, 1)
        ppt_years  = st.number_input("PPT (years)", 1, 100, ppt_years, 1)
        mode       = st.selectbox(
            "Premium Mode",
            ["Annual","Semi-Annual","Quarterly","Monthly"],
            index=["Annual","Semi-Annual","Quarterly","Monthly"].index(mode),
        )
        annual_premium = st.number_input("Annualised Premium (₹)", 0.0, 1e12, annual_premium, 1000.0)
        sum_assured    = st.number_input("Sum Assured (₹)",       0.0, 1e12, sum_assured,    50000.0)
        la_name        = st.text_input("Life Assured Name", value=la_name)

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

with st.expander("Life Assured details (only if needed)"):
    la_dob    = st.date_input("Life Assured DOB", value=dt.date(1990,1,1))
    la_gender = st.selectbox("Life Assured Gender", ["Female","Male"], index=0)

run = st.button("Generate In-Force Projection", type="primary")

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

    # Totals of additions from today -> PT end
    total_GA_BA = float(df_m["CI_raw"].sum())
    total_Loyalty = float(df_m["CJ_raw"].sum())
    total_additions = total_GA_BA + total_Loyalty

    # KPIs & helpers
    py_today = policy_year_today(issue_date, valuation_date, pt_years)
    py_in_2 = policy_year_today(issue_date, add_years(valuation_date, 2), pt_years)
    py_in_5 = policy_year_today(issue_date, add_years(valuation_date, 5), pt_years)
    fv_2y = fv_at_year_eoy(df_m, py_in_2)
    fv_5y = fv_at_year_eoy(df_m, py_in_5)

    invested_so_far = premiums_paid_till_ptd(issue_date, ptd_date, ppt_years, annual_premium, mode)

    # Partial withdrawal now (this policy year)
    pw_tbl = partial_withdrawal_availability(df_m, pt_years=pt_years)
    pw_now = pw_tbl.loc[pw_tbl["Policy_Year"] == py_today].iloc[0] if (py_today in pw_tbl["Policy_Year"].values) else None
    pw_now_amt = int(pw_now["Max_Available"]) if pw_now is not None and pw_now["Eligible"] == "Yes" else 0
    pw_eligible_now = (pw_now is not None and pw_now["Eligible"] == "Yes")

    # DF comparison if discontinued within lock-in
    df_y5 = discontinuance_y5_value_from_today(fv0_seed, issue_date, valuation_date) if status == "DISCONTINUED" else None

    # ---------------- Snapshot ----------------
    st.subheader("Snapshot")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.write(f"**Life Assured:** {la_name}")
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
        st.write(f"**PT / PPT:** {pt_years} / {ppt_years}")
    with c4:
        st.write("**Projected Additions (from today → PT end)**")
        st.write(f"- GA + BA: ₹{format_in_indian_system(total_GA_BA)}")
        st.write(f"- Loyalty: ₹{format_in_indian_system(total_Loyalty)}")
        st.write(f"**Total:** ₹{format_in_indian_system(total_additions)}")

    # ---------------- Retention Pitch (bullets + dynamic highlights) ----------------
    st.subheader("Retention Pitch (for caller)")

    # Build bullet lines
    pitch_lines = []
    pitch_lines.append(f"- **You’ve invested** about **₹{format_in_indian_system(invested_so_far)}** so far.")
    pitch_lines.append(f"- **Your current fund value** is **₹{format_in_indian_system(fv0_seed)}**.")

    if fv_2y is not None:
        delta_2 = max(0, fv_2y - fv0_seed)
        per_month = int(round(delta_2 / 24.0)) if delta_2 > 0 else 0
        pitch_lines.append(
            f"- At **8% projection**, your fund could grow to:\n"
            f"  - **₹{format_in_indian_system(fv_2y)} in 2 years** "
            f"(_~₹{format_in_indian_system(per_month)} per month growth from here_)."
        )
    if fv_5y is not None:
        delta_5 = max(0, fv_5y - fv0_seed)
        pitch_lines.append(
            f"  - **₹{format_in_indian_system(fv_5y)} in 5 years** "
            f"(**₹{format_in_indian_system(delta_5)} more than today**)."
        )

    pre2021 = issue_date < ULIP_TAX_CHANGE_DATE
    if pre2021:
        pitch_lines.append(
            "- **Tax edge**: Issued **before Feb 2021**, maturity is **generally tax-free** (u/s 10(10D); "
            "assumes premiums within limits). Hard to replicate this outside."
        )
    else:
        pitch_lines.append(
            "- **Tax note**: ULIPs issued on/after **Feb 2021** may face different rules; "
            "staying invested preserves your current structure."
        )

    if py_today >= 6:
        pitch_lines.append("- **Low charges ahead**: You’ve already crossed **high-charge years (1–5)**. Admin charges now are **nil**.")

    pitch_lines.append(
        f"- **Extra additions ahead**: From today till policy end, you’re on track for "
        f"**₹{format_in_indian_system(total_GA_BA)} (GA+BA)** and **₹{format_in_indian_system(total_Loyalty)} (Loyalty)** — "
        f"**₹{format_in_indian_system(total_additions)} in total** added to your fund."
    )

    if pw_eligible_now and pw_now_amt > 0:
        pitch_lines.append(
            f"- **Need funds now?** Use a **partial withdrawal up to ₹{format_in_indian_system(pw_now_amt)}** "
            "instead of surrendering — keep your investment & future additions intact."
        )
    else:
        pitch_lines.append(
            "- **Need funds now?** From the **6th policy year**, partial withdrawals can be used instead of surrendering "
            "(subject to the 105% rule)."
        )

    if (df_y5 is not None) and (fv_5y is not None):
        delta_df = fv_5y - df_y5
        pitch_lines.append(
            f"- **If discontinued in lock-in**: Discontinuance Fund pays about **₹{format_in_indian_system(df_y5)}** at end of Year 5 "
            f"(@5%). Continuing at 8% projects **₹{format_in_indian_system(fv_5y)}** — "
            f"**₹{format_in_indian_system(delta_df)} more**."
        )

    pitch_lines.append(
        "- **Re-entry cost**: Buying new insurance later usually costs **more** (higher age, medicals, waiting periods). "
        "Keeping this policy preserves your benefits."
    )

    # Dynamic highlights (pick strongest 2–3)
    highlights = []
    if pre2021:
        highlights.append("✅ **Tax-free maturity** (pre-2021 policy).")
    if py_today >= 6:
        highlights.append("✅ **Low/no admin charges ahead** (past Year 5).")
    if fv_5y is not None and (fv_5y - fv0_seed) > 0:
        highlights.append(f"✅ **₹{format_in_indian_system(fv_5y - fv0_seed)} more in 5 years** by staying invested.")
    if (df_y5 is not None) and (fv_5y is not None) and ((fv_5y - df_y5) > 0):
        highlights.append(f"✅ **Continue vs DF**: **₹{format_in_indian_system(fv_5y - df_y5)} more** than DF payout.")

    highlights = highlights[:3]

    if highlights:
        st.success("**Key hooks for this customer:**\n\n" + "\n".join([f"- {h}" for h in highlights]))

    # Full bullet list
    st.markdown("\n".join(pitch_lines))
    st.caption("Notes: Projections use standard 8%/5% illustrations. Actual returns depend on markets. Tax treatment depends on prevailing laws and your specifics — please consult your tax advisor.")

    # ------- FV Graph (EOY) till PT end -------
    st.subheader("Fund Value at End of Each Policy Year")
    eoy_series = yearly_fv_series(df_m, pt_years=pt_years)
    graph_df = eoy_series.copy()
    graph_df["FV_EOY_Display"] = graph_df["FV_EOY"].apply(lambda x: None if pd.isna(x) else float(x))
    st.line_chart(data=graph_df.set_index("Year")["FV_EOY_Display"], use_container_width=True)
    st.caption("Graph shows projected Fund Value at the end of each policy year from current year until PT end.")

    # ------- Partial Withdrawals (full table for reference) -------
    st.subheader("Partial Withdrawal — Eligibility & Maximum Available (EOY)")
    st.caption("Rules: from 6th policy year; min ₹500; FV after withdrawal ≥ 105% of total premiums paid.")
    pw_fmt = pw_tbl.copy()
    pw_fmt["Max_Available"] = pw_fmt["Max_Available"].apply(lambda x: f"₹{format_in_indian_system(x)}" if x else "₹0")
    st.dataframe(
        pw_fmt.rename(columns={"Policy_Year":"Policy Year","Eligible":"Eligible?","Max_Available":"Max Available","Notes":"Notes"}),
        use_container_width=True
    )

else:
    st.caption("Upload the POS PDF, enter PTD & Current FV, then click **Generate In-Force Projection**.")
