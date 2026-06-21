"""
main.py — ValtoSpend: AI-Powered Personal Expense Tracker
Entry point for the Streamlit application.
Imports from: database.py, ai_models.py, charts.py, receipt.py, auth.py, live_data.py
"""
import os
import time
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from sklearn.linear_model import LinearRegression

from database  import (init_db, load_data, load_my_expenses, add_my_expense,
                        delete_my_expense, get_budgets, set_budget, export_expenses_csv)
from ai_models import (linear_regression_forecast, random_forest_model,
                       neural_network_scratch, FEATURES)
from charts    import (bar_chart_categories, pie_chart_categories, bracket_comparison_chart,
                       monthly_trend_chart, personal_category_chart, personal_weekly_chart,
                       forecast_chart, feature_importance_chart, nn_loss_chart,
                       community_category_chart)
from receipt   import read_receipt_with_ai
from auth      import init_users_table, register_user, login_user, update_profile
from live_data import get_live_rates, get_community_stats

EXCHANGE_API_KEY = "99e195f29733ad0ef48de417"

SPEND_COLS = ["Food","Groceries","Transport","Entertainment",
              "Shopping","Rent","Bills","Healthcare","Education"]
BRACKETS   = ["Low Income","Lower Middle Income","Middle Income",
              "Upper Middle Income","High Income"]
CURRENCIES = {
    "AED — UAE Dirham":          ("AED", "AED"),  "ARS — Argentine Peso":  ("ARS", "$"),
    "AUD — Australian Dollar":   ("AUD", "A$"),   "BDT — Bangladeshi Taka":("BDT", "Tk"),
    "BRL — Brazilian Real":      ("BRL", "R$"),   "CAD — Canadian Dollar": ("CAD", "C$"),
    "CHF — Swiss Franc":         ("CHF", "CHF"),  "CLP — Chilean Peso":    ("CLP", "$"),
    "CNY — Chinese Yuan":        ("CNY", "¥"),    "COP — Colombian Peso":  ("COP", "$"),
    "CZK — Czech Koruna":        ("CZK", "Kc"),   "DKK — Danish Krone":    ("DKK", "kr"),
    "EGP — Egyptian Pound":      ("EGP", "£"),    "EUR — Euro":            ("EUR", "EUR"),
    "GBP — British Pound":       ("GBP", "£"),    "HKD — Hong Kong Dollar":("HKD", "HK$"),
    "HUF — Hungarian Forint":    ("HUF", "Ft"),   "IDR — Indonesian Rupiah":("IDR","Rp"),
    "ILS — Israeli Shekel":      ("ILS", "ILS"),  "INR — Indian Rupee":    ("INR", "INR"),
    "JPY — Japanese Yen":        ("JPY", "JPY"),  "KES — Kenyan Shilling": ("KES", "KSh"),
    "KRW — South Korean Won":    ("KRW", "KRW"),  "KWD — Kuwaiti Dinar":   ("KWD", "KD"),
    "LKR — Sri Lankan Rupee":    ("LKR", "Rs"),   "MAD — Moroccan Dirham": ("MAD", "MAD"),
    "MXN — Mexican Peso":        ("MXN", "$"),    "MYR — Malaysian Ringgit":("MYR","RM"),
    "NGN — Nigerian Naira":      ("NGN", "NGN"),  "NOK — Norwegian Krone": ("NOK", "kr"),
    "NZD — New Zealand Dollar":  ("NZD", "NZ$"),  "PEN — Peruvian Sol":    ("PEN", "S/"),
    "PHP — Philippine Peso":     ("PHP", "PHP"),  "PKR — Pakistani Rupee": ("PKR", "Rs"),
    "PLN — Polish Zloty":        ("PLN", "zl"),   "QAR — Qatari Riyal":    ("QAR", "QR"),
    "RON — Romanian Leu":        ("RON", "lei"),  "RUB — Russian Ruble":   ("RUB", "RUB"),
    "SAR — Saudi Riyal":         ("SAR", "SR"),   "SEK — Swedish Krona":   ("SEK", "kr"),
    "SGD — Singapore Dollar":    ("SGD", "S$"),   "THB — Thai Baht":       ("THB", "THB"),
    "TRY — Turkish Lira":        ("TRY", "TRY"),  "TWD — Taiwan Dollar":   ("TWD", "NT$"),
    "TZS — Tanzanian Shilling":  ("TZS", "TSh"),  "UAH — Ukrainian Hryvnia":("UAH","UAH"),
    "USD — US Dollar":           ("USD", "$"),    "VND — Vietnamese Dong": ("VND", "VND"),
    "ZAR — South African Rand":  ("ZAR", "R"),    "ZMW — Zambian Kwacha":  ("ZMW", "ZK"),
}

st.set_page_config(page_title="ValtoSpend", page_icon=":bar_chart:", layout="wide")

# ── Global theme — pitch black + teal accent, applied everywhere ───────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"], .stApp, section[data-testid="stSidebar"] > div,
.stMainBlockContainer, [data-testid="stSidebarContent"] {
    background-color: #000000 !important;
}
.stButton > button[kind="primary"] {
    background: #00C9A7 !important; color: #000 !important;
    font-weight: 600 !important; border: none !important;
    border-radius: 6px !important; letter-spacing: 0.3px;
}
.stButton > button[kind="primary"]:hover { background: #00E5BF !important; }
.stButton > button[kind="secondary"] {
    background: #0a0a0a !important; border: 1px solid #1f1f1f !important;
    color: #cccccc !important; border-radius: 6px !important;
}
.stTabs [data-baseweb="tab"] { color: #666666 !important; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #00C9A7 !important; border-bottom-color: #00C9A7 !important; }
[data-testid="stMetricValue"] { color: #ffffff !important; }
[data-testid="stMetricLabel"] { color: #777777 !important; font-size: 0.8rem !important; }
h1, h2, h3 { color: #ffffff !important; font-weight: 600 !important; }
hr { border-color: #161616 !important; }
.stTextInput input, .stNumberInput input, .stDateInput input, .stSelectbox > div {
    background: #0a0a0a !important; border-color: #1f1f1f !important; color: #eeeeee !important;
}
.section-tag {
    display: inline-block; font-size: 0.7rem; letter-spacing: 2px;
    color: #00C9A7; text-transform: uppercase; margin-bottom: 0.3rem;
}
.card {
    background: #070707; border: 1px solid #161616;
    border-radius: 10px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
}
.insight-line {
    border-left: 2px solid #00C9A7; padding: 0.5rem 0 0.5rem 0.9rem;
    color: #cccccc; font-size: 0.92rem; margin-bottom: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

init_db()
init_users_table()
df = load_data()

# ── Session defaults ─────────────────────────────────────────────────────────
for k, v in [("logged_in", False), ("user_id", None), ("username", None),
             ("user_income", 2000.0), ("user_bracket", "Middle Income"),
             ("user_currency", "EUR"), ("splash_shown", False)]:
    if k not in st.session_state:
        st.session_state[k] = v


def sym():
    for label, (code, symbol) in CURRENCIES.items():
        if code == st.session_state.user_currency:
            return symbol
    return "EUR"

def fmt(amount):
    return f"{sym()} {amount:,.2f}"

def get_conversion_rate():
    target = st.session_state.user_currency
    if target == "EUR":
        return 1.0
    cache_key = f"rate_EUR_{target}"
    if cache_key not in st.session_state:
        try:
            rates_data = get_live_rates(EXCHANGE_API_KEY, "EUR")
            st.session_state[cache_key] = rates_data.get("rates", {}).get(target, 1.0)
        except Exception:
            st.session_state[cache_key] = 1.0
    return st.session_state[cache_key]


# ════════════════════════════════════════════════════════════════════════════
# AUTH SCREEN
# ════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown("""
    <style>
    .welcome-outer {
        min-height: 78vh; display: flex; flex-direction: column;
        align-items: center; justify-content: center; text-align: center;
        padding: 3rem 1rem 1.5rem;
    }
    .vs-mark {
        font-size: 110px; font-weight: 800; color: #00C9A7;
        letter-spacing: -5px; line-height: 1; margin-bottom: 0.4rem;
    }
    .brand-row { font-size: 26px; font-weight: 300; color: #ffffff; letter-spacing: 9px; }
    .brand-sub { font-size: 12px; color: #3a3a3a; letter-spacing: 4px; margin-top: 0.4rem; }
    .welcome-tagline {
        font-size: 0.95rem; color: #555555; max-width: 440px;
        margin: 1.6rem auto 0; line-height: 1.6;
    }
    .pill-row { display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; margin-top: 1.8rem; }
    .pill {
        background: #0a0a0a; border: 1px solid #1a1a1a; border-radius: 18px;
        padding: 6px 16px; font-size: 0.78rem; color: #00C9A7; letter-spacing: 0.3px;
    }
    .auth-box {
        background: #060606; border: 1px solid #161616; border-radius: 12px;
        padding: 2rem; max-width: 440px; margin: 0 auto;
    }
    </style>
    <div class="welcome-outer">
        <div class="vs-mark">VS</div>
        <div class="brand-row">VALTOSPEND</div>
        <div class="brand-sub">EXPENSE INTELLIGENCE</div>
        <div class="welcome-tagline">
            A finance companion that learns from your spending,
            forecasts what comes next, and helps you stay ahead of your budget.
        </div>
        <div class="pill-row">
            <div class="pill">Receipt Recognition</div>
            <div class="pill">Predictive Analytics</div>
            <div class="pill">Budget Tracking</div>
            <div class="pill">50+ Currencies</div>
            <div class="pill">Market Benchmarking</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 1.3, 1])
    with col_m:
        st.markdown('<div class="auth-box">', unsafe_allow_html=True)
        tabs = st.tabs(["Sign In", "Create Account"])

        with tabs[0]:
            with st.form("login_form"):
                lu = st.text_input("Username")
                lp = st.text_input("Password", type="password")
                if st.form_submit_button("Sign In", type="primary", use_container_width=True):
                    ok, result = login_user(lu, lp)
                    if ok:
                        uid, uname, income, bracket, currency = result
                        st.session_state.logged_in     = True
                        st.session_state.user_id       = uid
                        st.session_state.username      = uname
                        st.session_state.user_income   = income
                        st.session_state.user_bracket  = bracket
                        st.session_state.user_currency = currency
                        st.session_state.splash_shown  = False
                        st.rerun()
                    else:
                        st.error(result)

        with tabs[1]:
            with st.form("register_form"):
                ru       = st.text_input("Choose a username")
                rp       = st.text_input("Choose a password", type="password")
                rp2      = st.text_input("Confirm password",  type="password")
                rincome  = st.number_input("Monthly income", min_value=0.0, step=100.0, value=2000.0)
                rbracket = st.selectbox("Income bracket", BRACKETS)
                rcurr    = st.selectbox("Currency", list(CURRENCIES.keys()))
                if st.form_submit_button("Create Account", type="primary", use_container_width=True):
                    if rp != rp2:
                        st.error("Passwords do not match.")
                    else:
                        curr_code = CURRENCIES[rcurr][0]
                        ok, msg = register_user(ru, rp, rincome, rbracket, curr_code)
                        if ok:
                            st.success("Account created. Please sign in.")
                        else:
                            st.error(msg)
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# SPLASH — one-time transition shown after sign in
# ════════════════════════════════════════════════════════════════════════════
if not st.session_state.splash_shown:
    st.markdown("""
    <style>
    @keyframes riseIn { 0% {opacity:0; transform:translateY(24px);} 100% {opacity:1; transform:translateY(0);} }
    .splash-outer {
        background:#000; min-height: 75vh; display:flex; flex-direction:column;
        align-items:center; justify-content:center; text-align:center;
    }
    .splash-vs { font-size:120px; font-weight:800; color:#00C9A7; letter-spacing:-6px;
                 line-height:1; animation: riseIn .7s ease; }
    .splash-name { font-size:22px; font-weight:300; color:#fff; letter-spacing:9px;
                   margin-top:0.6rem; animation: riseIn .9s ease; }
    .splash-bar { width:70px; height:2px; background:#00C9A7; margin:1.4rem auto 0;
                  border-radius:2px; animation: riseIn 1.1s ease; }
    </style>
    <div class="splash-outer">
        <div class="splash-vs">VS</div>
        <div class="splash-name">VALTOSPEND</div>
        <div class="splash-bar"></div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(1.6)
    st.session_state.splash_shown = True
    st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ════════════════════════════════════════════════════════════════════════════
user_id      = st.session_state.user_id
username     = st.session_state.username
user_income  = st.session_state.user_income
user_bracket = st.session_state.user_bracket

with st.sidebar:
    st.markdown("<h1 style='color:#00C9A7;font-size:34px;font-weight:800;margin:0;padding:0;'>ValtoSpend</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#3a3a3a;font-size:10px;letter-spacing:3px;margin:0 0 1.2rem 0;'>EXPENSE INTELLIGENCE</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#cccccc;font-size:0.95rem;margin:0;'>{username.capitalize()}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#555555;font-size:0.78rem;margin:0 0 1rem 0;'>{fmt(user_income)} &middot; {user_bracket}</p>", unsafe_allow_html=True)

    st.markdown("<div class='section-tag'>Currency</div>", unsafe_allow_html=True)
    curr_keys    = list(CURRENCIES.keys())
    curr_default = next((i for i, (k,(c,s)) in enumerate(CURRENCIES.items())
                         if c == st.session_state.user_currency), 0)
    selected_curr = st.selectbox("Display currency", curr_keys, index=curr_default,
                                 label_visibility="collapsed")
    new_curr_code = CURRENCIES[selected_curr][0]
    if new_curr_code != st.session_state.user_currency:
        st.session_state.user_currency = new_curr_code
        update_profile(user_id, user_income, user_bracket, new_curr_code)
        st.rerun()

    c_a, c_b = st.columns(2)
    with c_a:
        if st.button("Profile", use_container_width=True):
            st.session_state["editing_profile"] = True
    with c_b:
        if st.button("Sign Out", use_container_width=True):
            for k in ["logged_in","user_id","username","user_income",
                      "user_bracket","user_currency","splash_shown"]:
                st.session_state[k] = False if k in ("logged_in","splash_shown") else None
            st.rerun()

    st.markdown("<div class='section-tag' style='margin-top:1.4rem;'>Filters</div>", unsafe_allow_html=True)
    brackets_list = ["All"] + sorted(df["Income_Bracket"].dropna().unique().tolist())
    sel_bracket   = st.selectbox("Income bracket filter", brackets_list, label_visibility="collapsed")

if st.session_state.get("editing_profile"):
    with st.expander("Edit profile", expanded=True):
        with st.form("edit_profile"):
            new_income  = st.number_input("Monthly income", value=float(user_income), step=100.0)
            new_bracket = st.selectbox("Income bracket", BRACKETS,
                           index=BRACKETS.index(user_bracket) if user_bracket in BRACKETS else 0)
            if st.form_submit_button("Save changes"):
                update_profile(user_id, new_income, new_bracket, st.session_state.user_currency)
                st.session_state.user_income  = new_income
                st.session_state.user_bracket = new_bracket
                st.session_state["editing_profile"] = False
                st.rerun()

filtered = df.copy()
if sel_bracket != "All":
    filtered = filtered[filtered["Income_Bracket"] == sel_bracket]

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Budget", "Market Insights", "Predictions", "Live Data"]
)

# ══════════════════════════════════════════════════════════════
# TAB 1 — MY EXPENSES
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-tag'>Personal Tracker</div>", unsafe_allow_html=True)
    st.header(f"{username.capitalize()}'s Expenses")

    with st.container():
        st.markdown("<div class='section-tag'>Add an Expense</div>", unsafe_allow_html=True)
        st.caption("Upload a receipt photo to auto-fill the details below, or enter them manually.")

        uploaded_file = st.file_uploader("Upload receipt", type=["jpg","jpeg","png"],
                                         label_visibility="collapsed")

        for k, v in [("prefill_amount",10.0),("prefill_category","Food"),
                     ("prefill_note",""),("scan_done",False)]:
            if k not in st.session_state:
                st.session_state[k] = v

        if uploaded_file and not st.session_state.scan_done:
            with st.spinner("Reading receipt..."):
                img_bytes  = uploaded_file.read()
                media_type = "image/png" if uploaded_file.name.endswith(".png") else "image/jpeg"
                result     = read_receipt_with_ai(img_bytes, media_type)
            if "error" in result:
                st.warning(f"Could not read this receipt automatically ({result['error']}). Please enter the details manually.")
            else:
                st.success(f"Detected {fmt(result.get('amount',0))} — {result.get('note','')}")
                st.session_state.prefill_amount   = float(result.get("amount", 10.0))
                st.session_state.prefill_category = result.get("category", "Food")
                st.session_state.prefill_note     = result.get("note", "")
                st.session_state.scan_done        = True
                st.image(img_bytes, width=180)
        if not uploaded_file:
            st.session_state.scan_done = False

        c1, c2, c3, c4 = st.columns(4)
        exp_date = c1.date_input("Date", value=date.today())
        cat_idx  = SPEND_COLS.index(st.session_state.prefill_category) \
                   if st.session_state.prefill_category in SPEND_COLS else 0
        exp_cat  = c2.selectbox("Category", SPEND_COLS, index=cat_idx)
        exp_amt  = c3.number_input("Amount", min_value=0.01, step=0.50,
                                    value=float(st.session_state.prefill_amount))
        exp_note = c4.text_input("Note", value=st.session_state.prefill_note,
                                  placeholder="Optional description")

        if st.button("Save Expense", type="primary"):
            add_my_expense(user_id, exp_date, exp_cat, exp_amt, exp_note)
            st.success(f"Saved {fmt(exp_amt)} for {exp_cat}")
            for k, v in [("prefill_amount",10.0),("prefill_category","Food"),
                         ("prefill_note",""),("scan_done",False)]:
                st.session_state[k] = v
            st.rerun()

    st.divider()
    my_df = load_my_expenses(user_id)

    if my_df.empty:
        st.info("No expenses recorded yet. Upload a receipt or add one manually above to get started.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Spent",       fmt(my_df['amount'].sum()))
        m2.metric("Entries",           len(my_df))
        m3.metric("Largest Expense",   fmt(my_df['amount'].max()))
        m4.metric("Top Category",      my_df.groupby("category")["amount"].sum().idxmax())

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<div class='section-tag'>By Category</div>", unsafe_allow_html=True)
            st.plotly_chart(personal_category_chart(my_df, symbol=sym()), use_container_width=True, key="personal_cat_chart")
        with col_b:
            st.markdown("<div class='section-tag'>Over Time</div>", unsafe_allow_html=True)
            st.plotly_chart(personal_weekly_chart(my_df, symbol=sym()), use_container_width=True, key="personal_weekly_chart")

        st.markdown("<div class='section-tag'>Insights</div>", unsafe_allow_html=True)
        cat_totals  = my_df.groupby("category")["amount"].sum()
        total_spent = my_df["amount"].sum()
        top_cat     = cat_totals.idxmax()
        top_pct     = cat_totals[top_cat] / total_spent * 100
        st.markdown(f"<div class='insight-line'><b>{top_cat}</b> accounts for <b>{top_pct:.0f}%</b> of total spending — your largest category.</div>", unsafe_allow_html=True)

        if "Food" in cat_totals.index or "Groceries" in cat_totals.index:
            food_total = cat_totals.get("Food",0) + cat_totals.get("Groceries",0)
            food_pct   = food_total / total_spent * 100
            if food_pct > 35:
                st.markdown(f"<div class='insight-line'>Food and groceries make up <b>{food_pct:.0f}%</b> of spending — above the typical range. Meal planning could reduce this.</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='insight-line'>Food and groceries spending is balanced at <b>{food_pct:.0f}%</b> of total.</div>", unsafe_allow_html=True)

        if len(cat_totals) >= 3:
            lowest_cat = cat_totals.idxmin()
            st.markdown(f"<div class='insight-line'><b>{lowest_cat}</b> is the lowest spending category — a sign of good balance elsewhere.</div>", unsafe_allow_html=True)

        my_df["month"] = my_df["date"].dt.to_period("M").astype(str)
        monthly_me = my_df.groupby("month")["amount"].sum().reset_index()
        monthly_me["idx"] = range(len(monthly_me))
        if len(monthly_me) >= 2:
            lr_me = LinearRegression()
            lr_me.fit(monthly_me[["idx"]], monthly_me["amount"])
            next_pred_me  = lr_me.predict([[len(monthly_me)]])[0]
            next_mo_label = (my_df["date"].max() + pd.DateOffset(months=1)).strftime("%B %Y")
            st.markdown(f"<div class='insight-line'>Projected spend for <b>{next_mo_label}</b>: <b>{fmt(next_pred_me)}</b>, based on your recent trend.</div>", unsafe_allow_html=True)

        st.divider()
        col_e1, col_e2 = st.columns([3,1])
        with col_e1:
            st.markdown("<div class='section-tag'>All Expenses</div>", unsafe_allow_html=True)
        with col_e2:
            csv_data = export_expenses_csv(user_id)
            st.download_button("Export CSV", data=csv_data,
                              file_name=f"valtospend_{username}_expenses.csv", mime="text/csv")

        for _, row in my_df.iterrows():
            c1, c2, c3, c4, c5 = st.columns([2,2,1,3,1])
            c1.write(row["date"].strftime("%d %b %Y"))
            c2.write(row["category"])
            c3.write(fmt(row["amount"]))
            c4.write(row["note"] if row["note"] else "—")
            if c5.button("Remove", key=f"del_{row['id']}"):
                delete_my_expense(row["id"])
                st.rerun()

# ══════════════════════════════════════════════════════════════
# TAB 2 — BUDGET
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-tag'>Spending Limits</div>", unsafe_allow_html=True)
    st.header("Monthly Budget")
    st.caption("Set a limit per category and track progress against it through the month.")

    budgets = get_budgets(user_id)
    my_df_b = load_my_expenses(user_id)

    this_month = date.today().strftime("%Y-%m")
    if not my_df_b.empty:
        my_df_b["month"] = my_df_b["date"].dt.strftime("%Y-%m")
        this_month_df    = my_df_b[my_df_b["month"] == this_month]
        monthly_cats     = this_month_df.groupby("category")["amount"].sum().to_dict()
    else:
        monthly_cats = {}

    with st.expander("Set category budgets"):
        cols = st.columns(3)
        for i, cat in enumerate(SPEND_COLS):
            with cols[i % 3]:
                current = budgets.get(cat, 0.0)
                new_val = st.number_input(cat, min_value=0.0, value=float(current),
                                          step=10.0, key=f"budget_{cat}")
                if new_val != current and new_val > 0:
                    set_budget(user_id, cat, new_val)

    st.divider()
    st.markdown(f"<div class='section-tag'>{date.today().strftime('%B %Y')}</div>", unsafe_allow_html=True)

    if not budgets:
        st.info("No budgets set yet. Expand the panel above to define monthly limits per category.")
    else:
        for cat in SPEND_COLS:
            if cat in budgets:
                limit     = budgets[cat]
                spent     = monthly_cats.get(cat, 0.0)
                pct       = min(spent / limit * 100, 100) if limit > 0 else 0
                remaining = max(limit - spent, 0)

                col_l, col_r = st.columns([3, 1])
                with col_l:
                    status = "Over budget" if pct >= 100 else "Approaching limit" if pct >= 80 else "On track"
                    st.markdown(f"**{cat}** — {status}")
                    st.progress(int(pct))
                with col_r:
                    st.metric("Spent",     fmt(spent))
                    st.metric("Remaining", fmt(remaining))
                st.caption(f"Limit {fmt(limit)} · {pct:.0f}% used")
                st.divider()

# ══════════════════════════════════════════════════════════════
# TAB 3 — MARKET INSIGHTS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-tag'>Benchmark</div>", unsafe_allow_html=True)
    st.header("Market Insights")
    st.caption("Patterns drawn from 3,655 household records (2021–2024), used as the model's training foundation. Category behaviour and income-bracket ratios remain consistent over time.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records",            f"{len(filtered):,}")
    c2.metric("Avg Monthly Spend",  fmt(filtered['Total_Expenses'].mean()))
    c3.metric("Avg Monthly Income", fmt(filtered['Income'].mean()))
    c4.metric("Avg Monthly Savings",fmt(filtered['Savings'].mean()))

    conv_rate = get_conversion_rate()
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("<div class='section-tag'>Average Spend per Category</div>", unsafe_allow_html=True)
        fig, avg_cats = bar_chart_categories(filtered, rate=conv_rate, symbol=sym())
        st.plotly_chart(fig, use_container_width=True, key="market_bar_chart")
    with col_r:
        st.markdown("<div class='section-tag'>Category Distribution</div>", unsafe_allow_html=True)
        st.plotly_chart(pie_chart_categories(avg_cats), use_container_width=True, key="market_pie_chart")

    st.markdown("<div class='section-tag'>Income Bracket Comparison</div>", unsafe_allow_html=True)
    st.plotly_chart(bracket_comparison_chart(df, rate=conv_rate, symbol=sym()), use_container_width=True, key="bracket_chart")

    st.markdown("<div class='section-tag'>Monthly Spending Trend</div>", unsafe_allow_html=True)
    st.plotly_chart(monthly_trend_chart(filtered, rate=conv_rate, symbol=sym()), use_container_width=True, key="monthly_trend_chart")

    if "Festivals" in df.columns:
        st.markdown("<div class='section-tag'>Festival Impact</div>", unsafe_allow_html=True)
        fest_s = (filtered.groupby("Festivals")["Total_Expenses"].mean()
                  .reset_index()
                  .rename(columns={"Festivals":"Festival","Total_Expenses":"Avg Expenses"})
                  .sort_values("Avg Expenses", ascending=False).reset_index(drop=True))
        st.dataframe(fest_s, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — PREDICTIONS
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-tag'>Machine Learning</div>", unsafe_allow_html=True)
    st.header("Prediction Engine")

    st.markdown("<div class='section-tag'>Next Month Forecast — Linear Regression</div>", unsafe_allow_html=True)
    st.caption("Trained on the monthly average spending trend across all records.")
    next_p, mae_ts, t_preds, next_lbl, monthly_ts = linear_regression_forecast(df)
    ma, mb = st.columns(2)
    ma.metric(f"Forecast for {next_lbl}", fmt(next_p))
    mb.metric("Mean Absolute Error", fmt(mae_ts))
    st.plotly_chart(forecast_chart(monthly_ts, t_preds, next_p, next_lbl, symbol=sym()), use_container_width=True, key="forecast_chart")

    st.divider()

    st.markdown(f"<div class='section-tag'>Personal Predictor — Random Forest</div>", unsafe_allow_html=True)
    st.caption("An ensemble of 10 decision trees trained on income, bracket, month, and category ratios.")
    rf, le, mae_rf, r2_rf = random_forest_model(df)
    mc, md = st.columns(2)
    mc.metric("Mean Absolute Error", fmt(mae_rf))
    md.metric("R-squared", f"{r2_rf:.3f}")
    st.plotly_chart(feature_importance_chart(rf, FEATURES + ["Income_Bracket"]), use_container_width=True, key="feature_importance_chart")

    p1, p2, p3 = st.columns(3)
    income_in = p1.slider("Monthly income", 500, 10000,
                          int(min(max(user_income,500),10000)), step=100)
    month_in  = p2.slider("Month", 1, 12, date.today().month)
    fest_in   = p3.slider("Festival count", 0, 5, 1)
    bracket_options = list(le.classes_)
    default_idx     = bracket_options.index(user_bracket) if user_bracket in bracket_options else 0
    bracket_in      = st.selectbox("Income bracket", bracket_options, index=default_idx)
    bracket_enc     = le.transform([bracket_in])[0]
    default_ratios  = [0.20,0.12,0.10,0.08,0.10,0.25,0.07,0.04,0.04]
    input_row       = np.array([[income_in,month_in,fest_in,*default_ratios,bracket_enc]])
    st.success(f"Predicted monthly expenses: {fmt(rf.predict(input_row)[0])}")

    st.divider()

    st.markdown("<div class='section-tag'>Neural Network — Built from Scratch</div>", unsafe_allow_html=True)
    st.caption("Implemented in NumPy only, without TensorFlow. Two hidden layers (64 and 32 units), ReLU activation, trained by gradient descent over 100 epochs.")
    with st.spinner("Training network..."):
        mae_nn, r2_nn, losses = neural_network_scratch(df)
    n1, n2 = st.columns(2)
    n1.metric("Mean Absolute Error", fmt(mae_nn))
    n2.metric("R-squared", f"{r2_nn:.3f}")
    st.plotly_chart(nn_loss_chart(losses), use_container_width=True, key="nn_loss_chart")

    st.markdown("<div class='section-tag'>Model Comparison</div>", unsafe_allow_html=True)
    comparison = pd.DataFrame({
        "Model":    ["Linear Regression","Random Forest","Neural Network"],
        "Type":     ["Statistical","Ensemble learning","Deep learning"],
        "MAE":      [fmt(mae_ts), fmt(mae_rf), fmt(mae_nn)],
        "R-squared":["—", f"{r2_rf:.3f}", f"{r2_nn:.3f}"]
    })
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    with st.expander("View raw records"):
        min_date = df["Date"].min().date()
        max_date = df["Date"].max().date()
        date_range = st.date_input("Filter by date range", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date, key="records_date")
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start, end = date_range
            records_df = df[(df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)]
        else:
            records_df = df
        st.dataframe(records_df[["UserID","Date","Income","Income_Bracket",
                          "Total_Expenses","Savings"]+SPEND_COLS].head(200), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 5 — LIVE DATA
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-tag'>Real-Time</div>", unsafe_allow_html=True)
    st.header("Live Data")

    st.markdown("<div class='section-tag'>Community Activity</div>", unsafe_allow_html=True)
    st.caption("Aggregated statistics from every ValtoSpend account, updated as users log expenses.")

    stats = get_community_stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registered Users", stats["total_users"])
    c2.metric("Expenses Logged",  stats["total_logged"])
    c3.metric("This Month",       stats["this_month"])
    c4.metric("Top Category",     stats["top_category"])

    if stats["total_logged"] > 0:
        m_a, m_b = st.columns(2)
        m_a.metric("Total Community Spend", fmt(stats["total_amount"]))
        m_b.metric("Average per Transaction", fmt(stats["avg_expense"]))

        if stats["category_totals"]:
            st.markdown("<div class='section-tag'>Community Spend by Category</div>", unsafe_allow_html=True)
            st.plotly_chart(community_category_chart(stats["category_totals"], symbol=sym()),
                           use_container_width=True, key="community_chart")
    else:
        st.info("No community data yet. Be the first to log an expense.")

    st.divider()

    st.markdown("<div class='section-tag'>Exchange Rates</div>", unsafe_allow_html=True)
    st.caption("Live rates refreshed every 24 hours via ExchangeRate-API.")

    base_curr = st.selectbox(
        "Base currency", list(CURRENCIES.keys()),
        index=next((i for i, (k,(c,s)) in enumerate(CURRENCIES.items())
                    if c == st.session_state.user_currency), 0)
    )
    base_code = CURRENCIES[base_curr][0]

    with st.spinner("Fetching rates..."):
        rates = get_live_rates(EXCHANGE_API_KEY, base_code)

    if "error" in rates:
        st.error(f"Could not fetch rates: {rates['error']}")
    elif rates:
        st.caption(f"Updated: {rates.get('updated', 'unavailable')}")

        key_currencies = ["USD","EUR","GBP","INR","JPY","CAD","AUD",
                          "CHF","CNY","KRW","BRL","MXN","SGD","HKD"]
        key_rates = {k: v for k, v in rates["rates"].items()
                     if k in key_currencies and k != base_code}
        rate_df = pd.DataFrame(
            [(k, f"{v:.4f}") for k, v in sorted(key_rates.items())],
            columns=["Currency", f"1 {base_code} ="]
        )

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.dataframe(rate_df, hide_index=True, use_container_width=True)
        with col_r2:
            amount_in  = st.number_input("Amount", value=100.0, step=10.0)
            to_curr    = st.selectbox("Convert to", list(CURRENCIES.keys()))
            to_code    = CURRENCIES[to_curr][0]
            if to_code in rates.get("rates", {}):
                converted = amount_in * rates["rates"].get(to_code, 1)
                st.success(f"{amount_in:.2f} {base_code} = {converted:.2f} {to_code}")

        with st.expander("All currency rates"):
            all_rates_df = pd.DataFrame(
                [(k, f"{v:.4f}") for k, v in sorted(rates["rates"].items())],
                columns=["Currency", f"1 {base_code} ="]
            )
            st.dataframe(all_rates_df, hide_index=True, use_container_width=True)