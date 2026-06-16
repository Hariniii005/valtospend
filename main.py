"""
main.py — ValtoSpend: AI-Powered Personal Expense Tracker
Entry point for the Streamlit application.
Imports from: database.py, ai_models.py, charts.py, receipt.py, auth.py
"""
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
                       forecast_chart, feature_importance_chart, nn_loss_chart)
from receipt   import read_receipt_with_ai
from auth      import init_users_table, register_user, login_user, update_profile
from live_data import get_live_rates, get_community_stats, convert_amount

# ── API Key ─────────────────────────────────────────────────────────────────
EXCHANGE_API_KEY = "99e195f29733ad0ef48de417"  

# ── Constants ───────────────────────────────────────────────────────────────
SPEND_COLS = ["Food","Groceries","Transport","Entertainment",
              "Shopping","Rent","Bills","Healthcare","Education"]
BRACKETS   = ["Low Income","Lower Middle Income","Middle Income",
              "Upper Middle Income","High Income"]
CURRENCIES = {
    "AED — UAE Dirham (د.إ)":          ("AED", "د.إ"),
    "ARS — Argentine Peso ($)":        ("ARS", "$"),
    "AUD — Australian Dollar (A$)":    ("AUD", "A$"),
    "BDT — Bangladeshi Taka (৳)":      ("BDT", "৳"),
    "BRL — Brazilian Real (R$)":       ("BRL", "R$"),
    "CAD — Canadian Dollar (C$)":      ("CAD", "C$"),
    "CHF — Swiss Franc (CHF)":         ("CHF", "CHF"),
    "CLP — Chilean Peso ($)":          ("CLP", "$"),
    "CNY — Chinese Yuan (¥)":          ("CNY", "¥"),
    "COP — Colombian Peso ($)":        ("COP", "$"),
    "CZK — Czech Koruna (Kč)":         ("CZK", "Kč"),
    "DKK — Danish Krone (kr)":         ("DKK", "kr"),
    "EGP — Egyptian Pound (£)":        ("EGP", "£"),
    "EUR — Euro (€)":                  ("EUR", "€"),
    "GBP — British Pound (£)":         ("GBP", "£"),
    "HKD — Hong Kong Dollar (HK$)":    ("HKD", "HK$"),
    "HUF — Hungarian Forint (Ft)":     ("HUF", "Ft"),
    "IDR — Indonesian Rupiah (Rp)":    ("IDR", "Rp"),
    "ILS — Israeli Shekel (₪)":        ("ILS", "₪"),
    "INR — Indian Rupee (₹)":          ("INR", "₹"),
    "JPY — Japanese Yen (¥)":          ("JPY", "¥"),
    "KES — Kenyan Shilling (KSh)":     ("KES", "KSh"),
    "KRW — South Korean Won (₩)":      ("KRW", "₩"),
    "KWD — Kuwaiti Dinar (KD)":        ("KWD", "KD"),
    "LKR — Sri Lankan Rupee (Rs)":     ("LKR", "Rs"),
    "MAD — Moroccan Dirham (MAD)":     ("MAD", "MAD"),
    "MXN — Mexican Peso ($)":          ("MXN", "$"),
    "MYR — Malaysian Ringgit (RM)":    ("MYR", "RM"),
    "NGN — Nigerian Naira (₦)":        ("NGN", "₦"),
    "NOK — Norwegian Krone (kr)":      ("NOK", "kr"),
    "NZD — New Zealand Dollar (NZ$)":  ("NZD", "NZ$"),
    "PEN — Peruvian Sol (S/)":         ("PEN", "S/"),
    "PHP — Philippine Peso (₱)":       ("PHP", "₱"),
    "PKR — Pakistani Rupee (₨)":       ("PKR", "₨"),
    "PLN — Polish Zloty (zł)":         ("PLN", "zł"),
    "QAR — Qatari Riyal (QR)":         ("QAR", "QR"),
    "RON — Romanian Leu (lei)":        ("RON", "lei"),
    "RUB — Russian Ruble (₽)":         ("RUB", "₽"),
    "SAR — Saudi Riyal (SR)":          ("SAR", "SR"),
    "SEK — Swedish Krona (kr)":        ("SEK", "kr"),
    "SGD — Singapore Dollar (S$)":     ("SGD", "S$"),
    "THB — Thai Baht (฿)":             ("THB", "฿"),
    "TRY — Turkish Lira (₺)":          ("TRY", "₺"),
    "TWD — Taiwan Dollar (NT$)":       ("TWD", "NT$"),
    "TZS — Tanzanian Shilling (TSh)":  ("TZS", "TSh"),
    "UAH — Ukrainian Hryvnia (₴)":     ("UAH", "₴"),
    "USD — US Dollar ($)":             ("USD", "$"),
    "VND — Vietnamese Dong (₫)":       ("VND", "₫"),
    "XAF — Central African CFA (CFA)": ("XAF", "CFA"),
    "XOF — West African CFA (CFA)":    ("XOF", "CFA"),
    "ZAR — South African Rand (R)":    ("ZAR", "R"),
    "ZMW — Zambian Kwacha (ZK)":       ("ZMW", "ZK"),
}

# ── Page config & DB init ───────────────────────────────────────────────────
st.set_page_config(page_title="ValtoSpend", page_icon="💰", layout="wide")

# ── Logo (shown in sidebar automatically) ───────────────────────────────────
import os
if os.path.exists("logo.png"):
    st.logo("logo.png")

init_db()
init_users_table()
df = load_data()

# ── Session state defaults ──────────────────────────────────────────────────
for k, v in [("logged_in", False), ("user_id", None), ("username", None),
             ("user_income", 2000.0), ("user_bracket", "Middle Income"),
             ("user_currency", "EUR"), ("auth_page", "login")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Currency helper ─────────────────────────────────────────────────────────
def sym():
    for label, (code, symbol) in CURRENCIES.items():
        if code == st.session_state.user_currency:
            return symbol
    return "€"

def fmt(amount):
    return f"{sym()}{amount:,.2f}"

# ════════════════════════════════════════════════════════════════════════════
# AUTH SCREEN — shown when not logged in
# ════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:

    # ── Top navbar style ──
    st.markdown("""
    <style>
    .navbar {
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.5rem 0 1.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 2rem;
    }
    .navbar-brand { font-size: 1.5rem; font-weight: 600; color: #00C9A7; }
    .hero-tag {
        display: inline-block; background: rgba(0,201,167,0.12);
        color: #00C9A7; border-radius: 20px; padding: 4px 14px;
        font-size: 0.8rem; font-weight: 500; margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.8rem; font-weight: 700; line-height: 1.2;
        color: var(--color-text-primary); margin-bottom: 1rem;
    }
    .hero-title span { color: #00C9A7; }
    .hero-sub {
        font-size: 1rem; color: var(--color-text-secondary);
        margin-bottom: 2rem; line-height: 1.6;
    }
    .feature-item {
        display: flex; align-items: flex-start; gap: 12px;
        margin-bottom: 1rem;
    }
    .feature-icon {
        background: rgba(0,201,167,0.12); border-radius: 10px;
        padding: 8px; font-size: 1.1rem; min-width: 40px;
        text-align: center;
    }
    .feature-text strong { color: var(--color-text-primary); }
    .feature-text p { color: var(--color-text-secondary); font-size: 0.85rem; margin: 0; }
    .auth-card {
        background: var(--color-background-secondary);
        border: 1px solid var(--color-border-tertiary);
        border-radius: 16px; padding: 2rem;
    }
    .stat-row { display: flex; gap: 1.5rem; margin-bottom: 2rem; }
    .stat-box {
        background: var(--color-background-secondary);
        border: 1px solid var(--color-border-tertiary);
        border-radius: 12px; padding: 1rem 1.5rem; flex: 1; text-align: center;
    }
    .stat-num { font-size: 1.6rem; font-weight: 700; color: #00C9A7; }
    .stat-label { font-size: 0.78rem; color: var(--color-text-secondary); }
    </style>
    """, unsafe_allow_html=True)

    # Navbar
    st.markdown("""
    <div class="navbar">
        <div class="navbar-brand">💰 ValtoSpend</div>
    </div>
    """, unsafe_allow_html=True)

    # Hero layout — left: tagline + features, right: auth form
    col_hero, col_auth = st.columns([1.2, 1], gap="large")

    with col_hero:
        if os.path.exists("logo.png"):
            st.image("logo.png", width=130)
        st.markdown('<div class="hero-tag">AI-Powered Finance</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="hero-title">
            The smarter way to<br>
            <span>track your money</span>
        </div>
        <div class="hero-sub">
            ValtoSpend uses AI to analyse your spending habits,
            predict future expenses, and help you stay on budget —
            all in one place.
        </div>
        """, unsafe_allow_html=True)

        # Stats row
        st.markdown("""
        <div class="stat-row">
            <div class="stat-box">
                <div class="stat-num">3,655</div>
                <div class="stat-label">Real user records</div>
            </div>
            <div class="stat-box">
                <div class="stat-num">3</div>
                <div class="stat-label">AI models</div>
            </div>
            <div class="stat-box">
                <div class="stat-num">50+</div>
                <div class="stat-label">Currencies</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Features list
        features = [
            ("📸", "Receipt Scanner", "AI reads your receipts and logs expenses automatically"),
            ("🤖", "AI Predictions",  "Predicts next month's spending using machine learning"),
            ("💰", "Budget Alerts",   "Get warned before you overspend any category"),
            ("📊", "Market Insights", "Compare your habits with thousands of real users"),
            ("🌍", "Multi-Currency",  "Switch between €, $, £, ₹ and more instantly"),
        ]
        for icon, title, desc in features:
            st.markdown(f"""
            <div class="feature-item">
                <div class="feature-icon">{icon}</div>
                <div class="feature-text">
                    <strong>{title}</strong>
                    <p>{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_auth:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        tabs = st.tabs(["🔑 Login", "📝 Register"])

        # ── LOGIN ──
        with tabs[0]:
            st.subheader("Welcome back!")
            with st.form("login_form"):
                lu = st.text_input("Username")
                lp = st.text_input("Password", type="password")
                if st.form_submit_button("Login →", type="primary", use_container_width=True):
                    ok, result = login_user(lu, lp)
                    if ok:
                        uid, uname, income, bracket, currency = result
                        st.session_state.logged_in     = True
                        st.session_state.user_id       = uid
                        st.session_state.username      = uname
                        st.session_state.user_income   = income
                        st.session_state.user_bracket  = bracket
                        st.session_state.user_currency = currency
                        st.rerun()
                    else:
                        st.error(result)

        # ── REGISTER ──
        with tabs[1]:
            st.subheader("Create your account")
            with st.form("register_form"):
                ru       = st.text_input("Choose a username")
                rp       = st.text_input("Choose a password", type="password")
                rp2      = st.text_input("Confirm password",  type="password")
                rincome  = st.number_input("Monthly income", min_value=0.0,
                                           step=100.0, value=2000.0)
                rbracket = st.selectbox("Income bracket", BRACKETS)
                rcurr    = st.selectbox("Currency", list(CURRENCIES.keys()))
                if st.form_submit_button("Create Account →", type="primary",
                                         use_container_width=True):
                    if rp != rp2:
                        st.error("Passwords don't match.")
                    else:
                        curr_code = CURRENCIES[rcurr][0]
                        ok, msg = register_user(ru, rp, rincome, rbracket, curr_code)
                        if ok:
                            st.success("✅ Account created! Please login.")
                        else:
                            st.error(msg)
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# MAIN APP — shown after login
# ════════════════════════════════════════════════════════════════════════════
user_id      = st.session_state.user_id
username     = st.session_state.username
user_income  = st.session_state.user_income
user_bracket = st.session_state.user_bracket

# Global black theme for main app
st.markdown("""
<style>
.stApp, section[data-testid="stSidebar"], .stMainBlockContainer {
    background-color: #000000 !important;
}
.stButton > button[kind="primary"] {
    background: #00C9A7 !important; color: #000 !important;
    font-weight: 600 !important; border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #00C9A7 !important; border-bottom-color: #00C9A7 !important;
}
</style>
""", unsafe_allow_html=True)

col_logo, col_title = st.columns([1, 9])
with col_logo:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=65)
with col_title:
    st.title(f"Welcome back, {username.capitalize()}! 👋")

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"**👤 {username.capitalize()}**")
    st.caption(f"Income: {fmt(user_income)} | {user_bracket}")

    # Currency selector
    st.subheader("🌍 Currency")
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

    # Profile edit
    if st.button("✏️ Edit Profile"):
        st.session_state["editing_profile"] = True

    # Logout
    if st.button("🚪 Logout"):
        for k in ["logged_in","user_id","username","user_income",
                  "user_bracket","user_currency"]:
            st.session_state[k] = None if k != "logged_in" else False
        st.rerun()

    st.header("🔍 Filters")
    brackets_list = ["All"] + sorted(df["Income_Bracket"].dropna().unique().tolist())
    sel_bracket   = st.selectbox("Income Bracket", brackets_list)

# Profile edit expander
if st.session_state.get("editing_profile"):
    with st.expander("✏️ Edit Profile", expanded=True):
        with st.form("edit_profile"):
            new_income  = st.number_input("Monthly Income", value=float(user_income), step=100.0)
            new_bracket = st.selectbox("Income Bracket", BRACKETS,
                           index=BRACKETS.index(user_bracket) if user_bracket in BRACKETS else 0)
            if st.form_submit_button("💾 Save"):
                update_profile(user_id, new_income, new_bracket,
                               st.session_state.user_currency)
                st.session_state.user_income  = new_income
                st.session_state.user_bracket = new_bracket
                st.session_state["editing_profile"] = False
                st.rerun()

# Apply filters
filtered = df.copy()
if sel_bracket != "All":
    filtered = filtered[filtered["Income_Bracket"] == sel_bracket]

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["👤 My Expenses", "💰 Budget", "📊 Market Insights", "🤖 AI Prediction", "🌐 Live Data"]
)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — MY EXPENSES
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header(f"👤 {username.capitalize()}'s Expenses")

    # Receipt scanner
    st.subheader("📸 Scan a Receipt")
    st.caption("Upload a receipt photo — AI reads it and fills in the details automatically.")
    uploaded_file = st.file_uploader("Upload", type=["jpg","jpeg","png"],
                                     label_visibility="collapsed")

    for k, v in [("prefill_amount",10.0),("prefill_category","Food"),
                 ("prefill_note",""),("scan_done",False)]:
        if k not in st.session_state:
            st.session_state[k] = v

    if uploaded_file and not st.session_state.scan_done:
        with st.spinner("🤖 Reading your receipt..."):
            img_bytes  = uploaded_file.read()
            media_type = "image/png" if uploaded_file.name.endswith(".png") else "image/jpeg"
            result     = read_receipt_with_ai(img_bytes, media_type)
        if "error" in result:
            st.warning(f"Couldn't read receipt: {result['error']}. Please fill in manually.")
        else:
            st.success(f"✅ Found: **{fmt(result.get('amount',0))}** — {result.get('note','')}")
            st.session_state.prefill_amount   = float(result.get("amount", 10.0))
            st.session_state.prefill_category = result.get("category", "Food")
            st.session_state.prefill_note     = result.get("note", "")
            st.session_state.scan_done        = True
            st.image(img_bytes, caption="Uploaded receipt", width=200)
    if not uploaded_file:
        st.session_state.scan_done = False

    # Add expense form
    st.subheader("➕ Add Expense")
    c1, c2, c3, c4 = st.columns(4)
    exp_date = c1.date_input("Date", value=date.today())
    cat_idx  = SPEND_COLS.index(st.session_state.prefill_category) \
               if st.session_state.prefill_category in SPEND_COLS else 0
    exp_cat  = c2.selectbox("Category", SPEND_COLS, index=cat_idx)
    exp_amt  = c3.number_input("Amount", min_value=0.01, step=0.50,
                                value=float(st.session_state.prefill_amount))
    exp_note = c4.text_input("Note", value=st.session_state.prefill_note,
                              placeholder="e.g. Lunch at work")

    if st.button("💾 Save Expense", type="primary"):
        add_my_expense(user_id, exp_date, exp_cat, exp_amt, exp_note)
        st.success(f"✅ Saved: {fmt(exp_amt)} for {exp_cat}")
        for k, v in [("prefill_amount",10.0),("prefill_category","Food"),
                     ("prefill_note",""),("scan_done",False)]:
            st.session_state[k] = v
        st.rerun()

    st.divider()
    my_df = load_my_expenses(user_id)

    if my_df.empty:
        st.info("No expenses yet! Upload a receipt or add one manually above ☝️")
    else:
        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Spent",       fmt(my_df['amount'].sum()))
        m2.metric("Number of Entries", len(my_df))
        m3.metric("Biggest Expense",   fmt(my_df['amount'].max()))
        m4.metric("Top Category",      my_df.groupby("category")["amount"].sum().idxmax())

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("My Spending by Category")
            st.pyplot(personal_category_chart(my_df))
        with col_b:
            st.subheader("My Spending Over Time")
            st.pyplot(personal_weekly_chart(my_df))

        # AI spending tips
        st.subheader("💡 AI Spending Tips")
        cat_totals  = my_df.groupby("category")["amount"].sum()
        total_spent = my_df["amount"].sum()
        tips = []

        top_cat = cat_totals.idxmax()
        top_pct = cat_totals[top_cat] / total_spent * 100
        tips.append(f"🔴 **{top_cat}** is your biggest expense at **{top_pct:.0f}%** of total spending.")

        if "Food" in cat_totals and "Groceries" in cat_totals:
            food_total = cat_totals.get("Food",0) + cat_totals.get("Groceries",0)
            food_pct   = food_total / total_spent * 100
            if food_pct > 35:
                tips.append(f"🍔 You spend **{food_pct:.0f}%** on Food & Groceries combined — consider meal planning to reduce this.")
            else:
                tips.append(f"✅ Your Food & Groceries spending is healthy at **{food_pct:.0f}%** of total.")

        if len(cat_totals) >= 3:
            lowest_cat = cat_totals.idxmin()
            tips.append(f"💚 Your lowest spending category is **{lowest_cat}** — good financial balance!")

        for tip in tips:
            st.markdown(f"> {tip}")

        # Personal next month prediction
        st.subheader("🤖 My Next Month Prediction")
        my_df["month"] = my_df["date"].dt.to_period("M").astype(str)
        monthly_me = my_df.groupby("month")["amount"].sum().reset_index()
        monthly_me["idx"] = range(len(monthly_me))
        if len(monthly_me) >= 2:
            lr_me = LinearRegression()
            lr_me.fit(monthly_me[["idx"]], monthly_me["amount"])
            next_pred_me  = lr_me.predict([[len(monthly_me)]])[0]
            next_mo_label = (my_df["date"].max() + pd.DateOffset(months=1)).strftime("%B %Y")
            st.success(f"💡 Predicted spend for **{next_mo_label}**: **{fmt(next_pred_me)}**")
        else:
            st.info("Add expenses across at least 2 months to unlock your personal prediction.")

        # Export CSV
        st.subheader("📥 Export My Expenses")
        csv_data = export_expenses_csv(user_id)
        st.download_button(
            label="⬇️ Download as CSV",
            data=csv_data,
            file_name=f"valtospend_{username}_expenses.csv",
            mime="text/csv"
        )

        # Expense table
        st.subheader("📋 All My Expenses")
        for _, row in my_df.iterrows():
            c1, c2, c3, c4, c5 = st.columns([2,2,1,3,1])
            c1.write(row["date"].strftime("%d %b %Y"))
            c2.write(row["category"])
            c3.write(fmt(row["amount"]))
            c4.write(row["note"] if row["note"] else "—")
            if c5.button("🗑️", key=f"del_{row['id']}"):
                delete_my_expense(row["id"])
                st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — BUDGET ALERTS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("💰 Monthly Budget Tracker")
    st.caption("Set spending limits per category. Get alerted when you're close to or over budget.")

    budgets = get_budgets(user_id)
    my_df_b = load_my_expenses(user_id)

    # Current month spending
    this_month = date.today().strftime("%Y-%m")
    if not my_df_b.empty:
        my_df_b["month"] = my_df_b["date"].dt.strftime("%Y-%m")
        this_month_df    = my_df_b[my_df_b["month"] == this_month]
        monthly_cats     = this_month_df.groupby("category")["amount"].sum().to_dict()
    else:
        monthly_cats = {}

    # Set budgets
    st.subheader("⚙️ Set Your Monthly Budgets")
    with st.expander("Click to set category budgets"):
        cols = st.columns(3)
        for i, cat in enumerate(SPEND_COLS):
            with cols[i % 3]:
                current = budgets.get(cat, 0.0)
                new_val = st.number_input(f"{cat}", min_value=0.0,
                                          value=float(current), step=10.0,
                                          key=f"budget_{cat}")
                if new_val != current and new_val > 0:
                    set_budget(user_id, cat, new_val)

    st.divider()
    st.subheader(f"📊 This Month — {date.today().strftime('%B %Y')}")

    if not budgets:
        st.info("Set your budgets above to see progress bars here!")
    else:
        for cat in SPEND_COLS:
            if cat in budgets:
                limit   = budgets[cat]
                spent   = monthly_cats.get(cat, 0.0)
                pct     = min(spent / limit * 100, 100) if limit > 0 else 0
                remaining = max(limit - spent, 0)

                col_l, col_r = st.columns([3, 1])
                with col_l:
                    if pct >= 100:
                        st.markdown(f"🔴 **{cat}** — OVER BUDGET!")
                        color = "red"
                    elif pct >= 80:
                        st.markdown(f"🟡 **{cat}** — Getting close!")
                        color = "orange"
                    else:
                        st.markdown(f"🟢 **{cat}**")
                        color = "green"
                    st.progress(int(pct))
                with col_r:
                    st.metric("Spent",     fmt(spent))
                    st.metric("Remaining", fmt(remaining))
                st.caption(f"Budget: {fmt(limit)} | Spent: {fmt(spent)} ({pct:.0f}%)")
                st.divider()

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MARKET INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📊 Market Insights")
    st.caption("Analysis based on 3,655 real user records (2021–2024).")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",       f"{len(filtered):,}")
    c2.metric("Avg Monthly Spend",   fmt(filtered['Total_Expenses'].mean()))
    c3.metric("Avg Monthly Income",  fmt(filtered['Income'].mean()))
    c4.metric("Avg Monthly Savings", fmt(filtered['Savings'].mean()))

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Average spend per category")
        fig, avg_cats = bar_chart_categories(filtered)
        st.pyplot(fig)
    with col_r:
        st.subheader("Category distribution")
        st.pyplot(pie_chart_categories(avg_cats))

    st.subheader("💼 Income Bracket Comparison")
    st.pyplot(bracket_comparison_chart(df))

    st.subheader("📅 Monthly Spending Trend")
    st.pyplot(monthly_trend_chart(filtered))

    if "Festivals" in df.columns:
        st.subheader("🎉 Festival Impact")
        fest_s = (filtered.groupby("Festivals")["Total_Expenses"].mean()
                  .reset_index()
                  .rename(columns={"Festivals":"Festival","Total_Expenses":"Avg Expenses"})
                  .sort_values("Avg Expenses", ascending=False).reset_index(drop=True))
        st.dataframe(fest_s)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI PREDICTION
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("🤖 AI Prediction Engine")

    # Model 1 — Linear Regression
    st.subheader("📆 Market Next Month Forecast")
    st.caption("Linear Regression on monthly average spending trend.")
    next_p, mae_ts, t_preds, next_lbl, monthly_ts = linear_regression_forecast(df)
    ma, mb = st.columns(2)
    ma.metric(f"Predicted avg spend for {next_lbl}", fmt(next_p))
    mb.metric("Model MAE", fmt(mae_ts))
    st.pyplot(forecast_chart(monthly_ts, t_preds, next_p, next_lbl))

    st.divider()

    # Model 2 — Random Forest
    st.subheader(f"🔮 {username.capitalize()}'s Personalised Predictor")
    st.caption("Random Forest Regressor — 10 decision trees, evaluated on 20% test set.")
    rf, le, mae_rf, r2_rf = random_forest_model(df)
    mc, md = st.columns(2)
    mc.metric("MAE (test set)", fmt(mae_rf))
    md.metric("R² Score",       f"{r2_rf:.3f}")
    st.pyplot(feature_importance_chart(rf, FEATURES + ["Income_Bracket"]))

    p1, p2, p3 = st.columns(3)
    income_in = p1.slider("Monthly Income", 500, 10000,
                          int(min(max(user_income,500),10000)), step=100)
    month_in  = p2.slider("Month", 1, 12, date.today().month)
    fest_in   = p3.slider("Festival Count", 0, 5, 1)
    bracket_options = list(le.classes_)
    default_idx     = bracket_options.index(user_bracket) if user_bracket in bracket_options else 0
    bracket_in      = st.selectbox("Income Bracket", bracket_options, index=default_idx)
    bracket_enc     = le.transform([bracket_in])[0]
    default_ratios  = [0.20,0.12,0.10,0.08,0.10,0.25,0.07,0.04,0.04]
    input_row       = np.array([[income_in,month_in,fest_in,*default_ratios,bracket_enc]])
    st.success(f"💡 Predicted expenses: **{fmt(rf.predict(input_row)[0])}**")

    st.divider()

    # Model 3 — Neural Network from scratch
    st.subheader("🧠 Neural Network (Built from Scratch)")
    st.caption(
        "NumPy-only neural network — no TensorFlow. "
        "Architecture: Input → Dense(64,ReLU) → Dense(32,ReLU) → Output. "
        "Trained with gradient descent for 100 epochs."
    )
    with st.spinner("🧠 Training neural network..."):
        mae_nn, r2_nn, losses = neural_network_scratch(df)
    n1, n2 = st.columns(2)
    n1.metric("Neural Network MAE", fmt(mae_nn))
    n2.metric("Neural Network R²",  f"{r2_nn:.3f}")
    st.write("**Training loss — decreasing curve proves the network is learning:**")
    st.pyplot(nn_loss_chart(losses))

    # Comparison table
    st.write("**All 3 AI Models Compared:**")
    comparison = pd.DataFrame({
        "Model":    ["Linear Regression","Random Forest","Neural Network (scratch)"],
        "Type":     ["Statistical","Ensemble ML","Deep Learning"],
        "MAE":      [fmt(mae_ts), fmt(mae_rf), fmt(mae_nn)],
        "R² Score": ["N/A", f"{r2_rf:.3f}", f"{r2_nn:.3f}"]
    })
    st.dataframe(comparison, hide_index=True)

    with st.expander("🗂️ View raw database records"):
        min_date = df["Date"].min().date()
        max_date = df["Date"].max().date()
        date_range = st.date_input("Filter by date range", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date,
                                   key="records_date")
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start, end = date_range
            records_df = df[(df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)]
        else:
            records_df = df
        st.dataframe(records_df[["UserID","Date","Income","Income_Bracket",
                          "Total_Expenses","Savings"]+SPEND_COLS].head(200))

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — LIVE DATA
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("🌐 Live Data")

    # ── SECTION 1: Community Stats ──────────────────────────────────────
    st.subheader("👥 ValtoSpend Community — Live Stats")
    st.caption("Real-time statistics from all registered ValtoSpend users. Updates every time someone logs an expense.")

    stats = get_community_stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👤 Registered Users",    stats["total_users"])
    c2.metric("📝 Expenses Logged",     stats["total_logged"])
    c3.metric("📅 Logged This Month",   stats["this_month"])
    c4.metric("🏆 Top Category",        stats["top_category"])

    if stats["total_logged"] > 0:
        st.metric("💰 Total Community Spend", fmt(stats["total_amount"]))
        st.metric("📊 Avg per Transaction",   fmt(stats["avg_expense"]))

        if stats["category_totals"]:
            st.write("**Community spending by category:**")
            import matplotlib.pyplot as plt
            cat_df = pd.DataFrame(
                list(stats["category_totals"].items()),
                columns=["Category", "Total"]
            ).sort_values("Total", ascending=False)
            fig_c, ax_c = plt.subplots(figsize=(8, 3))
            ax_c.bar(cat_df["Category"], cat_df["Total"], color="#00C9A7")
            ax_c.set_ylabel(f"Total ({sym()})")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig_c)
            plt.close()
    else:
        st.info("No community data yet — be the first to log expenses!")

    st.divider()

    # ── SECTION 2: Live Currency Rates ──────────────────────────────────
    st.subheader("💱 Live Currency Exchange Rates")
    st.caption("Live rates updated every 24 hours from ExchangeRate-API.")

    base_curr = st.selectbox(
        "Base currency",
        list(CURRENCIES.keys()),
        index=next((i for i, (k,(c,s)) in enumerate(CURRENCIES.items())
                    if c == st.session_state.user_currency), 0)
    )
    base_code = CURRENCIES[base_curr][0]

    with st.spinner("Fetching live rates..."):
        rates = get_live_rates(EXCHANGE_API_KEY, base_code)

    if "error" in rates:
        st.error(f"Could not fetch rates: {rates['error']}")
    elif rates:
        st.success(f"✅ Rates updated: {rates.get('updated', 'N/A')}")

        # Show key rates
        key_currencies = ["USD","EUR","GBP","INR","JPY","CAD","AUD",
                          "CHF","CNY","KRW","BRL","MXN","SGD","HKD"]
        key_rates = {k: v for k, v in rates["rates"].items()
                     if k in key_currencies and k != base_code}

        rate_df = pd.DataFrame(
            [(k, f"{v:.4f}") for k, v in sorted(key_rates.items())],
            columns=[f"Currency", f"1 {base_code} ="]
        )

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.dataframe(rate_df, hide_index=True, use_container_width=True)
        with col_r2:
            # Converter
            st.write("**Quick Converter:**")
            amount_in  = st.number_input("Amount", value=100.0, step=10.0)
            to_curr    = st.selectbox("Convert to", list(CURRENCIES.keys()))
            to_code    = CURRENCIES[to_curr][0]
            if to_code in rates.get("rates", {}):
                converted = amount_in * rates["rates"].get(to_code, 1)
                st.success(f"**{amount_in:.2f} {base_code} = {converted:.2f} {to_code}**")

        # Show all rates in expander
        with st.expander("📋 See all 50+ currency rates"):
            all_rates_df = pd.DataFrame(
                [(k, f"{v:.4f}") for k, v in sorted(rates["rates"].items())],
                columns=["Currency", f"1 {base_code} ="]
            )
            st.dataframe(all_rates_df, hide_index=True, use_container_width=True)
