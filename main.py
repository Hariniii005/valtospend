import pandas as pd
import streamlit as st
import sqlite3
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from datetime import date
import base64
import json
import urllib.request
import urllib.error

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="ValtoSpend", page_icon="💰", layout="wide")

SPEND_COLS = [
    "Food", "Groceries", "Transport", "Entertainment",
    "Shopping", "Rent", "Bills", "Healthcare", "Education"
]

BRACKETS = ["Low Income", "Lower Middle Income", "Middle Income",
            "Upper Middle Income", "High Income"]

# ============================================================
# DATABASE SETUP
# ============================================================
DB_PATH = "valtospend.db"
CSV_PATH = "expenses.csv"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            UserID TEXT, Year INTEGER, Month INTEGER,
            Income REAL, Income_Bracket TEXT,
            Festivals TEXT, Festival_Count INTEGER,
            Food REAL, Groceries REAL, Transport REAL,
            Entertainment REAL, Shopping REAL, Rent REAL,
            Bills REAL, Healthcare REAL, Education REAL,
            Total_Expenses REAL, Savings REAL,
            Food_Ratio REAL, Groceries_Ratio REAL,
            Transport_Ratio REAL, Entertainment_Ratio REAL,
            Shopping_Ratio REAL, Rent_Ratio REAL,
            Bills_Ratio REAL, Healthcare_Ratio REAL,
            Education_Ratio REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS my_expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT, category TEXT,
            amount REAL, note TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, income REAL, bracket TEXT
        )
    """)
    count = cursor.execute("SELECT COUNT(*) FROM expenses").fetchone()[0]
    if count == 0 and os.path.exists(CSV_PATH):
        df_csv = pd.read_csv(CSV_PATH)
        df_csv.to_sql("expenses", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM expenses", conn)
    conn.close()
    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2) + "-01"
    )
    return df

def load_my_expenses():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM my_expenses ORDER BY date DESC", conn)
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df

def add_my_expense(date_val, category, amount, note):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO my_expenses (date, category, amount, note) VALUES (?, ?, ?, ?)",
        (str(date_val), category, float(amount), note)
    )
    conn.commit()
    conn.close()

def delete_my_expense(row_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM my_expenses WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()

def get_profile():
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT name, income, bracket FROM user_profile LIMIT 1").fetchone()
    conn.close()
    return row  # (name, income, bracket) or None

def save_profile(name, income, bracket):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM user_profile")
    conn.execute(
        "INSERT INTO user_profile (name, income, bracket) VALUES (?, ?, ?)",
        (name, float(income), bracket)
    )
    conn.commit()
    conn.close()

# ============================================================
# AI RECEIPT READER — calls Claude API
# ============================================================
def read_receipt_with_ai(image_bytes, media_type="image/jpeg"):
    """Send receipt image to Claude and extract amount + category."""
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 300,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.b64encode(image_bytes).decode("utf-8")
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "Look at this receipt image. Extract the total amount spent and "
                        "the most appropriate spending category from this list: "
                        "Food, Groceries, Transport, Entertainment, Shopping, Rent, Bills, Healthcare, Education. "
                        "Also write a short note describing what was purchased. "
                        "Respond ONLY with valid JSON, no extra text, in this exact format: "
                        '{"amount": 12.50, "category": "Food", "note": "Lunch at cafe"}'
                    )
                }
            ]
        }]
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=data,
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            text = result["content"][0]["text"].strip()
            # Strip markdown fences if present
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
    except Exception as e:
        return {"error": str(e)}

# ============================================================
# INIT
# ============================================================
init_db()
df = load_data()
profile = get_profile()

# ============================================================
# PROFILE SETUP — shown if no profile exists
# ============================================================
if profile is None:
    st.title("💰 Welcome to ValtoSpend")
    st.subheader("Let's set up your profile first!")
    st.write("This takes 10 seconds and personalises the entire app for you.")

    with st.form("profile_form"):
        p_name    = st.text_input("Your first name", placeholder="e.g. Harini")
        p_income  = st.number_input("Your monthly income (€)", min_value=0.0, step=100.0, value=2000.0)
        p_bracket = st.selectbox("Income bracket", BRACKETS)
        submitted = st.form_submit_button("🚀 Start ValtoSpend", type="primary")
        if submitted and p_name.strip():
            save_profile(p_name.strip(), p_income, p_bracket)
            st.success(f"Welcome, {p_name}! 🎉")
            st.rerun()
        elif submitted:
            st.error("Please enter your name.")
    st.stop()

# Profile exists — unpack
user_name, user_income, user_bracket = profile

# ============================================================
# MAIN APP — shown after profile setup
# ============================================================
st.title(f"💰 ValtoSpend — Welcome back, {user_name}! 👋")

# Profile edit in sidebar
with st.sidebar:
    st.markdown(f"**👤 {user_name}**")
    st.caption(f"Income: €{user_income:,.0f} | {user_bracket}")
    if st.button("✏️ Edit Profile"):
        st.session_state["editing_profile"] = True

if st.session_state.get("editing_profile"):
    with st.expander("Edit your profile", expanded=True):
        with st.form("edit_profile"):
            new_name    = st.text_input("Name", value=user_name)
            new_income  = st.number_input("Monthly Income (€)", value=float(user_income), step=100.0)
            new_bracket = st.selectbox("Income Bracket", BRACKETS,
                                       index=BRACKETS.index(user_bracket) if user_bracket in BRACKETS else 0)
            if st.form_submit_button("💾 Save"):
                save_profile(new_name, new_income, new_bracket)
                st.session_state["editing_profile"] = False
                st.rerun()

tab1, tab2, tab3 = st.tabs(["👤 My Expenses", "📊 Market Insights", "🤖 AI Prediction"])

# ============================================================
# TAB 1 — MY EXPENSES
# ============================================================
with tab1:
    st.header(f"👤 {user_name}'s Expenses")

    # --- RECEIPT UPLOAD ---
    st.subheader("📸 Scan a Receipt")
    st.caption("Take a photo of any receipt — AI will read it and fill in the details automatically.")

    uploaded_file = st.file_uploader(
        "Upload receipt photo", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    # Pre-fill state
    if "prefill_amount"   not in st.session_state: st.session_state.prefill_amount   = 10.0
    if "prefill_category" not in st.session_state: st.session_state.prefill_category = "Food"
    if "prefill_note"     not in st.session_state: st.session_state.prefill_note     = ""
    if "scan_done"        not in st.session_state: st.session_state.scan_done        = False

    if uploaded_file and not st.session_state.scan_done:
        with st.spinner("🤖 Reading your receipt..."):
            img_bytes  = uploaded_file.read()
            ext        = uploaded_file.name.split(".")[-1].lower()
            media_type = "image/png" if ext == "png" else "image/jpeg"
            result     = read_receipt_with_ai(img_bytes, media_type)

        if "error" in result:
            st.warning(f"Couldn't read receipt automatically: {result['error']}. Please fill in manually below.")
        else:
            st.success(f"✅ Receipt scanned! Found: **€{result.get('amount', 0):.2f}** — {result.get('note', '')}")
            st.session_state.prefill_amount   = float(result.get("amount", 10.0))
            st.session_state.prefill_category = result.get("category", "Food")
            st.session_state.prefill_note     = result.get("note", "")
            st.session_state.scan_done        = True
            col_img, _ = st.columns([1, 2])
            with col_img:
                st.image(img_bytes, caption="Uploaded receipt", width=200)

    # Reset scan when uploader is cleared
    if not uploaded_file:
        st.session_state.scan_done = False

    # --- ADD EXPENSE FORM ---
    st.subheader("➕ Add Expense")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        exp_date = st.date_input("Date", value=date.today())
    with col2:
        cat_idx  = SPEND_COLS.index(st.session_state.prefill_category) \
                   if st.session_state.prefill_category in SPEND_COLS else 0
        exp_cat  = st.selectbox("Category", SPEND_COLS, index=cat_idx)
    with col3:
        exp_amt  = st.number_input("Amount (€)", min_value=0.01, step=0.50,
                                   value=float(st.session_state.prefill_amount))
    with col4:
        exp_note = st.text_input("Note", value=st.session_state.prefill_note,
                                 placeholder="e.g. Lunch at work")

    if st.button("💾 Save Expense", type="primary"):
        add_my_expense(exp_date, exp_cat, exp_amt, exp_note)
        st.success(f"✅ Saved: €{exp_amt:.2f} for {exp_cat}")
        # Reset prefills
        st.session_state.prefill_amount   = 10.0
        st.session_state.prefill_category = "Food"
        st.session_state.prefill_note     = ""
        st.session_state.scan_done        = False
        st.rerun()

    st.divider()

    # --- PERSONAL DASHBOARD ---
    my_df = load_my_expenses()

    if my_df.empty:
        st.info("No expenses yet! Upload a receipt or add one manually above ☝️")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Spent",       f"€{my_df['amount'].sum():,.2f}")
        m2.metric("Number of Entries", len(my_df))
        m3.metric("Biggest Expense",   f"€{my_df['amount'].max():,.2f}")
        m4.metric("Top Category",      my_df.groupby("category")["amount"].sum().idxmax())

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("My Spending by Category")
            cat_totals = my_df.groupby("category")["amount"].sum().sort_values(ascending=False)
            fig1, ax1  = plt.subplots(figsize=(6, 3))
            ax1.bar(cat_totals.index, cat_totals.values, color="#4C72B0")
            ax1.set_ylabel("Total (€)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close()

        with col_b:
            st.subheader("My Spending Over Time")
            my_df["week"] = my_df["date"].dt.to_period("W").astype(str)
            weekly = my_df.groupby("week")["amount"].sum()
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.plot(weekly.index, weekly.values, marker="o", color="#DD8452")
            ax2.set_ylabel("Total (€)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        # Personal prediction
        st.subheader("🤖 My Next Month Prediction")
        my_df["month"] = my_df["date"].dt.to_period("M").astype(str)
        monthly_me = my_df.groupby("month")["amount"].sum().reset_index()
        monthly_me["idx"] = range(len(monthly_me))
        if len(monthly_me) >= 2:
            lr_me = LinearRegression()
            lr_me.fit(monthly_me[["idx"]], monthly_me["amount"])
            next_pred_me = lr_me.predict([[len(monthly_me)]])[0]
            next_mo_label = (my_df["date"].max() + pd.DateOffset(months=1)).strftime("%B %Y")
            st.success(
                f"💡 {user_name}, based on your spending, "
                f"predicted spend for **{next_mo_label}**: **€{next_pred_me:,.2f}**"
            )
        else:
            st.info("Add expenses across at least 2 months to unlock your personal prediction.")

        # Expense table
        st.subheader("📋 All My Expenses")
        for _, row in my_df.iterrows():
            c1, c2, c3, c4, c5 = st.columns([2, 2, 1, 3, 1])
            c1.write(row["date"].strftime("%d %b %Y"))
            c2.write(row["category"])
            c3.write(f"€{row['amount']:.2f}")
            c4.write(row["note"] if row["note"] else "—")
            if c5.button("🗑️", key=f"del_{row['id']}"):
                delete_my_expense(row["id"])
                st.rerun()

# ============================================================
# TAB 2 — MARKET INSIGHTS
# ============================================================
with tab2:
    st.header("📊 Market Insights")
    st.caption("Analysis based on 3,655 real user records (2021–2024).")

    st.sidebar.header("🔍 Filters")
    min_date  = df["Date"].min().date()
    max_date  = df["Date"].max().date()
    date_range = st.sidebar.date_input(
        "Date range", value=(min_date, max_date),
        min_value=min_date, max_value=max_date
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = date_range
        filtered = df[(df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)].copy()
    else:
        filtered = df.copy()

    brackets_list = ["All"] + sorted(df["Income_Bracket"].dropna().unique().tolist())
    sel_bracket   = st.sidebar.selectbox("Income Bracket", brackets_list)
    if sel_bracket != "All":
        filtered = filtered[filtered["Income_Bracket"] == sel_bracket]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",       f"{len(filtered):,}")
    c2.metric("Avg Monthly Spend",   f"€{filtered['Total_Expenses'].mean():,.0f}")
    c3.metric("Avg Monthly Income",  f"€{filtered['Income'].mean():,.0f}")
    c4.metric("Avg Monthly Savings", f"€{filtered['Savings'].mean():,.0f}")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Average spend per category")
        avg_cats = filtered[SPEND_COLS].mean().sort_values(ascending=False)
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.bar(avg_cats.index, avg_cats.values, color="#4C72B0")
        ax3.set_ylabel("Average (€)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with col_r:
        st.subheader("Category distribution")
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        ax4.pie(avg_cats.values, labels=avg_cats.index, autopct="%1.1f%%", startangle=90)
        ax4.axis("equal")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    st.subheader("💼 Income Bracket Comparison")
    b_summary = df.groupby("Income_Bracket")[
        ["Income", "Total_Expenses", "Savings"]
    ].mean().reset_index().sort_values("Income")
    fig5, ax5 = plt.subplots(figsize=(8, 4))
    x = np.arange(len(b_summary))
    w = 0.25
    ax5.bar(x-w, b_summary["Income"],         w, label="Income",         color="#4C72B0")
    ax5.bar(x,   b_summary["Total_Expenses"], w, label="Total Expenses", color="#DD8452")
    ax5.bar(x+w, b_summary["Savings"],        w, label="Savings",        color="#55A868")
    ax5.set_xticks(x)
    ax5.set_xticklabels(b_summary["Income_Bracket"], rotation=15)
    ax5.set_ylabel("Average (€)")
    ax5.legend()
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

    st.subheader("📅 Monthly Spending Trend")
    m_trend = filtered.groupby("Date")["Total_Expenses"].mean().reset_index().sort_values("Date")
    m_trend["Label"] = m_trend["Date"].dt.strftime("%b %Y")
    fig6, ax6 = plt.subplots(figsize=(10, 3))
    ax6.plot(m_trend["Label"], m_trend["Total_Expenses"], marker="o", color="#4C72B0")
    ax6.set_ylabel("Avg Total Expenses (€)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()

    if "Festivals" in df.columns:
        st.subheader("🎉 Festival Impact")
        fest_s = (filtered.groupby("Festivals")["Total_Expenses"].mean()
                  .reset_index()
                  .rename(columns={"Festivals":"Festival","Total_Expenses":"Avg Expenses"})
                  .sort_values("Avg Expenses", ascending=False).reset_index(drop=True))
        st.dataframe(fest_s)

# ============================================================
# TAB 3 — AI PREDICTION
# ============================================================
with tab3:
    st.header("🤖 AI Prediction Engine")

    st.subheader("📆 Market Next Month Forecast")
    monthly_ts = df.groupby("Date")["Total_Expenses"].mean().reset_index().sort_values("Date")
    monthly_ts["idx"] = range(len(monthly_ts))
    if len(monthly_ts) >= 3:
        lr2       = LinearRegression()
        lr2.fit(monthly_ts[["idx"]], monthly_ts["Total_Expenses"])
        next_p    = lr2.predict([[len(monthly_ts)]])[0]
        t_preds   = lr2.predict(monthly_ts[["idx"]])
        mae_ts    = mean_absolute_error(monthly_ts["Total_Expenses"], t_preds)
        next_lbl  = (monthly_ts["Date"].max() + pd.DateOffset(months=1)).strftime("%B %Y")

        ma, mb = st.columns(2)
        ma.metric(f"Predicted avg spend for {next_lbl}", f"€{next_p:,.2f}")
        mb.metric("Model MAE", f"€{mae_ts:,.2f}")

        fig7, ax7 = plt.subplots(figsize=(10, 3))
        t_labels = monthly_ts["Date"].dt.strftime("%b %Y").tolist() + [next_lbl]
        ax7.plot(monthly_ts["Date"].dt.strftime("%b %Y"),
                 monthly_ts["Total_Expenses"], marker="o", label="Actual", color="#4C72B0")
        ax7.plot(t_labels, list(t_preds)+[next_p],
                 linestyle="--", label="Trend", color="#FF7F0E")
        ax7.scatter([next_lbl], [next_p], color="red", zorder=5,
                    label=f"Prediction: €{next_p:,.0f}", s=80)
        ax7.set_ylabel("Avg Total Expenses (€)")
        plt.xticks(rotation=45, ha="right")
        ax7.legend()
        plt.tight_layout()
        st.pyplot(fig7)
        plt.close()

    st.divider()

    st.subheader(f"🔮 {user_name}'s Personalised Predictor")
    st.caption(
        "Method: Random Forest Regressor. Uses your saved income and bracket automatically. "
        "Trained on 3,655 real user records."
    )

    FEATURES = [
        "Income", "Month", "Festival_Count",
        "Food_Ratio", "Groceries_Ratio", "Transport_Ratio",
        "Entertainment_Ratio", "Shopping_Ratio", "Rent_Ratio",
        "Bills_Ratio", "Healthcare_Ratio", "Education_Ratio"
    ]
    model_df = df[FEATURES + ["Income_Bracket", "Total_Expenses"]].dropna().copy()
    le = LabelEncoder()
    model_df["Income_Bracket_enc"] = le.fit_transform(model_df["Income_Bracket"])
    X_rf = model_df[FEATURES + ["Income_Bracket_enc"]].values
    y_rf = model_df["Total_Expenses"].values

    if len(model_df) >= 20:
        X_tr, X_te, y_tr, y_te = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_tr, y_tr)
        mae_rf = mean_absolute_error(y_te, rf.predict(X_te))
        r2_rf  = r2_score(y_te, rf.predict(X_te))

        mc, md = st.columns(2)
        mc.metric("MAE (test set)", f"€{mae_rf:,.2f}")
        md.metric("R² Score",       f"{r2_rf:.3f}")

        # Use user's saved profile as defaults
        bracket_enc_user = 0
        if user_bracket in le.classes_:
            bracket_enc_user = le.transform([user_bracket])[0]

        st.write(f"**Adjust for {user_name} — income and bracket pre-filled from your profile:**")
        p1, p2, p3 = st.columns(3)
        income_in = p1.slider("Monthly Income (€)", 500, 10000,
                              int(min(max(user_income, 500), 10000)), step=100)
        month_in  = p2.slider("Month", 1, 12, date.today().month)
        fest_in   = p3.slider("Festival Count", 0, 5, 1)

        bracket_options = list(le.classes_)
        default_idx     = bracket_options.index(user_bracket) \
                          if user_bracket in bracket_options else 0
        bracket_in  = st.selectbox("Income Bracket", bracket_options, index=default_idx)
        bracket_enc = le.transform([bracket_in])[0]

        default_ratios = [0.20, 0.12, 0.10, 0.08, 0.10, 0.25, 0.07, 0.04, 0.04]
        input_row = np.array([[income_in, month_in, fest_in, *default_ratios, bracket_enc]])
        rf_pred   = rf.predict(input_row)[0]
        st.success(f"💡 {user_name}'s predicted total expenses: **€{rf_pred:,.2f}**")

        # Feature importance
        st.write("**What drives expenses most?**")
        feat_names = FEATURES + ["Income_Bracket"]
        importance = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)
        fig8, ax8  = plt.subplots(figsize=(8, 4))
        ax8.barh(importance.index[::-1], importance.values[::-1], color="#4C72B0")
        ax8.set_xlabel("Importance score")
        plt.tight_layout()
        st.pyplot(fig8)
        plt.close()

    with st.expander("🗂️ View raw database records"):
        st.dataframe(df[["UserID","Date","Income","Income_Bracket",
                          "Total_Expenses","Savings"]+SPEND_COLS].head(200))