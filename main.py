"""
main.py — ValtoSpend: AI-Powered Personal Expense Tracker
Entry point for the Streamlit application.
Imports from: database.py, ai_models.py, charts.py, receipt.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from sklearn.linear_model import LinearRegression

from database import init_db, load_data, load_my_expenses, add_my_expense, delete_my_expense
from ai_models import linear_regression_forecast, random_forest_model, neural_network_scratch, FEATURES
from charts import (bar_chart_categories, pie_chart_categories, bracket_comparison_chart,
                    monthly_trend_chart, personal_category_chart, personal_weekly_chart,
                    forecast_chart, feature_importance_chart, nn_loss_chart)
from receipt import read_receipt_with_ai

SPEND_COLS = ["Food","Groceries","Transport","Entertainment",
              "Shopping","Rent","Bills","Healthcare","Education"]
BRACKETS   = ["Low Income","Lower Middle Income","Middle Income",
              "Upper Middle Income","High Income"]

st.set_page_config(page_title="ValtoSpend", page_icon="💰", layout="wide")
init_db()
df = load_data()

# ── Profile (session state — private per browser) ──────────────────────────
def get_profile():
    if "user_name" in st.session_state:
        return (st.session_state["user_name"],
                st.session_state["user_income"],
                st.session_state["user_bracket"])
    return None

def save_profile(name, income, bracket):
    st.session_state["user_name"]    = name
    st.session_state["user_income"]  = float(income)
    st.session_state["user_bracket"] = bracket

profile = get_profile()

# ── First-visit profile setup ───────────────────────────────────────────────
if profile is None:
    st.title("💰 Welcome to ValtoSpend")
    st.subheader("Let's set up your profile first!")
    with st.form("profile_form"):
        p_name    = st.text_input("Your first name")
        p_income  = st.number_input("Your monthly income (€)", min_value=0.0, step=100.0, value=2000.0)
        p_bracket = st.selectbox("Income bracket", BRACKETS)
        if st.form_submit_button("🚀 Start ValtoSpend", type="primary"):
            if p_name.strip():
                save_profile(p_name.strip(), p_income, p_bracket)
                st.rerun()
            else:
                st.error("Please enter your name.")
    st.stop()

user_name, user_income, user_bracket = profile

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.title(f"💰 ValtoSpend — Welcome back, {user_name}! 👋")

with st.sidebar:
    st.markdown(f"**👤 {user_name}**")
    st.caption(f"Income: €{user_income:,.0f} | {user_bracket}")
    if st.button("✏️ Edit Profile"):
        st.session_state["editing_profile"] = True
    st.header("🔍 Filters")
    min_date   = df["Date"].min().date()
    max_date   = df["Date"].max().date()
    date_range = st.date_input("Date range", value=(min_date, max_date),
                               min_value=min_date, max_value=max_date)
    brackets_list = ["All"] + sorted(df["Income_Bracket"].dropna().unique().tolist())
    sel_bracket   = st.selectbox("Income Bracket", brackets_list)

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

# Apply filters
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start, end = date_range
    filtered = df[(df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)].copy()
else:
    filtered = df.copy()
if sel_bracket != "All":
    filtered = filtered[filtered["Income_Bracket"] == sel_bracket]

tab1, tab2, tab3 = st.tabs(["👤 My Expenses", "📊 Market Insights", "🤖 AI Prediction"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — MY EXPENSES
# ══════════════════════════════════════════════════════════════
with tab1:
    st.header(f"👤 {user_name}'s Expenses")

    # Receipt scanner
    st.subheader("📸 Scan a Receipt")
    st.caption("Upload a receipt photo — AI reads it and fills in the details automatically.")
    uploaded_file = st.file_uploader("Upload", type=["jpg","jpeg","png"],
                                     label_visibility="collapsed")

    for key, val in [("prefill_amount",10.0),("prefill_category","Food"),
                     ("prefill_note",""),("scan_done",False)]:
        if key not in st.session_state:
            st.session_state[key] = val

    if uploaded_file and not st.session_state.scan_done:
        with st.spinner("🤖 Reading your receipt..."):
            img_bytes  = uploaded_file.read()
            media_type = "image/png" if uploaded_file.name.endswith(".png") else "image/jpeg"
            result     = read_receipt_with_ai(img_bytes, media_type)
        if "error" in result:
            st.warning(f"Couldn't read receipt: {result['error']}. Please fill in manually.")
        else:
            st.success(f"✅ Found: **€{result.get('amount',0):.2f}** — {result.get('note','')}")
            st.session_state.prefill_amount   = float(result.get("amount", 10.0))
            st.session_state.prefill_category = result.get("category", "Food")
            st.session_state.prefill_note     = result.get("note", "")
            st.session_state.scan_done        = True
            st.image(img_bytes, caption="Uploaded receipt", width=200)
    if not uploaded_file:
        st.session_state.scan_done = False

    # Add expense
    st.subheader("➕ Add Expense")
    c1, c2, c3, c4 = st.columns(4)
    exp_date = c1.date_input("Date", value=date.today())
    cat_idx  = SPEND_COLS.index(st.session_state.prefill_category) \
               if st.session_state.prefill_category in SPEND_COLS else 0
    exp_cat  = c2.selectbox("Category", SPEND_COLS, index=cat_idx)
    exp_amt  = c3.number_input("Amount (€)", min_value=0.01, step=0.50,
                                value=float(st.session_state.prefill_amount))
    exp_note = c4.text_input("Note", value=st.session_state.prefill_note,
                              placeholder="e.g. Lunch at work")

    if st.button("💾 Save Expense", type="primary"):
        add_my_expense(exp_date, exp_cat, exp_amt, exp_note)
        st.success(f"✅ Saved: €{exp_amt:.2f} for {exp_cat}")
        for k, v in [("prefill_amount",10.0),("prefill_category","Food"),
                     ("prefill_note",""),("scan_done",False)]:
            st.session_state[k] = v
        st.rerun()

    st.divider()
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
            st.pyplot(personal_category_chart(my_df))
        with col_b:
            st.subheader("My Spending Over Time")
            st.pyplot(personal_weekly_chart(my_df))

        st.subheader("🤖 My Next Month Prediction")
        my_df["month"] = my_df["date"].dt.to_period("M").astype(str)
        monthly_me = my_df.groupby("month")["amount"].sum().reset_index()
        monthly_me["idx"] = range(len(monthly_me))
        if len(monthly_me) >= 2:
            lr_me = LinearRegression()
            lr_me.fit(monthly_me[["idx"]], monthly_me["amount"])
            next_pred_me  = lr_me.predict([[len(monthly_me)]])[0]
            next_mo_label = (my_df["date"].max() + pd.DateOffset(months=1)).strftime("%B %Y")
            st.success(f"💡 {user_name}, predicted spend for **{next_mo_label}**: **€{next_pred_me:,.2f}**")
        else:
            st.info("Add expenses across at least 2 months to unlock your personal prediction.")

        st.subheader("📋 All My Expenses")
        for _, row in my_df.iterrows():
            c1, c2, c3, c4, c5 = st.columns([2,2,1,3,1])
            c1.write(row["date"].strftime("%d %b %Y"))
            c2.write(row["category"])
            c3.write(f"€{row['amount']:.2f}")
            c4.write(row["note"] if row["note"] else "—")
            if c5.button("🗑️", key=f"del_{row['id']}"):
                delete_my_expense(row["id"])
                st.rerun()

# ══════════════════════════════════════════════════════════════
# TAB 2 — MARKET INSIGHTS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.header("📊 Market Insights")
    st.caption("Analysis based on 3,655 real user records (2021–2024).")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",       f"{len(filtered):,}")
    c2.metric("Avg Monthly Spend",   f"€{filtered['Total_Expenses'].mean():,.0f}")
    c3.metric("Avg Monthly Income",  f"€{filtered['Income'].mean():,.0f}")
    c4.metric("Avg Monthly Savings", f"€{filtered['Savings'].mean():,.0f}")

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

# ══════════════════════════════════════════════════════════════
# TAB 3 — AI PREDICTION
# ══════════════════════════════════════════════════════════════
with tab3:
    st.header("🤖 AI Prediction Engine")

    # Model 1 — Linear Regression
    st.subheader("📆 Market Next Month Forecast")
    st.caption("Linear Regression on monthly average spending trend.")
    next_p, mae_ts, t_preds, next_lbl, monthly_ts = linear_regression_forecast(df)
    ma, mb = st.columns(2)
    ma.metric(f"Predicted avg spend for {next_lbl}", f"€{next_p:,.2f}")
    mb.metric("Model MAE", f"€{mae_ts:,.2f}")
    st.pyplot(forecast_chart(monthly_ts, t_preds, next_p, next_lbl))

    st.divider()

    # Model 2 — Random Forest
    st.subheader(f"🔮 {user_name}'s Personalised Predictor")
    st.caption("Random Forest Regressor — 10 decision trees, evaluated on 20% test set.")
    rf, le, mae_rf, r2_rf = random_forest_model(df)
    mc, md = st.columns(2)
    mc.metric("MAE (test set)", f"€{mae_rf:,.2f}")
    md.metric("R² Score",       f"{r2_rf:.3f}")
    st.pyplot(feature_importance_chart(rf, FEATURES + ["Income_Bracket"]))

    p1, p2, p3 = st.columns(3)
    income_in = p1.slider("Monthly Income (€)", 500, 10000,
                          int(min(max(user_income,500),10000)), step=100)
    month_in  = p2.slider("Month", 1, 12, date.today().month)
    fest_in   = p3.slider("Festival Count", 0, 5, 1)
    bracket_options = list(le.classes_)
    default_idx     = bracket_options.index(user_bracket) if user_bracket in bracket_options else 0
    bracket_in      = st.selectbox("Income Bracket", bracket_options, index=default_idx)
    bracket_enc     = le.transform([bracket_in])[0]
    default_ratios  = [0.20,0.12,0.10,0.08,0.10,0.25,0.07,0.04,0.04]
    input_row       = np.array([[income_in,month_in,fest_in,*default_ratios,bracket_enc]])
    st.success(f"💡 {user_name}'s predicted expenses: **€{rf.predict(input_row)[0]:,.2f}**")

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
    n1.metric("Neural Network MAE", f"€{mae_nn:,.2f}")
    n2.metric("Neural Network R²",  f"{r2_nn:.3f}")
    st.write("**Training loss — decreasing curve proves the network is learning:**")
    st.pyplot(nn_loss_chart(losses))

    # Comparison table
    st.write("**All 3 AI Models Compared:**")
    comparison = pd.DataFrame({
        "Model":    ["Linear Regression","Random Forest","Neural Network (scratch)"],
        "Type":     ["Statistical","Ensemble ML","Deep Learning"],
        "MAE (€)":  [f"€{mae_ts:,.2f}",f"€{mae_rf:,.2f}",f"€{mae_nn:,.2f}"],
        "R² Score": ["N/A",f"{r2_rf:.3f}",f"{r2_nn:.3f}"]
    })
    st.dataframe(comparison, hide_index=True)

    with st.expander("🗂️ View raw database records"):
        st.dataframe(df[["UserID","Date","Income","Income_Bracket",
                          "Total_Expenses","Savings"]+SPEND_COLS].head(200))