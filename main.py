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

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="ValtoSpend", page_icon="💰", layout="wide")
st.title("💰 ValtoSpend — Smart Expense Analyser")

SPEND_COLS = [
    "Food", "Groceries", "Transport", "Entertainment",
    "Shopping", "Rent", "Bills", "Healthcare", "Education"
]

# ============================================================
# PHASE 2: DATABASE — load CSV into SQLite
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
    count = cursor.execute("SELECT COUNT(*) FROM expenses").fetchone()[0]
    if count == 0 and os.path.exists(CSV_PATH):
        df_csv = pd.read_csv(CSV_PATH)
        df_csv.to_sql("expenses", conn, if_exists="append", index=False)
        st.sidebar.success(f"✅ Loaded {len(df_csv)} rows into SQLite database.")
    conn.commit()
    conn.close()

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM expenses", conn)
    conn.close()
    # Build a proper date from Year + Month
    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2) + "-01"
    )
    return df

init_db()
df = load_data()

# ============================================================
# SIDEBAR — DATE RANGE + FILTERS
# ============================================================
st.sidebar.header("🔍 Filters")

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start, end = date_range
    filtered = df[(df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)].copy()
else:
    filtered = df.copy()

brackets = ["All"] + sorted(df["Income_Bracket"].dropna().unique().tolist())
sel_bracket = st.sidebar.selectbox("Income Bracket", brackets)
if sel_bracket != "All":
    filtered = filtered[filtered["Income_Bracket"] == sel_bracket]

# ============================================================
# SECTION 1: OVERVIEW METRICS
# ============================================================
st.header("📊 Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records",       f"{len(filtered):,}")
c2.metric("Avg Monthly Spend",   f"€{filtered['Total_Expenses'].mean():,.0f}")
c3.metric("Avg Monthly Income",  f"€{filtered['Income'].mean():,.0f}")
c4.metric("Avg Monthly Savings", f"€{filtered['Savings'].mean():,.0f}")

# ============================================================
# SECTION 2: SPENDING BY CATEGORY
# ============================================================
st.header("📈 Spending by Category")
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("Average spend per category")
    avg_cats = filtered[SPEND_COLS].mean().sort_values(ascending=False)
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(avg_cats.index, avg_cats.values, color="#4C72B0")
    ax1.set_ylabel("Average (€)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

with col_r:
    st.subheader("Category distribution (pie)")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.pie(avg_cats.values, labels=avg_cats.index, autopct="%1.1f%%", startangle=90)
    ax2.axis("equal")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ============================================================
# SECTION 3: INCOME BRACKET COMPARISON
# ============================================================
st.header("💼 Income Bracket Analysis")
bracket_summary = df.groupby("Income_Bracket")[
    ["Income", "Total_Expenses", "Savings"]
].mean().reset_index().sort_values("Income")

fig3, ax3 = plt.subplots(figsize=(8, 4))
x = np.arange(len(bracket_summary))
w = 0.25
ax3.bar(x - w, bracket_summary["Income"],         w, label="Income",         color="#4C72B0")
ax3.bar(x,      bracket_summary["Total_Expenses"], w, label="Total Expenses", color="#DD8452")
ax3.bar(x + w,  bracket_summary["Savings"],        w, label="Savings",        color="#55A868")
ax3.set_xticks(x)
ax3.set_xticklabels(bracket_summary["Income_Bracket"], rotation=15)
ax3.set_ylabel("Average (€)")
ax3.legend()
plt.tight_layout()
st.pyplot(fig3)
plt.close()

# ============================================================
# SECTION 4: MONTHLY TREND WITH DATE FILTER
# ============================================================
st.header("📅 Monthly Spending Trend")
monthly_trend = filtered.groupby("Date")["Total_Expenses"].mean().reset_index()
monthly_trend = monthly_trend.sort_values("Date")
monthly_trend["Label"] = monthly_trend["Date"].dt.strftime("%b %Y")

fig4, ax4 = plt.subplots(figsize=(10, 3))
ax4.plot(monthly_trend["Label"], monthly_trend["Total_Expenses"],
         marker="o", color="#4C72B0")
ax4.set_ylabel("Avg Total Expenses (€)")
ax4.set_xlabel("Month")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig4)
plt.close()

# ============================================================
# SECTION 5: FESTIVAL IMPACT
# ============================================================
if "Festivals" in df.columns:
    st.header("🎉 Festival Impact on Spending")
    fest_summary = (
        filtered.groupby("Festivals")["Total_Expenses"]
        .mean()
        .reset_index()
        .rename(columns={"Festivals": "Festival", "Total_Expenses": "Avg Expenses"})
        .sort_values("Avg Expenses", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(fest_summary)

# ============================================================
# PHASE 3: AI MODEL — Predict NEXT MONTH's expenses
# Uses monthly average time series + Linear Regression trend
# PLUS a Random Forest for feature-based prediction
# ============================================================
st.header("🤖 AI: Predict Next Month's Expenses")

# --- Part A: Time-series next-month prediction ---
st.subheader("📆 Next Month Forecast (time series)")
st.caption(
    "Method: Linear Regression on the monthly average total expenses time series. "
    "Each point = average spend across all users for that month. "
    "The model extrapolates the trend to predict the next calendar month."
)

monthly_ts = df.groupby("Date")["Total_Expenses"].mean().reset_index().sort_values("Date")
monthly_ts["month_index"] = range(len(monthly_ts))

if len(monthly_ts) >= 3:
    X_ts = monthly_ts[["month_index"]].values
    y_ts = monthly_ts["Total_Expenses"].values

    lr = LinearRegression()
    lr.fit(X_ts, y_ts)

    next_idx = np.array([[len(monthly_ts)]])
    predicted_next = lr.predict(next_idx)[0]
    trend_preds    = lr.predict(X_ts)
    mae_ts = mean_absolute_error(y_ts, trend_preds)

    next_month_label = (monthly_ts["Date"].max() + pd.DateOffset(months=1)).strftime("%B %Y")

    m1, m2 = st.columns(2)
    m1.metric(f"Predicted spend for {next_month_label}", f"€{predicted_next:,.2f}")
    m2.metric("Model MAE", f"€{mae_ts:,.2f}", help="Average error on training months")

    fig5, ax5 = plt.subplots(figsize=(10, 3))
    labels = monthly_ts["Date"].dt.strftime("%b %Y").tolist() + [next_month_label]
    actuals    = list(y_ts) + [None]
    trend_line = list(trend_preds) + [predicted_next]
    ax5.plot(labels[:-1], y_ts, marker="o", label="Actual avg spend", color="#4C72B0")
    ax5.plot(labels,      trend_line, linestyle="--", label="Trend line", color="#FF7F0E")
    ax5.scatter([next_month_label], [predicted_next], color="red", zorder=5,
                label=f"Prediction: €{predicted_next:,.0f}", s=80)
    ax5.set_ylabel("Avg Total Expenses (€)")
    plt.xticks(rotation=45, ha="right")
    ax5.legend()
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()
else:
    st.info("Need at least 3 months of data for the time-series prediction.")

# --- Part B: Feature-based prediction (Random Forest) ---
st.subheader("🔮 Predict Your Own Expenses")
st.caption(
    "Method: Random Forest Regressor. "
    "Features: Income, Month, Festival_Count, Income Bracket, and all category ratios. "
    "Target: Total_Expenses. Evaluated on a held-out 20% test set."
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
    X_train, X_test, y_train, y_test = train_test_split(
        X_rf, y_rf, test_size=0.2, random_state=42
    )
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf  = r2_score(y_test, y_pred_rf)

    ma, mb = st.columns(2)
    ma.metric("MAE (test set)", f"€{mae_rf:,.2f}")
    mb.metric("R² Score",       f"{r2_rf:.3f}", help="1.0 = perfect. Above 0.85 is very good.")

    # Feature importance
    st.write("**What drives your expenses most?**")
    feat_names  = FEATURES + ["Income_Bracket"]
    importance  = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)
    fig6, ax6 = plt.subplots(figsize=(8, 4))
    ax6.barh(importance.index[::-1], importance.values[::-1], color="#4C72B0")
    ax6.set_xlabel("Importance score")
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()

    # Interactive sliders
    st.write("**Adjust your profile to get a personalised prediction:**")
    p1, p2, p3 = st.columns(3)
    income_in = p1.slider("Monthly Income (€)", 500,  10000, 3000, step=100)
    month_in  = p2.slider("Month",              1,    12,    6)
    fest_in   = p3.slider("Festival Count",     0,    5,     1)
    bracket_in    = st.selectbox("Income Bracket", le.classes_)
    bracket_enc   = le.transform([bracket_in])[0]

    default_ratios = [0.20, 0.12, 0.10, 0.08, 0.10, 0.25, 0.07, 0.04, 0.04]
    input_row = np.array([[income_in, month_in, fest_in,
                           *default_ratios, bracket_enc]])
    rf_prediction = rf.predict(input_row)[0]
    st.success(f"💡 Predicted Total Monthly Expenses: **€{rf_prediction:,.2f}**")

else:
    st.info("Not enough data rows to train the Random Forest model.")

# ============================================================
# SECTION 6: RAW DATA
# ============================================================
with st.expander("🗂️ View raw database records"):
    display_cols = ["UserID", "Date", "Income", "Income_Bracket",
                    "Total_Expenses", "Savings"] + SPEND_COLS
    st.dataframe(filtered[display_cols].head(200).reset_index(drop=True))