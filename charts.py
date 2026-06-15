"""
charts.py — All chart/visualisation functions for ValtoSpend
Each function returns a matplotlib figure ready for st.pyplot().
"""
import matplotlib.pyplot as plt
import numpy as np

SPEND_COLS = [
    "Food", "Groceries", "Transport", "Entertainment",
    "Shopping", "Rent", "Bills", "Healthcare", "Education"
]


def bar_chart_categories(filtered_df):
    """Bar chart of average spend per category."""
    avg_cats = filtered_df[SPEND_COLS].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(avg_cats.index, avg_cats.values, color="#4C72B0")
    ax.set_ylabel("Average (€)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig, avg_cats


def pie_chart_categories(avg_cats):
    """Pie chart of category spending distribution."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(avg_cats.values, labels=avg_cats.index, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    plt.tight_layout()
    return fig


def bracket_comparison_chart(df):
    """Grouped bar chart comparing income, expenses, savings by bracket."""
    b_summary = df.groupby("Income_Bracket")[
        ["Income", "Total_Expenses", "Savings"]
    ].mean().reset_index().sort_values("Income")

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(b_summary))
    w = 0.25
    ax.bar(x - w, b_summary["Income"],         w, label="Income",         color="#4C72B0")
    ax.bar(x,     b_summary["Total_Expenses"], w, label="Total Expenses", color="#DD8452")
    ax.bar(x + w, b_summary["Savings"],        w, label="Savings",        color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(b_summary["Income_Bracket"], rotation=15)
    ax.set_ylabel("Average (€)")
    ax.legend()
    plt.tight_layout()
    return fig


def monthly_trend_chart(filtered_df):
    """Line chart of average monthly total expenses over time."""
    m_trend = (
        filtered_df.groupby("Date")["Total_Expenses"]
        .mean().reset_index().sort_values("Date")
    )
    m_trend["Label"] = m_trend["Date"].dt.strftime("%b %Y")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(m_trend["Label"], m_trend["Total_Expenses"], marker="o", color="#4C72B0")
    ax.set_ylabel("Avg Total Expenses (€)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def personal_category_chart(my_df):
    """Bar chart of personal spending by category."""
    cat_totals = my_df.groupby("category")["amount"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(cat_totals.index, cat_totals.values, color="#4C72B0")
    ax.set_ylabel("Total (€)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def personal_weekly_chart(my_df):
    """Line chart of personal spending over time (weekly)."""
    my_df["week"] = my_df["date"].dt.to_period("W").astype(str)
    weekly = my_df.groupby("week")["amount"].sum()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(weekly.index, weekly.values, marker="o", color="#DD8452")
    ax.set_ylabel("Total (€)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def forecast_chart(monthly_ts, t_preds, next_p, next_lbl):
    """Line chart showing actual trend + predicted next month."""
    fig, ax = plt.subplots(figsize=(10, 3))
    t_labels = monthly_ts["Date"].dt.strftime("%b %Y").tolist() + [next_lbl]
    ax.plot(
        monthly_ts["Date"].dt.strftime("%b %Y"),
        monthly_ts["Total_Expenses"],
        marker="o", label="Actual", color="#4C72B0"
    )
    ax.plot(t_labels, list(t_preds) + [next_p],
            linestyle="--", label="Trend", color="#FF7F0E")
    ax.scatter([next_lbl], [next_p], color="red", zorder=5,
               label=f"Prediction: €{next_p:,.0f}", s=80)
    ax.set_ylabel("Avg Total Expenses (€)")
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    return fig


def feature_importance_chart(rf, feat_names):
    """Horizontal bar chart of Random Forest feature importances."""
    import pandas as pd
    importance = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(importance.index[::-1], importance.values[::-1], color="#4C72B0")
    ax.set_xlabel("Importance score")
    plt.tight_layout()
    return fig


def nn_loss_chart(losses):
    """Line chart showing neural network training loss over epochs."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(range(len(losses)), losses, color="#4C72B0", label="Training MAE loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE Loss")
    ax.legend()
    plt.tight_layout()
    return fig
