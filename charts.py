"""
charts.py — All chart/visualisation functions for ValtoSpend
Built with Plotly for an interactive, dark-themed experience that
blends with the application background. Hover over any chart to
see exact values.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

SPEND_COLS = [
    "Food", "Groceries", "Transport", "Entertainment",
    "Shopping", "Rent", "Bills", "Healthcare", "Education"
]

# ── Shared dark theme ────────────────────────────────────────────────────────
ACCENT      = "#00C9A7"
ACCENT_2    = "#00E676"
PALETTE     = ["#00C9A7", "#3FA7D6", "#9D7DD8", "#E8956D",
               "#E85D75", "#6FCF97", "#F2C14E", "#7B8FA1", "#C77DFF"]

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Arial, sans-serif", color="#cccccc", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    hoverlabel=dict(bgcolor="#111111", font_color="#ffffff",
                    bordercolor=ACCENT),
)


def _style(fig, showlegend=True):
    fig.update_layout(**LAYOUT_BASE, showlegend=showlegend,
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#cccccc")))
    fig.update_xaxes(showgrid=False, color="#888888", linecolor="#222222")
    fig.update_yaxes(showgrid=True, gridcolor="#161616", color="#888888", zeroline=False)
    return fig


def bar_chart_categories(filtered_df, rate=1.0, symbol="EUR"):
    """Interactive bar chart of average spend per category."""
    avg_cats = (filtered_df[SPEND_COLS].mean() * rate).sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=avg_cats.index, y=avg_cats.values,
        marker=dict(color=ACCENT, line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>" + symbol + " %{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(yaxis_title=f"Average ({symbol})")
    _style(fig, showlegend=False)
    return fig, avg_cats


def pie_chart_categories(avg_cats):
    """Interactive donut chart of category spending distribution."""
    fig = go.Figure(go.Pie(
        labels=avg_cats.index, values=avg_cats.values, hole=0.55,
        marker=dict(colors=PALETTE, line=dict(color="#000000", width=2)),
        textfont=dict(color="#ffffff", size=12),
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
    ))
    _style(fig)
    return fig


def bracket_comparison_chart(df, rate=1.0, symbol="EUR"):
    """Grouped bar chart comparing income, expenses, savings by bracket."""
    b = df.groupby("Income_Bracket")[["Income", "Total_Expenses", "Savings"]] \
          .mean().reset_index().sort_values("Income")
    b[["Income", "Total_Expenses", "Savings"]] *= rate

    fig = go.Figure()
    for col, color, label in [("Income", ACCENT, "Income"),
                               ("Total_Expenses", "#E8956D", "Expenses"),
                               ("Savings", ACCENT_2, "Savings")]:
        fig.add_bar(x=b["Income_Bracket"], y=b[col], name=label,
                   marker_color=color,
                   hovertemplate=f"<b>{label}</b><br>%{{x}}<br>{symbol} %{{y:,.2f}}<extra></extra>")
    fig.update_layout(barmode="group", yaxis_title=f"Average ({symbol})")
    _style(fig)
    return fig


def monthly_trend_chart(filtered_df, rate=1.0, symbol="EUR"):
    """Line chart of average monthly total expenses over time."""
    m = filtered_df.groupby("Date")["Total_Expenses"].mean().reset_index().sort_values("Date")
    m["Total_Expenses"] *= rate
    m["Label"] = m["Date"].dt.strftime("%b %Y")

    fig = go.Figure(go.Scatter(
        x=m["Label"], y=m["Total_Expenses"], mode="lines+markers",
        line=dict(color=ACCENT, width=2.5, shape="spline"),
        marker=dict(color=ACCENT, size=6),
        fill="tozeroy", fillcolor="rgba(0,201,167,0.08)",
        hovertemplate="<b>%{x}</b><br>" + symbol + " %{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(yaxis_title=f"Average Total Expenses ({symbol})")
    _style(fig, showlegend=False)
    return fig


def personal_category_chart(my_df, symbol="EUR"):
    """Bar chart of personal spending by category."""
    cat_totals = my_df.groupby("category")["amount"].sum().sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=cat_totals.index, y=cat_totals.values,
        marker=dict(color=ACCENT),
        hovertemplate="<b>%{x}</b><br>" + symbol + " %{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(yaxis_title=f"Total ({symbol})")
    _style(fig, showlegend=False)
    return fig


def personal_weekly_chart(my_df, symbol="EUR"):
    """Line chart of personal spending over time (weekly)."""
    my_df = my_df.copy()
    my_df["week"] = my_df["date"].dt.to_period("W").astype(str)
    weekly = my_df.groupby("week")["amount"].sum().reset_index()
    fig = go.Figure(go.Scatter(
        x=weekly["week"], y=weekly["amount"], mode="lines+markers",
        line=dict(color="#E8956D", width=2.5, shape="spline"),
        marker=dict(color="#E8956D", size=6),
        fill="tozeroy", fillcolor="rgba(232,149,109,0.08)",
        hovertemplate="<b>%{x}</b><br>" + symbol + " %{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(yaxis_title=f"Total ({symbol})")
    _style(fig, showlegend=False)
    return fig


def forecast_chart(monthly_ts, t_preds, next_p, next_lbl, symbol="EUR"):
    """Line chart showing actual trend + predicted next month."""
    labels = monthly_ts["Date"].dt.strftime("%b %Y").tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=monthly_ts["Total_Expenses"], mode="lines+markers",
        name="Actual", line=dict(color=ACCENT, width=2.5),
        marker=dict(size=6),
        hovertemplate="<b>%{x}</b><br>" + symbol + " %{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels + [next_lbl], y=list(t_preds) + [next_p], mode="lines",
        name="Trend", line=dict(color="#888888", width=1.5, dash="dash"),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[next_lbl], y=[next_p], mode="markers", name="Prediction",
        marker=dict(color="#E85D75", size=12, symbol="diamond"),
        hovertemplate=f"<b>Predicted — {next_lbl}</b><br>{symbol} %{{y:,.2f}}<extra></extra>",
    ))
    fig.update_layout(yaxis_title=f"Average Total Expenses ({symbol})")
    _style(fig)
    return fig


def feature_importance_chart(rf, feat_names):
    """Horizontal bar chart of Random Forest feature importances."""
    importance = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        x=importance.values, y=importance.index, orientation="h",
        marker=dict(color=ACCENT),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Importance score", height=380)
    _style(fig, showlegend=False)
    return fig


def nn_loss_chart(losses):
    """Line chart showing neural network training loss over epochs."""
    fig = go.Figure(go.Scatter(
        x=list(range(len(losses))), y=losses, mode="lines",
        line=dict(color=ACCENT, width=2.5),
        fill="tozeroy", fillcolor="rgba(0,201,167,0.08)",
        hovertemplate="Epoch %{x}<br>Loss: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Training Loss (MAE)")
    _style(fig, showlegend=False)
    return fig


def community_category_chart(category_totals, symbol="EUR"):
    """Bar chart of community-wide spending by category."""
    cat_df = pd.DataFrame(list(category_totals.items()), columns=["Category", "Total"]) \
              .sort_values("Total", ascending=False)
    fig = go.Figure(go.Bar(
        x=cat_df["Category"], y=cat_df["Total"],
        marker=dict(color=ACCENT),
        hovertemplate="<b>%{x}</b><br>" + symbol + " %{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(yaxis_title=f"Total ({symbol})")
    _style(fig, showlegend=False)
    return figS