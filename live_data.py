"""
live_data.py — Live data features for ValtoSpend
Option 1: Live currency exchange rates from ExchangeRate-API
Option 2: Community stats from registered users own expenses
"""
import sqlite3
import urllib.request
import json
import pandas as pd
from datetime import datetime

DB_PATH = "valtospend.db"


# ── OPTION 1: Live Currency Rates ──────────────────────────────────────────
def get_live_rates(api_key: str, base: str = "EUR") -> dict:
    """
    Fetch live exchange rates from ExchangeRate-API.
    Returns dict of {currency_code: rate} or {} on error.
    """
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{base}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("result") == "success":
                return {
                    "rates":      data["conversion_rates"],
                    "base":       data["base_code"],
                    "updated":    data["time_last_update_utc"],
                    "next_update": data["time_next_update_utc"]
                }
    except Exception as e:
        return {"error": str(e)}
    return {}


def convert_amount(amount: float, from_currency: str,
                   to_currency: str, rates: dict) -> float:
    """Convert amount between currencies using live rates."""
    if not rates or "rates" not in rates:
        return amount
    r = rates["rates"]
    if from_currency not in r or to_currency not in r:
        return amount
    in_base  = amount / r[from_currency]
    return in_base * r[to_currency]


# ── OPTION 2: Community Stats ───────────────────────────────────────────────
def get_community_stats() -> dict:
    """
    Calculate real-time community statistics from all users' expenses.
    Returns dict with total_users, total_expenses, avg_expense,
    top_category, total_logged, this_month_count.
    """
    conn = sqlite3.connect(DB_PATH)

    # Total registered users
    total_users = conn.execute(
        "SELECT COUNT(*) FROM users"
    ).fetchone()[0]

    # Total expenses logged
    total_logged = conn.execute(
        "SELECT COUNT(*) FROM my_expenses"
    ).fetchone()[0]

    # Total amount spent across all users
    total_amount = conn.execute(
        "SELECT SUM(amount) FROM my_expenses"
    ).fetchone()[0] or 0

    # Average expense per transaction
    avg_expense = conn.execute(
        "SELECT AVG(amount) FROM my_expenses"
    ).fetchone()[0] or 0

    # Top category across all users
    top_cat_row = conn.execute(
        "SELECT category, SUM(amount) as total FROM my_expenses "
        "GROUP BY category ORDER BY total DESC LIMIT 1"
    ).fetchone()
    top_category = top_cat_row[0] if top_cat_row else "N/A"

    # This month's expense count
    this_month = datetime.now().strftime("%Y-%m")
    this_month_count = conn.execute(
        "SELECT COUNT(*) FROM my_expenses WHERE date LIKE ?",
        (f"{this_month}%",)
    ).fetchone()[0]

    # Category breakdown across all users
    cat_rows = conn.execute(
        "SELECT category, SUM(amount) as total FROM my_expenses "
        "GROUP BY category ORDER BY total DESC"
    ).fetchall()
    category_totals = {row[0]: row[1] for row in cat_rows}

    # Most active user (most expenses logged)
    active_row = conn.execute(
        "SELECT u.username, COUNT(e.id) as cnt "
        "FROM my_expenses e JOIN users u ON e.user_id = u.id "
        "GROUP BY e.user_id ORDER BY cnt DESC LIMIT 1"
    ).fetchone()
    most_active = active_row[0] if active_row else "N/A"

    conn.close()

    return {
        "total_users":     total_users,
        "total_logged":    total_logged,
        "total_amount":    total_amount,
        "avg_expense":     avg_expense,
        "top_category":    top_category,
        "this_month":      this_month_count,
        "category_totals": category_totals,
        "most_active":     most_active
    }
