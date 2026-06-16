"""
database.py — All database operations for ValtoSpend
Handles SQLite setup, data loading, and expense management.
"""
import sqlite3
import os
import pandas as pd

DB_PATH = "valtospend.db"
CSV_PATH = "expenses.csv"


def init_db():
    """Create tables and load CSV into SQLite on first run."""
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
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   INTEGER DEFAULT 0,
            date      TEXT,
            category  TEXT,
            amount    REAL,
            note      TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS budgets (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   INTEGER NOT NULL,
            category  TEXT NOT NULL,
            amount    REAL NOT NULL,
            UNIQUE(user_id, category)
        )
    """)
    count = cursor.execute("SELECT COUNT(*) FROM expenses").fetchone()[0]
    if count == 0 and os.path.exists(CSV_PATH):
        df_csv = pd.read_csv(CSV_PATH)
        df_csv.to_sql("expenses", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()


def load_data():
    """Load main dataset from SQLite and add Date column."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM expenses", conn)
    conn.close()
    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2) + "-01"
    )
    return df


def load_my_expenses(user_id: int):
    """Load personal expenses for the logged-in user only."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT * FROM my_expenses WHERE user_id=? ORDER BY date DESC",
        conn, params=(user_id,)
    )
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def add_my_expense(user_id: int, date_val, category, amount, note):
    """Insert a new personal expense for the logged-in user."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO my_expenses (user_id, date, category, amount, note) VALUES (?,?,?,?,?)",
        (user_id, str(date_val), category, float(amount), note)
    )
    conn.commit()
    conn.close()


def delete_my_expense(row_id: int):
    """Delete a personal expense by ID."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM my_expenses WHERE id=?", (row_id,))
    conn.commit()
    conn.close()


def get_budgets(user_id: int) -> dict:
    """Get all budget limits for a user as {category: amount}."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT category, amount FROM budgets WHERE user_id=?", (user_id,)
    ).fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


def set_budget(user_id: int, category: str, amount: float):
    """Set or update a budget limit for a category."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO budgets (user_id, category, amount) VALUES (?,?,?) "
        "ON CONFLICT(user_id, category) DO UPDATE SET amount=excluded.amount",
        (user_id, category, amount)
    )
    conn.commit()
    conn.close()


def export_expenses_csv(user_id: int) -> str:
    """Return personal expenses as CSV string for download."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT date, category, amount, note FROM my_expenses WHERE user_id=? ORDER BY date DESC",
        conn, params=(user_id,)
    )
    conn.close()
    return df.to_csv(index=False)
