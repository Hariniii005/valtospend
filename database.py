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
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT, category TEXT,
            amount REAL, note TEXT
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


def load_my_expenses():
    """Load personal expenses for the current user."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM my_expenses ORDER BY date DESC", conn)
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def add_my_expense(date_val, category, amount, note):
    """Insert a new personal expense into the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO my_expenses (date, category, amount, note) VALUES (?, ?, ?, ?)",
        (str(date_val), category, float(amount), note)
    )
    conn.commit()
    conn.close()


def delete_my_expense(row_id):
    """Delete a personal expense by ID."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM my_expenses WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()
