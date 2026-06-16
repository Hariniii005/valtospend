"""
auth.py — Authentication system for ValtoSpend
Handles user registration, login, and password hashing.
Passwords are hashed using SHA-256 — never stored in plain text.
"""
import sqlite3
import hashlib
import os

DB_PATH = "valtospend.db"


def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with a salt."""
    salt = "valtospend_salt_2024"
    return hashlib.sha256((password + salt).encode()).hexdigest()


def init_users_table():
    """Create users table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            income   REAL DEFAULT 2000,
            bracket  TEXT DEFAULT 'Middle Income',
            currency TEXT DEFAULT 'EUR'
        )
    """)
    conn.commit()
    conn.close()


def register_user(username: str, password: str, income: float,
                  bracket: str, currency: str) -> tuple:
    """
    Register a new user.
    Returns (True, 'success') or (False, 'error message').
    """
    if len(username.strip()) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO users (username, password, income, bracket, currency) VALUES (?,?,?,?,?)",
            (username.strip().lower(), hash_password(password), income, bracket, currency)
        )
        conn.commit()
        conn.close()
        return True, "success"
    except sqlite3.IntegrityError:
        return False, "Username already exists. Please choose another."


def login_user(username: str, password: str) -> tuple:
    """
    Verify login credentials.
    Returns (True, user_row) or (False, 'error message').
    """
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT id, username, income, bracket, currency FROM users WHERE username=? AND password=?",
        (username.strip().lower(), hash_password(password))
    ).fetchone()
    conn.close()
    if row:
        return True, row  # (id, username, income, bracket, currency)
    return False, "Incorrect username or password."


def update_profile(user_id: int, income: float, bracket: str, currency: str):
    """Update user profile settings."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE users SET income=?, bracket=?, currency=? WHERE id=?",
        (income, bracket, currency, user_id)
    )
    conn.commit()
    conn.close()
