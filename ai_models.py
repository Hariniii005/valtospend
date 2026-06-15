"""
ai_models.py — All AI models for ValtoSpend
Contains Linear Regression, Random Forest, and Neural Network (from scratch).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

FEATURES = [
    "Income", "Month", "Festival_Count",
    "Food_Ratio", "Groceries_Ratio", "Transport_Ratio",
    "Entertainment_Ratio", "Shopping_Ratio", "Rent_Ratio",
    "Bills_Ratio", "Healthcare_Ratio", "Education_Ratio"
]


def linear_regression_forecast(df):
    """
    Predict next month's average total expenses using Linear Regression.
    Uses monthly time series — one data point per month.
    Returns: predicted value, MAE, trend predictions, next month label.
    """
    monthly_ts = df.groupby("Date")["Total_Expenses"].mean().reset_index().sort_values("Date")
    monthly_ts["idx"] = range(len(monthly_ts))

    lr = LinearRegression()
    lr.fit(monthly_ts[["idx"]], monthly_ts["Total_Expenses"])

    next_p   = lr.predict([[len(monthly_ts)]])[0]
    t_preds  = lr.predict(monthly_ts[["idx"]])
    mae      = mean_absolute_error(monthly_ts["Total_Expenses"], t_preds)
    next_lbl = (monthly_ts["Date"].max() + pd.DateOffset(months=1)).strftime("%B %Y")

    return next_p, mae, t_preds, next_lbl, monthly_ts


def random_forest_model(df):
    """
    Train a Random Forest Regressor to predict total monthly expenses.
    Features: income, month, festival count, income bracket, category ratios.
    Returns: model, label encoder, MAE, R2 score.
    """
    model_df = df[FEATURES + ["Income_Bracket", "Total_Expenses"]].dropna().copy()
    le = LabelEncoder()
    model_df["Income_Bracket_enc"] = le.fit_transform(model_df["Income_Bracket"])

    X = model_df[FEATURES + ["Income_Bracket_enc"]].values
    y = model_df["Total_Expenses"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    rf.fit(X_tr, y_tr)

    mae = mean_absolute_error(y_te, rf.predict(X_te))
    r2  = r2_score(y_te, rf.predict(X_te))

    return rf, le, mae, r2


def neural_network_scratch(df):
    """
    Neural Network built from scratch using NumPy only — no TensorFlow.
    Architecture: Input → Dense(64, ReLU) → Dense(32, ReLU) → Output(1)
    Optimiser: Gradient Descent | Loss: MAE | Epochs: 100
    Returns: MAE, R2 score, training losses list.
    """
    model_df = df[FEATURES + ["Income_Bracket", "Total_Expenses"]].dropna().copy()
    le_nn = LabelEncoder()
    model_df["Income_Bracket_enc"] = le_nn.fit_transform(model_df["Income_Bracket"])

    X = model_df[FEATURES + ["Income_Bracket_enc"]].values.astype(float)
    y = model_df["Total_Expenses"].values.astype(float)

    # Normalise features and target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_mean, y_std = y.mean(), y.std()
    y_scaled = (y - y_mean) / y_std

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    # Weight initialisation (He initialisation for ReLU)
    np.random.seed(42)
    n_in = X_tr.shape[1]
    W1 = np.random.randn(n_in, 64) * np.sqrt(2.0 / n_in)
    b1 = np.zeros((1, 64))
    W2 = np.random.randn(64, 32)   * np.sqrt(2.0 / 64)
    b2 = np.zeros((1, 32))
    W3 = np.random.randn(32, 1)    * np.sqrt(2.0 / 32)
    b3 = np.zeros((1, 1))

    def relu(x):      return np.maximum(0, x)
    def relu_d(x):    return (x > 0).astype(float)

    # Training loop — gradient descent
    lr_rate = 0.001
    losses  = []
    for _ in range(100):
        # Forward pass
        Z1 = X_tr @ W1 + b1;  A1 = relu(Z1)
        Z2 = A1    @ W2 + b2;  A2 = relu(Z2)
        Z3 = A2    @ W3 + b3
        diff = Z3 - y_tr.reshape(-1, 1)
        losses.append(np.mean(np.abs(diff)))

        # Backward pass
        dZ3 = np.sign(diff) / len(y_tr)
        dW3 = A2.T @ dZ3;    db3 = dZ3.sum(axis=0, keepdims=True)
        dA2 = dZ3 @ W3.T
        dZ2 = dA2 * relu_d(Z2)
        dW2 = A1.T @ dZ2;    db2 = dZ2.sum(axis=0, keepdims=True)
        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * relu_d(Z1)
        dW1 = X_tr.T @ dZ1;  db1 = dZ1.sum(axis=0, keepdims=True)

        # Weight update
        W1 -= lr_rate * dW1;  b1 -= lr_rate * db1
        W2 -= lr_rate * dW2;  b2 -= lr_rate * db2
        W3 -= lr_rate * dW3;  b3 -= lr_rate * db3

    # Evaluate on test set
    A1t = relu(X_te @ W1 + b1)
    A2t = relu(A1t  @ W2 + b2)
    preds_scaled = (A2t @ W3 + b3).flatten()
    preds  = preds_scaled * y_std + y_mean
    y_actual = y_te * y_std + y_mean

    mae = mean_absolute_error(y_actual, preds)
    r2  = r2_score(y_actual, preds)

    return mae, r2, losses
