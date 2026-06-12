# ValtoSpend 
### AI-Powered Personal Expense Tracker

## Live App
 https://valtospend.streamlit.app/

##  Project Progress
| Phase | Task | Status |
|-------|------|--------|
| 1 | Business idea & app definition | Done |
| 2 | Real dataset (3,655 rows) + SQLite database | Done |
| 3 | AI model (Random Forest + Linear Regression) | Done |
| 4 | Streamlit web app + deployment | Done |
| 5 | Presentation slides | In progress |

## Description
ValtoSpend is an AI-powered expense tracking web application that helps 
users understand and manage their everyday finances.

- **Problem:** People struggle to track where their money goes each month
- **Use case:** Upload receipts, track expenses, visualise spending patterns and get AI-based spending forecasts
- **Expected outcome:** Actionable financial insights and monthly spend predictions

## Setup
Create and activate a virtual environment, then install dependencies:

```
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

## Usage
Run the project:

```
python -m streamlit run main.py
```

- **Input:** 3,655 real household expense records used to train the AI model. Users add their own real-time expenses directly in the app.
- **Output:** Interactive dashboard with spending charts, AI-predicted next month expenses, and personalised financial insights

## Project Structure

```
ss26-valtospend-rajaram/
├── README.md
├── .gitignore
├── requirements.txt
├── expenses.csv
└── main.py
```

## AI Components
- **Linear Regression** — predicts next month's average spending trend
- **Random Forest Regressor** — predicts total monthly expenses based on income, bracket, and spending ratios. Evaluated with MAE and R² score on a 20% held-out test set.
- **Claude AI** — reads receipt photos and extracts amount, category, and description

## Tech Stack
Python, Streamlit, SQLite, scikit-learn, pandas, matplotlib, Claude AI.

## Requirements
- Python >= 3.9
- All dependencies listed in `requirements.txt`

## Notes
- The project is reproducible on another system using only this repository
- `.venv/`, cache files, and system files are excluded via `.gitignore`

## Author
Harini Rajaram — ss26-valtospend-rajaram