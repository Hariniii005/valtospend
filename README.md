# ValtoSpend

## Description
ValtoSpend is an AI-powered expense tracking web app that helps users 
understand spending habits and predict future expenses.

- Problem: People struggle to track where their money goes
- Use case: Upload expenses, see dashboards, get AI-based spending forecasts
- Expected outcome: Actionable financial insights + monthly spend predictions


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

- **Input data:** `expenses.csv` — 3,655 real household expense records 
  (2021–2024) with monthly spending by category, income, and savings
- **Output/results:** Interactive dashboard with spending charts, 
  AI-predicted next month expenses, and personalised financial insights


## Project Structure

```
ss26-valtospend-rajaram/
├── README.md
├── .gitignore
├── requirements.txt
├── expenses.csv
└── main.py
```

## Requirements

- Python >= 3.10
- All dependencies are listed in `requirements.txt`

## Notes

- The project must be reproducible on another system using only this repository.
- Do not commit unnecessary files, e.g. `.venv/`, cache files, or system files.

## Authors

- Harini Rajaram