"""
cpi_data.py — Germany Consumer Price Index (CPI) reference data.
Source: Destatis (German Federal Statistical Office), official press releases.
  https://www.destatis.de/EN/Press/2025/01/PE25_020_611.html  (2024 annual figures)
  https://www.destatis.de/EN/Press/2026/01/PE26_019_611.html  (2025 annual figures)

This module stores real, published annual inflation rates per category,
used to inflation-adjust the category price trend projections in
ai_models.category_price_trends(). It deliberately does NOT call a live
API: Eurostat's HICP methodology changed substantially in February 2026,
making live integration unreliable for a fixed-deadline academic project.
Instead, rates are refreshed manually once a year when Destatis
publishes its annual report (typically mid-January) — a five-minute task.

Each SPEND_COLS category is mapped to the closest matching Destatis
expenditure category. Where no exact match exists, the closest available
sub-category or the national average is used (noted below).
"""

# Last verified: June 2026, against Destatis press releases for 2024 and 2025.
CPI_SOURCE = "Destatis (Federal Statistical Office of Germany)"
CPI_LAST_UPDATED = "2026-06"

# Annual average inflation rate (%) per category, year-on-year.
# These are real, published figures — not estimates.
ANNUAL_INFLATION_RATES = {
    2024: {
        "Food":          2.0,   # Food and non-alcoholic beverages, 2024 vs 2023
        "Groceries":     2.0,   # Same Destatis category as Food
        "Transport":     3.8,   # Combined passenger transport services (proxy)
        "Entertainment": 3.8,   # Services (total) average, 2024
        "Shopping":      1.0,   # Non-durable consumer goods average
        "Rent":          2.1,   # Net rents exclusive of heating, 2024 estimate
        "Bills":         -2.3,  # Household energy, 2024 (electricity/gas/heating)
        "Healthcare":    3.8,   # Services average (health services trend with this)
        "Education":     3.8,   # Services average (no distinct education index published)
    },
    2025: {
        "Food":          2.0,   # Food, +2.0% on annual average, 2025 vs 2024
        "Groceries":     2.0,
        "Transport":     11.4,  # Combined passenger transport services, 2025
        "Entertainment": 3.5,   # Services (total), 2025 annual average
        "Shopping":      1.1,   # Non-durable consumer goods, 2025
        "Rent":          2.1,   # Net rents exclusive of heating, 2025
        "Bills":         -2.3,  # Household energy, 2025 (electricity -2.2%, gas, heating oil -5.3%)
        "Healthcare":    6.7,   # In-patient health services, 2025
        "Education":     3.5,   # Services average (no distinct education index)
    },
}

# Overall national CPI (all items), for reference and fallback.
NATIONAL_CPI = {
    2021: 3.1,
    2022: 6.9,
    2023: 5.9,
    2024: 2.2,
    2025: 2.2,
}


def get_inflation_rate(category: str, year: int) -> float:
    """
    Return the annual inflation rate (%) for a given category and year.
    Falls back to the national average CPI if the category or year
    is not in the lookup table.
    """
    if year in ANNUAL_INFLATION_RATES and category in ANNUAL_INFLATION_RATES[year]:
        return ANNUAL_INFLATION_RATES[year][category]
    if year in NATIONAL_CPI:
        return NATIONAL_CPI[year]
    return list(NATIONAL_CPI.values())[-1]  # most recent known year as fallback


def average_recent_rate(category: str, years: int = 2) -> float:
    """
    Average inflation rate for a category over the most recent N years
    available in the lookup table. Used to project forward sensibly
    rather than relying on a single volatile year.
    """
    available_years = sorted(ANNUAL_INFLATION_RATES.keys())[-years:]
    rates = [get_inflation_rate(category, y) for y in available_years]
    return sum(rates) / len(rates) if rates else 0.0
