"""
Centralised configuration for the DK Power Monitor.

All tunables live here so the rest of the code base never hard-codes
magic numbers.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
DB_PATH: Path = Path("dk_power.db")
GAS_CSV_PATH: Path = Path("Dutch_TTF_Natural_Gas_Futures_Historical_Data.csv")

# ── Price areas ──────────────────────────────────────────────────────────────
DK_AREAS: list = ["DK1", "DK2"]
NEIGHBOUR_AREAS: list = ["NO2", "SE3", "SE4", "DE-LU", "DE"]
ALL_PRICE_AREAS: list = DK_AREAS + NEIGHBOUR_AREAS

# ── API ──────────────────────────────────────────────────────────────────────
API_BASE: str = "https://api.energidataservice.dk/dataset"
PAGE_SIZE: int = 100_000

# ── History / refresh ────────────────────────────────────────────────────────
HISTORY_YEARS: int = 26  # Elspotprices goes back to ~2000
REFRESH_SECONDS: int = 300

# ── Charting ─────────────────────────────────────────────────────────────────
MAX_CHART_PTS: int = 800  # downsample beyond this per series

# ── Elspotprices → DayAheadPrices transition ─────────────────────────────────
ELSPOT_CUTOFF: str = "2025-10-01T00:00:00"

# ── Column auto-detection specs ──────────────────────────────────────────────
PROD_SPEC: dict = {
    "ts":          ["hourutc"],
    "area":        ["pricearea"],
    "central":     ["centralpower", "centralmwh"],
    "decentral":   ["decentralpower", "decentralmwh"],
    "onshore":     ["onshorewind", "windpoweronshore", "onshore"],
    "offshore":    ["offshorewind", "windpoweroffshore", "offshore"],
    "solar":       ["solar"],
    "consumption": ["grossconsumption", "consumption"],
    "ex_no":       ["exchangeno", "exno"],
    "ex_se":       ["exchangese", "exse"],
    "ex_de":       ["exchangede", "exchangege"],
}

RT_SPEC: dict = {
    "ts":        ["minutes5utc", "minutes5dk", "minutes5"],
    "area":      ["pricearea"],
    "central":   ["centralpower", "central"],
    "decentral": ["decentralpower", "decentral"],
    "onshore":   ["onshorewind", "onshore"],
    "offshore":  ["offshorewind", "offshore"],
    "solar":     ["solar"],
    "ex_no":     ["exchangeno", "exno"],
    "ex_se":     ["exchangese", "exse"],
    "ex_de":     ["exchangede", "exchangege"],
}

# ── Open-Meteo temperature source ───────────────────────────────────────────
TEMP_LAT: float = 55.6761  # Copenhagen
TEMP_LON: float = 12.5683
TEMP_HISTORY_DAYS: int = 9500  # ~26 years to match electricity data
