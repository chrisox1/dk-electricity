# DK Power Monitor

Real-time dashboard for Denmark's DK1 (West) and DK2 (East) electricity markets.

## Quick Start

```bash
pip install -r requirements.txt
python run.py
# → http://127.0.0.1:8050
```

First launch backfills 20 years of data (~60-120 seconds). Subsequent launches are instant.

## Project Structure

```
dk_power/
├── run.py                     # Entry point
├── requirements.txt
├── dk_power/                  # Package
│   ├── __init__.py            # Public API: create_app(), startup()
│   ├── config.py              # All constants & tunables
│   ├── logger.py              # Logging setup
│   ├── db.py                  # SQLite schema, connections, meta KV
│   ├── api.py                 # Energi Data Service client & column detection
│   ├── fetchers.py            # Data ingest (prices, production, gas, temp)
│   ├── loaders.py             # DB → DataFrame readers, downsample, smooth
│   ├── theme.py               # Colours, Plotly base layout, UI atoms
│   ├── layout.py              # Dash page structure (no logic)
│   ├── app.py                 # App factory
│   ├── startup.py             # Backfill orchestration & background thread
│   ├── callbacks_dashboard.py # Dashboard tab callbacks
│   └── callbacks_analytics.py # Analytics & forecast tab callbacks
└── tests/
```

## Module Responsibilities

| Module | Lines | What it does |
|--------|------:|--------------|
| `config.py` | ~60 | Every magic number in one place |
| `db.py` | ~100 | Schema, WAL connections, meta get/set |
| `api.py` | ~120 | HTTP pagination, column auto-detection |
| `fetchers.py` | ~250 | All remote data ingest |
| `loaders.py` | ~100 | Read from DB, resample, downsample, smooth |
| `theme.py` | ~120 | Colour palette, Plotly base layout, KPI cards |
| `layout.py` | ~260 | Pure Dash component tree |
| `callbacks_dashboard.py` | ~260 | Price/production/exchange/mix charts |
| `callbacks_analytics.py` | ~280 | Regression, correlation, forecast charts |
| `startup.py` | ~110 | First-run backfill + background updater |

## Data Sources

- **Energi Data Service** (energidataservice.dk) — spot prices, production, realtime
- **Open-Meteo** — historical temperature (Copenhagen)
- **TTF Gas CSV** — optional Dutch TTF natural gas futures (place CSV in project root)

## Optional Dependencies

`statsmodels` and `scipy` enable proper p-values and F-statistics in the analytics tab. Without them, the dashboard falls back to scikit-learn (no statistical inference).
