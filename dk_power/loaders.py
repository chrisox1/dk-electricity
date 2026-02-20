"""
DataFrame loaders: read from the local DB, resample, downsample, smooth.

These are the functions that the Dash callbacks call to get chart-ready data.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from .config import MAX_CHART_PTS
from .db import get_conn
from .logger import log


def _cutoff(days: float | str | None) -> str:
    if not days:
        days = 7
    return (
        datetime.now(timezone.utc) - timedelta(days=float(days))
    ).strftime("%Y-%m-%dT%H:%M:%S")


def load_prices(days: float | str) -> pd.DataFrame:
    cutoff = _cutoff(days)
    db = get_conn()
    total = db.execute("SELECT COUNT(*) FROM spot_prices").fetchone()[0]
    sample = db.execute(
        "SELECT ts_utc FROM spot_prices ORDER BY ts_utc DESC LIMIT 3"
    ).fetchall()
    log.info(
        "load_prices: total=%d cutoff=%s recent=%s",
        total, cutoff, [r[0] for r in sample],
    )
    df = pd.read_sql(
        "SELECT ts_utc,price_area,price_eur,price_dkk FROM spot_prices"
        " WHERE SUBSTR(ts_utc,1,19)>=? ORDER BY ts_utc",
        db,
        params=(cutoff,),
        parse_dates=["ts_utc"],
    )
    db.close()
    log.info("load_prices: rows returned=%d for days=%s", len(df), days)
    return df


def load_production(days: float | str) -> pd.DataFrame:
    db = get_conn()
    df = pd.read_sql(
        "SELECT * FROM production WHERE SUBSTR(ts_utc,1,19)>=? ORDER BY ts_utc",
        db,
        params=(_cutoff(days),),
        parse_dates=["ts_utc"],
    )
    db.close()
    return df


def load_gas_prices() -> pd.DataFrame:
    db = get_conn()
    df = pd.read_sql(
        "SELECT date, price_eur as gas_price_eur FROM gas_prices ORDER BY date", db
    )
    db.close()
    return df


def load_temperature() -> pd.DataFrame:
    db = get_conn()
    df = pd.read_sql(
        "SELECT date, temp_avg_c FROM temperature ORDER BY date", db
    )
    db.close()
    return df


# ── Downsample & smooth ─────────────────────────────────────────────────────

def downsample(df: pd.DataFrame, n: int = MAX_CHART_PTS) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.iloc[:: max(1, len(df) // n)].copy()


def apply_smoothing(
    df: pd.DataFrame,
    ts_col: str,
    smooth_mode: str,
    days: float,
) -> pd.DataFrame:
    """Rolling-average smoothing.  *auto* picks a window based on time span."""
    if smooth_mode == "none":
        return df

    if smooth_mode == "auto":
        if days <= 7:
            return df
        elif days <= 90:
            window = "7D"
        elif days <= 365:
            window = "30D"
        else:
            window = "90D"
    else:
        window = smooth_mode

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols or ts_col not in df.columns:
        return df

    df_smooth = df.set_index(ts_col).copy()
    df_smooth[numeric_cols] = (
        df_smooth[numeric_cols].rolling(window, min_periods=1).mean()
    )
    return df_smooth.reset_index()
