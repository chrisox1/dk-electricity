"""
DataFrame loaders with aggressive in-memory caching.

Cache strategy:
  - Each loader caches the full result for a given `days` parameter.
  - Cache is invalidated after CACHE_TTL_SECONDS (60s by default).
  - Dashboard ticks (every 300s) get instant cache hits most of the time.
  - Background refresh thread updates DB; next cache miss picks it up.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from .config import MAX_CHART_PTS
from .db import get_conn
from .logger import log


# ── Cache infrastructure ────────────────────────────────────────────────────

CACHE_TTL_SECONDS = 60  # How long cached DataFrames are valid

_cache: dict[str, tuple[float, object]] = {}
_cache_lock = threading.Lock()


def _cache_get(key: str):
    """Return cached value if still valid, else None."""
    with _cache_lock:
        entry = _cache.get(key)
        if entry and (time.monotonic() - entry[0]) < CACHE_TTL_SECONDS:
            return entry[1]
    return None


def _cache_set(key: str, value):
    """Store value in cache with current timestamp."""
    with _cache_lock:
        _cache[key] = (time.monotonic(), value)


def invalidate_cache():
    """Force-clear all cached data (called after DB writes)."""
    with _cache_lock:
        _cache.clear()


# ── Cutoff helper ────────────────────────────────────────────────────────────

def _cutoff(days: float | str | None) -> str:
    if not days:
        days = 7
    return (
        datetime.now(timezone.utc) - timedelta(days=float(days))
    ).strftime("%Y-%m-%dT%H:%M:%S")


# ── Price loader ─────────────────────────────────────────────────────────────

def load_prices(days: float | str) -> pd.DataFrame:
    days = float(days) if days else 7.0
    key = f"prices:{days}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    cutoff = _cutoff(days)
    db = get_conn()
    df = pd.read_sql(
        "SELECT ts_utc,price_area,price_eur,price_dkk FROM spot_prices"
        " WHERE ts_utc>=? ORDER BY ts_utc",
        db,
        params=(cutoff,),
        parse_dates=["ts_utc"],
    )
    db.close()
    log.info("load_prices: %d rows for days=%s (cache miss)", len(df), days)
    _cache_set(key, df)
    return df


# ── Production loader ────────────────────────────────────────────────────────

def load_production(days: float | str) -> pd.DataFrame:
    days = float(days) if days else 7.0
    key = f"prod:{days}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    db = get_conn()
    df = pd.read_sql(
        "SELECT * FROM production WHERE ts_utc>=? ORDER BY ts_utc",
        db,
        params=(_cutoff(days),),
        parse_dates=["ts_utc"],
    )
    db.close()
    _cache_set(key, df)
    return df


# ── Gas prices ───────────────────────────────────────────────────────────────

def load_gas_prices() -> pd.DataFrame:
    key = "gas"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    db = get_conn()
    df = pd.read_sql(
        "SELECT date, price_eur as gas_price_eur FROM gas_prices ORDER BY date", db
    )
    db.close()
    _cache_set(key, df)
    return df


# ── Temperature ──────────────────────────────────────────────────────────────

def load_temperature() -> pd.DataFrame:
    key = "temp"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    db = get_conn()
    df = pd.read_sql(
        "SELECT date, temp_avg_c FROM temperature ORDER BY date", db
    )
    db.close()
    _cache_set(key, df)
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
