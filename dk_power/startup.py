"""
Startup sequence: DB init, first-run backfill, and background refresh thread.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from .config import GAS_CSV_PATH, REFRESH_SECONDS
from .db import get_conn, init_db
from .fetchers import (
    fetch_prices,
    fetch_production,
    fetch_realtime,
    fetch_temperature_data,
    load_gas_prices_from_csv,
)
from .fetchers_macro import fetch_all_macro
from .logger import log


def _background_updater() -> None:
    """Runs in a daemon thread: refreshes realtime data every cycle,
    daily data once per calendar day."""
    last_day = None
    log.info("Background updater started (every %ds)", REFRESH_SECONDS)
    while True:
        try:
            fetch_realtime()
        except Exception as exc:
            log.error("RT: %s", exc)

        today = datetime.now(timezone.utc).date()
        if last_day != today:
            try:
                fetch_prices()
            except Exception as exc:
                log.error("Prices: %s", exc)
            try:
                fetch_production()
            except Exception as exc:
                log.error("Prod: %s", exc)
            last_day = today

        time.sleep(REFRESH_SECONDS)


def startup() -> None:
    """Called once before the web server starts."""
    init_db()

    # Gas CSV import (searches project root + dk_power/ subfolder)
    try:
        load_gas_prices_from_csv()
    except Exception as exc:
        log.error("Gas price CSV import failed: %s", exc)

    # Temperature
    try:
        fetch_temperature_data()
    except Exception as exc:
        log.error("Temperature fetch failed: %s", exc)

    # Macro economics (CPI, GDP, Industrial Production)
    try:
        fetch_all_macro()
    except Exception as exc:
        log.error("Macro data fetch failed: %s", exc)

    # Decide backfill vs. incremental
    db = get_conn()
    n = db.execute("SELECT COUNT(*) FROM spot_prices").fetchone()[0]
    db.close()

    if n == 0:
        log.info("=== First run: full backfill (~2000 → now, may take 3-8 min) ===")
        try:
            fetch_prices(full=True)
        except Exception as exc:
            log.error("Price backfill: %s", exc)
        try:
            fetch_production(full=True)
        except Exception as exc:
            log.error("Prod backfill: %s", exc)
    else:
        log.info("=== Incremental update ===")
        db2 = get_conn()
        p_latest = db2.execute(
            "SELECT value FROM meta WHERE key='price_latest'"
        ).fetchone()
        db2.close()

        if p_latest:
            latest_dt = datetime.fromisoformat(p_latest[0].rstrip("Z").split("+")[0])
            age_days = (
                datetime.now(timezone.utc)
                - latest_dt.replace(tzinfo=timezone.utc)
            ).days
            log.info("Price data age: %d days (latest: %s)", age_days, p_latest[0])
            if age_days > 2:
                log.warning(
                    "Price data is %d days stale — running yearly-chunk backfill",
                    age_days,
                )
                try:
                    fetch_prices(full=True)
                except Exception as exc:
                    log.error("Price backfill: %s", exc)
            else:
                try:
                    fetch_prices(full=False)
                except Exception as exc:
                    log.error("Price update: %s", exc)
        else:
            try:
                fetch_prices(full=True)
            except Exception as exc:
                log.error("Price update: %s", exc)

        try:
            fetch_production()
        except Exception as exc:
            log.error("Prod update: %s", exc)
        try:
            fetch_realtime()
        except Exception as exc:
            log.error("RT update: %s", exc)

    threading.Thread(target=_background_updater, daemon=True).start()
