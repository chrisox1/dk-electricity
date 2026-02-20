"""
SQLite database helpers: connection pool, schema bootstrap, meta KV store.
"""

import sqlite3
from datetime import datetime, timezone

from .config import DB_PATH
from .logger import log


def get_conn() -> sqlite3.Connection:
    """Return a WAL-mode connection (thread-safe for reads)."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS spot_prices (
    ts_utc     TEXT NOT NULL,
    price_area TEXT NOT NULL,
    price_eur  REAL,
    price_dkk  REAL,
    PRIMARY KEY (ts_utc, price_area)
);
CREATE INDEX IF NOT EXISTS ix_spot_ts ON spot_prices(ts_utc);

CREATE TABLE IF NOT EXISTS production (
    ts_utc        TEXT NOT NULL,
    price_area    TEXT NOT NULL,
    central_mw    REAL,
    decentral_mw  REAL,
    onshore_mw    REAL,
    offshore_mw   REAL,
    solar_mw      REAL,
    consumption_mw REAL,
    ex_no         REAL,
    ex_se         REAL,
    ex_de         REAL,
    PRIMARY KEY (ts_utc, price_area)
);
CREATE INDEX IF NOT EXISTS ix_prod_ts ON production(ts_utc);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS gas_prices (
    date       TEXT PRIMARY KEY,
    price_eur  REAL
);

CREATE TABLE IF NOT EXISTS temperature (
    date       TEXT PRIMARY KEY,
    temp_avg_c REAL
);
"""


def init_db() -> None:
    """Create tables / indices and normalise legacy timestamps."""
    db = get_conn()
    db.executescript(_SCHEMA)

    # Strip Z / +00:00 suffixes left by older versions
    db.execute(
        "UPDATE spot_prices SET ts_utc=SUBSTR(ts_utc,1,19) "
        "WHERE ts_utc LIKE '%Z' OR ts_utc LIKE '%+%'"
    )
    db.execute(
        "UPDATE production SET ts_utc=SUBSTR(ts_utc,1,19) "
        "WHERE ts_utc LIKE '%Z' OR ts_utc LIKE '%+%'"
    )

    # Clear stale price_latest bookmark (>2 days) to force re-fetch
    row = db.execute("SELECT value FROM meta WHERE key='price_latest'").fetchone()
    if row:
        try:
            ts = row[0].rstrip("Z").split("+")[0]
            age = (
                datetime.now(timezone.utc)
                - datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
            ).days
            if age > 2:
                db.execute("DELETE FROM meta WHERE key='price_latest'")
                log.info(
                    "Cleared stale price_latest (%s, %d days old) — will full re-fetch",
                    row[0],
                    age,
                )
        except Exception as exc:
            log.warning("Could not check price_latest age: %s", exc)
            db.execute("DELETE FROM meta WHERE key='price_latest'")

    db.commit()
    db.close()
    log.info("DB ready: %s", DB_PATH)


# ── Meta key-value helpers ───────────────────────────────────────────────────

def meta_get(key: str) -> str | None:
    db = get_conn()
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    db.close()
    return row[0] if row else None


def meta_set(key: str, value: str) -> None:
    db = get_conn()
    db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, value))
    db.commit()
    db.close()
