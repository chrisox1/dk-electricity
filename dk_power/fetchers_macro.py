"""
Macro data ingest: CPI (inflation), GDP, and industrial production from
Statistics Denmark's free StatBank API (api.statbank.dk).

The fetcher probes each table's metadata at runtime to discover the exact
variable codes, so it never hard-codes values that could become stale.
All series are cached in SQLite.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import pandas as pd
import requests

from .db import get_conn
from .logger import log

DST_API = "https://api.statbank.dk/v1"

# ── Schema extension ─────────────────────────────────────────────────────────

_MACRO_SCHEMA = """
CREATE TABLE IF NOT EXISTS macro_cpi (
    date       TEXT PRIMARY KEY,
    cpi_index  REAL
);
CREATE TABLE IF NOT EXISTS macro_gdp (
    date       TEXT PRIMARY KEY,
    gdp_real   REAL
);
CREATE TABLE IF NOT EXISTS macro_indpro (
    date       TEXT PRIMARY KEY,
    indpro_index REAL
);
"""


def init_macro_tables() -> None:
    db = get_conn()
    db.executescript(_MACRO_SCHEMA)
    db.commit()
    db.close()


# ── DST helpers ──────────────────────────────────────────────────────────────

def _dst_tableinfo(table: str) -> dict:
    """Fetch table metadata (variable codes & values)."""
    resp = requests.post(
        f"{DST_API}/tableinfo",
        json={"table": table, "format": "JSON", "lang": "en"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _dst_bulk(table: str, variables: list[dict] | None = None) -> pd.DataFrame:
    """POST to DST /data endpoint and return a DataFrame."""
    params: dict = {
        "table": table,
        "format": "BULK",
        "lang": "en",
    }
    if variables:
        params["variables"] = variables
    resp = requests.post(f"{DST_API}/data", json=params, timeout=60)

    if resp.status_code == 400:
        log.error("DST 400 for %s: %s", table, resp.text[:500])
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text), sep=";")


def _find_var(meta: dict, *substrings: str) -> dict | None:
    """Find a variable in table metadata whose id matches any substring."""
    for var in meta.get("variables", []):
        vid = var.get("id", "").lower()
        for sub in substrings:
            if sub.lower() in vid:
                return var
    return None


def _first_value_code(var: dict | None) -> str | None:
    if not var:
        return None
    vals = var.get("values", [])
    return vals[0]["id"] if vals else None


def _parse_dst_month(tid: str) -> str | None:
    try:
        return datetime.strptime(tid.strip(), "%YM%m").strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        return None


def _parse_dst_quarter(tid: str) -> str | None:
    try:
        tid = tid.strip()
        year = int(tid[:4])
        q = int(tid[-1])
        month = {1: 1, 2: 4, 3: 7, 4: 10}[q]
        return f"{year}-{month:02d}-01"
    except (ValueError, KeyError, AttributeError):
        return None


def _parse_dst_value(val) -> float | None:
    if pd.isna(val):
        return None
    s = str(val).strip().replace(",", ".").replace("..", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _is_fresh(table_name: str, date_col: str, min_rows: int, max_age_days: int) -> bool:
    db = get_conn()
    count = db.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    last = db.execute(f"SELECT MAX({date_col}) FROM {table_name}").fetchone()[0]
    db.close()
    if count >= min_rows and last:
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(last).replace(tzinfo=timezone.utc)).days
        if age < max_age_days:
            log.info("%s fresh (%d rows, latest %s, %d days old)", table_name, count, last, age)
            return True
    return False


# ── CPI (Consumer Price Index) ───────────────────────────────────────────────

def fetch_cpi() -> None:
    if _is_fresh("macro_cpi", "date", 24, 35):
        return

    log.info("Fetching CPI from Statistics Denmark (PRIS111)…")
    try:
        meta = _dst_tableinfo("PRIS111")
        log.info("PRIS111 variables: %s",
                 [(v["id"], v.get("text", "")) for v in meta.get("variables", [])])

        varegr = _find_var(meta, "VAREGR")
        enhed = _find_var(meta, "ENHED")

        variables = []
        if varegr:
            code = _first_value_code(varegr)
            if code:
                variables.append({"code": varegr["id"], "values": [code]})
        if enhed:
            code = _first_value_code(enhed)
            if code:
                variables.append({"code": enhed["id"], "values": [code]})
        variables.append({"code": "Tid", "values": ["*"]})

        log.info("PRIS111 query: %s", json.dumps(variables, ensure_ascii=False))
        df = _dst_bulk("PRIS111", variables)
        log.info("PRIS111 columns: %s, rows: %d", list(df.columns), len(df))
    except Exception as exc:
        log.error("CPI fetch failed: %s", exc)
        return

    rows = []
    for _, r in df.iterrows():
        date = _parse_dst_month(str(r.get("TID", "")))
        val = _parse_dst_value(r.get("INDHOLD", None))
        if date and val:
            rows.append((date, val))

    if rows:
        db = get_conn()
        db.executemany("INSERT OR REPLACE INTO macro_cpi(date, cpi_index) VALUES(?,?)", rows)
        db.commit()
        db.close()
        log.info("CPI: stored %d monthly observations", len(rows))
    else:
        log.warning("CPI: no valid rows parsed")


# ── GDP (quarterly, real, seasonally adjusted) ───────────────────────────────

def fetch_gdp() -> None:
    if _is_fresh("macro_gdp", "date", 8, 100):
        return

    log.info("Fetching GDP from Statistics Denmark (NKN1)…")
    try:
        meta = _dst_tableinfo("NKN1")
        log.info("NKN1 variables: %s",
                 [(v["id"], v.get("text", ""),
                   [vv["id"] for vv in v.get("values", [])[:5]])
                  for v in meta.get("variables", [])])

        transakt = _find_var(meta, "TRANSAKT")
        prisenhed = _find_var(meta, "PRISENHED")
        saesonkorr = _find_var(meta, "SÆSON", "SAESON", "KORRIG")

        variables = []
        if transakt:
            vals = transakt.get("values", [])
            gdp_code = next((v["id"] for v in vals if "B1GQ" in v["id"]), None)
            if not gdp_code:
                gdp_code = _first_value_code(transakt)
            if gdp_code:
                variables.append({"code": transakt["id"], "values": [gdp_code]})

        if prisenhed:
            vals = prisenhed.get("values", [])
            const_code = next(
                (v["id"] for v in vals
                 if any(kw in v.get("text", "").lower()
                        for kw in ["chain", "constant", "kæde", "2010"])),
                None,
            )
            if not const_code:
                const_code = _first_value_code(prisenhed)
            if const_code:
                variables.append({"code": prisenhed["id"], "values": [const_code]})

        if saesonkorr:
            vals = saesonkorr.get("values", [])
            sa_code = next(
                (v["id"] for v in vals
                 if any(kw in v.get("text", "").lower()
                        for kw in ["season", "sæson", "korri"])),
                None,
            )
            if not sa_code:
                sa_code = _first_value_code(saesonkorr)
            if sa_code:
                variables.append({"code": saesonkorr["id"], "values": [sa_code]})

        variables.append({"code": "Tid", "values": ["*"]})

        log.info("NKN1 query: %s", json.dumps(variables, ensure_ascii=False))
        df = _dst_bulk("NKN1", variables)
        log.info("NKN1 columns: %s, rows: %d", list(df.columns), len(df))
    except Exception as exc:
        log.error("GDP fetch failed: %s", exc)
        return

    rows = []
    for _, r in df.iterrows():
        date = _parse_dst_quarter(str(r.get("TID", "")))
        val = _parse_dst_value(r.get("INDHOLD", None))
        if date and val:
            rows.append((date, val))

    if rows:
        db = get_conn()
        db.executemany("INSERT OR REPLACE INTO macro_gdp(date, gdp_real) VALUES(?,?)", rows)
        db.commit()
        db.close()
        log.info("GDP: stored %d quarterly observations", len(rows))
    else:
        log.warning("GDP: no valid rows parsed")


# ── Industrial production index ──────────────────────────────────────────────

def fetch_industrial_production() -> None:
    if _is_fresh("macro_indpro", "date", 24, 35):
        return

    log.info("Fetching Industrial Production from Statistics Denmark (IPOP2021)…")
    try:
        meta = _dst_tableinfo("IPOP2021")
        log.info("IPOP2021 variables: %s",
                 [(v["id"], v.get("text", ""),
                   [vv["id"] for vv in v.get("values", [])[:5]])
                  for v in meta.get("variables", [])])

        # For all non-time variables, pick the first value (usually "total")
        variables = []
        for var in meta.get("variables", []):
            vid = var["id"].upper()
            if vid == "TID":
                variables.append({"code": "Tid", "values": ["*"]})
                continue
            code = _first_value_code(var)
            if code:
                variables.append({"code": var["id"], "values": [code]})

        log.info("IPOP2021 query: %s", json.dumps(variables, ensure_ascii=False))
        df = _dst_bulk("IPOP2021", variables)
        log.info("IPOP2021 columns: %s, rows: %d", list(df.columns), len(df))
    except Exception as exc:
        log.error("IndPro fetch failed: %s", exc)
        return

    rows = []
    for _, r in df.iterrows():
        date = _parse_dst_month(str(r.get("TID", "")))
        val = _parse_dst_value(r.get("INDHOLD", None))
        if date and val:
            rows.append((date, val))

    if rows:
        db = get_conn()
        db.executemany(
            "INSERT OR REPLACE INTO macro_indpro(date, indpro_index) VALUES(?,?)", rows
        )
        db.commit()
        db.close()
        log.info("IndPro: stored %d monthly observations", len(rows))
    else:
        log.warning("IndPro: no valid rows parsed")


# ── Convenience ──────────────────────────────────────────────────────────────

def fetch_all_macro() -> None:
    init_macro_tables()
    for fn in [fetch_cpi, fetch_gdp, fetch_industrial_production]:
        try:
            fn()
        except Exception as exc:
            log.error("Macro fetch error in %s: %s", fn.__name__, exc)
