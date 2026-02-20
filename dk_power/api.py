"""
Low-level helpers for the Energi Data Service REST API.

Handles pagination, column auto-detection, and caching of discovered schemas.
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests

from .config import API_BASE, PAGE_SIZE, PROD_SPEC, RT_SPEC
from .db import get_conn, meta_get, meta_set
from .logger import log

# ── Generic request / pagination ─────────────────────────────────────────────


def api_get(dataset: str, params: dict) -> dict:
    resp = requests.get(f"{API_BASE}/{dataset}", params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_pages(
    dataset: str,
    start: str,
    end: str,
    extra: dict | None = None,
) -> list[dict]:
    """Page through the API and return *all* records in the time range."""
    params: dict[str, Any] = {
        "start": start,
        "end": end,
        "limit": PAGE_SIZE,
        "timezone": "UTC",
        "offset": 0,
    }
    if extra:
        params.update(extra)

    records: list[dict] = []
    while True:
        data = api_get(dataset, params)
        batch = data.get("records", [])
        records.extend(batch)
        log.info(
            "  %s  offset=%-7d fetched=%-6d total=%s",
            dataset,
            params["offset"],
            len(batch),
            data.get("total", "?"),
        )
        if len(batch) < PAGE_SIZE:
            break
        params["offset"] += PAGE_SIZE
        time.sleep(0.25)
    return records


# ── Timestamp normalisation ──────────────────────────────────────────────────


def strip_z(ts: str | None) -> str | None:
    """Remove trailing ``Z`` or ``+...`` timezone suffixes."""
    if ts and isinstance(ts, str):
        return ts.rstrip("Z").split("+")[0]
    return ts


# ── Column auto-detection ────────────────────────────────────────────────────


def _probe_cols(dataset: str) -> list[str]:
    """Return the column names from one live API record."""
    data = api_get(
        dataset,
        {"limit": 1, "timezone": "UTC", "filter": '{"PriceArea":"DK1"}'},
    )
    recs = data.get("records", [])
    return list(recs[0].keys()) if recs else []


def _build_map(cols: list[str], mapping_spec: dict) -> dict:
    """Map our canonical names → actual API column names via substring matching."""
    cl = [c.lower() for c in cols]
    result: dict[str, str | None] = {}
    for our_name, candidates in mapping_spec.items():
        found = None
        for cand in candidates:
            for i, c in enumerate(cl):
                if cand in c:
                    found = cols[i]
                    break
            if found:
                break
        result[our_name] = found
    return result


_PROD_MAP_CACHE: dict | None = None
_RT_MAP_CACHE: dict | None = None


def get_prod_map() -> dict:
    global _PROD_MAP_CACHE
    if _PROD_MAP_CACHE:
        return _PROD_MAP_CACHE

    raw = meta_get("prod_map")
    if raw:
        _PROD_MAP_CACHE = json.loads(raw)
        return _PROD_MAP_CACHE

    log.info("Probing ProductionConsumptionSettlement columns…")
    cols = _probe_cols("ProductionConsumptionSettlement")
    log.info("  Raw cols: %s", cols)
    m = _build_map(cols, PROD_SPEC)
    log.info("  Mapped:   %s", m)
    meta_set("prod_map", json.dumps(m))
    _PROD_MAP_CACHE = m
    return m


def get_rt_map() -> dict:
    global _RT_MAP_CACHE
    if _RT_MAP_CACHE:
        return _RT_MAP_CACHE

    raw = meta_get("rt_map")
    if raw:
        _RT_MAP_CACHE = json.loads(raw)
        return _RT_MAP_CACHE

    log.info("Probing ElectricityProdex5MinRealtime columns…")
    cols = _probe_cols("ElectricityProdex5MinRealtime")
    log.info("  RT cols:  %s", cols)
    m = _build_map(cols, RT_SPEC)
    log.info("  RT mapped:%s", m)
    meta_set("rt_map", json.dumps(m))
    _RT_MAP_CACHE = m
    return m
