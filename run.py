#!/usr/bin/env python3
"""
DK1 & DK2 Power Dashboard
==========================
Priority:  1. Spot prices  2. Production mix  3. Exchange
Speed:     - Data downsampled before charting (never pushes 87k pts to browser)
           - DB indices for fast queries
           - Series with all-zero values skipped automatically

Run:  python run.py   →   http://127.0.0.1:8050

First launch backfills 10 years (~30-90 sec). Subsequent launches are instant.
"""

from dk_power import create_app, startup
from dk_power.logger import log

if __name__ == "__main__":
    startup()
    app = create_app()
    log.info("→ http://127.0.0.1:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)
