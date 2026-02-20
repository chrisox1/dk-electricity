"""
DK Power Monitor
================
Modular Dash dashboard for DK1/DK2 electricity prices, production mix,
cross-border exchange, analytics, and day-ahead forecasts.

Quick start::

    from dk_power import create_app, startup

    startup()
    app = create_app()
    app.run(host="0.0.0.0", port=8050)
"""

from .app import create_app
from .startup import startup

__all__ = ["create_app", "startup"]