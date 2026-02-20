"""
Application factory: builds the Dash app, registers layout and callbacks.
"""

import dash
import dash_bootstrap_components as dbc

from .callbacks_analytics import register_analytics_callbacks
from .callbacks_dashboard import register_dashboard_callbacks
from .callbacks_macro import register_macro_callbacks
from .layout import build_layout


def create_app() -> dash.Dash:
    """Construct and return a fully-wired Dash application."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        title="DK Power Monitor",
    )
    app.layout = build_layout()
    register_dashboard_callbacks(app)
    register_analytics_callbacks(app)
    register_macro_callbacks(app)
    return app
