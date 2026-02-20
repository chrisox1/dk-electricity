"""
Colour palette, Plotly base layout, and small Dash UI components.

Keeps visual constants out of callback code so charts stay consistent.
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import html

# ── Colour palette ───────────────────────────────────────────────────────────

C: dict[str, str] = {
    "bg":        "#0a0e1a",
    "card":      "#111827",
    "border":    "#1f2937",
    "dk1":       "#f59e0b",
    "dk2":       "#38bdf8",
    "offshore":  "#34d399",
    "onshore":   "#6ee7b7",
    "solar":     "#fde68a",
    "central":   "#f87171",
    "decentral": "#fb923c",
    "consume":   "#a78bfa",
    "text":      "#f1f5f9",
    "muted":     "#64748b",
    "green":     "#22c55e",
    "red":       "#ef4444",
}


def rgba(hex_color: str, alpha: float = 1.0) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Plotly base layout dict ──────────────────────────────────────────────────

BASE: dict = dict(
    paper_bgcolor=C["bg"],
    plot_bgcolor=C["card"],
    font=dict(color=C["text"], family="'IBM Plex Mono',monospace", size=11),
    xaxis=dict(gridcolor=C["border"], linecolor=C["border"], zeroline=False),
    yaxis=dict(gridcolor=C["border"], linecolor=C["border"], zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.2, font=dict(size=10)),
    margin=dict(l=55, r=15, t=35, b=60),
    hovermode="x unified",
    hoverlabel=dict(bgcolor=C["card"], font_size=11),
)

# ── Dropdown / label styles ──────────────────────────────────────────────────

DD_STYLE: dict = {
    "background": C["card"],
    "color": C["text"],
    "border": f"1px solid {C['border']}",
    "fontSize": "0.85rem",
}

LABEL_STYLE: dict = {
    "color": C["muted"],
    "fontSize": "0.68rem",
    "textTransform": "uppercase",
    "letterSpacing": "0.08em",
}


# ── Reusable UI atoms ───────────────────────────────────────────────────────

def no_data(height: int = 380, msg: str = "No data — fetching…") -> go.Figure:
    """Return an empty Plotly figure with a centred message."""
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(color=C["muted"], size=13),
    )
    layout = {
        **BASE,
        "height": height,
        "xaxis": {**BASE["xaxis"], "visible": False},
        "yaxis": {**BASE["yaxis"], "visible": False},
    }
    fig.update_layout(**layout)
    return fig


def kpi_card(
    label: str,
    val: str,
    unit: str,
    color: str,
    border: str | None = None,
) -> html.Div:
    return html.Div(
        [
            html.P(
                label,
                style={
                    "color": C["muted"],
                    "fontSize": "0.65rem",
                    "letterSpacing": "0.1em",
                    "textTransform": "uppercase",
                    "marginBottom": "2px",
                },
            ),
            html.Span(
                val,
                style={
                    "color": color,
                    "fontSize": "1.4rem",
                    "fontFamily": "'IBM Plex Mono',monospace",
                    "fontWeight": 700,
                },
            ),
            html.Span(f" {unit}", style={"color": C["muted"], "fontSize": "0.75rem"}),
        ],
        style={
            "background": C["card"],
            "border": f"1px solid {C['border']}",
            "borderLeft": f"3px solid {border or color}",
            "borderRadius": "6px",
            "padding": "8px 14px",
        },
    )


def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    elif p_value < 0.10:
        return "†"
    return ""
