"""
Dash layout definition.

Pure structure — no callbacks, no data loading.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from .config import REFRESH_SECONDS
from .theme import C, DD_STYLE, LABEL_STYLE


def build_layout() -> dbc.Container:
    return dbc.Container(
        [
            # ── Header ───────────────────────────────────────────────────────
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H1(
                                    "⚡ DK Power Monitor",
                                    style={
                                        "fontFamily": "'IBM Plex Mono',monospace",
                                        "fontSize": "1.35rem",
                                        "color": C["text"],
                                        "marginBottom": "1px",
                                        "fontWeight": 700,
                                    },
                                ),
                                html.Span(
                                    "DK1 West  ·  DK2 East  ·  Energi Data Service",
                                    style={"color": C["muted"], "fontSize": "0.72rem"},
                                ),
                            ]
                        ),
                        width=8,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.Span(
                                    "● LIVE",
                                    style={
                                        "color": C["green"],
                                        "fontFamily": "monospace",
                                        "fontSize": "0.8rem",
                                        "fontWeight": 700,
                                    },
                                ),
                                html.Br(),
                                html.Span(
                                    id="ts-lbl",
                                    style={"color": C["muted"], "fontSize": "0.68rem"},
                                ),
                            ],
                            style={"textAlign": "right", "paddingTop": "4px"},
                        ),
                        width=4,
                    ),
                ],
                className="my-3 align-items-center",
            ),
            # ── Tabs ─────────────────────────────────────────────────────────
            dbc.Tabs(
                [
                    dbc.Tab(
                        label="Dashboard",
                        tab_id="tab-dashboard",
                        label_style={"color": C["muted"]},
                        active_label_style={"color": C["text"]},
                    ),
                    dbc.Tab(
                        label="Analytics & Forecast",
                        tab_id="tab-analytics",
                        label_style={"color": C["muted"]},
                        active_label_style={"color": C["text"]},
                    ),
                    dbc.Tab(
                        label="Macro",
                        tab_id="tab-macro",
                        label_style={"color": C["muted"]},
                        active_label_style={"color": C["text"]},
                    ),
                    dbc.Tab(
                        label="OLS Diagnostics",
                        tab_id="tab-diagnostics",
                        label_style={"color": C["muted"]},
                        active_label_style={"color": C["text"]},
                    ),
                ],
                id="tabs",
                active_tab="tab-dashboard",
                className="mb-3",
                style={"borderBottom": f"1px solid {C['border']}"},
            ),
            html.Div(id="page-content"),
            # ═════════════════════════════════════════════════════════════════
            # DASHBOARD PAGE
            # ═════════════════════════════════════════════════════════════════
            html.Div(
                id="dashboard-page",
                children=[
                    # Controls
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Time window", style=LABEL_STYLE),
                                    dbc.Select(
                                        id="tw",
                                        value="7",
                                        style={**DD_STYLE, "cursor": "pointer"},
                                        options=[
                                            {"label": "6 h", "value": "0.25"},
                                            {"label": "24 h", "value": "1"},
                                            {"label": "7 d", "value": "7"},
                                            {"label": "14 d", "value": "14"},
                                            {"label": "30 d", "value": "30"},
                                            {"label": "90 d", "value": "90"},
                                            {"label": "1 yr", "value": "365"},
                                            {"label": "5 yr", "value": "1825"},
                                            {"label": "10 yr", "value": "3650"},
                                            {"label": "All data", "value": "9999"},
                                        ],
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Price area", style=LABEL_STYLE),
                                    dbc.Select(
                                        id="area",
                                        value="both",
                                        style={**DD_STYLE, "cursor": "pointer"},
                                        options=[
                                            {"label": "DK1 & DK2", "value": "both"},
                                            {"label": "DK1 West", "value": "DK1"},
                                            {"label": "DK2 East", "value": "DK2"},
                                        ],
                                    ),
                                ],
                                md=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Currency", style=LABEL_STYLE),
                                    dbc.Select(
                                        id="ccy",
                                        value="dkk",
                                        style={**DD_STYLE, "cursor": "pointer"},
                                        options=[
                                            {"label": "DKK / kWh", "value": "dkk"},
                                            {"label": "EUR / MWh", "value": "eur"},
                                        ],
                                    ),
                                ],
                                md=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Smoothing", style=LABEL_STYLE),
                                    dbc.Select(
                                        id="smooth",
                                        value="auto",
                                        style={**DD_STYLE, "cursor": "pointer"},
                                        options=[
                                            {"label": "Auto (recommended)", "value": "auto"},
                                            {"label": "None (raw data)", "value": "none"},
                                            {"label": "7-day average", "value": "7D"},
                                            {"label": "30-day average", "value": "30D"},
                                        ],
                                    ),
                                ],
                                md=3,
                            ),
                        ],
                        className="mb-3 g-2",
                    ),
                    # KPI row
                    dbc.Row(id="kpi-row", className="mb-3 g-2"),
                    # Primary: spot price
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Graph(id="price-chart", config={"scrollZoom": True, "displayModeBar": True}),
                                width=12,
                            ),
                        ],
                        className="mb-1",
                    ),
                    dbc.Row(id="price-stats", className="mb-3 g-2"),
                    # Secondary: production + exchange
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id="prod-chart", config={"scrollZoom": True}), md=8),
                            dbc.Col(dcc.Graph(id="exch-chart", config={"scrollZoom": True}), md=4),
                        ],
                        className="mb-3",
                    ),
                    # Tertiary: energy mix + self-sufficiency
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id="mix-chart", config={"scrollZoom": True}), md=8),
                            dbc.Col(dcc.Graph(id="gap-chart", config={"scrollZoom": True}), md=4),
                        ],
                        className="mb-3",
                    ),
                    # Green energy % of total production
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id="green-pct-chart", config={"scrollZoom": True}), md=12),
                        ],
                        className="mb-3",
                    ),
                ],
                style={"display": "block"},
            ),
            # ═════════════════════════════════════════════════════════════════
            # ANALYTICS PAGE
            # ═════════════════════════════════════════════════════════════════
            html.Div(
                id="analytics-page",
                children=[
                    html.Hr(style={"borderColor": C["border"], "margin": "40px 0 30px 0"}),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H3(
                                    "📊 Analytics & Price Forecast",
                                    style={"fontSize": "1.2rem", "color": C["text"], "fontWeight": 600},
                                ),
                                width=8,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Refresh Analysis",
                                    id="refresh-analytics",
                                    color="secondary",
                                    size="sm",
                                    style={"float": "right"},
                                ),
                                width=4,
                            ),
                        ],
                        className="mb-3",
                    ),
                    # Analytics controls
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Analysis window", style=LABEL_STYLE),
                                    dbc.Select(
                                        id="analysis-window",
                                        value="30",
                                        style={**DD_STYLE, "cursor": "pointer"},
                                        options=[
                                            {"label": "7 days", "value": "7"},
                                            {"label": "30 days", "value": "30"},
                                            {"label": "90 days", "value": "90"},
                                            {"label": "1 year", "value": "365"},
                                            {"label": "2 years", "value": "730"},
                                            {"label": "5 years", "value": "1825"},
                                            {"label": "All data", "value": "9999"},
                                        ],
                                    ),
                                ],
                                md=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Price area", style=LABEL_STYLE),
                                    dbc.Select(
                                        id="analysis-area",
                                        value="DK1",
                                        style={**DD_STYLE, "cursor": "pointer"},
                                        options=[
                                            {"label": "DK1 West", "value": "DK1"},
                                            {"label": "DK2 East", "value": "DK2"},
                                        ],
                                    ),
                                ],
                                md=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Data frequency", style=LABEL_STYLE),
                                    dbc.Select(
                                        id="analysis-freq",
                                        value="hourly",
                                        style={**DD_STYLE, "cursor": "pointer"},
                                        options=[
                                            {"label": "Hourly (time features, forward-fill gas)", "value": "hourly"},
                                            {"label": "Daily (honest gas, no time features)", "value": "daily"},
                                        ],
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Regression features", style=LABEL_STYLE),
                                    dbc.Checklist(
                                        id="reg-features",
                                        options=[
                                            {"label": " Price Lag-1", "value": "price_lag1"},
                                            {"label": " Price Lag-7", "value": "price_lag7"},
                                            {"label": " Weekdays", "value": "weekday_dummies"},
                                            {"label": " Wind", "value": "total_wind"},
                                            {"label": " Solar", "value": "solar_mw"},
                                            {"label": " Gas price", "value": "gas_price_eur"},
                                            {"label": " Consumption", "value": "consumption_mw"},
                                            {"label": " Net imports", "value": "net_imports"},
                                            {"label": " Temperature", "value": "temp_avg_c"},
                                            {"label": " 🇩🇪 DE price", "value": "de_price_eur"},
                                            {"label": " 🇳🇴 NO price", "value": "no_price_eur"},
                                            {"label": " 🇸🇪 SE price", "value": "se_price_eur"},
                                        ],
                                        value=["price_lag1", "price_lag7", "weekday_dummies",
                                               "total_wind", "solar_mw", "gas_price_eur",
                                               "consumption_mw", "net_imports", "temp_avg_c"],
                                        inline=True,
                                        style={"fontSize": "0.8rem", "color": C["text"]},
                                        input_style={"marginRight": "3px"},
                                        label_style={"marginRight": "12px", "cursor": "pointer"},
                                    ),
                                ],
                                md=5,
                            ),
                        ],
                        className="mb-3 g-2",
                    ),
                    # Charts
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id="corr-chart", config={"scrollZoom": True}), md=6),
                            dbc.Col(dcc.Graph(id="scatter-chart", config={"scrollZoom": True}), md=6),
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id="wind-regression", config={"scrollZoom": True}), md=3),
                            dbc.Col(dcc.Graph(id="solar-regression", config={"scrollZoom": True}), md=3),
                            dbc.Col(dcc.Graph(id="gas-regression", config={"scrollZoom": True}), md=3),
                            dbc.Col(dcc.Graph(id="temp-regression", config={"scrollZoom": True}), md=3),
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id="consumption-regression", config={"scrollZoom": True}), md=3),
                            dbc.Col(dcc.Graph(id="green-pct-regression", config={"scrollZoom": True}), md=3),
                            dbc.Col(dcc.Graph(id="peak-effect", config={"scrollZoom": True}), md=3),
                            dbc.Col(dcc.Graph(id="weekend-effect", config={"scrollZoom": True}), md=3),
                        ],
                        className="mb-3",
                    ),
                    # Neighbour price regressions
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id="de-price-regression", config={"scrollZoom": True}), md=4),
                            dbc.Col(dcc.Graph(id="no-price-regression", config={"scrollZoom": True}), md=4),
                            dbc.Col(dcc.Graph(id="se-price-regression", config={"scrollZoom": True}), md=4),
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [dbc.Col(dcc.Graph(id="forecast-chart", config={"scrollZoom": True}), md=12)],
                        className="mb-3",
                    ),
                    dbc.Row(id="forecast-summary", className="mb-3 g-2"),
                    html.P(
                        [
                            "Significance: *** p<0.001, ** p<0.01, * p<0.05, † p<0.10  ·  ",
                            "F-statistic tests overall model significance  ·  ",
                            "Adjusted R² penalizes model complexity",
                        ],
                        className="text-muted",
                        style={"fontSize": "10px", "marginTop": "15px", "textAlign": "center"},
                    ),
                    dcc.Interval(id="tick", interval=REFRESH_SECONDS * 1000, n_intervals=0),
                ],
                style={"display": "none"},
            ),
            # ═════════════════════════════════════════════════════════════════
            # MACRO ECONOMICS PAGE
            # ═════════════════════════════════════════════════════════════════
            html.Div(
                id="macro-page",
                children=[
                    html.Hr(style={"borderColor": C["border"], "margin": "40px 0 30px 0"}),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H3(
                                    "Macro — Electricity & the Danish Economy",
                                    style={"fontSize": "1.2rem", "color": C["text"], "fontWeight": 600},
                                ),
                                width=12,
                            ),
                        ],
                        className="mb-2",
                    ),
                    html.P(
                        "Monthly CPI (inflation) and industrial production from Statistics Denmark (PRIS111 / INDPRO1). "
                        "Quarterly real GDP from NKN1. Electricity data aggregated to monthly averages for alignment.",
                        style={"color": C["muted"], "fontSize": "0.72rem"},
                    ),
                    # Controls
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Price area", style=LABEL_STYLE),
                                    dbc.Select(
                                        id="macro-area",
                                        value="DK1",
                                        style={**DD_STYLE, "cursor": "pointer"},
                                        options=[
                                            {"label": "DK1 West", "value": "DK1"},
                                            {"label": "DK2 East", "value": "DK2"},
                                        ],
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("History", style=LABEL_STYLE),
                                    dbc.Select(
                                        id="macro-window",
                                        value="5",
                                        style={**DD_STYLE, "cursor": "pointer"},
                                        options=[
                                            {"label": "2 years", "value": "2"},
                                            {"label": "5 years", "value": "5"},
                                            {"label": "10 years", "value": "10"},
                                        ],
                                    ),
                                ],
                                md=3,
                            ),
                        ],
                        className="mb-3 g-2",
                    ),
                    # ── Time series overlays ─────────────────────────────────
                    html.H5("Time Series Overlays", style={"color": C["text"], "fontSize": "0.9rem", "marginTop": "15px"}),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id="macro-ts-price-cpi", config={"scrollZoom": True}), md=4),
                            dbc.Col(dcc.Graph(id="macro-ts-price-gdp", config={"scrollZoom": True}), md=4),
                            dbc.Col(dcc.Graph(id="macro-ts-price-indpro", config={"scrollZoom": True}), md=4),
                        ],
                        className="mb-3",
                    ),
                    # ── Electricity price regressions ────────────────────────
                    html.H5("Electricity Price vs Macro Variables", style={"color": C["text"], "fontSize": "0.9rem"}),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id="macro-scatter-price-cpi", config={"scrollZoom": True}), md=6),
                            dbc.Col(dcc.Graph(id="macro-scatter-price-indpro", config={"scrollZoom": True}), md=6),
                        ],
                        className="mb-3",
                    ),
                    # ── Renewables vs inflation ──────────────────────────────
                    html.H5("Renewable Generation vs Inflation", style={"color": C["text"], "fontSize": "0.9rem"}),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id="macro-scatter-wind-cpi", config={"scrollZoom": True}), md=6),
                            dbc.Col(dcc.Graph(id="macro-scatter-solar-cpi", config={"scrollZoom": True}), md=6),
                        ],
                        className="mb-3",
                    ),
                    # ── Output / production regressions ──────────────────────
                    html.H5("Electricity Output vs Economic Output", style={"color": C["text"], "fontSize": "0.9rem"}),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id="macro-scatter-gen-indpro", config={"scrollZoom": True}), md=6),
                            dbc.Col(dcc.Graph(id="macro-scatter-gen-gdp", config={"scrollZoom": True}), md=6),
                        ],
                        className="mb-3",
                    ),
                    # Summary cards
                    dbc.Row(id="macro-summary", className="mb-3 g-2"),
                    html.P(
                        [
                            "Data: Statistics Denmark (StatBank API)  ·  ",
                            "Significance: *** p<0.001, ** p<0.01, * p<0.05, † p<0.10  ·  ",
                            "Monthly CPI & IndPro, Quarterly GDP (forward-filled to monthly)",
                        ],
                        className="text-muted",
                        style={"fontSize": "10px", "marginTop": "15px", "textAlign": "center"},
                    ),
                ],
                style={"display": "none"},
            ),
            # ═════════════════════════════════════════════════════════════════
            # OLS DIAGNOSTICS PAGE
            # ═════════════════════════════════════════════════════════════════
            html.Div(
                id="diagnostics-page",
                children=[
                    html.Hr(style={"borderColor": C["border"], "margin": "40px 0 30px 0"}),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H3(
                                    "🔬 OLS Regression Diagnostics",
                                    style={"fontSize": "1.2rem", "color": C["text"], "fontWeight": 600},
                                ),
                                width=8,
                            ),
                            dbc.Col(
                                dbc.Button("Run Diagnostics", id="diag-refresh", color="secondary", size="sm",
                                           style={"float": "right"}),
                                width=4,
                            ),
                        ],
                        className="mb-2",
                    ),
                    html.P(
                        "Checking the 6 classical OLS assumptions: linearity, independence, "
                        "homoscedasticity, normality, multicollinearity, and influential observations.",
                        style={"color": C["muted"], "fontSize": "0.72rem"},
                    ),
                    # Controls
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Analysis window", style=LABEL_STYLE),
                                    dbc.Select(
                                        id="diag-window", value="90",
                                        style={**DD_STYLE, "cursor": "pointer"},
                                        options=[
                                            {"label": "30 days", "value": "30"},
                                            {"label": "90 days", "value": "90"},
                                            {"label": "1 year", "value": "365"},
                                            {"label": "5 years", "value": "1825"},
                                            {"label": "All data", "value": "9999"},
                                        ],
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Price area", style=LABEL_STYLE),
                                    dbc.Select(
                                        id="diag-area", value="DK1",
                                        style={**DD_STYLE, "cursor": "pointer"},
                                        options=[
                                            {"label": "DK1 West", "value": "DK1"},
                                            {"label": "DK2 East", "value": "DK2"},
                                        ],
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Data frequency", style=LABEL_STYLE),
                                    dbc.Select(
                                        id="diag-freq", value="daily",
                                        style={**DD_STYLE, "cursor": "pointer"},
                                        options=[
                                            {"label": "Hourly", "value": "hourly"},
                                            {"label": "Daily", "value": "daily"},
                                        ],
                                    ),
                                ],
                                md=3,
                            ),
                        ],
                        className="mb-3 g-2",
                    ),
                    # Row 1: Linearity + Homoscedasticity
                    html.H5("1. Linearity  &  3. Homoscedasticity",
                            style={"color": C["muted"], "fontSize": "0.85rem", "marginTop": "10px"}),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id="diag-resid-fitted", config={"scrollZoom": True}), md=6),
                            dbc.Col(dcc.Graph(id="diag-scale-loc", config={"scrollZoom": True}), md=6),
                        ],
                        className="mb-3",
                    ),
                    # Row 2: Normality
                    html.H5("4. Normality of Errors",
                            style={"color": C["muted"], "fontSize": "0.85rem"}),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id="diag-qq", config={"scrollZoom": True}), md=6),
                            dbc.Col(dcc.Graph(id="diag-resid-hist", config={"scrollZoom": True}), md=6),
                        ],
                        className="mb-3",
                    ),
                    # Row 3: Influential + Multicollinearity
                    html.H5("5. Multicollinearity  &  6. Influential Observations",
                            style={"color": C["muted"], "fontSize": "0.85rem"}),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id="diag-corr-heatmap", config={"scrollZoom": True}), md=4),
                            dbc.Col(dcc.Graph(id="diag-cooks", config={"scrollZoom": True}), md=4),
                            dbc.Col(dcc.Graph(id="diag-leverage", config={"scrollZoom": True}), md=4),
                        ],
                        className="mb-3",
                    ),
                    # Row 4: Partial regressions (linearity deep dive)
                    html.H5("1. Linearity — Partial Regression (Added-Variable) Plots",
                            style={"color": C["muted"], "fontSize": "0.85rem"}),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id="diag-partial-wind", config={"scrollZoom": True}), md=4),
                            dbc.Col(dcc.Graph(id="diag-partial-solar", config={"scrollZoom": True}), md=4),
                            dbc.Col(dcc.Graph(id="diag-partial-gas", config={"scrollZoom": True}), md=4),
                        ],
                        className="mb-3",
                    ),
                    # Test statistic summary
                    html.H5("2. Independence  &  Test Statistics Summary",
                            style={"color": C["muted"], "fontSize": "0.85rem"}),
                    dbc.Row(id="diag-summary", className="mb-3 g-2"),
                    html.P(
                        [
                            "Green = assumption satisfied (p>0.05)  ·  Red = assumption violated (p<0.05)  ·  ",
                            "DW ≈ 2 = no autocorrelation  ·  VIF < 5 = low multicollinearity  ·  ",
                            "Cook's D > 4/n = influential observation",
                        ],
                        className="text-muted",
                        style={"fontSize": "10px", "marginTop": "15px", "textAlign": "center"},
                    ),
                ],
                style={"display": "none"},
            ),
            # Footer
            html.Hr(style={"borderColor": C["border"]}),
            html.P(
                "Data: Energinet / Energi Data Service  ·  Statistics Denmark (StatBank API)  ·  "
                f"Refreshes every {REFRESH_SECONDS // 60} min  ·  "
                "Max 1500 pts/series (downsampled for speed)",
                style={"color": C["muted"], "fontSize": "0.63rem", "textAlign": "center"},
            ),
        ],
        fluid=True,
        style={
            "background": C["bg"],
            "minHeight": "100vh",
            "color": C["text"],
            "paddingBottom": "30px",
        },
    )
