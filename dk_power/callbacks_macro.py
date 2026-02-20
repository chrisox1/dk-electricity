"""
Macro Economics tab callbacks.

Regresses Danish electricity prices and renewable generation against
macroeconomic variables: CPI (inflation), real GDP, and industrial production.
All data sourced from Statistics Denmark's StatBank API (monthly / quarterly).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

from .db import get_conn
from .theme import BASE, C, kpi_card, no_data, rgba, significance_stars

try:
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False


# ── Data loaders ─────────────────────────────────────────────────────────────

def _load_macro_table(table: str, date_col: str = "date", val_col: str = None) -> pd.DataFrame:
    db = get_conn()
    df = pd.read_sql(f"SELECT * FROM {table} ORDER BY {date_col}", db)
    db.close()
    if not df.empty:
        df[date_col] = pd.to_datetime(df[date_col])
    return df


def _monthly_elec(area: str) -> pd.DataFrame:
    """Monthly average electricity price (DKK/kWh) and generation from local DB."""
    db = get_conn()
    prices = pd.read_sql(
        "SELECT ts_utc, price_dkk FROM spot_prices WHERE price_area=? ORDER BY ts_utc",
        db, params=(area,), parse_dates=["ts_utc"],
    )
    prod = pd.read_sql(
        "SELECT ts_utc, onshore_mw, offshore_mw, solar_mw, central_mw, decentral_mw, consumption_mw "
        "FROM production WHERE price_area=? ORDER BY ts_utc",
        db, params=(area,), parse_dates=["ts_utc"],
    )
    db.close()

    if prices.empty:
        return pd.DataFrame()

    prices["price_dkk_kwh"] = prices["price_dkk"] * 0.001
    monthly_price = prices.set_index("ts_utc").resample("MS").agg(
        price_dkk_kwh=("price_dkk_kwh", "mean")
    )

    if not prod.empty:
        prod["total_wind"] = prod[["onshore_mw", "offshore_mw"]].fillna(0).sum(axis=1)
        prod["total_solar"] = prod["solar_mw"].fillna(0)
        prod["total_gen"] = prod[["onshore_mw", "offshore_mw", "solar_mw", "central_mw", "decentral_mw"]].fillna(0).sum(axis=1)
        prod["consumption_mw"] = prod["consumption_mw"].fillna(0)
        monthly_prod = prod.set_index("ts_utc").resample("MS").agg(
            total_wind=("total_wind", "mean"),
            total_solar=("total_solar", "mean"),
            total_gen=("total_gen", "mean"),
            consumption_mw=("consumption_mw", "mean"),
        )
        monthly = monthly_price.join(monthly_prod, how="left")
    else:
        monthly = monthly_price

    return monthly.reset_index().rename(columns={"ts_utc": "date"})


# ── Regression helper ────────────────────────────────────────────────────────

def _ols_pair(x: np.ndarray, y: np.ndarray):
    """Fit OLS, return (model, r2, beta, p_value)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return None, 0, 0, 1.0
    X = x.reshape(-1, 1)
    lr = LinearRegression().fit(X, y)
    r2 = lr.score(X, y)
    beta = lr.coef_[0]

    if HAS_SM:
        res = sm.OLS(y, sm.add_constant(X)).fit()
        p = res.pvalues[1] if len(res.pvalues) > 1 else 1.0
    else:
        p = 1.0
    return lr, r2, beta, p


def _scatter_with_trend(fig, x, y, xlabel, ylabel, color, name, row=None, col=None):
    """Add scatter + OLS trend to a figure (optionally in subplot)."""
    mask = np.isfinite(x) & np.isfinite(y)
    xc, yc = x[mask], y[mask]
    kw = {"row": row, "col": col} if row else {}

    fig.add_trace(go.Scatter(
        x=xc, y=yc, mode="markers",
        marker=dict(size=4, color=color, opacity=0.5),
        name=name, showlegend=False,
    ), **kw)

    if len(xc) >= 5:
        lr, r2, beta, p = _ols_pair(xc, yc)
        xr = np.linspace(xc.min(), xc.max(), 80)
        fig.add_trace(go.Scatter(
            x=xr, y=lr.predict(xr.reshape(-1, 1)),
            mode="lines", line=dict(color=C["red"], width=2),
            name=f"β={beta:.4f}  R²={r2:.3f}  {significance_stars(p)}",
            showlegend=True,
        ), **kw)
        return r2, beta, p
    return 0, 0, 1.0


# ── Dual-axis time series helper ─────────────────────────────────────────────

def _dual_axis_ts(dates, y1, y2, y1_name, y2_name, y1_color, y2_color, title):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=dates, y=y1, name=y1_name, mode="lines",
        line=dict(color=y1_color, width=1.5),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=dates, y=y2, name=y2_name, mode="lines",
        line=dict(color=y2_color, width=1.5),
    ), secondary_y=True)
    fig.update_layout(**BASE, title=title, height=340)
    fig.update_yaxes(title_text=y1_name, secondary_y=False,
                     gridcolor=C["border"], linecolor=C["border"])
    fig.update_yaxes(title_text=y2_name, secondary_y=True,
                     gridcolor="rgba(0,0,0,0)", linecolor=C["border"])
    return fig


# ── Register callbacks ───────────────────────────────────────────────────────

def register_macro_callbacks(app) -> None:

    @app.callback(
        Output("macro-ts-price-cpi", "figure"),
        Output("macro-ts-price-gdp", "figure"),
        Output("macro-ts-price-indpro", "figure"),
        Output("macro-scatter-price-cpi", "figure"),
        Output("macro-scatter-price-indpro", "figure"),
        Output("macro-scatter-wind-cpi", "figure"),
        Output("macro-scatter-solar-cpi", "figure"),
        Output("macro-scatter-gen-indpro", "figure"),
        Output("macro-scatter-gen-gdp", "figure"),
        Output("macro-summary", "children"),
        Input("macro-area", "value"),
        Input("macro-window", "value"),
        Input("tabs", "active_tab"),
        prevent_initial_call=False,
    )
    def update_macro(area, window, active_tab):
        area = area or "DK1"
        years = int(window) if window else 5
        empty = no_data(340, "No macro data — fetching on next refresh…")

        # Only compute when tab is active
        if active_tab != "tab-macro":
            return (empty,) * 9 + ([],)

        elec = _monthly_elec(area)
        cpi_df = _load_macro_table("macro_cpi")
        gdp_df = _load_macro_table("macro_gdp")
        indpro_df = _load_macro_table("macro_indpro")

        if elec.empty:
            return (empty,) * 9 + ([],)

        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=years * 365)
        elec["date"] = pd.to_datetime(elec["date"]).dt.tz_localize(None)
        elec = elec[elec["date"] >= cutoff].copy()

        # ── Merge electricity with CPI (monthly) ────────────────────────
        if not cpi_df.empty:
            cpi_df["date"] = pd.to_datetime(cpi_df["date"]).dt.tz_localize(None)
            # compute YoY inflation rate from index
            cpi_df = cpi_df.sort_values("date")
            cpi_df["cpi_yoy"] = cpi_df["cpi_index"].pct_change(12) * 100
            m_cpi = pd.merge(elec, cpi_df[["date", "cpi_index", "cpi_yoy"]],
                             on="date", how="inner")
        else:
            m_cpi = pd.DataFrame()

        # ── Merge electricity with GDP (quarterly → monthly ffill) ──────
        if not gdp_df.empty:
            gdp_df["date"] = pd.to_datetime(gdp_df["date"]).dt.tz_localize(None)
            gdp_monthly = gdp_df.set_index("date").resample("MS").ffill().reset_index()
            m_gdp = pd.merge(elec, gdp_monthly, on="date", how="inner")
        else:
            m_gdp = pd.DataFrame()

        # ── Merge electricity with Industrial Production (monthly) ──────
        if not indpro_df.empty:
            indpro_df["date"] = pd.to_datetime(indpro_df["date"]).dt.tz_localize(None)
            m_indpro = pd.merge(elec, indpro_df, on="date", how="inner")
        else:
            m_indpro = pd.DataFrame()

        # ═══ TIME SERIES CHARTS ═══
        # 1. Price vs CPI
        fig_ts_cpi = empty
        if not m_cpi.empty and "cpi_yoy" in m_cpi.columns:
            fig_ts_cpi = _dual_axis_ts(
                m_cpi["date"], m_cpi["price_dkk_kwh"], m_cpi["cpi_yoy"],
                "Elec Price (DKK/kWh)", "CPI YoY %",
                C["dk1"] if area == "DK1" else C["dk2"], C["red"],
                f"{area} Electricity Price vs Inflation",
            )

        # 2. Price vs GDP
        fig_ts_gdp = empty
        if not m_gdp.empty:
            fig_ts_gdp = _dual_axis_ts(
                m_gdp["date"], m_gdp["price_dkk_kwh"], m_gdp["gdp_real"],
                "Elec Price (DKK/kWh)", "Real GDP (Mio DKK)",
                C["dk1"] if area == "DK1" else C["dk2"], C["green"],
                f"{area} Electricity Price vs Real GDP",
            )

        # 3. Price vs IndPro
        fig_ts_indpro = empty
        if not m_indpro.empty:
            fig_ts_indpro = _dual_axis_ts(
                m_indpro["date"], m_indpro["price_dkk_kwh"], m_indpro["indpro_index"],
                "Elec Price (DKK/kWh)", "Industrial Prod. Index",
                C["dk1"] if area == "DK1" else C["dk2"], C["solar"],
                f"{area} Electricity Price vs Industrial Production",
            )

        # ═══ SCATTER / REGRESSION CHARTS ═══
        results = {}

        # Price vs CPI
        fig_sc_cpi = go.Figure()
        if not m_cpi.empty and "cpi_yoy" in m_cpi.columns:
            r2, beta, p = _scatter_with_trend(
                fig_sc_cpi,
                m_cpi["cpi_yoy"].values, m_cpi["price_dkk_kwh"].values,
                "CPI YoY %", "Price (DKK/kWh)", C["red"], "Price vs CPI",
            )
            fig_sc_cpi.update_layout(**BASE, title=f"Elec Price vs Inflation (β={beta:.4f})",
                                     height=340, xaxis_title="CPI YoY %", yaxis_title="DKK/kWh")
            results["price_cpi"] = (r2, beta, p, len(m_cpi))
        else:
            fig_sc_cpi = empty

        # Price vs IndPro
        fig_sc_indpro = go.Figure()
        if not m_indpro.empty:
            r2, beta, p = _scatter_with_trend(
                fig_sc_indpro,
                m_indpro["indpro_index"].values, m_indpro["price_dkk_kwh"].values,
                "IndPro Index", "Price (DKK/kWh)", C["solar"], "Price vs IndPro",
            )
            fig_sc_indpro.update_layout(**BASE, title=f"Elec Price vs Ind. Production (β={beta:.4f})",
                                        height=340, xaxis_title="Industrial Prod. Index", yaxis_title="DKK/kWh")
            results["price_indpro"] = (r2, beta, p, len(m_indpro))
        else:
            fig_sc_indpro = empty

        # Wind vs CPI
        fig_wind_cpi = go.Figure()
        if not m_cpi.empty and "total_wind" in m_cpi.columns:
            r2, beta, p = _scatter_with_trend(
                fig_wind_cpi,
                m_cpi["total_wind"].values, m_cpi["cpi_yoy"].values,
                "Wind (MW)", "CPI YoY %", C["offshore"], "Wind vs CPI",
            )
            fig_wind_cpi.update_layout(**BASE, title=f"Wind Generation vs Inflation (β={beta:.4f})",
                                       height=340, xaxis_title="Avg Wind Gen (MW)", yaxis_title="CPI YoY %")
            results["wind_cpi"] = (r2, beta, p, len(m_cpi))
        else:
            fig_wind_cpi = empty

        # Solar vs CPI
        fig_solar_cpi = go.Figure()
        if not m_cpi.empty and "total_solar" in m_cpi.columns:
            r2, beta, p = _scatter_with_trend(
                fig_solar_cpi,
                m_cpi["total_solar"].values, m_cpi["cpi_yoy"].values,
                "Solar (MW)", "CPI YoY %", C["solar"], "Solar vs CPI",
            )
            fig_solar_cpi.update_layout(**BASE, title=f"Solar Generation vs Inflation (β={beta:.4f})",
                                        height=340, xaxis_title="Avg Solar Gen (MW)", yaxis_title="CPI YoY %")
            results["solar_cpi"] = (r2, beta, p, len(m_cpi))
        else:
            fig_solar_cpi = empty

        # Total generation vs IndPro
        fig_gen_indpro = go.Figure()
        if not m_indpro.empty and "total_gen" in m_indpro.columns:
            r2, beta, p = _scatter_with_trend(
                fig_gen_indpro,
                m_indpro["total_gen"].values, m_indpro["indpro_index"].values,
                "Total Gen (MW)", "IndPro Index", C["onshore"], "Gen vs IndPro",
            )
            fig_gen_indpro.update_layout(**BASE, title=f"Electricity Output vs Ind. Production (β={beta:.4f})",
                                         height=340, xaxis_title="Avg Total Gen (MW)", yaxis_title="IndPro Index")
            results["gen_indpro"] = (r2, beta, p, len(m_indpro))
        else:
            fig_gen_indpro = empty

        # Total generation vs GDP
        fig_gen_gdp = go.Figure()
        if not m_gdp.empty and "total_gen" in m_gdp.columns:
            r2, beta, p = _scatter_with_trend(
                fig_gen_gdp,
                m_gdp["total_gen"].values, m_gdp["gdp_real"].values,
                "Total Gen (MW)", "Real GDP", C["green"], "Gen vs GDP",
            )
            fig_gen_gdp.update_layout(**BASE, title=f"Electricity Output vs Real GDP (β={beta:.4f})",
                                      height=340, xaxis_title="Avg Total Gen (MW)", yaxis_title="Real GDP (Mio DKK)")
            results["gen_gdp"] = (r2, beta, p, len(m_gdp))
        else:
            fig_gen_gdp = empty

        # ═══ SUMMARY CARDS ═══
        summary = []
        labels = {
            "price_cpi": ("Price ↔ CPI", "DKK/kWh per 1% CPI"),
            "price_indpro": ("Price ↔ IndPro", "DKK/kWh per index pt"),
            "wind_cpi": ("Wind → CPI", "% CPI per MW wind"),
            "solar_cpi": ("Solar → CPI", "% CPI per MW solar"),
            "gen_indpro": ("Output ↔ IndPro", "IndPro per MW gen"),
            "gen_gdp": ("Output ↔ GDP", "GDP per MW gen"),
        }
        for key, (label, unit) in labels.items():
            if key in results:
                r2, beta, p, n = results[key]
                summary.append(
                    dbc.Col(kpi_card(
                        f"{label} {significance_stars(p)}",
                        f"{beta:.5f}",
                        f"R²={r2:.3f}  n={n}",
                        C["green"] if p < 0.05 else C["muted"],
                    ), lg=2, md=4)
                )

        return (fig_ts_cpi, fig_ts_gdp, fig_ts_indpro,
                fig_sc_cpi, fig_sc_indpro,
                fig_wind_cpi, fig_solar_cpi,
                fig_gen_indpro, fig_gen_gdp,
                summary)
