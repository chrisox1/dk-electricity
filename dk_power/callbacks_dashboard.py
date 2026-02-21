"""
Dashboard tab callbacks: spot prices, production mix, exchange, energy mix,
self-sufficiency, and KPI cards.
"""

from __future__ import annotations

import hashlib
import threading
import time
from datetime import datetime, timezone

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output
from plotly.subplots import make_subplots

from .loaders import apply_smoothing, downsample, load_prices, load_production
from .theme import BASE, C, kpi_card, no_data, rgba
import pandas as pd

# ── Figure cache ─────────────────────────────────────────────────────────────
_fig_cache: dict[str, tuple[float, tuple]] = {}
_fig_lock = threading.Lock()
_FIG_TTL = 45  # seconds; covers most tick intervals


def _fig_cache_key(days, area_sel, ccy, smooth):
    return f"{days}:{area_sel}:{ccy}:{smooth}"


def register_dashboard_callbacks(app) -> None:
    """Attach all dashboard-tab callbacks to *app*."""

    @app.callback(
        Output("dashboard-page", "style"),
        Output("analytics-page", "style"),
        Output("macro-page", "style"),
        Output("diagnostics-page", "style"),
        Input("tabs", "active_tab"),
    )
    def switch_tab(active_tab):
        hide = {"display": "none"}
        show = {"display": "block"}
        if active_tab == "tab-analytics":
            return hide, show, hide, hide
        elif active_tab == "tab-macro":
            return hide, hide, show, hide
        elif active_tab == "tab-diagnostics":
            return hide, hide, hide, show
        return show, hide, hide, hide

    @app.callback(
        Output("price-chart", "figure"),
        Output("prod-chart", "figure"),
        Output("exch-chart", "figure"),
        Output("mix-chart", "figure"),
        Output("gap-chart", "figure"),
        Output("green-pct-chart", "figure"),
        Output("kpi-row", "children"),
        Output("price-stats", "children"),
        Output("ts-lbl", "children"),
        Input("tick", "n_intervals"),
        Input("tw", "value"),
        Input("area", "value"),
        Input("ccy", "value"),
        Input("smooth", "value"),
    )
    def update(_, days, area_sel, ccy, smooth):
        now_str = datetime.now(timezone.utc).strftime("Updated %H:%M UTC")
        try:
            days = float(days) if days else 7.0
        except (TypeError, ValueError):
            days = 7.0
        area_sel = area_sel or "both"
        ccy = ccy or "eur"

        # Check figure cache (skip expensive chart rebuild if params unchanged)
        cache_key = _fig_cache_key(days, area_sel, ccy, smooth)
        with _fig_lock:
            cached = _fig_cache.get(cache_key)
            if cached and (time.monotonic() - cached[0]) < _FIG_TTL:
                # Return cached figures with updated timestamp
                return cached[1][:-1] + (now_str,)

        areas = ["DK1", "DK2"] if area_sel == "both" else [area_sel]
        pcol = "price_eur" if ccy == "eur" else "price_dkk"
        clabel = "EUR/MWh" if ccy == "eur" else "DKK/kWh"
        price_scale = 1.0 if ccy == "eur" else 0.001

        pdf = load_prices(days)
        qdf = load_production(days)

        # Normalise production frequency
        if not qdf.empty and "ts_utc" in qdf.columns:
            prod_freq = "5min" if days <= 1 else "1h"
            numeric_cols = qdf.select_dtypes(include="number").columns.tolist()
            parts = []
            for area, grp in qdf.groupby("price_area"):
                resampled = (
                    grp.set_index("ts_utc")[numeric_cols]
                    .resample(prod_freq)
                    .mean()
                    .reset_index()
                )
                resampled["price_area"] = area
                parts.append(resampled)
            if parts:
                qdf = pd.concat(parts, ignore_index=True)

        fig_p = _build_price_chart(pdf, areas, pcol, clabel, price_scale, smooth, days)
        p_stats = _compute_price_stats(pdf, areas, pcol, price_scale, smooth, days)
        fig_q = _build_production_chart(qdf, areas, smooth, days)
        fig_e = _build_exchange_chart(qdf, areas)
        fig_mix, fig_gap = _build_mix_charts(qdf, areas, smooth, days)
        fig_green = _build_green_pct_chart(qdf, areas, smooth, days)

        kpis = _build_kpis(p_stats, price_scale, clabel, qdf, areas)
        stats = _build_price_stats_row(p_stats, price_scale, clabel)

        result = (fig_p, fig_q, fig_e, fig_mix, fig_gap, fig_green, kpis, stats, now_str)
        with _fig_lock:
            _fig_cache[cache_key] = (time.monotonic(), result)
        return result


# ── Chart builders (pure functions) ──────────────────────────────────────────

def _build_price_chart(pdf, areas, pcol, clabel, price_scale, smooth, days):
    if pdf.empty:
        return no_data(400)
    fig = go.Figure()
    any_data = False
    for a, color in [("DK1", C["dk1"]), ("DK2", C["dk2"])]:
        if a not in areas:
            continue
        sub = pdf[pdf["price_area"] == a].sort_values("ts_utc").copy()
        sub = sub.set_index("ts_utc")
        freq = "15min" if days <= 1 else "1h"
        sub = sub[[pcol]].resample(freq).mean().dropna().reset_index()
        sub = apply_smoothing(sub, "ts_utc", smooth, days)
        if days > 2:
            sub = downsample(sub)
        prices = (sub[pcol] * price_scale).dropna()
        if prices.empty:
            continue
        any_data = True
        fig.add_trace(go.Scatter(
            x=sub["ts_utc"], y=prices,
            name=a, mode="lines",
            line=dict(color=color, width=2),
            fill="tozeroy", fillcolor=rgba(color, 0.07),
        ))
    if not any_data:
        return no_data(400, "Price data loading…")
    fig.add_hline(y=0, line_color=rgba(C["muted"], 0.35), line_width=1, line_dash="dot")
    fig.update_layout(**BASE, title=f"Spot Price  ({clabel})", yaxis_title=clabel, height=400)
    return fig


def _compute_price_stats(pdf, areas, pcol, price_scale, smooth, days):
    stats = {}
    if pdf.empty:
        return stats
    for a, color in [("DK1", C["dk1"]), ("DK2", C["dk2"])]:
        if a not in areas:
            continue
        sub = pdf[pdf["price_area"] == a].sort_values("ts_utc").copy()
        sub = sub.set_index("ts_utc")
        freq = "15min" if days <= 1 else "1h"
        sub = sub[[pcol]].resample(freq).mean().dropna().reset_index()
        sub = apply_smoothing(sub, "ts_utc", smooth, days)
        if days > 2:
            sub = downsample(sub)
        prices = (sub[pcol] * price_scale).dropna()
        if prices.empty:
            continue
        stats[a] = {
            "now": prices.iloc[-1],
            "min": prices.min(),
            "max": prices.max(),
            "mean": prices.mean(),
            "color": color,
        }
    return stats


def _build_production_chart(qdf, areas, smooth, days):
    if qdf.empty:
        return no_data(300)
    valid_areas = [a for a in areas if a in qdf["price_area"].unique()]
    ncols = len(valid_areas)
    if ncols == 0:
        return no_data(300)
    if ncols == 2:
        fig = make_subplots(rows=1, cols=2, subplot_titles=valid_areas, shared_yaxes=True)
    else:
        fig = go.Figure()

    for ci, a in enumerate(valid_areas, start=1):
        d = qdf[qdf["price_area"] == a].sort_values("ts_utc").copy()
        d = apply_smoothing(d, "ts_utc", smooth, days)
        if days > 2:
            d = downsample(d)
        kw = {"row": 1, "col": ci} if ncols == 2 else {}

        gen_series = [
            ("offshore_mw", C["offshore"], "Offshore Wind"),
            ("onshore_mw", C["onshore"], "Onshore Wind"),
            ("solar_mw", C["solar"], "Solar"),
            ("central_mw", C["central"], "Central"),
            ("decentral_mw", C["decentral"], "Decentral"),
        ]
        for s_col, s_color, s_name in gen_series:
            if s_col not in d.columns:
                continue
            vals = d[s_col].fillna(0)
            if vals.astype(float).abs().sum() == 0:
                continue
            tr = go.Scatter(
                x=d["ts_utc"], y=vals,
                name=s_name, mode="lines",
                line=dict(color=s_color, width=0.5),
                fill="tonexty", fillcolor=rgba(s_color, 0.4),
                stackgroup=f"g{ci}",
                legendgroup=s_name, showlegend=(ci == 1),
            )
            fig.add_trace(tr, **kw) if ncols == 2 else fig.add_trace(tr)

        if "consumption_mw" in d.columns and d["consumption_mw"].notna().any():
            cons = d["consumption_mw"]
            if cons.notna().sum() > 0:
                tr2 = go.Scatter(
                    x=d["ts_utc"], y=cons,
                    name=f"{a} Consumption", mode="lines",
                    line=dict(color=rgba(C["consume"], 0.9), width=1.8, dash="dot"),
                    legendgroup=f"cons{a}", showlegend=True,
                    hovertemplate="%{y:.0f} MW<extra>Consumption</extra>",
                )
                fig.add_trace(tr2, **kw) if ncols == 2 else fig.add_trace(tr2)

    fig.update_layout(**BASE, title="Generation Mix (MW)", height=300, yaxis_title="MW")
    return fig


def _build_exchange_chart(qdf, areas):
    if qdf.empty:
        return no_data(300)
    has_any = False
    fig = go.Figure()
    for a, color in [("DK1", C["dk1"]), ("DK2", C["dk2"])]:
        if a not in areas:
            continue
        d = qdf[qdf["price_area"] == a].sort_values("ts_utc")
        d = downsample(d)
        for ex_col, partner, alpha in [("ex_no", "NO", 1.0), ("ex_se", "SE", 0.65), ("ex_de", "DE", 0.4)]:
            if ex_col not in d.columns:
                continue
            if d[ex_col].fillna(0).astype(float).abs().sum() == 0:
                continue
            has_any = True
            fig.add_trace(go.Scatter(
                x=d["ts_utc"], y=d[ex_col].fillna(0),
                name=f"{a}↔{partner}", mode="lines",
                line=dict(color=rgba(color, alpha), width=1),
            ))
    if not has_any:
        fig = no_data(300)
        fig.add_annotation(
            text="No exchange data for this period",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color=C["muted"], size=11),
        )
    else:
        fig.add_hline(y=0, line_color=rgba(C["muted"], 0.35), line_width=1, line_dash="dot")
        fig.update_layout(**BASE, title="Cross-border Exchange (MW)", height=300, yaxis_title="MW  (+export)")
    return fig


def _build_mix_charts(qdf, areas, smooth, days):
    if qdf.empty:
        return no_data(300), no_data(300)

    valid_areas = [a for a in areas if a in qdf["price_area"].unique()]
    ncols = len(valid_areas)

    if ncols == 2:
        fig_mix = make_subplots(rows=1, cols=2,
                                subplot_titles=[f"{a} Energy Mix %" for a in valid_areas],
                                shared_yaxes=True)
        fig_gap = make_subplots(rows=1, cols=2,
                                subplot_titles=[f"{a} Import Gap" for a in valid_areas],
                                shared_yaxes=True)
    elif ncols == 1:
        fig_mix = go.Figure()
        fig_gap = go.Figure()
    else:
        return no_data(300), no_data(300)

    gen_mix_series = [
        ("offshore_mw", C["offshore"], "Offshore Wind"),
        ("onshore_mw", C["onshore"], "Onshore Wind"),
        ("solar_mw", C["solar"], "Solar"),
        ("central_mw", C["central"], "Central"),
        ("decentral_mw", C["decentral"], "Decentral"),
    ]
    import_colors = {"ex_no": "#a78bfa", "ex_se": "#34d399", "ex_de": "#fb923c"}
    import_labels = {"ex_no": "Import NO", "ex_se": "Import SE", "ex_de": "Import DE"}

    any_mix = False
    any_gap = False

    for ci, a in enumerate(valid_areas, start=1):
        d = qdf[qdf["price_area"] == a].sort_values("ts_utc").copy()
        d = apply_smoothing(d, "ts_utc", smooth, days)
        if days > 2:
            d = downsample(d)
        kw = {"row": 1, "col": ci} if ncols == 2 else {}

        gen_cols = [c for c in ["offshore_mw", "onshore_mw", "solar_mw",
                                "central_mw", "decentral_mw"] if c in d.columns]
        d["_total_gen"] = d[gen_cols].clip(lower=0).fillna(0).sum(axis=1)

        ex_cols = [c for c in ["ex_no", "ex_se", "ex_de"] if c in d.columns]
        for ec in ex_cols:
            d[f"_imp_{ec}"] = (-d[ec]).clip(lower=0).fillna(0)
        d["_total_imp"] = sum(d[f"_imp_{ec}"] for ec in ex_cols) if ex_cols else 0
        d["_total_supply"] = d["_total_gen"] + d["_total_imp"]

        if d["_total_supply"].sum() == 0:
            continue

        # Mix %
        for s_col, s_color, s_name in gen_mix_series:
            if s_col not in d.columns:
                continue
            pct = (d[s_col].clip(lower=0).fillna(0) / d["_total_supply"].replace(0, float("nan")) * 100)
            if pct.fillna(0).abs().sum() == 0:
                continue
            any_mix = True
            tr = go.Scatter(
                x=d["ts_utc"], y=pct.round(1),
                name=s_name, mode="lines",
                line=dict(color=s_color, width=0.5),
                fill="tonexty", fillcolor=rgba(s_color, 0.5),
                stackgroup=f"mix{ci}",
                legendgroup=s_name, showlegend=(ci == 1),
                hovertemplate="%{y:.1f}%<extra>" + s_name + "</extra>",
            )
            fig_mix.add_trace(tr, **kw) if ncols == 2 else fig_mix.add_trace(tr)

        for ec in ex_cols:
            pct = (d[f"_imp_{ec}"] / d["_total_supply"].replace(0, float("nan")) * 100)
            if pct.fillna(0).abs().sum() == 0:
                continue
            any_mix = True
            tr = go.Scatter(
                x=d["ts_utc"], y=pct.round(1),
                name=import_labels[ec], mode="lines",
                line=dict(color=import_colors[ec], width=0.5),
                fill="tonexty", fillcolor=rgba(import_colors[ec], 0.4),
                stackgroup=f"mix{ci}",
                legendgroup=import_labels[ec], showlegend=(ci == 1),
                hovertemplate="%{y:.1f}%<extra>" + import_labels[ec] + "</extra>",
            )
            fig_mix.add_trace(tr, **kw) if ncols == 2 else fig_mix.add_trace(tr)

        # Self-sufficiency
        if "consumption_mw" in d.columns:
            cons = d["consumption_mw"].replace(0, float("nan"))
            suff = (d["_total_gen"] / cons * 100)
            if suff.notna().sum() > 0:
                any_gap = True
                color_suff = C["dk1"] if a == "DK1" else C["dk2"]
                tr_suff = go.Scatter(
                    x=d["ts_utc"], y=suff.round(1),
                    name=f"{a} Self-Sufficiency", mode="lines",
                    line=dict(color=color_suff, width=1.5),
                    fill="tozeroy", fillcolor=rgba(color_suff, 0.15),
                    hovertemplate="%{y:.1f}%<extra>" + a + "</extra>",
                )
                fig_gap.add_trace(tr_suff, **kw) if ncols == 2 else fig_gap.add_trace(tr_suff)

    if any_mix:
        fig_mix.update_layout(**{**BASE, "title": "Energy Mix  (%  of total supply)",
                                 "height": 300, "yaxis": {**BASE["yaxis"], "title": "%", "range": [0, 100]}})
    else:
        fig_mix = no_data(300, "No mix data")

    if any_gap:
        fig_gap.add_hline(y=100, line_color=rgba(C["green"], 0.5), line_dash="dot", line_width=1,
                          annotation_text="100% (self-sufficient)", annotation_position="right")
        fig_gap.update_layout(**{**BASE, "title": "Self-Sufficiency  (% of consumption met by local gen)",
                                 "height": 300, "yaxis": {**BASE["yaxis"], "title": "%", "range": [0, None]}})
    else:
        fig_gap = no_data(300, "No generation data")

    return fig_mix, fig_gap


def _build_green_pct_chart(qdf, areas, smooth, days):
    """Running green energy (wind + solar) as % of total MW production."""
    if qdf.empty:
        return no_data(300)

    fig = go.Figure()
    any_data = False

    for a, color in [("DK1", C["dk1"]), ("DK2", C["dk2"])]:
        if a not in areas:
            continue
        d = qdf[qdf["price_area"] == a].sort_values("ts_utc").copy()
        d = apply_smoothing(d, "ts_utc", smooth, days)
        if days > 2:
            d = downsample(d)

        green_cols = [c for c in ["offshore_mw", "onshore_mw", "solar_mw"] if c in d.columns]
        all_gen_cols = [c for c in ["offshore_mw", "onshore_mw", "solar_mw",
                                     "central_mw", "decentral_mw"] if c in d.columns]
        if not green_cols or not all_gen_cols:
            continue

        green_mw = d[green_cols].fillna(0).clip(lower=0).sum(axis=1)
        total_mw = d[all_gen_cols].fillna(0).clip(lower=0).sum(axis=1)
        pct = (green_mw / total_mw.replace(0, float("nan")) * 100)

        if pct.notna().sum() == 0:
            continue
        any_data = True

        fig.add_trace(go.Scatter(
            x=d["ts_utc"], y=pct.round(1),
            name=a, mode="lines",
            line=dict(color=color, width=1.5),
            fill="tozeroy", fillcolor=rgba(color, 0.08),
            hovertemplate="%{y:.1f}%<extra>" + a + "</extra>",
        ))

    if not any_data:
        return no_data(300, "No generation data")

    fig.add_hline(y=100, line_color=rgba(C["green"], 0.4), line_dash="dot", line_width=1)
    fig.update_layout(
        **BASE,
        title="Green Energy  (% of total MW production — wind + solar)",
        height=300,
        yaxis_title="%",
        yaxis_range=[0, None],
    )
    return fig


def _build_kpis(p_stats, price_scale, clabel, qdf, areas):
    kpis = []
    for a, st in p_stats.items():
        v = st["now"]
        if v is None:
            continue
        tlo, thi = (0.05, 0.20) if price_scale < 1 else (50, 200)
        color = C["green"] if v < tlo else (C["red"] if v > thi else st["color"])
        fmt = f"{v:.3f}" if price_scale < 1 else f"{v:.1f}"
        kpis.append(dbc.Col(kpi_card(f"{a} now", fmt, clabel, color, st["color"]), lg=2, md=4))

    if not qdf.empty:
        for a, color in [("DK1", C["dk1"]), ("DK2", C["dk2"])]:
            if a not in areas:
                continue
            d = qdf[qdf["price_area"] == a]
            if d.empty:
                continue
            last = d.sort_values("ts_utc").iloc[-1]
            gen_cols = ["offshore_mw", "onshore_mw", "solar_mw", "central_mw", "decentral_mw"]
            total = sum(float(last[c] or 0) for c in gen_cols if c in last.index and last[c])
            if total > 0:
                kpis.append(dbc.Col(kpi_card(f"{a} gen", f"{total:,.0f}", "MW", color), lg=2, md=4))
    return kpis


def _build_price_stats_row(p_stats, price_scale, clabel):
    stats = []
    for a, st in p_stats.items():
        col = st["color"]
        fmt = ".3f" if price_scale < 1 else ".1f"
        stats += [
            dbc.Col(kpi_card(f"{a} min", f"{st['min']:{fmt}}", clabel, C["green"], col), lg=2, md=4),
            dbc.Col(kpi_card(f"{a} avg", f"{st['mean']:{fmt}}", clabel, col, col), lg=2, md=4),
            dbc.Col(kpi_card(f"{a} max", f"{st['max']:{fmt}}", clabel, C["red"], col), lg=2, md=4),
        ]
    return stats
