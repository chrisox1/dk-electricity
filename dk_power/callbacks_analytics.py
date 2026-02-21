"""
Analytics & Forecast tab callbacks: correlation, multi-variable regression,
individual regressions (wind / solar / gas / temp / demand), peak / weekend
effects, and day-ahead forecast display.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output
from sklearn.linear_model import LinearRegression

from .db import get_conn
from .loaders import load_gas_prices, load_prices, load_production, load_temperature
from .theme import BASE, C, kpi_card, no_data, rgba, significance_stars

# Try optional statsmodels
try:
    import statsmodels.api as sm

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def register_analytics_callbacks(app) -> None:
    """Attach all analytics-tab callbacks to *app*."""

    @app.callback(
        Output("corr-chart", "figure"),
        Output("scatter-chart", "figure"),
        Output("wind-regression", "figure"),
        Output("solar-regression", "figure"),
        Output("gas-regression", "figure"),
        Output("temp-regression", "figure"),
        Output("consumption-regression", "figure"),
        Output("green-pct-regression", "figure"),
        Output("peak-effect", "figure"),
        Output("weekend-effect", "figure"),
        Output("de-price-regression", "figure"),
        Output("no-price-regression", "figure"),
        Output("se-price-regression", "figure"),
        Output("forecast-chart", "figure"),
        Output("forecast-summary", "children"),
        Input("refresh-analytics", "n_clicks"),
        Input("analysis-window", "value"),
        Input("analysis-area", "value"),
        Input("analysis-freq", "value"),
        Input("reg-features", "value"),
        Input("tick", "n_intervals"),
        prevent_initial_call=False,
    )
    def update_analytics(_, days_str, area, freq, selected_features, __):
        days = float(days_str) if days_str else 30.0
        area = area or "DK1"
        selected_features = selected_features or []
        empty = no_data(350, "Insufficient data")

        # Force daily mode for large windows to avoid OOM
        if days > 365 and freq == "hourly":
            freq = "daily"

        pdf = load_prices(days)
        qdf = load_production(days)

        if pdf.empty or qdf.empty:
            return (empty,) * 14 + ([],)

        merged = _prepare_merged_data(pdf, qdf, area, freq, days)

        if merged is None or len(merged) < 10:
            return (empty,) * 14 + ([],)

        fig_corr = _build_correlation_chart(merged, area, freq)
        model, coeffs, p_values, stats, used_features = _fit_regression(merged, freq, selected_features)
        fig_scatter = _build_scatter_chart(merged, model, area, freq, stats, used_features)

        fig_wind = _univariate_regression(merged, "total_wind", "Wind (MW)", C["offshore"], area)
        fig_solar = _univariate_regression(merged, "solar_mw", "Solar (MW)", C["solar"], area)
        fig_gas = _univariate_regression(merged, "gas_price_eur", "Gas Price (EUR/MWh)", C["decentral"], area)
        fig_temp = _univariate_regression(merged, "temp_avg_c", "Temperature (°C)", C["central"], area)
        fig_consumption = _univariate_regression(merged, "consumption_mw", "Consumption (MW)", C["consume"], area)
        fig_green_pct = _univariate_regression(merged, "green_pct", "Green Energy (% of production)", C["green"], area)

        fig_peak = _build_box_effect(merged, "is_peak", ["Off-Peak", "Peak Hours"], area, "Peak") if freq == "hourly" else no_data(300, "Peak hours: only in hourly mode")
        fig_weekend = _build_box_effect(merged, "is_weekend", ["Weekday", "Weekend"], area, "Weekend") if freq == "hourly" else no_data(300, "Weekend: only in hourly mode")

        # Neighbour price regressions
        fig_de = _safe_univariate(merged, "de_price_eur", "DE-LU Price (EUR/MWh)", "#fb923c", area)
        fig_no = _safe_univariate(merged, "no_price_eur", "NO2 Price (EUR/MWh)", "#a78bfa", area)
        fig_se = _safe_univariate(merged, "se_price_eur", "SE Price (EUR/MWh)", "#34d399", area)

        fig_forecast = _build_forecast_chart(area)
        summary = _build_summary_cards(merged, stats, coeffs, p_values, freq, area, used_features)

        return (fig_corr, fig_scatter, fig_wind, fig_solar, fig_gas,
                fig_temp, fig_consumption, fig_green_pct, fig_peak, fig_weekend,
                fig_de, fig_no, fig_se,
                fig_forecast, summary)


# ── Data preparation ─────────────────────────────────────────────────────────

def _prepare_merged_data(pdf, qdf, area, freq, days):
    """Merge price + production + gas + temperature into one analysis DataFrame."""
    pdf_area = pdf[pdf["price_area"] == area].copy()
    qdf_area = qdf[qdf["price_area"] == area].copy()

    price_num = pdf_area.select_dtypes(include="number").columns.tolist()
    prod_num = qdf_area.select_dtypes(include="number").columns.tolist()

    if freq == "daily":
        pdf_area = pdf_area.set_index("ts_utc")[price_num].resample("1D").mean().reset_index()
        qdf_area = qdf_area.set_index("ts_utc")[prod_num].resample("1D").mean().reset_index()
    else:
        pdf_area = pdf_area.set_index("ts_utc")[price_num].resample("1h").mean().reset_index()
        qdf_area = qdf_area.set_index("ts_utc")[prod_num].resample("1h").mean().reset_index()

    merged = pd.merge(pdf_area, qdf_area, on="ts_utc", how="inner", suffixes=("", "_prod"))

    # Gas prices
    gas_df = load_gas_prices()
    if not gas_df.empty:
        gas_df["date"] = pd.to_datetime(gas_df["date"])
        if freq == "daily":
            merged["ts_utc"] = pd.to_datetime(merged["ts_utc"]).dt.tz_localize(None)
            gas_df["date"] = gas_df["date"].dt.tz_localize(None)
            merged["date"] = merged["ts_utc"].dt.date
            gas_df["date_only"] = gas_df["date"].dt.date
            merged = pd.merge(merged, gas_df[["date_only", "gas_price_eur"]],
                              left_on="date", right_on="date_only", how="left")
            merged["gas_price_eur"] = merged["gas_price_eur"].ffill().fillna(merged["gas_price_eur"].mean())
            merged.drop(columns=["date", "date_only"], inplace=True)
        else:
            gas_hourly = gas_df.set_index("date").resample("1h").ffill().reset_index()
            gas_hourly.rename(columns={"date": "ts_utc"}, inplace=True)
            gas_hourly["ts_utc"] = pd.to_datetime(gas_hourly["ts_utc"]).dt.tz_localize(None)
            merged["ts_utc"] = pd.to_datetime(merged["ts_utc"]).dt.tz_localize(None)
            merged = pd.merge(merged, gas_hourly, on="ts_utc", how="left")
            merged["gas_price_eur"] = merged["gas_price_eur"].ffill().fillna(merged["gas_price_eur"].mean())
    else:
        merged["gas_price_eur"] = 30.0

    # Temperature
    temp_df = load_temperature()
    if not temp_df.empty:
        temp_df["date"] = pd.to_datetime(temp_df["date"])
        if freq == "daily":
            temp_df["date"] = temp_df["date"].dt.tz_localize(None)
            merged["date_temp"] = merged["ts_utc"].dt.date
            temp_df["date_only"] = temp_df["date"].dt.date
            merged = pd.merge(merged, temp_df[["date_only", "temp_avg_c"]],
                              left_on="date_temp", right_on="date_only", how="left")
            merged.drop(columns=["date_temp", "date_only"], inplace=True, errors="ignore")
        else:
            temp_hourly = temp_df.set_index("date").resample("1h").ffill().reset_index()
            temp_hourly.rename(columns={"date": "ts_utc"}, inplace=True)
            temp_hourly["ts_utc"] = pd.to_datetime(temp_hourly["ts_utc"]).dt.tz_localize(None)
            merged = pd.merge(merged, temp_hourly, on="ts_utc", how="left")
        merged["temp_avg_c"] = merged["temp_avg_c"].ffill().fillna(merged["temp_avg_c"].mean())
    else:
        merged["temp_avg_c"] = 10.0

    # ── Neighbour electricity prices (DE-LU/DE, NO2, SE3/SE4) ────────────
    _nb_map = [
        (["DE-LU", "DE"], "de_price_eur"),
        (["NO2"], "no_price_eur"),
        (["SE3", "SE4"], "se_price_eur"),
    ]
    for nb_areas, col_name in _nb_map:
        if col_name in merged.columns:
            continue
        parts = []
        for nb_area in nb_areas:
            nb_pdf = pdf[pdf["price_area"] == nb_area].copy()
            if nb_pdf.empty:
                continue
            nb_num = nb_pdf.select_dtypes(include="number").columns.tolist()
            if freq == "daily":
                nb_pdf = nb_pdf.set_index("ts_utc")[nb_num].resample("1D").mean().reset_index()
            else:
                nb_pdf = nb_pdf.set_index("ts_utc")[nb_num].resample("1h").mean().reset_index()
            nb_pdf["ts_utc"] = pd.to_datetime(nb_pdf["ts_utc"]).dt.tz_localize(None)
            nb_pdf = nb_pdf[["ts_utc", "price_eur"]].rename(columns={"price_eur": col_name})
            parts.append(nb_pdf)
        if parts:
            # Merge all parts, averaging overlaps
            combined = pd.concat(parts).groupby("ts_utc").mean().reset_index()
            merged = pd.merge(merged, combined, on="ts_utc", how="left")
        else:
            merged[col_name] = float("nan")
    # Forward-fill and fill remaining NaN with means
    for col_name in ["de_price_eur", "no_price_eur", "se_price_eur"]:
        if col_name in merged.columns:
            merged[col_name] = merged[col_name].ffill().bfill()
            col_mean = merged[col_name].mean()
            merged[col_name] = merged[col_name].fillna(col_mean if pd.notna(col_mean) else 0)

    # Derived columns
    wind_cols = [c for c in ["offshore_mw", "onshore_mw"] if c in merged.columns]
    merged["total_wind"] = merged[wind_cols].fillna(0).sum(axis=1) if wind_cols else 0
    merged["solar_mw"] = merged.get("solar_mw", pd.Series(0, index=merged.index)).fillna(0)
    merged["renewable_mw"] = merged["total_wind"] + merged["solar_mw"]
    merged["price_dkk_kwh"] = merged["price_dkk"] * 0.001

    # Green energy as % of total production
    all_gen_cols = [c for c in ["offshore_mw", "onshore_mw", "solar_mw",
                                 "central_mw", "decentral_mw"] if c in merged.columns]
    total_gen = merged[all_gen_cols].fillna(0).clip(lower=0).sum(axis=1)
    merged["green_pct"] = (merged["renewable_mw"].clip(lower=0) / total_gen.replace(0, float("nan")) * 100).fillna(0)

    if "consumption_mw" not in merged.columns:
        merged["consumption_mw"] = 0
    merged["consumption_mw"] = merged["consumption_mw"].fillna(0)

    # Clean consumption outliers
    merged = merged[merged["consumption_mw"] > 100].copy()
    if merged["consumption_mw"].std() > 0:
        mu, sigma = merged["consumption_mw"].mean(), merged["consumption_mw"].std()
        merged = merged[(merged["consumption_mw"] >= mu - 3 * sigma)
                        & (merged["consumption_mw"] <= mu + 3 * sigma)].copy()

    exchange_cols = [c for c in ["ex_no", "ex_se", "ex_de"] if c in merged.columns]
    merged["net_imports"] = merged[exchange_cols].fillna(0).sum(axis=1) if exchange_cols else 0

    merged["ts_utc"] = pd.to_datetime(merged["ts_utc"])
    if freq == "hourly":
        merged["hour"] = merged["ts_utc"].dt.hour
        merged["is_peak"] = merged["hour"].isin(range(7, 22)).astype(int)
        merged["is_weekend"] = merged["ts_utc"].dt.dayofweek.isin([5, 6]).astype(int)
    else:
        merged["is_peak"] = 0
        merged["is_weekend"] = 0

    merged = merged[(merged["renewable_mw"] > 0) & (merged["price_dkk_kwh"] > 0)].copy()

    # ── ARX model features ──────────────────────────────────────────────
    # asinh transform for dependent variable (handles negative prices)
    merged["log_price"] = np.arcsinh(merged["price_dkk_kwh"])

    # Sort by time for correct lag computation
    merged = merged.sort_values("ts_utc").reset_index(drop=True)

    # Lag features (autoregressive component)
    if freq == "daily":
        merged["price_lag1"] = merged["log_price"].shift(1)
        merged["price_lag7"] = merged["log_price"].shift(7)
    else:
        merged["price_lag1"] = merged["log_price"].shift(1)
        merged["price_lag7"] = merged["log_price"].shift(24)  # ~1 day in hourly

    # Weekday dummies (Mon=0 … Sun=6 → 6 dummies, Sunday = reference)
    merged["ts_utc"] = pd.to_datetime(merged["ts_utc"])
    dow = merged["ts_utc"].dt.dayofweek  # 0=Mon, 6=Sun
    for d in range(6):  # Mon–Sat, Sun is reference
        merged[f"dow_{d}"] = (dow == d).astype(int)

    # Drop rows with NaN from lagging
    merged = merged.dropna(subset=["price_lag1", "price_lag7"]).copy()

    return merged if len(merged) >= 10 else None


# ── Regression ───────────────────────────────────────────────────────────────

# All possible base features and their labels
ALL_BASE_FEATURES = [
    "total_wind", "solar_mw", "gas_price_eur", "consumption_mw",
    "net_imports", "temp_avg_c", "de_price_eur", "no_price_eur", "se_price_eur",
    "price_lag1", "price_lag7",
    "dow_0", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5",
]

FEATURE_LABELS = {
    "total_wind": "Wind",
    "solar_mw": "Solar",
    "gas_price_eur": "Gas",
    "consumption_mw": "Demand",
    "net_imports": "Imports",
    "temp_avg_c": "Temp",
    "de_price_eur": "DE Price",
    "no_price_eur": "NO Price",
    "se_price_eur": "SE Price",
    "is_peak": "Peak",
    "is_weekend": "Weekend",
    "price_lag1": "Price Lag-1",
    "price_lag7": "Price Lag-7",
    "dow_0": "Mon", "dow_1": "Tue", "dow_2": "Wed",
    "dow_3": "Thu", "dow_4": "Fri", "dow_5": "Sat",
}

# Weekday dummy column names (always added/removed as a group)
DOW_COLS = [f"dow_{d}" for d in range(6)]


def _feature_cols(freq: str, selected: list[str] | None = None) -> list[str]:
    if selected:
        features = [f for f in selected if f in ALL_BASE_FEATURES]
        # If "weekday_dummies" meta-toggle is in selected, expand to all 6
        if "weekday_dummies" in selected:
            features = [f for f in features if f != "weekday_dummies"]
            features += DOW_COLS
    else:
        # Default: ARX model with lags + weekday dummies
        features = ["price_lag1", "price_lag7",
                     "total_wind", "solar_mw", "gas_price_eur",
                     "consumption_mw", "net_imports", "temp_avg_c"]
        features += DOW_COLS
    if freq == "hourly":
        features += ["is_peak", "is_weekend"]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for f in features:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def _fit_regression(df, freq, selected_features=None):
    features = _feature_cols(freq, selected_features)
    # Only keep features that actually exist and have variance in the data
    features = [f for f in features if f in df.columns and df[f].notna().sum() > 10 and df[f].std() > 0]
    if not features:
        features = ["total_wind", "solar_mw"]  # fallback minimum

    # Standardize continuous features (z-scores) to fix condition number
    # Binary dummies (dow_*, is_peak, is_weekend) are NOT standardized
    BINARY_FEATS = set(DOW_COLS) | {"is_peak", "is_weekend"}
    X_raw = df[features].values.copy()
    means = np.zeros(len(features))
    stds = np.ones(len(features))
    for i, f in enumerate(features):
        if f not in BINARY_FEATS:
            m, s = X_raw[:, i].mean(), X_raw[:, i].std()
            if s > 0:
                means[i] = m
                stds[i] = s
                X_raw[:, i] = (X_raw[:, i] - m) / s

    X = X_raw
    # Use asinh-transformed price as dependent variable
    y = df["log_price"].values

    if HAS_STATSMODELS:
        X_const = sm.add_constant(X)
        # Newey-West HAC standard errors (robust to heteroskedasticity + autocorrelation)
        # maxlags ~ T^(1/3) is standard rule of thumb
        maxlags = max(1, int(len(y) ** (1 / 3)))
        ols = sm.OLS(y, X_const).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        stats = {
            "r2": ols.rsquared,
            "adj_r2": ols.rsquared_adj,
            "f_stat": ols.fvalue if hasattr(ols.fvalue, '__float__') else float(ols.fvalue),
            "f_pvalue": ols.f_pvalue if hasattr(ols.f_pvalue, '__float__') else float(ols.f_pvalue),
            "n": len(y),
            "aic": ols.aic,
            "bic": ols.bic,
            "nw_lags": maxlags,
        }
        coeffs = ols.params[1:]  # standardized coefficients
        p_values = ols.pvalues[1:]
    else:
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)
        stats = {"r2": r2, "adj_r2": r2, "f_stat": 0.0, "f_pvalue": 1.0,
                 "n": len(y), "aic": 0, "bic": 0, "nw_lags": 0}
        coeffs = model.coef_
        p_values = np.ones(len(coeffs))

    sklearn_model = LinearRegression().fit(X, y)
    # Store standardization params for scatter chart prediction
    sklearn_model._feat_means = means
    sklearn_model._feat_stds = stds
    return sklearn_model, coeffs, p_values, stats, features


# ── Chart builders ───────────────────────────────────────────────────────────

def _build_correlation_chart(df, area, freq):
    # Use ~15% of data as window, min 3, max 90 — and allow partial windows
    n = len(df)
    window = max(3, min(n // 7, 90))
    if n < 5:
        return no_data(350, "Not enough data for rolling correlation")
    df = df.copy()
    df["corr"] = df["renewable_mw"].rolling(window, min_periods=max(2, window // 2)).corr(df["price_dkk_kwh"])
    color = C["dk1"] if area == "DK1" else C["dk2"]
    unit = "days" if freq == "daily" else "hours"
    fig = go.Figure()
    # Correlation line — no fill
    fig.add_trace(go.Scatter(
        x=df["ts_utc"], y=df["corr"],
        name="Renewable ↔ Price", mode="lines",
        line=dict(color=color, width=1.5),
    ))
    # Zero reference
    fig.add_hline(y=0, line_dash="dot", line_color=C["muted"], line_width=1)
    # Annotation explaining the chart
    fig.add_annotation(
        text="< 0 = more renewables → lower prices",
        xref="paper", yref="paper", x=0.01, y=0.02,
        showarrow=False, font=dict(size=10, color=C["muted"]),
    )
    fig.update_layout(**BASE, title=f"{area} Rolling Correlation: Renewables vs Price ({window} {unit} window)",
                      height=350, yaxis_title="Pearson Correlation", yaxis_range=[-1, 1])
    return fig


def _build_scatter_chart(df, model, area, freq, stats, features):
    solar_pct = (df["solar_mw"] / df["renewable_mw"] * 100).fillna(0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["renewable_mw"], y=df["log_price"], mode="markers",
        marker=dict(size=4, color=solar_pct, colorscale="Viridis",
                    showscale=True, colorbar=dict(title="Solar %", len=0.5, x=1.12), opacity=0.5),
        name="Actual",
        customdata=np.column_stack((df["total_wind"], df["solar_mw"])),
        hovertemplate="Total: %{x:.0f} MW<br>Wind: %{customdata[0]:.0f} MW<br>Solar: %{customdata[1]:.0f} MW<br>asinh(Price): %{y:.3f}<extra></extra>",
    ))

    # Trend line
    ren_range = np.linspace(df["renewable_mw"].min(), df["renewable_mw"].max(), 100)
    avg_solar_ratio = df["solar_mw"].mean() / max(df["renewable_mw"].mean(), 1)
    means = {f: df[f].mean() for f in features}

    pred_rows = []
    for r in ren_range:
        row = {f: means[f] for f in features}
        if "total_wind" in row:
            row["total_wind"] = r * (1 - avg_solar_ratio)
        if "solar_mw" in row:
            row["solar_mw"] = r * avg_solar_ratio
        pred_rows.append([row[f] for f in features])

    # Standardize prediction rows using same params as training
    pred_arr = np.array(pred_rows)
    if hasattr(model, '_feat_means'):
        BINARY_FEATS = set(DOW_COLS) | {"is_peak", "is_weekend"}
        for i, f in enumerate(features):
            if f not in BINARY_FEATS and model._feat_stds[i] > 0:
                pred_arr[:, i] = (pred_arr[:, i] - model._feat_means[i]) / model._feat_stds[i]
    y_pred = model.predict(pred_arr)
    fig.add_trace(go.Scatter(
        x=ren_range, y=y_pred, mode="lines",
        line=dict(color=C["red"], width=2),
        name=f"Model (R²={stats['r2']:.3f})",
    ))
    nw_info = f", NW-HAC({stats.get('nw_lags', '?')})" if stats.get("nw_lags") else ""
    fig.update_layout(**BASE, title=f"{area} ARX {len(features)}-Var (R²={stats['r2']:.3f}, {freq}{nw_info})",
                      height=350, xaxis_title="Total Renewable Generation (MW)",
                      yaxis_title="asinh(Price)")
    return fig


def _univariate_regression(df, col, xlabel, color, area):
    X = df[[col]].values
    y = df["price_dkk_kwh"].values
    m = LinearRegression().fit(X, y)
    r2 = m.score(X, y)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[col], y=y, mode="markers",
        marker=dict(size=3, color=color, opacity=0.4),
        name="Actual",
        hovertemplate=f"{xlabel.split('(')[0].strip()}: %{{x:.1f}}<br>Price: %{{y:.3f}} DKK/kWh<extra></extra>",
    ))
    x_range = np.linspace(max(X.min(), 0.01), X.max(), 100)
    fig.add_trace(go.Scatter(
        x=x_range, y=m.predict(x_range.reshape(-1, 1)),
        mode="lines", line=dict(color=C["red"], width=2),
        name=f"R²={r2:.3f}",
    ))
    fig.update_layout(**BASE, title=f"{area} Price vs {xlabel.split('(')[0].strip()} (β={m.coef_[0]:.5f})",
                      height=300, xaxis_title=xlabel, yaxis_title="Price (DKK/kWh)")
    return fig


def _safe_univariate(df, col, xlabel, color, area):
    """Like _univariate_regression but handles missing/constant columns gracefully."""
    if col not in df.columns or df[col].isna().all() or df[col].std() == 0:
        return no_data(300, f"No {xlabel.split('(')[0].strip()} data")
    valid = df.dropna(subset=[col, "price_dkk_kwh"])
    if len(valid) < 10:
        return no_data(300, f"Too few {xlabel.split('(')[0].strip()} data points")
    return _univariate_regression(valid, col, xlabel, color, area)


def _build_box_effect(df, col, labels, area, effect_name):
    fig = go.Figure()
    group0 = df[df[col] == 0]["price_dkk_kwh"]
    group1 = df[df[col] == 1]["price_dkk_kwh"]
    fig.add_trace(go.Box(y=group0, name=labels[0], marker_color=C["muted"]))
    fig.add_trace(go.Box(y=group1, name=labels[1], marker_color=C["central"]))
    delta = group1.mean() - group0.mean()
    sign = "+" if delta >= 0 else ""
    fig.update_layout(**BASE, title=f"{area} {effect_name}: {sign}{delta:.3f} DKK/kWh",
                      height=300, yaxis_title="Price (DKK/kWh)", showlegend=False)
    return fig


def _build_forecast_chart(area):
    now_utc = datetime.now(timezone.utc)
    db = get_conn()
    future = pd.read_sql(
        "SELECT ts_utc, price_area, price_eur, price_dkk FROM spot_prices"
        " WHERE price_area = ? ORDER BY ts_utc DESC LIMIT 200",
        db, params=(area,), parse_dates=["ts_utc"],
    )
    db.close()

    if not future.empty and future["ts_utc"].dt.tz is None:
        future["ts_utc"] = pd.to_datetime(future["ts_utc"], utc=True)

    historical_cutoff = now_utc - timedelta(days=3)
    forecast_only = future[future["ts_utc"] > now_utc].copy()
    recent = future[(future["ts_utc"] <= now_utc) & (future["ts_utc"] > historical_cutoff)].copy()

    for df in [forecast_only, recent]:
        df["price_dkk_kwh"] = df["price_dkk"] * 0.001
    forecast_only = forecast_only.sort_values("ts_utc")
    recent = recent.sort_values("ts_utc")

    fig = go.Figure()
    color = C["dk1"] if area == "DK1" else C["dk2"]
    if not recent.empty:
        fig.add_trace(go.Scatter(
            x=recent["ts_utc"], y=recent["price_dkk_kwh"],
            mode="lines", name="Historical Price",
            line=dict(color=color, width=1.5),
        ))
    if not forecast_only.empty:
        fig.add_trace(go.Scatter(
            x=forecast_only["ts_utc"], y=forecast_only["price_dkk_kwh"],
            mode="lines+markers", name="Day-Ahead (Published)",
            line=dict(color=C["green"], width=2, dash="dash"),
            marker=dict(size=4),
        ))
    fig.add_shape(type="line",
                  x0=now_utc.replace(tzinfo=None), x1=now_utc.replace(tzinfo=None),
                  y0=0, y1=1, yref="paper",
                  line=dict(color=C["muted"], width=2, dash="dot"), layer="below")
    fig.update_layout(**BASE, title=f"{area} Day-Ahead Prices (Nord Pool Published Forecast)",
                      height=350, yaxis_title="Spot Price (DKK/kWh)")
    return fig


def _build_summary_cards(df, stats, coeffs, p_values, freq, area, features=None):
    now_utc = datetime.now(timezone.utc)
    db = get_conn()
    future = pd.read_sql(
        "SELECT ts_utc, price_dkk FROM spot_prices WHERE price_area=? ORDER BY ts_utc DESC LIMIT 200",
        db, params=(area,), parse_dates=["ts_utc"],
    )
    db.close()
    if not future.empty and future["ts_utc"].dt.tz is None:
        future["ts_utc"] = pd.to_datetime(future["ts_utc"], utc=True)

    forecast_only = future[future["ts_utc"] > now_utc].copy()
    forecast_only["price_dkk_kwh"] = forecast_only["price_dkk"] * 0.001

    c24 = now_utc + timedelta(hours=24)
    c48 = now_utc + timedelta(hours=48)
    n24 = forecast_only[forecast_only["ts_utc"] <= c24]
    n48 = forecast_only[forecast_only["ts_utc"] <= c48]
    avg24 = n24["price_dkk_kwh"].mean() if not n24.empty else 0
    avg48 = n48["price_dkk_kwh"].mean() if not n48.empty else 0

    # Named coefficient/p-value extraction using actual features used
    if features is None:
        features = _feature_cols(freq)
    c = {n: (coeffs[i] if i < len(coeffs) else 0) for i, n in enumerate(features)}
    p = {n: (p_values[i] if i < len(p_values) else 1.0) for i, n in enumerate(features)}

    def _kpi(label, coef_key, positive_good=False):
        if coef_key not in c:
            return dbc.Col(kpi_card(f"{label} (off)", "—", "", C["muted"]), lg=2, md=2)
        cv, pv = c[coef_key], p[coef_key]
        if positive_good:
            clr = C["green"] if cv > 0 and pv < 0.05 else C["muted"]
        else:
            clr = C["green"] if cv < 0 and pv < 0.05 else (C["green"] if pv < 0.05 else C["muted"])
        return dbc.Col(kpi_card(
            f"{label} {significance_stars(pv)}", f"{cv:.5f}", f"p={pv:.3f}", clr,
        ), lg=2, md=2)

    nw_lags = stats.get("nw_lags", 0)
    summary = [
        dbc.Col(kpi_card("Next 24h Avg", f"{avg24:.3f}", "DKK/kWh",
                         C["green"] if avg24 < 0.80 else C["red"]), lg=2, md=4),
        dbc.Col(kpi_card("Next 48h Avg", f"{avg48:.3f}", "DKK/kWh",
                         C["green"] if avg48 < 0.90 else C["red"]), lg=2, md=4),
        dbc.Col(kpi_card("R² / Adj-R²", f"{stats['r2']:.3f} / {stats['adj_r2']:.3f}",
                         "dep var: asinh(price)",
                         C["green"] if stats["r2"] > 0.5 else C["muted"]), lg=2, md=4),
        dbc.Col(kpi_card("F-statistic", f"{stats['f_stat']:.1f}", f"p={stats['f_pvalue']:.3e}",
                         C["green"] if stats["f_pvalue"] < 0.05 else C["red"]), lg=2, md=4),
        dbc.Col(kpi_card("Sample", f"n={stats['n']}", f"NW-HAC({nw_lags})" if nw_lags else "",
                         C["muted"]), lg=2, md=4),
    ]
    if stats.get("aic"):
        summary.append(dbc.Col(kpi_card("AIC / BIC",
            f"{stats['aic']:.0f} / {stats['bic']:.0f}", "", C["muted"]), lg=2, md=4))

    # Add a KPI card for each active feature (skip weekday dummies — show as group)
    dow_feats = [f for f in features if f.startswith("dow_")]
    non_dow = [f for f in features if not f.startswith("dow_")]
    for feat in non_dow:
        label = FEATURE_LABELS.get(feat, feat)
        positive_good = feat in ("gas_price_eur", "de_price_eur", "no_price_eur",
                                  "se_price_eur", "price_lag1", "price_lag7")
        summary.append(_kpi(label, feat, positive_good=positive_good))

    # Weekday dummies as one summary card
    if dow_feats:
        dow_info = []
        for d in dow_feats:
            if d in c:
                lbl = FEATURE_LABELS.get(d, d)
                dow_info.append(f"{lbl}={c[d]:.4f}")
        any_sig = any(p.get(d, 1) < 0.05 for d in dow_feats)
        dow_clr = C["green"] if any_sig else C["muted"]
        summary.append(dbc.Col(kpi_card(
            "Weekday FE", "6 dummies", "  ".join(dow_info), dow_clr
        ), lg=4, md=6))

    return summary
