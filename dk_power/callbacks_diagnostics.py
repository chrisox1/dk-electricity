"""
OLS Diagnostics tab: comprehensive regression diagnostic plots and tests.

Covers the 6 classical OLS assumptions:
  1. Linearity (residuals vs fitted, partial regression)
  2. Independence of errors (Durbin–Watson)
  3. Homoscedasticity (residuals vs fitted, Breusch–Pagan, Scale-Location)
  4. Normality of errors (Q–Q plot, histogram, Shapiro–Wilk)
  5. Multicollinearity (VIF, correlation heatmap)
  6. Influential observations (Cook's distance, leverage)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, html
from sklearn.linear_model import LinearRegression

from .db import get_conn
from .loaders import load_gas_prices, load_prices, load_production, load_temperature
from .theme import BASE, C, kpi_card, no_data, rgba

try:
    import statsmodels.api as sm
    from scipy import stats as sp_stats
    HAS_SM = True
except ImportError:
    HAS_SM = False


# ── Feature / data prep (shared with analytics) ────────────────────────────

ALL_BASE_FEATURES = [
    "total_wind", "solar_mw", "gas_price_eur", "consumption_mw",
    "net_imports", "temp_avg_c", "de_price_eur", "no_price_eur", "se_price_eur",
]

FEAT_LABELS = {
    "total_wind": "Wind", "solar_mw": "Solar", "gas_price_eur": "Gas",
    "consumption_mw": "Demand", "net_imports": "Imports", "temp_avg_c": "Temp",
    "de_price_eur": "DE Price", "no_price_eur": "NO Price", "se_price_eur": "SE Price",
    "is_peak": "Peak", "is_weekend": "Weekend",
}


def _prepare_data(pdf, qdf, area, freq, days):
    """Reuse analytics data prep logic — returns (merged_df, features, X, y, ols_results)."""
    from .callbacks_analytics import _prepare_merged_data, _feature_cols

    merged = _prepare_merged_data(pdf, qdf, area, freq, days)
    if merged is None or len(merged) < 20:
        return None

    features = _feature_cols(freq)
    features = [f for f in features if f in merged.columns and merged[f].notna().sum() > 10 and merged[f].std() > 0]
    if len(features) < 2:
        return None

    X = merged[features].values
    y = merged["price_dkk_kwh"].values

    ols = None
    if HAS_SM:
        try:
            ols = sm.OLS(y, sm.add_constant(X)).fit()
        except Exception:
            pass

    sklearn_model = LinearRegression().fit(X, y)
    y_pred = sklearn_model.predict(X)
    residuals = y - y_pred

    return {
        "df": merged, "features": features, "X": X, "y": y,
        "y_pred": y_pred, "residuals": residuals,
        "model": sklearn_model, "ols": ols,
    }


# ── Register callbacks ──────────────────────────────────────────────────────

def register_diagnostics_callbacks(app) -> None:

    @app.callback(
        Output("diag-resid-fitted", "figure"),
        Output("diag-qq", "figure"),
        Output("diag-resid-hist", "figure"),
        Output("diag-scale-loc", "figure"),
        Output("diag-cooks", "figure"),
        Output("diag-leverage", "figure"),
        Output("diag-corr-heatmap", "figure"),
        Output("diag-partial-wind", "figure"),
        Output("diag-partial-solar", "figure"),
        Output("diag-partial-gas", "figure"),
        Output("diag-summary", "children"),
        Input("diag-refresh", "n_clicks"),
        Input("diag-window", "value"),
        Input("diag-area", "value"),
        Input("diag-freq", "value"),
        prevent_initial_call=False,
    )
    def update_diagnostics(_, days_str, area, freq):
        days = float(days_str) if days_str else 90.0
        area = area or "DK1"
        empty = no_data(300, "Insufficient data — need ≥20 observations")

        if days > 365 and freq == "hourly":
            freq = "daily"

        pdf = load_prices(days)
        qdf = load_production(days)

        if pdf.empty or qdf.empty:
            return (empty,) * 10 + ([],)

        result = _prepare_data(pdf, qdf, area, freq, days)
        if result is None:
            return (empty,) * 10 + ([],)

        df = result["df"]
        features = result["features"]
        X, y = result["X"], result["y"]
        y_pred = result["y_pred"]
        resid = result["residuals"]
        ols = result["ols"]
        model = result["model"]

        fig_rf = _resid_vs_fitted(y_pred, resid, area)
        fig_qq = _qq_plot(resid, area)
        fig_hist = _resid_histogram(resid, area)
        fig_sl = _scale_location(y_pred, resid, area)
        fig_cook = _cooks_distance(ols, X, y, area)
        fig_lev = _leverage_plot(ols, X, y, resid, area)
        fig_corr = _correlation_heatmap(df, features)
        fig_pw = _partial_regression(df, "total_wind", features, area)
        fig_ps = _partial_regression(df, "solar_mw", features, area)
        fig_pg = _partial_regression(df, "gas_price_eur", features, area)
        summary = _build_test_cards(resid, y_pred, ols, df, features, area, freq)

        return (fig_rf, fig_qq, fig_hist, fig_sl, fig_cook, fig_lev,
                fig_corr, fig_pw, fig_ps, fig_pg, summary)


# ── 1. Linearity: Residuals vs Fitted ──────────────────────────────────────

def _resid_vs_fitted(y_pred, resid, area):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=resid, mode="markers",
        marker=dict(size=3, color=C["dk1"], opacity=0.4),
        name="Residuals",
    ))
    # LOWESS smoothing line
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smooth = lowess(resid, y_pred, frac=0.3)
        fig.add_trace(go.Scatter(
            x=smooth[:, 0], y=smooth[:, 1], mode="lines",
            line=dict(color=C["red"], width=2), name="LOWESS",
        ))
    except ImportError:
        pass
    fig.add_hline(y=0, line_dash="dot", line_color=C["muted"], line_width=1)
    fig.update_layout(
        **BASE, title=f"{area} Residuals vs Fitted (Linearity Check)",
        height=320, xaxis_title="Fitted Values (DKK/kWh)", yaxis_title="Residuals",
    )
    return fig


# ── 4. Normality: Q–Q Plot ─────────────────────────────────────────────────

def _qq_plot(resid, area):
    sorted_resid = np.sort(resid)
    n = len(sorted_resid)
    theoretical = np.array([sp_stats.norm.ppf((i + 0.5) / n) for i in range(n)]) if HAS_SM else np.linspace(-3, 3, n)
    std_resid = (sorted_resid - sorted_resid.mean()) / (sorted_resid.std() or 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theoretical, y=std_resid, mode="markers",
        marker=dict(size=3, color=C["dk2"], opacity=0.5),
        name="Residuals",
    ))
    rng = [min(theoretical.min(), std_resid.min()), max(theoretical.max(), std_resid.max())]
    fig.add_trace(go.Scatter(
        x=rng, y=rng, mode="lines",
        line=dict(color=C["red"], width=1.5, dash="dash"), name="Normal",
    ))
    fig.update_layout(
        **BASE, title=f"{area} Q–Q Plot (Normality Check)",
        height=320, xaxis_title="Theoretical Quantiles", yaxis_title="Standardized Residuals",
    )
    return fig


# ── 4. Normality: Histogram ────────────────────────────────────────────────

def _resid_histogram(resid, area):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=resid, nbinsx=50,
        marker_color=rgba(C["dk1"], 0.6),
        name="Residuals",
    ))
    # Normal curve overlay
    x_range = np.linspace(resid.min(), resid.max(), 200)
    mu, sigma = resid.mean(), resid.std()
    if sigma > 0:
        normal_curve = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
        # Scale to match histogram
        bin_width = (resid.max() - resid.min()) / 50
        fig.add_trace(go.Scatter(
            x=x_range, y=normal_curve * len(resid) * bin_width,
            mode="lines", line=dict(color=C["red"], width=2), name="Normal Fit",
        ))
    fig.update_layout(
        **BASE, title=f"{area} Residual Distribution (Normality)",
        height=320, xaxis_title="Residual", yaxis_title="Count",
    )
    return fig


# ── 3. Homoscedasticity: Scale-Location ────────────────────────────────────

def _scale_location(y_pred, resid, area):
    std_resid = np.sqrt(np.abs(resid / (resid.std() or 1)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=std_resid, mode="markers",
        marker=dict(size=3, color=C["solar"], opacity=0.4),
        name="√|Standardized Residuals|",
    ))
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smooth = lowess(std_resid, y_pred, frac=0.3)
        fig.add_trace(go.Scatter(
            x=smooth[:, 0], y=smooth[:, 1], mode="lines",
            line=dict(color=C["red"], width=2), name="LOWESS",
        ))
    except ImportError:
        pass
    fig.update_layout(
        **BASE, title=f"{area} Scale-Location (Homoscedasticity)",
        height=320, xaxis_title="Fitted Values", yaxis_title="√|Std. Residuals|",
    )
    return fig


# ── 6. Influential: Cook's Distance ────────────────────────────────────────

def _cooks_distance(ols, X, y, area):
    if ols is None:
        return no_data(320, "Cook's Distance requires statsmodels")
    try:
        influence = ols.get_influence()
        cooks_d = influence.cooks_distance[0]
        n = len(cooks_d)
        threshold = 4 / n

        colors = [C["red"] if c > threshold else C["dk1"] for c in cooks_d]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(n)), y=cooks_d,
            marker_color=colors, name="Cook's D",
        ))
        fig.add_hline(y=threshold, line_dash="dot", line_color=C["red"],
                       annotation_text=f"4/n = {threshold:.4f}")
        n_influential = sum(1 for c in cooks_d if c > threshold)
        fig.update_layout(
            **BASE, title=f"{area} Cook's Distance ({n_influential} influential of {n})",
            height=320, xaxis_title="Observation", yaxis_title="Cook's Distance",
        )
        return fig
    except Exception:
        return no_data(320, "Cook's Distance calculation failed")


# ── 6. Influential: Leverage vs Residuals ──────────────────────────────────

def _leverage_plot(ols, X, y, resid, area):
    if ols is None:
        return no_data(320, "Leverage plot requires statsmodels")
    try:
        influence = ols.get_influence()
        leverage = influence.hat_matrix_diag
        std_resid = resid / (resid.std() or 1)
        cooks_d = influence.cooks_distance[0]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=leverage, y=std_resid, mode="markers",
            marker=dict(
                size=np.clip(cooks_d * 1000, 3, 20),
                color=cooks_d, colorscale="YlOrRd",
                showscale=True, colorbar=dict(title="Cook's D", len=0.5),
                opacity=0.6,
            ),
            name="Observations",
            hovertemplate="Leverage: %{x:.4f}<br>Std Resid: %{y:.2f}<extra></extra>",
        ))
        fig.add_hline(y=0, line_dash="dot", line_color=C["muted"], line_width=1)
        p = X.shape[1] + 1  # predictors + intercept
        n = len(y)
        fig.add_vline(x=2 * p / n, line_dash="dot", line_color=C["red"],
                       annotation_text=f"2p/n = {2*p/n:.3f}")
        fig.update_layout(
            **BASE, title=f"{area} Leverage vs Std. Residuals (size=Cook's D)",
            height=320, xaxis_title="Leverage (hat value)", yaxis_title="Standardized Residuals",
        )
        return fig
    except Exception:
        return no_data(320, "Leverage calculation failed")


# ── 5. Multicollinearity: Correlation Heatmap ──────────────────────────────

def _correlation_heatmap(df, features):
    labels = [FEAT_LABELS.get(f, f) for f in features]
    corr_matrix = df[features].corr().values

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix, x=labels, y=labels,
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=np.round(corr_matrix, 2), texttemplate="%{text}",
        textfont=dict(size=9, color="white"),
        colorbar=dict(title="r", len=0.6),
    ))
    fig.update_layout(
        **BASE, title="Feature Correlation Matrix (Multicollinearity)",
        height=380,
    )
    fig.update_xaxes(tickangle=-45)
    return fig


# ── 1. Linearity: Partial Regression ───────────────────────────────────────

def _partial_regression(df, target_feat, features, area):
    if target_feat not in features or target_feat not in df.columns:
        return no_data(300, f"No {target_feat} data")
    label = FEAT_LABELS.get(target_feat, target_feat)

    other_feats = [f for f in features if f != target_feat]
    if not other_feats:
        return no_data(300, "Need ≥2 features")

    y = df["price_dkk_kwh"].values
    X_target = df[target_feat].values
    X_others = df[other_feats].values

    # Residualize y on others
    y_resid = y - LinearRegression().fit(X_others, y).predict(X_others)
    # Residualize target on others
    x_resid = X_target - LinearRegression().fit(X_others, X_target).predict(X_others)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_resid, y=y_resid, mode="markers",
        marker=dict(size=3, color=C["dk1"], opacity=0.3),
        name="Partial Residuals",
    ))
    # Fit line
    m = LinearRegression().fit(x_resid.reshape(-1, 1), y_resid)
    x_range = np.linspace(x_resid.min(), x_resid.max(), 100)
    fig.add_trace(go.Scatter(
        x=x_range, y=m.predict(x_range.reshape(-1, 1)),
        mode="lines", line=dict(color=C["red"], width=2),
        name=f"β={m.coef_[0]:.5f}",
    ))
    fig.update_layout(
        **BASE, title=f"{area} Partial Regression: {label}",
        height=300, xaxis_title=f"{label} | Others", yaxis_title="Price | Others",
    )
    return fig


# ── Test statistic summary cards ────────────────────────────────────────────

def _build_test_cards(resid, y_pred, ols, df, features, area, freq):
    cards = []

    n = len(resid)

    # ── Durbin–Watson (Independence) ────────────────────────────────────
    if HAS_SM and ols is not None:
        try:
            from statsmodels.stats.stattools import durbin_watson
            dw = durbin_watson(resid)
            dw_color = C["green"] if 1.5 < dw < 2.5 else C["red"]
            verdict = "OK (≈ no autocorrelation)" if 1.5 < dw < 2.5 else "⚠ Autocorrelation likely"
            cards.append(dbc.Col(kpi_card("Durbin–Watson", f"{dw:.3f}", verdict, dw_color), lg=2, md=3))
        except Exception:
            pass

    # ── Breusch–Pagan (Homoscedasticity) ────────────────────────────────
    if HAS_SM and ols is not None:
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            bp_stat, bp_p, _, _ = het_breuschpagan(ols.resid, ols.model.exog)
            bp_color = C["green"] if bp_p > 0.05 else C["red"]
            verdict = "OK (homoscedastic)" if bp_p > 0.05 else "⚠ Heteroscedastic"
            cards.append(dbc.Col(kpi_card("Breusch–Pagan", f"χ²={bp_stat:.2f}", f"p={bp_p:.4f} {verdict}", bp_color), lg=2, md=3))
        except Exception:
            pass

    # ── White's Test (Homoscedasticity) ─────────────────────────────────
    if HAS_SM and ols is not None:
        try:
            from statsmodels.stats.diagnostic import het_white
            w_stat, w_p, _, _ = het_white(ols.resid, ols.model.exog)
            w_color = C["green"] if w_p > 0.05 else C["red"]
            verdict = "OK" if w_p > 0.05 else "⚠ Heteroscedastic"
            cards.append(dbc.Col(kpi_card("White's Test", f"χ²={w_stat:.2f}", f"p={w_p:.4f} {verdict}", w_color), lg=2, md=3))
        except Exception:
            pass

    # ── Shapiro–Wilk (Normality) ────────────────────────────────────────
    if HAS_SM:
        try:
            sample = resid[:5000] if len(resid) > 5000 else resid
            sw_stat, sw_p = sp_stats.shapiro(sample)
            sw_color = C["green"] if sw_p > 0.05 else C["red"]
            verdict = "OK (normal)" if sw_p > 0.05 else "⚠ Non-normal"
            cards.append(dbc.Col(kpi_card("Shapiro–Wilk", f"W={sw_stat:.4f}", f"p={sw_p:.4f} {verdict}", sw_color), lg=2, md=3))
        except Exception:
            pass

    # ── Jarque–Bera (Normality) ─────────────────────────────────────────
    if HAS_SM and ols is not None:
        try:
            jb_stat, jb_p, skew, kurt = sm.stats.jarque_bera(ols.resid)
            jb_color = C["green"] if jb_p > 0.05 else C["red"]
            cards.append(dbc.Col(kpi_card("Jarque–Bera", f"JB={jb_stat:.1f}", f"p={jb_p:.4f}  skew={skew:.2f}  kurt={kurt:.2f}", jb_color), lg=2, md=3))
        except Exception:
            pass

    # ── Condition Number (Multicollinearity) ────────────────────────────
    if HAS_SM and ols is not None:
        try:
            cn = ols.condition_number
            cn_color = C["green"] if cn < 30 else (C["solar"] if cn < 100 else C["red"])
            verdict = "OK" if cn < 30 else ("⚠ Moderate" if cn < 100 else "⚠ Severe")
            cards.append(dbc.Col(kpi_card("Condition #", f"{cn:.1f}", verdict, cn_color), lg=2, md=3))
        except Exception:
            pass

    # ── VIF (Multicollinearity) ─────────────────────────────────────────
    if HAS_SM:
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            X_with_const = sm.add_constant(df[features].values)
            vif_vals = []
            for i in range(1, X_with_const.shape[1]):  # skip constant
                vif = variance_inflation_factor(X_with_const, i)
                vif_vals.append((features[i - 1], vif))

            max_vif = max(v for _, v in vif_vals) if vif_vals else 0
            vif_color = C["green"] if max_vif < 5 else (C["solar"] if max_vif < 10 else C["red"])
            vif_text = "  ".join(f"{FEAT_LABELS.get(f,f)}={v:.1f}" for f, v in vif_vals)
            cards.append(dbc.Col(kpi_card(
                "VIF (max)", f"{max_vif:.1f}", vif_text, vif_color
            ), lg=4, md=6))
        except Exception:
            pass

    # ── Influential observations ────────────────────────────────────────
    if HAS_SM and ols is not None:
        try:
            influence = ols.get_influence()
            cooks_d = influence.cooks_distance[0]
            threshold = 4 / n
            n_infl = sum(1 for c in cooks_d if c > threshold)
            pct = n_infl / n * 100
            infl_color = C["green"] if pct < 5 else (C["solar"] if pct < 10 else C["red"])
            cards.append(dbc.Col(kpi_card(
                "Influential Obs", f"{n_infl} / {n}", f"{pct:.1f}% (threshold=4/n)", infl_color
            ), lg=2, md=3))
        except Exception:
            pass

    # ── Model summary ───────────────────────────────────────────────────
    if HAS_SM and ols is not None:
        cards.append(dbc.Col(kpi_card("R²", f"{ols.rsquared:.4f}", f"Adj-R²={ols.rsquared_adj:.4f}", C["muted"]), lg=2, md=3))
        cards.append(dbc.Col(kpi_card("F-statistic", f"{ols.fvalue:.2f}", f"p={ols.f_pvalue:.2e}", C["muted"]), lg=2, md=3))
        cards.append(dbc.Col(kpi_card("AIC / BIC", f"{ols.aic:.0f}", f"BIC={ols.bic:.0f}", C["muted"]), lg=2, md=3))
        cards.append(dbc.Col(kpi_card("Sample", f"n={n}", f"{len(features)} features", C["muted"]), lg=2, md=3))

    return cards
