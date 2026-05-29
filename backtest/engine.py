"""Portfolio backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change()


def run_backtest(
    asset_rets: pd.DataFrame,
    target_weights: pd.DataFrame,
    cost_bps: float = 1.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Run backtest with daily target weights (rows aligned to asset_rets).
    cost_bps: one-way cost applied to sum(|Δw|) on each day.
    """
    assets = [c for c in target_weights.columns if c in asset_rets.columns]
    w = target_weights[assets].reindex(asset_rets.index).ffill().fillna(0.0)
    r = asset_rets[assets].fillna(0.0)

    port_ret = pd.Series(0.0, index=r.index)
    prev_w = w.iloc[0].values.astype(float)
    tc = cost_bps / 10000.0

    for i, dt in enumerate(r.index):
        w_t = w.loc[dt].values.astype(float)
        if i > 0:
            turnover = np.abs(w_t - prev_w).sum()
            cost = turnover * tc
        else:
            cost = np.abs(w_t).sum() * tc
        port_ret.loc[dt] = float(np.dot(prev_w, r.loc[dt].values)) - cost
        prev_w = w_t

    return port_ret.iloc[1:], w.loc[port_ret.index]


def align_period(
    prices: pd.DataFrame,
    required: list[str],
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    sub = prices[required].dropna(how="any")
    if sub.empty:
        raise ValueError(f"No overlapping data for {required}")
    return sub, sub.index[0], sub.index[-1]


def month_end_mask(index: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series(index=index, data=index)
    me = s.groupby([s.index.year, s.index.month]).transform("max")
    return pd.Series(index=index, data=index == me)


def week_end_mask(index: pd.DatetimeIndex) -> pd.Series:
    """Last trading day of each ISO week."""
    df = pd.DataFrame({"d": index}, index=index)
    wk = index.isocalendar().week.to_numpy()
    yr = index.isocalendar().year.to_numpy()
    key = list(zip(yr, wk))
    last: dict[tuple, pd.Timestamp] = {}
    for dt, k in zip(index, key):
        last[k] = dt
    return pd.Series(index=index, data=[dt == last[k] for dt, k in zip(index, key)])
