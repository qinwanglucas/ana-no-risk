"""Strategy weight schedules (README V2.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.engine import month_end_mask, week_end_mask

R1 = "511880"
R2_SHORT = "511360"
R2_RATE = "511580"
R2_CREDIT = "511070"
R2_5Y = "511010"
SIG_SPREAD = "511030"

VOL_TARGET = 0.025
E_MAX = 0.70
R2_INNER = {R2_RATE: 0.40, R2_SHORT: 0.35, R2_CREDIT: 0.25}


def _blank_weights(index: pd.DatetimeIndex, assets: list[str]) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=index, columns=assets)


def fixed_weights(
    index: pd.DatetimeIndex,
    weights: dict[str, float],
    assets: list[str],
) -> pd.DataFrame:
    w = _blank_weights(index, assets)
    for k, v in weights.items():
        if k in w.columns:
            w[k] = v
    return w


def s1_weights(index: pd.DatetimeIndex, assets: list[str], lite: bool = False) -> pd.DataFrame:
    if lite:
        tgt = {R1: 0.75, R2_RATE: 0.15, R2_CREDIT: 0.10}
    else:
        tgt = {R1: 0.70, R2_SHORT: 0.105, R2_RATE: 0.12, R2_CREDIT: 0.075}
    w = fixed_weights(index, tgt, assets)
    me = month_end_mask(index)
    # Hold constant between month-ends; rebalance on month-end close -> next day weights
    return w


def s2_weights(
    index: pd.DatetimeIndex,
    asset_rets: pd.DataFrame,
    assets: list[str],
) -> pd.DataFrame:
    pool = [R1, R2_RATE, R2_SHORT, R2_CREDIT, R2_5Y]
    pool = [p for p in pool if p in assets]
    vol = asset_rets[pool].rolling(60).std() * np.sqrt(252)
    inv = 1.0 / vol.replace(0, np.nan)
    w = _blank_weights(index, assets)
    biweekly = set(index[::10])
    current = None
    for dt in index:
        if dt in biweekly or current is None:
            row = inv.loc[dt]
            if row.isna().all():
                current = {R1: 1.0}
            else:
                raw = row.fillna(0)
                s = raw.sum()
                raw = raw / s if s > 0 else raw
                raw = raw.clip(upper=0.35)
                s = raw.sum()
                raw = raw / s if s > 0 else raw
                if R1 in raw.index and raw[R1] < 0.50:
                    rest = raw.drop(labels=[R1], errors="ignore")
                    raw[R1] = 0.50
                    rem = 0.50
                    if rest.sum() > 0:
                        raw[rest.index] = rem * rest / rest.sum()
                current = raw.to_dict()
        for k, v in current.items():
            w.at[dt, k] = v
    return w


def s3_weights(
    index: pd.DatetimeIndex,
    asset_rets: pd.DataFrame,
    assets: list[str],
) -> pd.DataFrame:
    inner_assets = [a for a in R2_INNER if a in assets]
    inner_w = np.array([R2_INNER[a] for a in inner_assets])
    inner_w = inner_w / inner_w.sum()

    probe = pd.Series(0.0, index=index)
    for i, a in enumerate(inner_assets):
        probe += inner_w[i] * asset_rets[a].fillna(0)
    probe = probe + (1 - inner_w.sum()) * 0  # full risk bucket
    vol20 = probe.rolling(20).std() * np.sqrt(252)

    w = _blank_weights(index, assets)
    weekly = set(index[week_end_mask(index)])
    e_val = E_MAX
    for dt in index:
        if dt in weekly or dt == index[0]:
            v = vol20.loc[dt]
            if pd.notna(v) and v > 0:
                e_val = min(E_MAX, VOL_TARGET / v)
            else:
                e_val = E_MAX
        w.at[dt, R1] = 1 - e_val
        for i, a in enumerate(inner_assets):
            w.at[dt, a] = e_val * inner_w[i]
    return w


def s4_weights(
    index: pd.DatetimeIndex,
    prices: pd.DataFrame,
    assets: list[str],
) -> pd.DataFrame:
    ma580 = prices[R2_RATE].rolling(60).mean()
    ma010 = prices[R2_5Y].rolling(60).mean()
    full = (prices[R2_RATE] > ma580) & (prices[R2_5Y] > ma010)
    w = _blank_weights(index, assets)
    for dt in index:
        if pd.isna(full.loc[dt]):
            tgt = {R1: 0.70, R2_RATE: 0.20, R2_5Y: 0.10}
        elif full.loc[dt]:
            tgt = {R1: 0.70, R2_RATE: 0.20, R2_5Y: 0.10}
        else:
            tgt = {R1: 0.85, R2_RATE: 0.10, R2_5Y: 0.05}
        for k, v in tgt.items():
            w.at[dt, k] = v
    return w


def s5_weights(
    index: pd.DatetimeIndex,
    prices: pd.DataFrame,
    assets: list[str],
) -> pd.DataFrame:
    r20 = prices[SIG_SPREAD] / prices[SIG_SPREAD].shift(20) - 1
    stress = r20 < -0.003
    w = _blank_weights(index, assets)
    week_last = week_end_mask(index)
    state = False
    for dt in index:
        if week_last.loc[dt] and pd.notna(stress.loc[dt]):
            state = bool(stress.loc[dt])
        if state:
            tgt = {R1: 0.60, R2_CREDIT: 0.10, R2_RATE: 0.30}
        else:
            tgt = {R1: 0.60, R2_CREDIT: 0.25, R2_RATE: 0.15}
        for k, v in tgt.items():
            w.at[dt, k] = v
    return w


def s6_weights(
    index: pd.DatetimeIndex,
    prices: pd.DataFrame,
    assets: list[str],
) -> pd.DataFrame:
    ma = prices[R2_5Y].rolling(60).mean()
    friendly = prices[R2_5Y] > ma
    w = _blank_weights(index, assets)
    for dt in index:
        if pd.isna(friendly.loc[dt]) or friendly.loc[dt]:
            tgt = {R2_SHORT: 0.50, R2_RATE: 0.30, R2_5Y: 0.20}
        else:
            tgt = {R2_SHORT: 0.55, R2_RATE: 0.35, R2_5Y: 0.10}
        for k, v in tgt.items():
            w.at[dt, k] = v
    return w


STRATEGIES = {
    "S1": {
        "fn": lambda idx, rets, prices, assets: s1_weights(idx, assets, lite=False),
        "required": [R1, R2_SHORT, R2_RATE, R2_CREDIT],
    },
    "S1-lite": {
        "fn": lambda idx, rets, prices, assets: s1_weights(idx, assets, lite=True),
        "required": [R1, R2_RATE, R2_CREDIT],
    },
    "S2": {
        "fn": lambda idx, rets, prices, assets: s2_weights(idx, rets, assets),
        "required": [R1, R2_RATE, R2_SHORT, R2_CREDIT, R2_5Y],
    },
    "S3": {
        "fn": lambda idx, rets, prices, assets: s3_weights(idx, rets, assets),
        "required": [R1, R2_RATE, R2_SHORT, R2_CREDIT],
    },
    "S4": {
        "fn": lambda idx, rets, prices, assets: s4_weights(idx, prices, assets),
        "required": [R1, R2_RATE, R2_5Y],
    },
    "S5": {
        "fn": lambda idx, rets, prices, assets: s5_weights(idx, prices, assets),
        "required": [R1, R2_RATE, R2_CREDIT, SIG_SPREAD],
    },
    "S6": {
        "fn": lambda idx, rets, prices, assets: s6_weights(idx, prices, assets),
        "required": [R2_SHORT, R2_RATE, R2_5Y],
    },
}

# Planned execution order
RUN_ORDER = ["S1", "S1-lite", "S3", "S2", "S4", "S5", "S6"]
