"""Fetch and cache ETF daily returns (Eastmoney NAV / 日增长率)."""

from __future__ import annotations

import time
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
START_DATE = "2020-01-01"
END_DATE = "20260528"
# Eastmoney 日增长率口径可靠的货币 ETF
MONEY_GROWTH_CODES = frozenset({"511880"})
# 日增长率为 0、需用收盘价的 ETF
MONEY_CLOSE_CODES = frozenset({"511990"})


def _sina_symbol(exchange: str, code: str) -> str:
    return f"{exchange.lower()}{code}"


def _fetch_nav_em(code: str) -> pd.DataFrame:
    df = ak.fund_etf_fund_info_em(fund=code, start_date="20200101", end_date=END_DATE)
    df = df.rename(
        columns={
            "净值日期": "date",
            "单位净值": "nav",
            "日增长率": "growth_pct",
        }
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df["growth_pct"] = pd.to_numeric(df["growth_pct"], errors="coerce")
    return df[["date", "nav", "growth_pct"]]


def _load_cached_close(exchange: str, code: str) -> pd.DataFrame | None:
    path = CACHE_DIR / f"{exchange}_{code}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns={"close": "nav"})
    df["growth_pct"] = np.nan
    return df[["date", "nav", "growth_pct"]]


def _fetch_close_sina(exchange: str, code: str) -> pd.DataFrame:
    cached = _load_cached_close(exchange, code)
    if cached is not None:
        return cached
    sym = _sina_symbol(exchange, code)
    raw = ak.fund_etf_hist_sina(symbol=sym)
    df = raw.rename(columns={"date": "date", "close": "nav"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df["growth_pct"] = np.nan
    return df[["date", "nav", "growth_pct"]]


def _daily_returns_from_nav(df: pd.DataFrame, code: str) -> pd.Series:
    """Prefer 日增长率 (%); fallback to NAV/收盘价 pct_change."""
    g = df["growth_pct"] / 100.0
    nav_ret = df["nav"].pct_change()
    if code in MONEY_CLOSE_CODES:
        ret = nav_ret
    elif (g.abs() > 1e-8).sum() >= max(30, int(0.1 * len(g))):
        ret = g
    else:
        ret = nav_ret
    ret = ret.mask(ret.abs() > 0.05, np.nan).ffill().fillna(0.0)
    s = pd.Series(ret.values, index=df["date"], dtype=float)
    return s[s.index >= pd.Timestamp(START_DATE)]


def fetch_etf_returns(exchange: str, code: str, refresh: bool = False) -> pd.Series:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"ret_{code}.csv"
    if path.exists() and not refresh:
        df = pd.read_csv(path, parse_dates=["date"])
        return df.set_index("date")["ret"]

    if code in MONEY_CLOSE_CODES:
        df = _fetch_close_sina(exchange, code)
    else:
        for attempt in range(3):
            try:
                df = _fetch_nav_em(code)
                if df["nav"].notna().sum() < 30:
                    raise ValueError("insufficient NAV rows")
                break
            except Exception:  # noqa: BLE001
                time.sleep(2 * (attempt + 1))
        else:
            try:
                df = _fetch_close_sina(exchange, code)
            except Exception:
                cached = _load_cached_close(exchange, code)
                if cached is None:
                    raise
                df = cached

    ret = _daily_returns_from_nav(df, code)
    out = pd.DataFrame({"date": ret.index, "ret": ret.values})
    out.to_csv(path, index=False)
    return ret


def load_returns(codes: dict[str, tuple[str, str]], refresh: bool = False) -> pd.DataFrame:
    """codes: {name: (exchange, code)} -> daily return DataFrame."""
    series: dict[str, pd.Series] = {}
    for name, (ex, code) in codes.items():
        series[name] = fetch_etf_returns(ex, code, refresh=refresh)
    return pd.DataFrame(series).sort_index()


ETF_UNIVERSE: dict[str, tuple[str, str]] = {
    "511880": ("sh", "511880"),
    "511990": ("sh", "511990"),
    "511360": ("sh", "511360"),
    "511580": ("sh", "511580"),
    "511070": ("sh", "511070"),
    "511010": ("sh", "511010"),
    "511030": ("sh", "511030"),
}
