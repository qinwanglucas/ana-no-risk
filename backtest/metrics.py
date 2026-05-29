"""V2 evaluation metrics for backtest results."""

from __future__ import annotations

import numpy as np
import pandas as pd

RB_HIGH = 0.0286
RB_LOW = 0.0268
ER_HIGH_TARGET = 0.006

# Moderate risk budget
VOL_STAR = 0.03
MDD_STAR = -0.05
CVAR95_STAR = -0.015
P_LOSS_STAR = 0.40
T_REC_STAR = 8

LAMBDA1 = 0.10
LAMBDA2 = 0.15
EPS = 0.0005

CR_MDD_MIN = 0.20
CR_CVAR_MIN = 0.30
CR_VOL_MIN = 0.20


def daily_to_annual_return(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if len(r) == 0:
        return np.nan
    return (1 + r).prod() ** (252 / len(r)) - 1


def annualized_vol(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if len(r) < 2:
        return np.nan
    return r.std() * np.sqrt(252)


def max_drawdown(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if len(r) == 0:
        return np.nan
    wealth = (1 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1
    return dd.min()


def monthly_returns(daily_returns: pd.Series) -> pd.Series:
    r = daily_returns.dropna()
    if r.empty:
        return pd.Series(dtype=float)
    return r.resample("ME").apply(lambda x: (1 + x).prod() - 1)


def cvar95_monthly(monthly: pd.Series) -> float:
    if len(monthly) < 5:
        return np.nan
    q = monthly.quantile(0.05)
    tail = monthly[monthly <= q]
    return tail.mean() if len(tail) else np.nan


def max_recovery_months(monthly: pd.Series) -> int:
    if monthly.empty:
        return np.nan
    wealth = (1 + monthly).cumprod()
    peak = wealth.cummax()
    underwater = wealth < peak
    if not underwater.any():
        return 0
    max_run = 0
    run = 0
    for u in underwater:
        if u:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return int(max_run)


def bootstrap_er_high_p5(daily_returns: pd.Series, n_boot: int = 1000) -> float:
    m = monthly_returns(daily_returns)
    if len(m) < 12:
        return np.nan
    rng = np.random.default_rng(42)
    ann_excess = []
    for _ in range(n_boot):
        sample = m.sample(n=len(m), replace=True, random_state=rng.integers(1e9))
        n = len(sample)
        rs = (1 + sample).prod() ** (12 / n) - 1
        ann_excess.append(rs - RB_HIGH)
    return float(np.percentile(ann_excess, 5))


def evaluate_v2(
    daily_returns: pd.Series,
    oos_fraction: float = 0.30,
) -> dict:
    rs = daily_to_annual_return(daily_returns)
    er_high = rs - RB_HIGH
    vol = annualized_vol(daily_returns)
    mdd = max_drawdown(daily_returns)
    mret = monthly_returns(daily_returns)
    cvar_m = cvar95_monthly(mret)
    cvar_a = abs(cvar_m) * np.sqrt(12) if pd.notna(cvar_m) else np.nan
    p_loss = (mret < 0).mean() if len(mret) else np.nan
    t_rec = max_recovery_months(mret)

    cr_mdd = er_high / abs(mdd) if abs(mdd) >= EPS else np.nan
    cr_cvar = er_high / cvar_a if pd.notna(cvar_a) and cvar_a >= EPS and cvar_m < 0 else np.nan
    cr_vol = er_high / vol if pd.notna(vol) and vol >= EPS else np.nan
    u = np.nan
    if pd.notna(cvar_a) and pd.notna(mdd):
        u = er_high - LAMBDA1 * abs(mdd) - LAMBDA2 * (cvar_a if cvar_m < 0 else 0)

    n = len(daily_returns.dropna())
    split = int(n * (1 - oos_fraction))
    oos_ret = daily_returns.iloc[split:] if split > 0 else pd.Series(dtype=float)
    er_high_oos = daily_to_annual_return(oos_ret) - RB_HIGH if len(oos_ret) > 20 else np.nan
    er_p5 = bootstrap_er_high_p5(daily_returns)

    pass_return = er_high >= ER_HIGH_TARGET
    pass_risk = (
        pd.notna(vol)
        and vol <= VOL_STAR
        and pd.notna(mdd)
        and mdd >= MDD_STAR
        and (pd.isna(cvar_m) or cvar_m >= CVAR95_STAR)
        and (pd.isna(p_loss) or p_loss <= P_LOSS_STAR)
        and (pd.isna(t_rec) or t_rec <= T_REC_STAR)
    )
    pass_comp = (
        (pd.isna(cr_mdd) or cr_mdd >= CR_MDD_MIN)
        and (pd.isna(cr_cvar) or cr_cvar >= CR_CVAR_MIN)
        and (pd.notna(cr_vol) and cr_vol >= CR_VOL_MIN)
        and pd.notna(u)
        and u > 0
    )
    pass_robust = (
        pd.notna(er_high_oos)
        and er_high_oos > 0
        and pd.notna(er_p5)
        and er_p5 > 0
    )
    n_months = len(mret)
    formal_ok = n_months >= 24

    if not formal_ok:
        verdict = "观察（样本<24月）"
    elif pass_return and pass_risk and pass_comp and pass_robust:
        verdict = "值得"
    elif pass_return and pass_risk and not (pass_comp and pass_robust):
        verdict = "谨慎"
    elif not pass_risk:
        verdict = "不做"
    else:
        verdict = "谨慎"

    return {
        "Rs": rs,
        "ER_high": er_high,
        "Vol": vol,
        "MDD": mdd,
        "CVaR95_m": cvar_m,
        "CVaR95_a": cvar_a,
        "P_loss_m": p_loss,
        "T_rec": t_rec,
        "CR_MDD": cr_mdd,
        "CR_CVaR": cr_cvar,
        "CR_Vol": cr_vol,
        "U": u,
        "ER_high_oos": er_high_oos,
        "ER_high_p5": er_p5,
        "n_days": n,
        "n_months": n_months,
        "verdict": verdict,
        "pass_return": pass_return,
        "pass_risk": pass_risk,
        "pass_comp": pass_comp,
        "pass_robust": pass_robust,
    }
