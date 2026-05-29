#!/usr/bin/env python3
"""Run R1/R2 ETF strategy backtests per README V2.1 plan."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.data import ETF_UNIVERSE, load_returns
from backtest.engine import align_period, run_backtest
from backtest.metrics import RB_HIGH, evaluate_v2
from backtest.strategies import RUN_ORDER, STRATEGIES

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "backtest" / "output"


def fmt_pct(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x * 100:.{digits}f}%"


def run_one(
    name: str,
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    cost_bps: float = 1.0,
) -> dict:
    spec = STRATEGIES[name]
    required = spec["required"]
    sub, start, end = align_period(returns, required)
    rets = sub
    px = prices.reindex(rets.index).ffill()
    assets = list(sub.columns)
    idx = sub.index
    weights = spec["fn"](idx, rets, px, assets)
    port_ret, _ = run_backtest(rets, weights, cost_bps=cost_bps)
    metrics = evaluate_v2(port_ret)
    metrics.update(
        {
            "strategy": name,
            "cost_bps": cost_bps,
            "start": str(start.date()),
            "end": str(end.date()),
        }
    )
    return metrics


def appendix_511990(returns: pd.DataFrame) -> dict:
    rets = returns["511990"].dropna()
    m = evaluate_v2(rets)
    m["strategy"] = "511990_BH"
    m["start"] = str(rets.index[0].date())
    m["end"] = str(rets.index[-1].date())
    return m


def main(refresh: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading ETF returns (Eastmoney 日增长率, cached in data/cache/)...")
    returns = load_returns(ETF_UNIVERSE, refresh=refresh)
    # Price levels for MA signals: rebuild from returns
    prices = (1 + returns.fillna(0)).cumprod() * 100
    availability = {}
    for col in returns.columns:
        first = returns[col].first_valid_index()
        availability[col] = str(first.date()) if first else None
    print("First valid dates:", json.dumps(availability, indent=2))

    rows: list[dict] = []
    for name in RUN_ORDER:
        print(f"Running {name} @ 1bp...")
        try:
            rows.append(run_one(name, returns, prices, cost_bps=1.0))
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {name}: {exc}")
            rows.append({"strategy": name, "verdict": f"失败: {exc}"})

    rows.append(appendix_511990(returns))

    # Cost sensitivity S4 / S5
    sens_rows = []
    for name in ["S4", "S5"]:
        for bps in [0.5, 1.0, 2.0]:
            try:
                m = run_one(name, returns, prices, cost_bps=bps)
                sens_rows.append(
                    {
                        "strategy": name,
                        "cost_bps": bps,
                        "Rs": m.get("Rs"),
                        "ER_high": m.get("ER_high"),
                        "verdict": m.get("verdict"),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                sens_rows.append({"strategy": name, "cost_bps": bps, "error": str(exc)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "results_summary.csv", index=False)
    pd.DataFrame(sens_rows).to_csv(OUT_DIR / "cost_sensitivity_s4_s5.csv", index=False)

    report_lines = [
        "# 回测结果报告",
        "",
        f"> 生成自 `backtest/run_backtest.py`；收益口径：akshare `fund_etf_fund_info_em` 日增长率（%），缺失时回退净值涨跌幅。",
        f"> 基准上沿 `Rb_high = {RB_HIGH:.2%}`；成本基准 **1 bp** 单边；OOS 最近 30%。",
        "",
        "## 数据可用性（各 ETF 首个有效交易日）",
        "",
        "| 代码 | 首个有效日 |",
        "|---|---|",
    ]
    for k, v in sorted(availability.items()):
        report_lines.append(f"| {k} | {v} |")

    report_lines.extend(
        [
            "",
            "## 策略回测摘要（1 bp）",
            "",
            "| 策略 | 区间 | 月数 | Rs | ER_high | Vol | MDD | CVaR95_m | 结论 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, r in df.iterrows():
        if r.get("strategy") == "511990_BH":
            continue
        report_lines.append(
            "| {strategy} | {start} ~ {end} | {n_months} | {Rs} | {ER_high} | {Vol} | {MDD} | {CVaR95_m} | {verdict} |".format(
                strategy=r.get("strategy", ""),
                start=r.get("start", ""),
                end=r.get("end", ""),
                n_months=int(r["n_months"]) if pd.notna(r.get("n_months")) else "",
                Rs=fmt_pct(r.get("Rs")),
                ER_high=fmt_pct(r.get("ER_high")),
                Vol=fmt_pct(r.get("Vol")),
                MDD=fmt_pct(r.get("MDD")),
                CVaR95_m=fmt_pct(r.get("CVaR95_m")),
                verdict=r.get("verdict", ""),
            )
        )

    report_lines.extend(["", "## 511990 买入持有（附录对照）", ""])
    bh = df[df["strategy"] == "511990_BH"].iloc[0] if (df["strategy"] == "511990_BH").any() else None
    if bh is not None:
        report_lines.append(
            f"- 区间：{bh['start']} ~ {bh['end']}；Rs={fmt_pct(bh['Rs'])}；ER_high={fmt_pct(bh['ER_high'])}"
        )

    report_lines.extend(["", "## S4/S5 成本敏感性", ""])
    sens_df = pd.DataFrame(sens_rows)
    if not sens_df.empty:
        report_lines.append(sens_df.to_markdown(index=False))

    report_lines.extend(["", "## V2 验收明细（1 bp）", ""])
    for r in rows:
        if "Rs" not in r or pd.isna(r.get("Rs")):
            continue
        name = r.get("strategy", "")
        report_lines.append(f"### {name}")
        report_lines.append("")
        report_lines.append("| 项目 | 数值 | 达标 |")
        report_lines.append("|---|---:|---|")
        checks = [
            ("Rs", fmt_pct(r.get("Rs")), ""),
            ("ER_high", fmt_pct(r.get("ER_high")), "Y" if r.get("pass_return") else "N"),
            ("Vol", fmt_pct(r.get("Vol")), "Y" if r.get("pass_risk") else "—"),
            ("MDD", fmt_pct(r.get("MDD")), ""),
            ("CR_MDD", f"{r.get('CR_MDD', float('nan')):.3f}" if pd.notna(r.get("CR_MDD")) else "N/A", ""),
            ("CR_Vol", f"{r.get('CR_Vol', float('nan')):.3f}" if pd.notna(r.get("CR_Vol")) else "N/A", ""),
            ("U", f"{r.get('U', float('nan')):.4f}" if pd.notna(r.get("U")) else "N/A", ""),
            ("ER_high_oos", fmt_pct(r.get("ER_high_oos")), ""),
            ("ER_high p5", fmt_pct(r.get("ER_high_p5")), ""),
            ("结论", r.get("verdict", ""), ""),
        ]
        for item, val, ok in checks:
            report_lines.append(f"| {item} | {val} | {ok} |")
        report_lines.append("")

    report_lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 含 **511070 / 511580** 的策略受 ETF 上市时间限制，全规格样本约自 **2025-01** 起；`n_months < 24` 时 V2 结论为「观察」。",
            "- 正式投决需结合上表与 README「快速填表模板」逐项核对 pass 标志。",
        ]
    )

    report_path = OUT_DIR / "backtest_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    (OUT_DIR / "results_summary.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nWrote {report_path}")
    print(df[["strategy", "start", "end", "n_months", "Rs", "ER_high", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
