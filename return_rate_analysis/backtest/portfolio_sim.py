"""Portfolio simulation helpers (no vol_screen / edge deps)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DayRecord:
    prob_sum: float = 0.0
    count: int = 0
    y_org: float = 0.0
    y_bin: float = 0.0
    strong: bool = False

    @property
    def prob(self) -> float:
        return self.prob_sum / max(1, self.count)


@dataclass
class PredictionStore:
    day_table: dict[str, dict[str, DayRecord]] = field(default_factory=lambda: defaultdict(dict))


@dataclass
class PortfolioResult:
    name: str
    dates: list[str]
    budget: list[float]
    daily_ret: list[float]
    picks: list[list[str]]
    metrics: dict[str, float]


def _calc_fees(transaction_vol: float, fees_pct: float) -> float:
    return transaction_vol * fees_pct * 2.0 / 100.0


def _portfolio_metrics(budget: list[float]) -> dict[str, float]:
    from backtest.metrics import calculate_ARR, calculate_Calmar_Ratio, calculate_IR, calculate_MDD, calculate_SR

    arr = float(calculate_ARR(budget))
    mdd = float(calculate_MDD(budget))
    return {
        "arr": arr,
        "sr": float(calculate_SR(budget)),
        "mdd": mdd,
        "cr": float(calculate_Calmar_Ratio(arr, mdd)),
        "ir": float(calculate_IR(budget)),
        "final_budget": float(budget[-1]),
    }


def run_portfolio_strategy(
    store: PredictionStore,
    *,
    name: str,
    mode: str = "model_topk",
    budget: float = 10_000.0,
    fees: float = 0.0,
    topk: int = 5,
    prob_threshold: float = 0.0,
    model_ew_threshold: float = 0.5,
    max_days: int | None = None,
) -> PortfolioResult:
    dates = sorted(store.day_table.keys())
    if max_days is not None and max_days > 0:
        dates = dates[:max_days]
    asset = [float(budget)]
    daily_ret: list[float] = []
    picks: list[list[str]] = []
    out_dates: list[str] = []

    for date_str in dates:
        stocks = store.day_table[date_str]
        items = [
            (comp, rec.prob, rec.y_org)
            for comp, rec in stocks.items()
            if rec.count > 0 and rec.prob >= prob_threshold
        ]
        if not items:
            continue

        if mode == "label_ew":
            day_ret = float(np.mean([x[2] for x in items]))
            day_picks = [x[0] for x in items]
        elif mode == "model_ew":
            sel = [x for x in items if x[1] >= model_ew_threshold]
            day_ret = float(np.mean([x[2] for x in sel])) if sel else 0.0
            day_picks = [x[0] for x in sel]
        elif mode == "model_topk":
            sel = sorted(items, key=lambda x: x[1], reverse=True)[:topk]
            day_ret = float(np.mean([x[2] for x in sel]))
            day_picks = [x[0] for x in sel]
        else:
            raise ValueError(f"unknown portfolio mode: {mode}")

        cur = asset[-1]
        asset.append(cur * (1.0 + day_ret) - _calc_fees(cur, fees))
        daily_ret.append(day_ret)
        picks.append(day_picks)
        out_dates.append(date_str)

    if len(asset) < 2:
        raise RuntimeError(f"portfolio mode={mode}: not enough trading days")

    metrics = _portfolio_metrics(asset)
    metrics["n_trading_days"] = len(out_dates)
    return PortfolioResult(
        name=name,
        dates=out_dates,
        budget=asset[1:],
        daily_ret=daily_ret,
        picks=picks,
        metrics=metrics,
    )
