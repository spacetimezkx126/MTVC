"""News post-5d impact triples (dir, mag, lag) from realized returns.

Oracle is ONLY for pretrain supervision on train(/val). Downstream test must use
model predictions — never these labels as features.
"""

from __future__ import annotations
import json
import math
import os
from bisect import bisect_right
from typing import Any

HORIZON = 5


def _f(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def post5_returns(
    label_by_date: dict, dates_sorted: list[str], pub_date: str, horizon: int = HORIZON
) -> list[float] | None:
    """Next ``horizon`` trading-day returns strictly after ``pub_date``."""
    if not dates_sorted or not label_by_date:
        return None
    i = bisect_right(dates_sorted, str(pub_date))
    fut = dates_sorted[i : i + int(horizon)]
    if len(fut) < int(horizon):
        return None
    out: list[float] = []
    for d in fut:
        r = _f(label_by_date.get(d))
        if not math.isfinite(r):
            return None
        out.append(r)
    return out


def triple_from_returns(returns: list[float]) -> tuple[float, float, float]:
    """(dir=sum, mag=|sum|, lag_norm=(argmax|r|+1-1)/4 in [0,1])."""
    if not returns:
        return (0.0, 0.0, 0.0)
    s = float(sum(returns))
    mag = abs(s)
    abs_r = [abs(r) for r in returns]
    lag_i = int(max(range(len(abs_r)), key=lambda k: abs_r[k]))
    lag_norm = float(lag_i) / float(max(1, len(returns) - 1))
    return (s, mag, lag_norm)


def compute_company_date_oracle(
    label: dict[str, dict], *, horizon: int = HORIZON
) -> dict[str, dict[str, list[float]]]:
    """{company: {pub_date: [dir, mag, lag_norm]}} for dates with full 5-day future."""
    out: dict[str, dict[str, list[float]]] = {}
    for comp, by_d in (label or {}).items():
        if not isinstance(by_d, dict) or not by_d:
            continue
        dates_sorted = sorted(by_d.keys())
        bucket: dict[str, list[float]] = {}
        for pub in dates_sorted:
            rets = post5_returns(by_d, dates_sorted, pub, horizon=horizon)
            if rets is None:
                continue
            (d, m, ell) = triple_from_returns(rets)
            bucket[str(pub)] = [float(d), float(m), float(ell)]
        if bucket:
            out[str(comp)] = bucket
    return out


def fit_norm_stats(
    oracle: dict[str, dict[str, list[float]]], *, date_start: str, date_end: str
) -> dict[str, float]:
    """Train-split mean/std for dir & mag (lag already in [0,1])."""
    dirs: list[float] = []
    mags: list[float] = []
    for _comp, by_d in oracle.items():
        for d, trip in by_d.items():
            if not date_start <= str(d) <= date_end:
                continue
            dirs.append(float(trip[0]))
            mags.append(float(trip[1]))

    def _ms(xs: list[float]) -> tuple[float, float]:
        if not xs:
            return (0.0, 1.0)
        mu = sum(xs) / len(xs)
        var = sum(((x - mu) ** 2 for x in xs)) / max(1, len(xs))
        sd = math.sqrt(var)
        return (float(mu), float(sd if sd > 1e-08 else 1.0))

    (d_mu, d_sd) = _ms(dirs)
    (m_mu, m_sd) = _ms(mags)
    return {
        "dir_mean": d_mu,
        "dir_std": d_sd,
        "mag_mean": m_mu,
        "mag_std": m_sd,
        "n": float(len(dirs)),
        "date_start": str(date_start),
        "date_end": str(date_end),
    }


def normalize_triple(
    trip: list[float] | tuple[float, float, float], stats: dict[str, float]
) -> list[float]:
    d = (float(trip[0]) - float(stats.get("dir_mean", 0.0))) / float(
        stats.get("dir_std", 1.0) or 1.0
    )
    m = (float(trip[1]) - float(stats.get("mag_mean", 0.0))) / float(
        stats.get("mag_std", 1.0) or 1.0
    )
    ell = float(trip[2])
    return [d, m, ell]


def denormalize_triple(
    trip: list[float] | tuple[float, float, float], stats: dict[str, float]
) -> list[float]:
    d = float(trip[0]) * float(stats.get("dir_std", 1.0) or 1.0) + float(stats.get("dir_mean", 0.0))
    m = float(trip[1]) * float(stats.get("mag_std", 1.0) or 1.0) + float(stats.get("mag_mean", 0.0))
    return [d, m, float(trip[2])]


def save_oracle_cache(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def load_oracle_cache(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def lookup_oracle(
    cache: dict, company: str, pub_date: str, *, normalized: bool = True
) -> list[float] | None:
    """Return [dir, mag, lag] or None if missing."""
    oracle = cache.get("oracle") or {}
    by = oracle.get(str(company)) or {}
    trip = by.get(str(pub_date))
    if trip is None:
        return None
    if normalized:
        stats = cache.get("norm") or {}
        return normalize_triple(trip, stats)
    return [float(trip[0]), float(trip[1]), float(trip[2])]
