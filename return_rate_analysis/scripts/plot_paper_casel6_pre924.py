#!/usr/bin/env python3
"""Paper portfolio curves + ARR/SR for MTVC L6 (pre-2024-09-24 on CSMD).

Writes ``plots/paper_casel6_pre924/``:
  - return_{csmd50,csmd300,massive}.png  (JMTVC / PEN / StockNet)
  - paper_portfolio_summary.json         (main_port / arch_port / token_port)

Uses prediction caches under ``cache/prediction_stores_casel6`` (MTVC L6)
and dual_tf ``prediction_stores_novol_accmcc`` (baselines).
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RRA = SCRIPT_DIR.parent
REPRO = RRA.parent
DUAL_TF = REPRO.parent

OUT = RRA / "plots" / "paper_casel6_pre924"
CK = REPRO / "checkpoints"
NOVOL_LOG = DUAL_TF / "contrast_exp_novol_s42_46/ablation_logs/novol_accmcc_s42_46"
NOVOL_CACHE = DUAL_TF / "return_rate_analysis/cache/prediction_stores_novol_accmcc"
MTVC_CACHE = RRA / "cache" / "prediction_stores_casel6"

sys.path.insert(0, str(RRA))
from backtest.portfolio_sim import (  # noqa: E402
    DayRecord,
    PredictionStore,
    _portfolio_metrics,
    run_portfolio_strategy,
)

SEEDS = [42, 43, 44, 45, 46]
BREAK = "2024-09-24"
BUDGET = 10000.0
TOPK = 5

DISPLAY = {
    "lstm": "LSTM",
    "bi_lstm": "BiLSTM",
    "alstm": "ALSTM",
    "adv_alstm": "Adv-ALSTM",
    "dtml": "DTML",
    "pen": "PEN",
    "stocknet": "StockNet",
}
FULL_SEED = {"csmd50": 42, "csmd300": 42, "massive": 46}
COLORS = {"JMTVC": "#d62728", "PEN": "#1f77b4", "StockNet": "#ff7f0e"}

# Formal L6 checkpoint dirs under checkpoints/{ds}/
FULL_EXP = {ds: "full_casel6" for ds in ("csmd50", "csmd300", "massive")}
ARCH_EXP = {
    "csmd50": {
        "Cross-Attention": ("crossattn_casel6", "crossattn"),
        "Fully Unified": ("fullunified_casel6", "fullunified"),
        "JMTVC (Partially Unified)": ("full_casel6", "full"),
    },
    "csmd300": {
        "Cross-Attention": ("crossattn_casel6", "crossattn"),
        "Fully Unified": ("fullunified_casel6", "fullunified"),
        "JMTVC (Partially Unified)": ("full_casel6", "full"),
    },
    "massive": {
        "Cross-Attention": ("crossattn_casel6", "crossattn"),
        "Fully Unified": ("fullunified_casel6", "fullunified"),
        "JMTVC (Partially Unified)": ("full_casel6", "full"),
    },
}
TOKEN_EXP = {ds: ("multinews_casel6", "multinews") for ds in ("csmd50", "csmd300", "massive")}


def load_store(path: Path) -> PredictionStore:
    store = PredictionStore()
    with path.open("rb") as f:
        raw = pickle.load(f)
    for d, stocks in raw.items():
        for comp, rec in stocks.items():
            if isinstance(rec, DayRecord):
                store.day_table[d][comp] = rec
            else:
                dr = DayRecord()
                dr.prob_sum = float(rec.get("prob_sum", 0))
                dr.count = int(rec.get("count", 0))
                dr.y_org = float(rec.get("y_org", 0))
                dr.y_bin = float(rec.get("y_bin", 0))
                store.day_table[d][comp] = dr
    return store


def filter_store(store: PredictionStore, *, end_before: str | None = None) -> PredictionStore:
    out = PredictionStore()
    for d, stocks in store.day_table.items():
        if end_before is not None and d >= end_before:
            continue
        out.day_table[d] = stocks
    return out


def scope_store(dataset: str, store: PredictionStore) -> PredictionStore:
    if dataset in ("csmd50", "csmd300"):
        return filter_store(store, end_before=BREAK)
    return store


def portfolio(store: PredictionStore, name: str) -> dict:
    res = run_portfolio_strategy(
        store, name=name, mode="model_topk", topk=TOPK, budget=BUDGET, max_days=None
    )
    return {
        "label": name,
        "dates": list(res.dates),
        "budget": list(res.budget),
        "metrics": dict(res.metrics),
    }


def align_runs(runs: list[dict]) -> list[dict]:
    common = set(runs[0]["dates"])
    for r in runs[1:]:
        common &= set(r["dates"])
    dates = sorted(common)
    out = []
    for r in runs:
        d2b = dict(zip(r["dates"], r["budget"]))
        b = [float(d2b[d]) for d in dates]
        met = _portfolio_metrics([BUDGET] + b)
        met["n_trading_days"] = len(dates)
        out.append({**r, "dates": dates, "budget": b, "metrics": met})
    return out


def parse_novol_acc_mcc(dataset: str, method: str, seed: int) -> tuple[float, float]:
    text = (NOVOL_LOG / f"{method}_{dataset}_seed{seed}.log").read_text(errors="replace")
    if method == "stocknet":
        hits = re.findall(
            r"\[FINAL TEST @ best epoch \d+\].*?acc=([0-9.]+).*?mcc=([+-]?[0-9.]+)", text
        )
        if not hits:
            raise RuntimeError(f"no stocknet metrics {dataset} {seed}")
        a, m = hits[-1]
        return float(a), float(m)
    if method == "pen":
        hits = re.findall(r"\[TEST\].*?acc=([0-9.]+).*?mcc=([+-]?[0-9.]+)", text)
        if not hits:
            raise RuntimeError(f"no pen metrics {dataset} {seed}")
        a, m = hits[-1]
        return float(a), float(m)
    hits = re.findall(r"\[FINAL_TEST\].*?acc=([0-9.]+).*?mcc=([+-]?[0-9.]+)", text)
    if not hits:
        raise RuntimeError(f"no FINAL_TEST {method} {dataset} {seed}")
    a, m = hits[-1]
    return float(a), float(m)


def best_mcc_seed(dataset: str, method: str) -> int:
    best_s, best_m = None, None
    for s in SEEDS:
        _, m = parse_novol_acc_mcc(dataset, method, s)
        if best_m is None or m > best_m:
            best_s, best_m = s, m
    assert best_s is not None
    return best_s


def read_ckpt_metrics(dataset: str, exp: str, seed: int) -> tuple[float, float]:
    meta = CK / dataset / exp / f"s{seed}" / "best.pt.meta.json"
    d = json.loads(meta.read_text())
    m = d["metrics"]
    return float(m["best_test_acc"]), float(m["best_test_mcc"])


def best_ckpt_mcc_seed(dataset: str, exp: str) -> int:
    best_s, best_m = None, None
    for s in SEEDS:
        try:
            _, m = read_ckpt_metrics(dataset, exp, s)
        except Exception:
            continue
        if best_m is None or m > best_m:
            best_s, best_m = s, m
    if best_s is None:
        raise RuntimeError(f"no metrics for {dataset}/{exp}")
    return best_s


def store_baseline(dataset: str, method: str, seed: int) -> PredictionStore:
    path = NOVOL_CACHE / f"{dataset}_{method}_s{seed}.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    return load_store(path)


def store_mtvc(dataset: str, cache_tag: str, seed: int) -> PredictionStore:
    # casel6 cache naming: {ds}_{tag}_s{seed}_s{seed}.pkl
    path = MTVC_CACHE / f"{dataset}_{cache_tag}_s{seed}_s{seed}.pkl"
    if not path.exists():
        # older single-suffix fallback
        alt = MTVC_CACHE / f"{dataset}_{cache_tag}_s{seed}.pkl"
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(path)
    return load_store(path)


def plot_equity(runs: list[dict], out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    by = {r["label"]: r for r in runs}
    for name in ("JMTVC", "PEN", "StockNet"):
        r = by[name]
        b = np.asarray(r["budget"], dtype=np.float64)
        ax.plot(
            np.arange(len(b)),
            b,
            label=f"{name} (ARR={r['metrics']['arr']*100:.1f}%, SR={r['metrics']['sr']:.2f})",
            color=COLORS[name],
            linewidth=2.3 if name == "JMTVC" else 1.5,
        )
    ax.axhline(BUDGET, color="gray", linestyle="--", linewidth=1.0, alpha=0.55)
    ax.set_title(title)
    ax.set_xlabel("trading day index")
    ax.set_ylabel("portfolio value")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[plot] {out_png}")


def _port_row(r: dict) -> dict:
    return {
        "seed": r["seed"],
        "arr": r["metrics"]["arr"],
        "sr": r["metrics"]["sr"],
        "final": r["metrics"].get("final_budget"),
        "n_days": r["metrics"]["n_trading_days"],
        "first": r["dates"][0],
        "last": r["dates"][-1],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    titles = {
        "csmd50": "CSMD50 top-5 portfolio (test days before 2024-09-24)",
        "csmd300": "CSMD300 top-5 portfolio (test days before 2024-09-24)",
        "massive": "Massive Source top-5 portfolio (full test window)",
    }

    main_port: dict = {}
    equity_runs: dict = {}
    for ds in ("csmd50", "csmd300", "massive"):
        runs = []
        s = FULL_SEED[ds]
        st = scope_store(ds, store_mtvc(ds, "full", s))
        r = portfolio(st, "JMTVC")
        r["seed"] = s
        runs.append(r)
        for method, name in DISPLAY.items():
            bs = best_mcc_seed(ds, method)
            rr = portfolio(scope_store(ds, store_baseline(ds, method, bs)), name)
            rr["seed"] = bs
            runs.append(rr)
        runs = align_runs(runs)
        main_port[ds] = {x["label"]: _port_row(x) for x in runs}
        equity_runs[ds] = [x for x in runs if x["label"] in ("JMTVC", "PEN", "StockNet")]
        print(f"[{ds}] days={runs[0]['metrics']['n_trading_days']} {runs[0]['dates'][0]}..{runs[0]['dates'][-1]}")
        for x in runs:
            m = x["metrics"]
            print(
                f"  {x['label']:12s} s{x['seed']} ARR={m['arr']*100:7.2f}% SR={m['sr']:6.3f}"
            )

    arch_port: dict = {}
    for ds in ("csmd50", "csmd300", "massive"):
        arch_port[ds] = {}
        for name, (exp, tag) in ARCH_EXP[ds].items():
            seed = best_ckpt_mcc_seed(ds, exp)
            r = portfolio(scope_store(ds, store_mtvc(ds, tag, seed)), name)
            r["seed"] = seed
            arch_port[ds][name] = {
                "seed": seed,
                "arr": r["metrics"]["arr"],
                "sr": r["metrics"]["sr"],
                "n_days": r["metrics"]["n_trading_days"],
                "first": r["dates"][0],
                "last": r["dates"][-1],
            }
            print(
                f"[arch {ds}] {name}: s{seed} ARR={r['metrics']['arr']*100:.2f}% "
                f"SR={r['metrics']['sr']:.3f}"
            )

    token_port: dict = {}
    for ds in ("csmd50", "csmd300", "massive"):
        exp, tag = TOKEN_EXP[ds]
        seed = best_ckpt_mcc_seed(ds, exp)
        r = portfolio(scope_store(ds, store_mtvc(ds, tag, seed)), "Token-level")
        r["seed"] = seed
        token_port[ds] = {
            "Token-level": {
                "seed": seed,
                "arr": r["metrics"]["arr"],
                "sr": r["metrics"]["sr"],
                "n_days": r["metrics"]["n_trading_days"],
            }
        }
        print(f"[token {ds}] multinews s{seed} ARR={r['metrics']['arr']*100:.2f}%")

    for ds in ("csmd50", "csmd300", "massive"):
        plot_equity(equity_runs[ds], OUT / f"return_{ds}.png", titles[ds])

    summary = {
        "scope": {
            "csmd50_csmd300": "portfolio dates strictly before 2024-09-24",
            "massive": "full test window (no day cap)",
            "mtvc": "full_casel6 / crossattn_casel6 / fullunified_casel6 / multinews_casel6",
        },
        "main_port": main_port,
        "arch_port": arch_port,
        "token_port": token_port,
    }
    out_json = OUT / "paper_portfolio_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[write] {out_json}")


if __name__ == "__main__":
    main()
