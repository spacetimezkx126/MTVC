"""Produce aligned top-5 portfolio tables and equity curves from dated CSVs."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.portfolio_sim import run_portfolio_strategy
from backtest.predictions import load_predictions


def run(manifest, output):
    spec = json.loads(manifest.read_text())
    output.mkdir(parents=True, exist_ok=True)
    report = {"metric_convention": "paper equations; 252 days/year; SR sample std (ddof=1)",
              "calendar_alignment": "Intersect dates before compounding; no carry-in profit from excluded dates.",
              "datasets": {}}
    for dataset, entries in spec["datasets"].items():
        stores = {}
        for entry in entries:
            label = entry["label"]
            if label in stores:
                raise ValueError(f"Duplicate label: {dataset}/{label}")
            path = (manifest.parent / entry["predictions"]).resolve()
            stores[label] = load_predictions(path)
        end_before = "2024-09-24" if dataset in ("csmd50", "csmd300") else None
        calendars = [{d for d in store.day_table if end_before is None or d < end_before}
                     for store in stores.values()]
        common = sorted(set.intersection(*calendars))
        if len(common) < 2:
            raise ValueError(f"No usable common trading calendar for {dataset}")
        rows = []
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        for entry in entries:
            label = entry["label"]
            store = stores[label]
            store.day_table = {d: store.day_table[d] for d in common}
            result = run_portfolio_strategy(store, name=label, topk=spec.get("topk", 5),
                                            budget=spec.get("initial_budget", 10000), fees=0)
            row = {"label": label, "seed": entry["seed"], "selection": entry["selection"],
                   "metrics": result.metrics, "dates": result.dates, "budget": result.budget,
                   "daily_return": result.daily_ret, "picks": result.picks}
            rows.append(row)
            if entry.get("plot", label in ("MTVC", "PEN", "StockNet")):
                ax.plot(range(1, len(common) + 1), result.budget, label=label)
        ax.axhline(spec.get("initial_budget", 10000), color="gray", linestyle="--", linewidth=1)
        ax.set(xlabel="Trading day", ylabel="Portfolio value", title=f"{dataset.upper()} equal-weighted top-5")
        ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(output / f"return_{dataset}.png", dpi=180)
        fig.savefig(output / f"return_{dataset}.pdf")
        plt.close(fig)
        report["datasets"][dataset] = {"n_days": len(common), "first": common[0], "last": common[-1],
                                        "end_before": end_before, "runs": rows}
    (output / "portfolio_results.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["| Dataset | Method | Seed | Days | ARR (%) | SR |", "| --- | --- | --- | --- | --- | --- |"]
    for dataset, result in report["datasets"].items():
        for row in result["runs"]:
            m = row["metrics"]
            lines.append(f"| {dataset} | {row['label']} | {row['seed']} | {result['n_days']} | {100*m['arr']:.4f} | {m['sr']:.6f} |")
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    run(a.manifest.resolve(), a.output.resolve())
