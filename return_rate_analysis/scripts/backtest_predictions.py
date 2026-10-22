"""Backtest a dated prediction CSV without choosing a seed by test performance."""

import argparse, csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.portfolio_sim import DayRecord, PredictionStore, run_portfolio_strategy

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--predictions", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--topk", type=int, default=5)
p.add_argument("--fees", type=float, default=0.0)
a = p.parse_args()
from backtest.predictions import load_predictions
s = load_predictions(a.predictions)
result = run_portfolio_strategy(s, name=a.predictions.stem, topk=a.topk, fees=a.fees)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(
    json.dumps(
        {
            "metrics": result.metrics,
            "dates": result.dates,
            "budget": result.budget,
            "daily_return": result.daily_ret,
            "picks": result.picks,
        },
        indent=2,
    )
    + "\n"
)
