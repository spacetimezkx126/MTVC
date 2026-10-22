"""Validate and aggregate dated prediction CSV records."""
from __future__ import annotations
import csv
import math
from .portfolio_sim import DayRecord, PredictionStore


def load_predictions(path):
    store = PredictionStore()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            date, company = row["date"], row["company"]
            probability, realized = float(row["probability"]), float(row["return"])
            if not (math.isfinite(probability) and 0 <= probability <= 1 and math.isfinite(realized)):
                raise ValueError(f"Nonfinite or invalid prediction: {date}/{company}")
            stocks = store.day_table[date]
            if company in stocks and abs(stocks[company].y_org - realized) > 1e-10:
                raise ValueError(f"Conflicting realized returns: {date}/{company}")
            record = stocks.setdefault(company, DayRecord(y_org=realized))
            record.prob_sum += probability
            record.count += 1
    if not store.day_table:
        raise ValueError(f"Empty prediction CSV: {path}")
    return store
