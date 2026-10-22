"""Convert a trusted historical prediction-store pickle into a portable CSV.

Only load caches from trusted experiment runs; pickle can execute code.
"""
import argparse
import csv
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    with a.cache.open("rb") as handle:
        rows = pickle.load(handle)
    if hasattr(rows, "day_table"):
        rows = rows.day_table
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "company", "probability", "return", "label", "strong"])
        for date, stocks in sorted(rows.items()):
            for company, record in stocks.items():
                get = record.get if isinstance(record, dict) else lambda key, default=0: getattr(record, key, default)
                count = int(get("count", 0))
                if count:
                    writer.writerow([date, company, get("prob_sum", 0) / count,
                                     get("y_org", 0), get("y_bin", 0), int(get("strong", False))])
