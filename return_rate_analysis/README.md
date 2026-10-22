# Portfolio reproduction

This directory computes the paper's daily equal-weighted long-only top-5 ARR/SR and produces `return_csmd50.png`, `return_csmd300.png`, and `return_massive.png` (plus PDF figures, dated holdings, daily returns, and JSON/Markdown tables). It does not reconstruct unrelated paper figures such as architecture diagrams, regime breaks, attention visualizations, or layer sweeps.

## Inputs and fixed selections

Use one dated CSV per dataset/method/seed. Required columns: `date`, `company`, `probability`, `return`. Returns are decimal simple returns, not percentage values. Duplicate company/day probabilities are averaged; conflicting realized returns are rejected. All valid stock-days, including weak-label rows, are portfolio candidates. Classification retains its separate strong-sample filter.

Export MTVC predictions from the repository root:

```bash
python scripts/export_predictions.py --dataset csmd50 --case full_casel6 --seed 42 \
  --output results/portfolio_inputs/csmd50_mtvc_s42.csv
python scripts/evaluate_baseline.py --method pen --dataset csmd50 --seed 42 \
  --checkpoint-root ../contrast_exp_novol_s42_46/checkpoints/novol_accmcc_s42_46 \
  --output results/baselines/pen_csmd50_s42.json \
  --predictions results/portfolio_inputs/csmd50_pen_s42.csv
```

Repeat with `stocknet` and the other datasets. Declare selected seeds before inspecting test returns. The manifest records the selected seed and its selection rule. This code does not select seeds by test returns.

For trusted historical caches, use `scripts/cache_to_predictions.py --cache CACHE.pkl --output OUTPUT.csv`. Cache conversion reproduces cached predictions, not independent checkpoint inference. Keep cache provenance separate from newly exported predictions.

## Manifest and run

Paths in a manifest are relative to that manifest. Example:

```json
{
  "topk": 5,
  "initial_budget": 10000,
  "datasets": {
    "csmd50": [
      {"label": "MTVC", "seed": 42, "selection": "fixed before evaluation", "predictions": "csmd50_mtvc_s42.csv"},
      {"label": "PEN", "seed": 42, "selection": "fixed before evaluation", "predictions": "csmd50_pen_s42.csv"},
      {"label": "StockNet", "seed": 42, "selection": "fixed before evaluation", "predictions": "csmd50_stocknet_s42.csv"}
    ]
  }
}
```

```bash
python return_rate_analysis/scripts/reproduce_portfolios.py \
  --manifest results/portfolio_inputs/manifest.json --output results/portfolios
```

Add `csmd300` and `massive` entries to generate all three figures. Additional methods appear in the tables; set `"plot": true` to add their curves.

## Calculation conventions

- CSMD50/CSMD300: retain dates strictly before 2024-09-24; Massive: full provided test window.
- Intersect participating models' trading calendars **before** compounding; each series starts with the same initial capital. Historical captions cite 170/170/245 aligned days, but actual input coverage is audited and reported rather than forced to those counts.
- Buy the five highest scores each day, equal weight, daily rebalance, zero transaction costs. Score ties preserve CSV insertion order. No test-based threshold or seed optimization is performed.
- ARR uses `(final / initial) ** (252 / T) - 1`, where T is the number of returns, excluding the initial wealth point.
- SR uses `(252 * mean(daily_returns) - 0.01) / (std(daily_returns, ddof=1) * sqrt(252))`, matching the appendix's sample standard deviation.
- Maximum drawdown uses each day's running peak.

The output records dates, selected seeds, holdings and calculation conventions. Preserve the input predictions and manifest to reproduce an existing curve.
