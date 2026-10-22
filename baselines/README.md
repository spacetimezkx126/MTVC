# Baselines

`main_pen.py`, `main_stocknet.py`, and `main_single.py` are the training/evaluation entry points. `common.py` combines the former dataset and PEN batch helpers. Calendar definitions, sample ordering, model parameter names, and checkpoint tensor layouts are retained.

From the repository root:

```bash
python scripts/evaluate_baseline.py --method pen --dataset csmd50 --seed 42 \
  --checkpoint-root ../contrast_exp_novol_s42_46/checkpoints/novol_accmcc_s42_46 \
  --output results/baselines/pen_csmd50_s42.json
```

Methods: `lstm`, `bi_lstm`, `alstm`, `adv_alstm`, `dtml`, `pen`, `stocknet`. Datasets: `csmd50`, `csmd300`, `massive`. Historical seeds: 42–46. The script loads tensors strictly and reports actual checkpoint inference metrics separately from stored references. Missing checkpoints raise an error.

PEN samples latent variables even in evaluation mode. Old PEN checkpoints contain test metrics but do not contain the corresponding sampling state, so standalone inference cannot guarantee the original logged numbers. File consolidation does not repair missing historical sampling state. New PEN checkpoints save the state immediately before test evaluation and restore it when evaluated independently. No historical weights or metric references are rewritten. StockNet checkpoints that contain `test_rng_state` now restore it in `--mode eval`, using the same final-test helper as training.

For portfolio inputs, append `--predictions results/portfolio_inputs/csmd50_pen_s42.csv` to PEN/StockNet evaluation. Export includes all valid stock-days with an explicitly seeded, separate inference pass and batch size 256. Its sampling stream is recorded separately from classification evaluation. Preserve CSVs when reproducing an existing curve; stochastic baseline rankings can differ across sampling streams.

The recorded baseline catalog is `configs/baseline_recipes.json` (105 runs). It retains original metric references, their precision, and source GPU indices without private paths. The evaluation helper distinguishes exact checkpoint references from rounded log references. Replaying StockNet requires mapping the saved source-device stream to the selected inference device.
