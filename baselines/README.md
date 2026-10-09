# Baselines

Training code for the paper main-table baselines
(LSTM / BiLSTM / ALSTM / Adv-ALSTM / DTML / PEN / StockNet), no-volume Acc/MCC setting.

Checkpoints are **not** included. Paths are relative to this package:

```bash
# from MTVC/
bash baselines/run.sh

# optional overrides
PYTHON=python3 MASSIVE_ROOT=/path/to/full_massive_data bash baselines/run.sh
```

Defaults: `../data/CSMD50`, `../data/CSMD300`, `../data/massive_data` (AAPL sample),
vocab under `../dict/`, Massive oversample jsonl under `../assets/massive_oversample/`.
