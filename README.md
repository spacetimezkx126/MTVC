# MTVC — MultiModal Transformer with Virtual-node and Contrast

Official training code for **MTVC**:
a price–news stock movement model with partially unified transformer, virtual nodes and contrastive learning.

This repository is a release package: formal code, launch scripts, baseline code. 

## Repository layout

```
MTVC/
  README.md
  code/                      # Training package
    run_mtvc.py              # Entrypoint
    mtvc/                    # Model / data / train / CLI
  scripts/                   # launcher codes (L1–L6 + ablations)
  data_preparation/          # Dataset build / industry remap / phase align
  return_rate_analysis/      # Portfolio backtest + return plots
  phase_periodicity/         # Market-periodicity visualization helpers
  assets/                    # Massive oversample extras, etc.
  baselines/                 # Baseline *code* only (run.sh + models; no checkpoints)
  dict/
    dict_csmd.pkl
    dict_massive.pkl
  data/
    CSMD50/                  # price + news (raw CSV) + comp_industry
    CSMD300/                 # same layout
    massive_data_sample_AAPL/# Single-company Massive slice (AAPL)
    massive_data -> …        # Symlink to the AAPL sample
```

## Data included

Paper training uses **`csmd_news_source=raw`** → company/day CSVs under `news/` (not `news_txt/`, not `llm_extract/`).

| Path | Contents |
|------|----------|
| `data/CSMD50/` | `price/`, `news/`, `comp_industry/` only |
| `data/CSMD300/` | same minimal layout |
| `data/massive_data_sample_AAPL/` | **AAPL only** (`price/`, `news/`, industry / name metadata). For inspection — **not** enough for full Massive training. |
| `dict/` | Character vocabs for CSMD and Massive |

Omitted on purpose (not required by the formal L6 stack): `news_txt/`, `llm_extract/`, `pattern/`, `precomputed/`, `comp_supply_chain/`, and stray `train_*_probs_preds_labels.txt` dumps.

For the complete **Massive Source** multi-ticker corpus:

```bash
export MASSIVE_ROOT=/path/to/massive_data
```

## Environment

- Python 3.8+ with PyTorch (CUDA recommended)
- Example conda env used in our runs: `CAMEF`

```bash
export PY=/path/to/python   # e.g. ~/miniconda3/envs/CAMEF/bin/python
export MTVC_ROOT=/path/to/MTVC
cd "$MTVC_ROOT"
```

`code/run_mtvc.py` sets `DUAL_TF_MODEL_DIR` to the parent of this package (the surrounding `dual_tf` tree in our lab layout) so shared assets such as FinBERT can be resolved. Override if needed:

```bash
export DUAL_TF_MODEL_DIR=/path/to/dual_tf   # optional
```

## Quick start (CSMD50, Full MTVC, case-mine L6)

```bash
$PY $MTVC_ROOT/code/run_mtvc.py \
  --dataset csmd50 --mode train \
  --dataset_root $MTVC_ROOT/data/CSMD50 \
  --vocab_path $MTVC_ROOT/dict/dict_csmd.pkl \
  --contrast_mode mtvc --mtvc_case_layer 6 \
  --price_encoder partial_unified \
  --news_word_emb random --news_text_tf_layers 3 --news_padding_k 5 \
  --mtvc_lambda_pair 0.2 --contrast_aux_weight 1.0 \
  --split_mode contiguous --bce_pos_weight 1.0 \
  --train_start 2021-01-01 --train_end 2023-01-01 \
  --val_start 2023-01-02 --val_end 2024-01-02 \
  --test_start 2024-01-03 --test_end 2024-12-31 \
  --seed 42 --device cuda:0 \
  --ckpt_dir $MTVC_ROOT/checkpoints/csmd50/full_casel6/s42
```

Or via the wrapper:

```bash
bash $MTVC_ROOT/scripts/run_mtvc_example.sh --dataset csmd50 --mode train ...
```

## Multi-run launchers

Scripts write under `checkpoints/{dataset}/{formal_name}/s{seed}/` and match the paper L6 stack:

| Script | Checkpoint dirs |
|--------|-----------------|
| `scripts/launch_full_casel_layers_3ds.sh` | `full_casel{1..6}` |
| `scripts/launch_abl_casel6_3ds.sh` | `abl_{nocontrast,nogate,noglobal\|addglobal,noindustry,noprice}_casel6` |
| `scripts/launch_arch_casel6_3ds.sh` | `multinews_casel6`, `crossattn_casel6`, `fullunified_casel6` |
| `scripts/launch_vin_casel6_3ds.sh` | `vin_{mkt_as_ind,ind_as_mkt}_casel6` |
| `scripts/verify_checkpoints.sh` | Sanity-check `best.pt` / code layout |

Defaults use seeds `42–46`. Override with env vars, e.g. `DATASETS=csmd50 SEEDS="42" ALLOWED_GPUS="0"`.

Bundled data includes **CSMD50** and **CSMD300** (+ AAPL Massive sample). For full Massive training set `MASSIVE_ROOT`:

```bash
DATASETS="csmd50 csmd300" bash scripts/launch_full_casel_layers_3ds.sh
```

## Evaluation protocol (paper)

| Metric | CSMD50 / CSMD300 | Massive Source |
|--------|------------------|----------------|
| Acc / MCC | Full test window | Full test window |
| ARR / SR (top-5 long-only) | Test days **strictly before 2024-09-24** | Full test window |

Portfolio / return figures:

```bash
$PY $MTVC_ROOT/return_rate_analysis/scripts/plot_paper_casel6_pre924.py
$PY $MTVC_ROOT/return_rate_analysis/scripts/summarize_acc_mcc_casel6.py
```

Acc/MCC alignment vs checkpoint meta: `code/eval_acc_mcc_vs_meta.py`.


## Citation

If you use this code or the CSMD / Massive setups, please cite the MTVC paper (Partially Unified / JMTVC).

## License / notes

- `baselines/` ships **source code only** (no baseline checkpoints); paths are package-relative.
- The AAPL Massive slice is a **sample**; redistribute only what your data license allows.
- Training logs and new runs go to `checkpoints/` and `launch_logs/` (create as needed; not pre-populated).
