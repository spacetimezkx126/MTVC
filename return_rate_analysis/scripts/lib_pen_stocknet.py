#!/usr/bin/env python3
"""Compare portfolio return curves: PEN / StockNet / TE-on (gt_full or no_gt) best-seed checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
RETURN_ROOT = SCRIPT_DIR.parent
DUAL_TF = RETURN_ROOT.parent
CONTRAST_ROOT = DUAL_TF / "contrast_exp_backup_20260816/contrast_exp"
BACKUP_GT = DUAL_TF / "backups/formal_gt_five_p5_20260820_112933/gt_five_p5/checkpoints"
CONTRAST_VOL = (
    DUAL_TF / "backup_ckpt_logs_20260817_152357/contrast_checkpoints/contrast_batch_with_volume_valloss"
)
CONTRAST_MASSIVE = CONTRAST_ROOT / "checkpoints/contrast_batch_contiguous_os_massive_valloss"

for p in (str(DUAL_TF), str(RETURN_ROOT), str(CONTRAST_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from backtest.portfolio_sim import (  # noqa: E402
    DayRecord,
    PredictionStore,
    run_portfolio_strategy,
)
from checkpoint import load_model_state_for_eval  # noqa: E402
from cuda_guard import cuda_device_lock, forward_with_cuda_retry  # noqa: E402
from data import MarketWindowDataset, get_profile, resolve_vocab_path  # noqa: E402
from model import Model as SimpleModel  # noqa: E402
from pen1_train import collate_pen_batch, load_ckpt  # noqa: E402
from main_pen import CminCnPenDataset, _build_model, _vocab_info  # noqa: E402
from model_stocknet import StockNetModel  # noqa: E402
from train import (  # noqa: E402
    _builder_dataset,
    dataset_kwargs_from_args,
    device_from_args,
    make_model_dataloader,
    model_finbert_kwargs,
    model_size_kwargs,
    normalize_run_args,
    set_all_seeds,
    wrap_model_devices,
)

_MODEL_NEWS_MODE = {
    "finbert": "finbert_padding",
    "vocab": "vocab_padding",
    "finbert_embedding": "finbert_embedding",
    "finbert_tok": "finbert_tok",
}

DATASET_ROOTS = {
    "csmd50": Path("/home/zhaokx/Pattern/Pattern_Mining/dataset/CSMD50"),
    "csmd300": Path("/home/zhaokx/Pattern/Pattern_Mining/dataset/CSMD300"),
    "massive": Path("/home/zhaokx/Pattern/Pattern_Mining/dataset/massive_data"),
}

BEST_SEEDS_GT_FULL = {
    "csmd50": {"pen": 45, "stocknet": 45, "teon": 46},
    "csmd300": {"pen": 43, "stocknet": 45, "teon": 45},
    "massive": {"pen": 44, "stocknet": 41, "teon": 45},
}
BEST_SEEDS_NO_GT = {
    "csmd50": {"pen": 45, "stocknet": 45, "teon": 41},
    "csmd300": {"pen": 43, "stocknet": 45, "teon": 41},
    "massive": {"pen": 44, "stocknet": 41, "teon": 42},
}
# Best seed per method (test MCC) from backup reproduce / meta.json
BEST_SEEDS_ABLATION = {
    "csmd50": {
        "pen": 45,
        "stocknet": 45,
        "gt_full": 46,
        "no_gt": 41,
        "gt_nonews": 45,
        "gt_noallvn": 44,
        "gt_noprice": 47,
    },
    "csmd300": {
        "pen": 43,
        "stocknet": 45,
        "gt_full": 45,
        "no_gt": 41,
        "gt_nonews": 47,
        "gt_noallvn": 45,
        "gt_noprice": 41,
    },
    "massive": {
        "pen": 44,
        "stocknet": 41,
        "gt_full": 42,
        "no_gt": 42,
        "gt_nonews": 47,
        "gt_noallvn": 43,
        "gt_noprice": 44,
    },
}
GT_ABLATION_METHODS = ("gt_full", "no_gt", "gt_nonews", "gt_noallvn", "gt_noprice")
GT_DISPLAY_LABELS = {
    "gt_full": "gt_full",
    "no_gt": "no_gt",
    "gt_nonews": "no_news",
    "gt_noallvn": "no_allvn",
    "gt_noprice": "no_price",
}
TEON_LABELS = {"gt_full": "TE-on (gt_full)", "no_gt": "no_gt"}

COLORS = {
    "PEN": "#1f77b4",
    "StockNet": "#ff7f0e",
    "TE-on": "#2ca02c",
    "TE-on (gt_full)": "#2ca02c",
    "no_gt": "#9467bd",
    "gt_full": "#2ca02c",
    "no_news": "#d62728",
    "no_allvn": "#8c564b",
    "no_price": "#e377c2",
}


def _resolve_news_sum_init_from_edge_tail(args):
    return (
        float(getattr(args, "edge_low_tail_ratio_window", 0.1)),
        float(getattr(args, "edge_high_tail_ratio_window", 0.3)),
        float(getattr(args, "edge_low_tail_ratio_type", 0.2)),
        float(getattr(args, "edge_high_tail_ratio_type", 0.2)),
    )


def _pen_ckpt(ds: str, seed: int) -> Path:
    if ds == "massive":
        return CONTRAST_MASSIVE / f"pen_massive_seed{seed}" / "best_massive.pt"
    return CONTRAST_VOL / f"pen_{ds}_seed{seed}" / f"best_{ds}.pt"


def _stocknet_ckpt(ds: str, seed: int) -> Path:
    if ds == "massive":
        return CONTRAST_MASSIVE / f"stocknet_massive_seed{seed}" / "best_massive.pt"
    return CONTRAST_VOL / f"stocknet_{ds}_seed{seed}" / f"best_{ds}.pt"


def _teon_ckpt_dir(ds: str, seed: int, method: str = "gt_full") -> Path:
    return BACKUP_GT / ds / method / f"s{seed}"


def _build_teon_model(args, profile, train_dataset, device):
    vocab_path = resolve_vocab_path(profile.vocab_path, "dict_csmd.pkl", "dict_cn.pkl")
    l2h_w, h2l_w, l2h_t, h2l_t = _resolve_news_sum_init_from_edge_tail(args)
    companies = list(getattr(_builder_dataset(train_dataset), "companies", []) or [])
    n_stocks = len(companies) if companies else int(profile.grid_num_stocks)
    model = SimpleModel(
        news_encode_mode=_MODEL_NEWS_MODE[str(args.news_encode_mode)],
        vocab_path=vocab_path,
        grid_num_stocks=n_stocks,
        price_input_dim=3,
        no_price_token=bool(getattr(args, "no_price_token", False)),
        news_ablation_mode=str(getattr(args, "news_ablation_mode", "full") or "full"),
        pred_fusion_ablation_mode=getattr(args, "pred_fusion_ablation_mode", "full"),
        vin_ablation_mode=getattr(args, "vin_ablation_mode", "full"),
        news_word_emb=str(getattr(args, "news_word_emb", "random") or "random"),
        news_process_mode=getattr(args, "news_process_mode", "default"),
        dual_own_prop_news=bool(getattr(args, "dual_own_prop_news", False)),
        own_news_only=bool(getattr(args, "own_news_only", False)),
        prop_same_trend_only=bool(getattr(args, "prop_same_trend_only", False)),
        companies=companies,
        price_encoder=str(getattr(args, "price_encoder", "lstm")),
        news_top_k=int(getattr(args, "news_padding_k", 5) or 5),
        te_dataset_root=(
            str(getattr(args, "te_dataset_root", "") or "").strip()
            or str(getattr(args, "dataset_root", "") or "").strip()
            or None
        ),
        **model_size_kwargs(args),
    )
    return wrap_model_devices(model, args, device)


@torch.no_grad()
def collect_teon_store(ckpt_dir: Path, device: torch.device) -> PredictionStore:
    man = json.loads((ckpt_dir / "run_manifest.json").read_text())
    args = normalize_run_args(SimpleNamespace(**dict(man.get("args") or {})), get_profile(str(man["args"]["dataset"])))
    args.device = str(device)
    set_all_seeds(int(args.seed))
    profile = get_profile(str(args.dataset))
    ds_kw = dataset_kwargs_from_args(args, profile)
    train_ds = MarketWindowDataset(profile, mode="train", **ds_kw)
    ds_eval = dict(ds_kw)
    if int(getattr(args, "subset_top_companies", 0) or 0) > 0:
        keep = list(getattr(_builder_dataset(train_ds), "companies", []) or [])
        ds_eval["subset_top_companies"] = 0
        ds_eval["companies_whitelist"] = keep
    test_loader = make_model_dataloader(
        MarketWindowDataset(profile, mode="test", **ds_eval), batch_size=1, shuffle=False
    )
    model = _build_teon_model(args, profile, train_ds, device)
    load_model_state_for_eval(model, str(ckpt_dir / "best.pt"), device)
    companies = list(getattr(_builder_dataset(train_ds), "companies", []) or profile.companies or [])

    store = PredictionStore()
    model.eval()
    for batch in test_loader:
        if batch["price"].x.numel() == 0 or batch["label"].x.numel() == 0:
            continue
        for nt in batch.node_types:
            for key, val in batch[nt].__dict__.items():
                if isinstance(val, torch.Tensor):
                    batch[nt][key] = val.to(device)

        def _fwd():
            return model(batch)

        with cuda_device_lock(device):
            logits = forward_with_cuda_retry(_fwd, device=device, tag="teon-portfolio")
        if logits.numel() == 0:
            continue

        price_dates = getattr(batch["price"], "date", None)
        if isinstance(price_dates, (list, tuple)) and len(price_dates) == 1 and isinstance(price_dates[0], (list, tuple)):
            price_dates = price_dates[0]
        if price_dates is None:
            continue

        labels = batch["label"].x.view(-1).cpu().numpy()
        label_org = (
            batch["label"].org.view(-1).cpu().numpy()
            if hasattr(batch["label"], "org")
            else labels
        )
        valid = (
            batch["label"].valid_mask.view(-1).cpu().numpy().astype(bool)
            if hasattr(batch["label"], "valid_mask")
            else np.ones_like(labels, dtype=bool)
        )
        comp_ids = batch["price"].company_id.view(-1).cpu().numpy()
        probs = torch.sigmoid(logits.view(-1)).cpu().numpy()

        for i in range(int(probs.shape[0])):
            if not valid[i]:
                continue
            if i >= len(price_dates):
                continue
            date_str = str(price_dates[i])
            ci = int(comp_ids[i])
            comp = companies[ci] if 0 <= ci < len(companies) else str(ci)
            y_org = float(label_org[i])
            rec = store.day_table[date_str].get(comp)
            if rec is None:
                rec = DayRecord()
                store.day_table[date_str][comp] = rec
            rec.prob_sum += float(probs[i])
            rec.count += 1
            rec.y_org = y_org
            rec.y_bin = float(labels[i])
    return store


@torch.no_grad()
def collect_pen_or_stocknet(
    model_kind: str,
    dataset: str,
    ckpt_path: Path,
    device: torch.device,
    *,
    use_volume: bool = True,
    batch_size: int = 256,
) -> PredictionStore:
    root = str(DATASET_ROOTS[dataset])
    ds = CminCnPenDataset(
        dataset_root=root,
        mode="test",
        use_volume=use_volume,
        split_mode="contiguous",
        seed=42,
    )

    def _getitem_with_mv(idx):
        item = ds[idx]
        item["main_mv"] = float(ds.samples[idx]["main_mv"])
        return item

    loader = torch.utils.data.DataLoader(
        range(len(ds)),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda idxs: [_getitem_with_mv(i) for i in idxs],
    )
    vocab_path, vocab_size, pad_id = _vocab_info(root)

    if model_kind == "pen":
        from argparse import Namespace

        args_ns = Namespace(
            max_n_days=5,
            max_n_msgs=5,
            max_n_words=30,
            use_volume=use_volume,
            price_input_size=4 if use_volume else 3,
            variant_type="hedge",
            vmd_rec="zh",
            mel_cell_type="gru",
            vmd_cell_type="gru",
            daily_att="y",
            alpha=0.5,
            dropout_mel_in=0.3,
            dropout_mel=0.0,
            dropout_vmd_in=0.3,
            dropout_vmd=0.0,
            word_embed_size=50,
            mel_h_size=100,
            msin_h_size=100,
            h_size=150,
            g_size=50,
        )
        model = _build_model(vocab_size, pad_id, args_ns).to(device)
    else:
        price_dim = 4 if use_volume else 3
        model = StockNetModel(
            vocab_size=vocab_size,
            pad_id=pad_id,
            word_embed_size=50,
            mel_h_size=100,
            h_size=150,
            g_size=50,
            max_n_days=5,
            max_n_msgs=5,
            max_n_words=30,
            variant_type="hedge",
            vmd_rec="zh",
            daily_att="y",
            alpha=0.5,
            price_input_size=price_dim,
        ).to(device)

    state, meta = load_ckpt(str(ckpt_path), device)
    model.load_state_dict(state)
    if "kl_lambda" in meta:
        model.set_kl_lambda(meta["kl_lambda"])
    model.eval()

    store = PredictionStore()
    max_n_days = 5
    for raw in loader:
        batch = collate_pen_batch(raw, max_n_days)
        out = model(
            batch["word_ids"].to(device),
            batch["n_words"].to(device),
            batch["ss_index"].to(device),
            batch["n_msgs"].to(device),
            batch["price"].to(device),
            batch["y"].to(device),
            batch["t_idx"].to(device),
            compute_loss=False,
        )
        y_T = torch.softmax(out["y_T"], dim=-1)
        probs = y_T[:, 1].cpu().numpy()
        for i in range(len(batch["companies"])):
            comp = batch["companies"][i]
            date_str = batch["dates"][i]
            y_org = float(batch["main_mv"][i].cpu().item())
            if not np.isfinite(y_org):
                continue
            prob = float(probs[i])
            rec = store.day_table[date_str].get(comp)
            if rec is None:
                rec = DayRecord()
                store.day_table[date_str][comp] = rec
            rec.prob_sum += prob
            rec.count += 1
            rec.y_org = y_org
            rec.y_bin = 1.0 if y_org > 0 else 0.0
    return store


def plot_dataset(dataset: str, runs: list[dict], out_dir: Path, budget_init: float, *, topk: int, max_days: int | None) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for run in runs:
        b = np.array(run["budget"], dtype=np.float64)
        if len(b) == 0:
            continue
        ax.plot(
            np.arange(len(b)),
            b,
            label=f"{run['label']} (s{run['seed']}, ARR={run['metrics']['arr']:.3f})",
            color=COLORS.get(run["label"]),
            linewidth=1.8,
        )
    ax.axhline(budget_init, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, label="initial")
    day_note = f"first {max_days} days" if max_days else "full test"
    ax.set_title(f"{dataset.upper()} — portfolio top-{topk} cumulative budget (test, {day_note})")
    ax.set_xlabel("trading day index")
    ax.set_ylabel("portfolio value")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = out_dir / f"compare_pen_stocknet_teon_{dataset}.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"[plot] {png}")


def plot_first100idx_dataset(
    dataset: str,
    runs: list[dict],
    out_dir: Path,
    budget_init: float,
    *,
    topk: int,
    n_idx: int = 100,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for run in runs:
        b = np.array(run["budget"][:n_idx], dtype=np.float64)
        if len(b) == 0:
            continue
        final = float(b[-1])
        ax.plot(
            np.arange(len(b)),
            b,
            label=f"{run['label']} (s{run['seed']}, final={final:.0f})",
            color=COLORS.get(run["label"]),
            linewidth=1.8,
        )
    ax.axhline(budget_init, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, label="initial")
    ax.set_title(
        f"{dataset.upper()} — top-{topk} portfolio (first {n_idx} trading-day indices, test)"
    )
    ax.set_xlabel("trading day index (0-based)")
    ax.set_ylabel("portfolio value")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = out_dir / f"compare_ablation_topk1_{dataset}_first100idx.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"[plot] {png}")


def _reuse_topk1_runs(dataset: str) -> list[dict]:
    topk1_path = RETURN_ROOT / "plots/compare_pen_stocknet_teon_topk1/compare_summary.json"
    if not topk1_path.exists():
        return []
    rows = json.loads(topk1_path.read_text()).get(dataset, [])
    out = []
    for r in rows:
        label = r["label"]
        if label in ("TE-on", "TE-on (gt_full)"):
            label = "gt_full"
        out.append({**r, "label": label})
    return out


def run_first100idx_ablation(args, device: torch.device, out_dir: Path) -> None:
    """Full-test top-1 curves; plot first N indices (aligned with topk1 full-plot morphology)."""
    n_idx = int(args.first100_idx or 100)
    all_summary: dict = {}
    summary_path = out_dir / "compare_summary_full.json"

    for dataset in args.datasets:
        seeds = BEST_SEEDS_ABLATION[dataset]
        use_volume = True
        print(f"\n=== {dataset} ===")
        reused = _reuse_topk1_runs(dataset)
        reused_labels = {r["label"] for r in reused}
        runs = list(reused)

        if "PEN" not in reused_labels:
            pen_ckpt = _pen_ckpt(dataset, seeds["pen"])
            print(f"PEN      s{seeds['pen']}: {pen_ckpt}")
            store = collect_pen_or_stocknet("pen", dataset, pen_ckpt, device, use_volume=use_volume)
            res = run_portfolio_strategy(store, name="PEN", mode="model_topk", topk=args.topk, budget=args.budget)
            runs.append(
                {"label": "PEN", "seed": seeds["pen"], "ckpt": str(pen_ckpt), "dates": res.dates, "budget": res.budget, "metrics": res.metrics}
            )

        if "StockNet" not in reused_labels:
            sn_ckpt = _stocknet_ckpt(dataset, seeds["stocknet"])
            print(f"StockNet s{seeds['stocknet']}: {sn_ckpt}")
            store = collect_pen_or_stocknet("stocknet", dataset, sn_ckpt, device, use_volume=use_volume)
            res = run_portfolio_strategy(store, name="StockNet", mode="model_topk", topk=args.topk, budget=args.budget)
            runs.append(
                {"label": "StockNet", "seed": seeds["stocknet"], "ckpt": str(sn_ckpt), "dates": res.dates, "budget": res.budget, "metrics": res.metrics}
            )

        have_gt = {r.get("method", r["label"]) for r in runs}
        for method in GT_ABLATION_METHODS:
            label = GT_DISPLAY_LABELS[method]
            if method in have_gt or label in reused_labels:
                print(f"{label:10s} reused from topk1")
                continue
            seed = seeds[method]
            ckpt_dir = _teon_ckpt_dir(dataset, seed, method)
            print(f"{label:10s} s{seed}: {ckpt_dir}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            store = collect_teon_store(ckpt_dir, device)
            res = run_portfolio_strategy(store, name=label, mode="model_topk", topk=args.topk, budget=args.budget)
            runs.append(
                {"label": label, "method": method, "seed": seed, "ckpt": str(ckpt_dir), "dates": res.dates, "budget": res.budget, "metrics": res.metrics}
            )

        # stable legend order
        order = ["PEN", "StockNet", "gt_full", "no_gt", "no_news", "no_allvn", "no_price"]
        runs.sort(key=lambda r: order.index(r["label"]) if r["label"] in order else 99)
        plot_first100idx_dataset(dataset, runs, out_dir, args.budget, topk=args.topk, n_idx=n_idx)
        all_summary[dataset] = runs

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)
    print(f"[done] full summary -> {summary_path}")


def run_ablation_suite(args, device: torch.device, out_dir: Path) -> None:
    all_summary = {}
    summary_path = out_dir / "compare_summary.json"
    if summary_path.exists():
        all_summary = json.loads(summary_path.read_text())

    for dataset in args.datasets:
        seeds = BEST_SEEDS_ABLATION[dataset]
        use_volume = True
        print(f"\n=== {dataset} ===")
        runs = []

        pen_ckpt = _pen_ckpt(dataset, seeds["pen"])
        print(f"PEN      s{seeds['pen']}: {pen_ckpt}")
        store = collect_pen_or_stocknet("pen", dataset, pen_ckpt, device, use_volume=use_volume)
        res = run_portfolio_strategy(
            store, name="PEN", mode="model_topk", topk=args.topk, budget=args.budget, max_days=args.max_days
        )
        runs.append(
            {"label": "PEN", "seed": seeds["pen"], "ckpt": str(pen_ckpt), "dates": res.dates, "budget": res.budget, "metrics": res.metrics}
        )

        sn_ckpt = _stocknet_ckpt(dataset, seeds["stocknet"])
        print(f"StockNet s{seeds['stocknet']}: {sn_ckpt}")
        store = collect_pen_or_stocknet("stocknet", dataset, sn_ckpt, device, use_volume=use_volume)
        res = run_portfolio_strategy(
            store, name="StockNet", mode="model_topk", topk=args.topk, budget=args.budget, max_days=args.max_days
        )
        runs.append(
            {"label": "StockNet", "seed": seeds["stocknet"], "ckpt": str(sn_ckpt), "dates": res.dates, "budget": res.budget, "metrics": res.metrics}
        )

        for method in GT_ABLATION_METHODS:
            label = GT_DISPLAY_LABELS[method]
            seed = seeds[method]
            ckpt_dir = _teon_ckpt_dir(dataset, seed, method)
            print(f"{label:10s} s{seed}: {ckpt_dir}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            store = collect_teon_store(ckpt_dir, device)
            res = run_portfolio_strategy(
                store, name=label, mode="model_topk", topk=args.topk, budget=args.budget, max_days=args.max_days
            )
            runs.append(
                {"label": label, "method": method, "seed": seed, "ckpt": str(ckpt_dir), "dates": res.dates, "budget": res.budget, "metrics": res.metrics}
            )

        plot_dataset(dataset, runs, out_dir, args.budget, topk=args.topk, max_days=args.max_days)
        all_summary[dataset] = runs

    summary_path = out_dir / "compare_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)
    print(f"[done] summary -> {summary_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max_days", type=int, default=None, help="limit to first N trading days in test")
    ap.add_argument("--budget", type=float, default=10000.0)
    ap.add_argument("--teon_method", choices=("gt_full", "no_gt"), default="gt_full")
    ap.add_argument("--suite", choices=("default", "ablation", "first100idx_ablation"), default="default")
    ap.add_argument("--first100_idx", type=int, default=100, help="plot first N trading-day indices from full-test curves")
    ap.add_argument("--datasets", nargs="+", default=["csmd50", "csmd300", "massive"])
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.suite == "first100idx_ablation":
        if args.out_dir is None:
            args.out_dir = RETURN_ROOT / "plots/compare_pen_stocknet_teon_topk1_first100idx"
        out_dir = args.out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        args.topk = 1
        run_first100idx_ablation(args, device, out_dir)
        return

    if args.suite == "ablation":
        if args.out_dir is None:
            args.out_dir = RETURN_ROOT / "plots/compare_ablation_suite"
        out_dir = args.out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        run_ablation_suite(args, device, out_dir)
        return

    best_seeds = BEST_SEEDS_NO_GT if args.teon_method == "no_gt" else BEST_SEEDS_GT_FULL
    teon_label = TEON_LABELS[args.teon_method]
    if args.out_dir is None:
        suffix = "no_gt" if args.teon_method == "no_gt" else "teon"
        args.out_dir = RETURN_ROOT / f"plots/compare_pen_stocknet_{suffix}"
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summary = {}
    for dataset in args.datasets:
        seeds = best_seeds[dataset]
        use_volume = True  # contrast checkpoints trained with volume (4-d price)
        print(f"\n=== {dataset} ===")
        runs = []

        pen_ckpt = _pen_ckpt(dataset, seeds["pen"])
        print(f"PEN      s{seeds['pen']}: {pen_ckpt}")
        store = collect_pen_or_stocknet("pen", dataset, pen_ckpt, device, use_volume=use_volume)
        res = run_portfolio_strategy(store, name="PEN", mode="model_topk", topk=args.topk, budget=args.budget, max_days=args.max_days)
        runs.append({"label": "PEN", "seed": seeds["pen"], "ckpt": str(pen_ckpt), "dates": res.dates, "budget": res.budget, "metrics": res.metrics})

        sn_ckpt = _stocknet_ckpt(dataset, seeds["stocknet"])
        print(f"StockNet s{seeds['stocknet']}: {sn_ckpt}")
        store = collect_pen_or_stocknet("stocknet", dataset, sn_ckpt, device, use_volume=use_volume)
        res = run_portfolio_strategy(store, name="StockNet", mode="model_topk", topk=args.topk, budget=args.budget, max_days=args.max_days)
        runs.append({"label": "StockNet", "seed": seeds["stocknet"], "ckpt": str(sn_ckpt), "dates": res.dates, "budget": res.budget, "metrics": res.metrics})

        teon_dir = _teon_ckpt_dir(dataset, seeds["teon"], args.teon_method)
        print(f"{teon_label} s{seeds['teon']}: {teon_dir}")
        store = collect_teon_store(teon_dir, device)
        res = run_portfolio_strategy(store, name=teon_label, mode="model_topk", topk=args.topk, budget=args.budget, max_days=args.max_days)
        runs.append({"label": teon_label, "seed": seeds["teon"], "ckpt": str(teon_dir), "dates": res.dates, "budget": res.budget, "metrics": res.metrics})

        plot_dataset(dataset, runs, out_dir, args.budget, topk=args.topk, max_days=args.max_days)
        all_summary[dataset] = runs

    summary_path = out_dir / "compare_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)
    print(f"[done] summary -> {summary_path}")


if __name__ == "__main__":
    main()
