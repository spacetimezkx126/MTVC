#!/usr/bin/env python3
"""Optional helper: rebuild MTVC L6 prediction stores / PEN·StockNet·MTVC compare curves.

Paper figures: use ``plot_paper_casel6_pre924.py`` → ``plots/paper_casel6_pre924/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
# Compatible with:
#   MTVC_paper_repro/eval/plot_compare_*.py
#   MTVC_paper_repro/return_rate_analysis/scripts/plot_compare_*.py
if SCRIPT_DIR.name == "scripts" and SCRIPT_DIR.parent.name == "return_rate_analysis":
    REPRO = SCRIPT_DIR.parent.parent
    RETURN_ROOT = SCRIPT_DIR.parent
else:
    REPRO = SCRIPT_DIR.parent
    RETURN_ROOT = REPRO / "return_rate_analysis"
DUAL_TF = REPRO.parent
MTVC_PKG = REPRO / "code" / "mtvc"
MTVC_SUPPORT = REPRO / "code" / "support"
MTVC_CKPT = REPRO / "checkpoints"
HUB_CKPT = DUAL_TF / "formal_release/diagram_pn_joint_contrast/checkpoints"
CACHE_DIR = RETURN_ROOT / "cache/prediction_stores_casel6"

# Match prior PEN/StockNet/TE-on compare: top-5, full test period
DEFAULT_TOPK = 5

# Best seed by test MCC (best.pt.meta.json)
MTVC_BEST = {
    "csmd50": {
        "exp": "full_casel6",
        "seed": 42,
    },
    "csmd300": {
        "exp": "full_casel6",
        "seed": 42,
    },
    "massive": {
        "exp": "full_casel6",
        "seed": 46,
    },
}
BREAK = "2024-09-24"

PEN_STOCKNET_BEST = {
    "csmd50": {"pen": 45, "stocknet": 45},
    "csmd300": {"pen": 43, "stocknet": 45},
    "massive": {"pen": 44, "stocknet": 41},
}

COLORS = {
    "PEN": "#1f77b4",
    "StockNet": "#ff7f0e",
    "MTVC": "#2ca02c",
}


def _setup_env() -> None:
    os.chdir(str(DUAL_TF))
    os.environ["DUAL_TF_MODEL_DIR"] = str(DUAL_TF)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["MTVC_USE_TYPE_EMB"] = "1"
    os.environ["MTVC_UNIFIED_CAUSAL"] = "1"
    os.environ["MTVC_UNIFIED_SAME_DAY"] = "0"
    os.environ["MTVC_SOFT_PRUNE"] = "0"
    os.environ["MTVC_SHARE_PRICE_NEWS_ENC"] = "0"
    os.environ["MTVC_TRANSFORMER_LAYERS"] = "6"
    os.environ.pop("MTVC_UNIFIED_LAYERS", None)
    os.environ.pop("MTVC_HARD_FINETUNE", None)
    os.environ.pop("MTVC_LOOP_ROUNDS", None)
    for p in (str(DUAL_TF), str(RETURN_ROOT), str(SCRIPT_DIR), str(MTVC_SUPPORT), str(MTVC_PKG)):
        if p not in sys.path:
            sys.path.insert(0, p)


_setup_env()

from lib_portfolio_store import (  # noqa: E402
    _get_store,
    _resolve_news_sum_init_from_edge_tail,
    load_store,
    save_store,
    store_cache_path,
)
from lib_pen_stocknet import (  # noqa: E402
    _pen_ckpt,
    _stocknet_ckpt,
    collect_pen_or_stocknet,
)
from backtest.portfolio_sim import DayRecord, PredictionStore, run_portfolio_strategy  # noqa: E402


def _mtvc_import(name: str):
    import importlib

    for mod in (
        "model",
        "train",
        "checkpoint",
        "data",
        "tool.cuda_guard",
        "contrast_aux",
        "partial_unified_backbone",
        "weighted_contrast_aux",
        "news_halt_aux",
    ):
        sys.modules.pop(mod, None)
    # Prefer MTVC package first
    while str(MTVC_PKG) in sys.path:
        sys.path.remove(str(MTVC_PKG))
    sys.path.insert(0, str(MTVC_PKG))
    return importlib.import_module(name)


def _build_mtvc_model(args, profile, train_dataset, device):
    ov_data = _mtvc_import("data")
    ov_train = _mtvc_import("train")
    ov_model = _mtvc_import("model")
    ov_mtvc = _mtvc_import("contrast_aux")
    SimpleModel = ov_model.Model

    resolve_vocab_path = ov_data.resolve_vocab_path
    _builder_dataset = ov_train._builder_dataset
    model_finbert_kwargs = ov_train.model_finbert_kwargs
    model_size_kwargs = ov_train.model_size_kwargs
    wrap_model_devices = ov_train.wrap_model_devices

    vocab_path = resolve_vocab_path(profile.vocab_path, "dict_csmd.pkl", "dict_cn.pkl")
    l2h_w, h2l_w, l2h_t, h2l_t = _resolve_news_sum_init_from_edge_tail(args)
    companies = list(getattr(_builder_dataset(train_dataset), "companies", []) or [])
    n_stocks = len(companies) if companies else int(profile.grid_num_stocks)
    model = SimpleModel(
        news_encode_mode={
            "finbert": "finbert_padding",
            "vocab": "vocab_padding",
            "finbert_embedding": "finbert_embedding",
            "finbert_tok": "finbert_tok",
        }[str(args.news_encode_mode)],
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
        price_encoder=str(getattr(args, "price_encoder", "partial_unified")),
        news_top_k=int(getattr(args, "news_padding_k", 5) or 5),
        te_dataset_root=(
            str(getattr(args, "te_dataset_root", "") or "").strip()
            or str(getattr(args, "dataset_root", "") or "").strip()
            or None
        ),
        **model_size_kwargs(args),
    )
    model = wrap_model_devices(model, args, device)
    ov_mtvc.enable_contrast_gates(model)
    return model


@torch.no_grad()
def collect_mtvc_store(ckpt_dir: Path, device: torch.device) -> PredictionStore:
    ov_data = _mtvc_import("data")
    ov_train = _mtvc_import("train")
    ov_ckpt = _mtvc_import("checkpoint")
    ov_cuda = _mtvc_import("tool.cuda_guard")

    MarketWindowDataset = ov_data.MarketWindowDataset
    get_profile = ov_data.get_profile
    dataset_kwargs_from_args = ov_train.dataset_kwargs_from_args
    normalize_run_args = ov_train.normalize_run_args
    set_all_seeds = ov_train.set_all_seeds
    _builder_dataset = ov_train._builder_dataset
    load_model_state_for_eval = ov_ckpt.load_model_state_for_eval
    cuda_device_lock = ov_cuda.cuda_device_lock
    forward_with_cuda_retry = ov_cuda.forward_with_cuda_retry

    man = json.loads((ckpt_dir / "run_manifest.json").read_text())
    args = normalize_run_args(
        SimpleNamespace(**dict(man.get("args") or {})),
        get_profile(str(man["args"]["dataset"])),
    )
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
    test_ds = MarketWindowDataset(profile, mode="test", **ds_eval)

    def _collate(xs):
        s = xs[0]
        return s.for_model() if hasattr(s, "for_model") else s

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=_collate)
    model = _build_mtvc_model(args, profile, train_ds, device)
    load_model_state_for_eval(model, str(ckpt_dir / "best.pt"), device)
    companies = list(getattr(_builder_dataset(train_ds), "companies", []) or profile.companies or [])

    store = PredictionStore()
    model.eval()
    move_batch = ov_train.move_model_batch_to_device
    for batch in test_loader:
        if not hasattr(batch, "node_types"):
            continue
        if batch["price"].x.numel() == 0 or batch["label"].x.numel() == 0:
            continue
        batch = move_batch(batch, device)

        def _fwd():
            return model(batch)

        with cuda_device_lock(device):
            logits = forward_with_cuda_retry(_fwd, device=device, tag="mtvc-portfolio")
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
            if not valid[i] or i >= len(price_dates):
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


def _reuse_baseline_store(dataset: str, label: str, seed: int) -> PredictionStore | None:
    """Reuse novol Acc/MCC prediction stores when present."""
    method = "pen" if label == "PEN" else "stocknet"
    novol = DUAL_TF / "return_rate_analysis/cache/prediction_stores_novol_accmcc"
    path = novol / f"{dataset}_{method}_s{seed}.pkl"
    if path.exists():
        print(f"[cache-reuse] {label} s{seed}: {path}")
        return load_store(path)
    return None


def _filter_store_before(store: PredictionStore, end_before: str) -> PredictionStore:
    out = PredictionStore()
    for d, stocks in store.day_table.items():
        if d >= end_before:
            continue
        out.day_table[d] = stocks
    return out


def ensure_stores(dataset: str, device: torch.device, *, use_cache: bool) -> dict[str, PredictionStore]:
    """Only MTVC needs inference; PEN/StockNet come from teon summary when available."""
    mtvc_cfg = MTVC_BEST[dataset]
    ckpt_dir = MTVC_CKPT / dataset / mtvc_cfg["exp"] / f"s{mtvc_cfg['seed']}"
    if not (ckpt_dir / "best.pt").is_file():
        # fallback to formal hub names
        ckpt_dir = HUB_CKPT / dataset / mtvc_cfg["exp"] / f"s{mtvc_cfg['seed']}"
    print(f"\n=== {dataset} ===")
    print(f"MTVC s{mtvc_cfg['seed']}: {ckpt_dir}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    store = _get_store(
        dataset=dataset,
        label=f"MTVC_{mtvc_cfg['exp']}",
        seed=mtvc_cfg["seed"],
        collect_fn=lambda: collect_mtvc_store(ckpt_dir, device),
        cache_dir=CACHE_DIR,
        use_cache=use_cache,
    )
    if dataset in ("csmd50", "csmd300"):
        store = _filter_store_before(store, BREAK)
    return {"MTVC": store}


def _align_runs_to_common_dates(runs: list[dict], budget_init: float) -> list[dict]:
    """Intersect trading dates across methods and recompute ARR/SR on the shared window."""
    from backtest.portfolio_sim import _portfolio_metrics

    common = None
    date_to_budget = []
    for run in runs:
        d2b = {d: b for d, b in zip(run["dates"], run["budget"])}
        date_to_budget.append(d2b)
        s = set(run["dates"])
        common = s if common is None else (common & s)
    if not common:
        return runs
    common_dates = sorted(common)
    out = []
    for run, d2b in zip(runs, date_to_budget):
        budget = [float(d2b[d]) for d in common_dates]
        # portfolio_sim metrics expect asset series starting from initial budget
        asset = [float(budget_init)] + budget
        metrics = _portfolio_metrics(asset)
        metrics["n_trading_days"] = len(common_dates)
        out.append({**run, "dates": common_dates, "budget": budget, "metrics": metrics})
    return out


def plot_dataset(dataset: str, runs: list[dict], out_dir: Path, budget_init: float, *, topk: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for run in runs:
        b = np.array(run["budget"], dtype=np.float64)
        if len(b) == 0:
            continue
        ax.plot(
            np.arange(len(b)),
            b,
            label=f"{run['label']} (s{run['seed']}, ARR={run['metrics']['arr']:.3f}, SR={run['metrics']['sr']:.3f})",
            color=COLORS.get(run["label"]),
            linewidth=1.8,
        )
    ax.axhline(budget_init, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, label="initial")
    ax.set_title(f"{dataset.upper()} — portfolio top-{topk} cumulative budget (full test)")
    ax.set_xlabel("trading day index")
    ax.set_ylabel("portfolio value")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = out_dir / f"return_{dataset}.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"[plot] {png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    ap.add_argument("--budget", type=float, default=10000.0)
    ap.add_argument("--datasets", nargs="+", default=["csmd50", "csmd300", "massive"])
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=RETURN_ROOT / "plots/paper_casel6_pre924",
    )
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summary: dict = {}
    for dataset in args.datasets:
        seeds = PEN_STOCKNET_BEST[dataset]
        mtvc_cfg = MTVC_BEST[dataset]
        runs: list[dict] = []

        for label, key in (("PEN", "pen"), ("StockNet", "stocknet")):
            seed = seeds[key]
            ckpt = _pen_ckpt(dataset, seed) if key == "pen" else _stocknet_ckpt(dataset, seed)
            reused = _reuse_baseline_store(dataset, label, seed)
            if reused is not None:
                store = reused
            else:
                store = _get_store(
                    dataset=dataset,
                    label=label,
                    seed=seed,
                    collect_fn=lambda k=key, c=ckpt: collect_pen_or_stocknet(
                        k, dataset, c, device
                    ),
                    cache_dir=CACHE_DIR,
                    use_cache=not args.no_cache,
                )
            if dataset in ("csmd50", "csmd300"):
                store = _filter_store_before(store, BREAK)
            res = run_portfolio_strategy(
                store, name=label, mode="model_topk", topk=args.topk, budget=args.budget, max_days=None
            )
            print(f"[baseline] {label} s{seed} days={res.metrics['n_trading_days']}")
            runs.append(
                {
                    "label": label,
                    "seed": seed,
                    "ckpt": str(ckpt),
                    "dates": res.dates,
                    "budget": res.budget,
                    "metrics": res.metrics,
                }
            )

        stores = ensure_stores(dataset, device, use_cache=not args.no_cache)
        res = run_portfolio_strategy(
            stores["MTVC"],
            name="MTVC",
            mode="model_topk",
            topk=args.topk,
            budget=args.budget,
            max_days=None,
        )
        runs.append(
            {
                "label": "MTVC",
                "seed": mtvc_cfg["seed"],
                "ckpt": str(MTVC_CKPT / dataset / mtvc_cfg["exp"] / f"s{mtvc_cfg['seed']}"),
                "dates": res.dates,
                "budget": res.budget,
                "metrics": res.metrics,
            }
        )

        runs = _align_runs_to_common_dates(runs, args.budget)
        print(f"[align] common days={runs[0]['metrics']['n_trading_days']}")
        for run in runs:
            m = run["metrics"]
            print(
                f"  {run['label']:9s} ARR={m['arr']*100:.2f}%  SR={m['sr']:.3f}  "
                f"final={m['final_budget']:.1f}  days={m['n_trading_days']}"
            )
        plot_dataset(dataset, runs, out_dir, args.budget, topk=args.topk)
        all_summary[dataset] = runs

    summary_path = out_dir / "compare_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)
    print(f"[done] summary -> {summary_path}")


if __name__ == "__main__":
    main()
