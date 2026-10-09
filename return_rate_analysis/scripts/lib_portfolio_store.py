"""Shared helpers: PEN / StockNet / expand_news_attn Diagram portfolio backtest."""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SCRIPT_DIR = Path(__file__).resolve().parent
RETURN_ROOT = SCRIPT_DIR.parent
# dual_tf/return_rate_analysis → parent=dual_tf
# MTVC_paper_repro/return_rate_analysis → parent=repro, grandparent=dual_tf
if (RETURN_ROOT.parent / "code" / "mtvc").is_dir():
    REPRO_ROOT = RETURN_ROOT.parent
    DUAL_TF = REPRO_ROOT.parent
else:
    REPRO_ROOT = None
    DUAL_TF = RETURN_ROOT.parent
EXPAND_ROOT = DUAL_TF / "formal_release/diagram_arch_notes/expand_news_attn"
OVERLAY = EXPAND_ROOT / "code_overlay"
EXPAND_CKPT = EXPAND_ROOT / "checkpoints"
CONTRAST_ROOT = DUAL_TF / "contrast_exp_backup_20260816/contrast_exp"
CACHE_DIR = RETURN_ROOT / "cache/prediction_stores_expand_diagram"

DATASETS = ("csmd50", "csmd300", "massive")
RETURN_METRICS = ("arr", "sr")

# Best seed by test Acc+MCC (best.pt.meta.json)
EXPAND_BEST = {
    "csmd50": {"method": "gt_full_notype", "seed": 43, "label": "gt_full_notype"},
    "csmd300": {"method": "gt_full_notype", "seed": 41, "label": "gt_full_notype"},
    "massive": {"method": "gt_noglobal", "seed": 44, "label": "gt_noglobal"},
}

PEN_STOCKNET_BEST = {
    "csmd50": {"pen": 45, "stocknet": 45},
    "csmd300": {"pen": 43, "stocknet": 45},
    "massive": {"pen": 44, "stocknet": 41},
}


def setup_import_paths() -> None:
    os.chdir(str(DUAL_TF))
    os.environ["DUAL_TF_MODEL_DIR"] = str(DUAL_TF)
    for p in (str(DUAL_TF), str(CONTRAST_ROOT), str(RETURN_ROOT), str(SCRIPT_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


setup_import_paths()

from backtest.portfolio_sim import DayRecord, PredictionStore, run_portfolio_strategy  # noqa: E402
from lib_pen_stocknet import (  # noqa: E402
    _pen_ckpt,
    _stocknet_ckpt,
    collect_pen_or_stocknet,
)


def diagram_label(dataset: str) -> str:
    return EXPAND_BEST[dataset]["label"]


def expand_ckpt_dir(dataset: str) -> Path:
    cfg = EXPAND_BEST[dataset]
    return EXPAND_CKPT / dataset / cfg["method"] / f"s{cfg['seed']}"


def store_cache_path(cache_dir: Path, dataset: str, label: str, seed: int) -> Path:
    safe = label.replace("/", "_")
    return cache_dir / f"{dataset}_{safe}_s{seed}.pkl"


def save_store(store: PredictionStore, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(dict(store.day_table), f)


def load_store(path: Path) -> PredictionStore:
    with path.open("rb") as f:
        day_table = pickle.load(f)
    store = PredictionStore()
    store.day_table = day_table
    return store


def _overlay_import(name: str):
    import importlib

    for mod in ("model", "train", "checkpoint", "data", "cuda_guard"):
        sys.modules.pop(mod, None)
    if str(OVERLAY) in sys.path:
        sys.path.remove(str(OVERLAY))
    sys.path.insert(0, str(OVERLAY))
    return importlib.import_module(name)


def _resolve_news_sum_init_from_edge_tail(args):
    return (
        float(getattr(args, "edge_low_tail_ratio_window", 0.1)),
        float(getattr(args, "edge_high_tail_ratio_window", 0.3)),
        float(getattr(args, "edge_low_tail_ratio_type", 0.2)),
        float(getattr(args, "edge_high_tail_ratio_type", 0.2)),
    )


def _build_diagram_model(args, profile, train_dataset, device):
    ov_data = _overlay_import("data")
    ov_train = _overlay_import("train")
    ov_model = _overlay_import("model")
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
        price_encoder=str(getattr(args, "price_encoder", "cross_attn")),
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
def collect_diagram_store(ckpt_dir: Path, device: torch.device) -> PredictionStore:
    ov_data = _overlay_import("data")
    ov_train = _overlay_import("train")
    ov_ckpt = _overlay_import("checkpoint")
    ov_cuda = _overlay_import("cuda_guard")

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
    test_ds = MarketWindowDataset(profile, mode="test", **ds_eval)

    def _collate(xs):
        s = xs[0]
        return s.for_model() if hasattr(s, "for_model") else s

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=_collate)
    model = _build_diagram_model(args, profile, train_ds, device)
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
            logits = forward_with_cuda_retry(_fwd, device=device, tag="diagram-portfolio")
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


def _get_store(
    *,
    dataset: str,
    label: str,
    seed: int,
    collect_fn,
    cache_dir: Path,
    use_cache: bool,
) -> PredictionStore:
    cache_path = store_cache_path(cache_dir, dataset, label, seed)
    if use_cache and cache_path.exists():
        print(f"[cache] {label} s{seed}: {cache_path}")
        return load_store(cache_path)
    store = collect_fn()
    if use_cache:
        save_store(store, cache_path)
        print(f"[cache] saved {cache_path}")
    return store


def ensure_dataset_stores(
    dataset: str,
    device: torch.device,
    *,
    cache_dir: Path = CACHE_DIR,
    use_cache: bool = True,
) -> dict[str, PredictionStore]:
    """Return {PEN, StockNet, diagram_label} -> PredictionStore."""
    seeds = PEN_STOCKNET_BEST[dataset]
    expand_cfg = EXPAND_BEST[dataset]
    diag_label = expand_cfg["label"]

    pen_ckpt = _pen_ckpt(dataset, seeds["pen"])
    sn_ckpt = _stocknet_ckpt(dataset, seeds["stocknet"])
    ckpt_dir = expand_ckpt_dir(dataset)

    print(f"\n=== {dataset} ===")
    print(f"PEN           s{seeds['pen']}: {pen_ckpt}")
    print(f"StockNet      s{seeds['stocknet']}: {sn_ckpt}")
    print(f"{diag_label:13s} s{expand_cfg['seed']}: {ckpt_dir}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "PEN": _get_store(
            dataset=dataset,
            label="PEN",
            seed=seeds["pen"],
            collect_fn=lambda: collect_pen_or_stocknet("pen", dataset, pen_ckpt, device, use_volume=True),
            cache_dir=cache_dir,
            use_cache=use_cache,
        ),
        "StockNet": _get_store(
            dataset=dataset,
            label="StockNet",
            seed=seeds["stocknet"],
            collect_fn=lambda: collect_pen_or_stocknet("stocknet", dataset, sn_ckpt, device, use_volume=True),
            cache_dir=cache_dir,
            use_cache=use_cache,
        ),
        diag_label: _get_store(
            dataset=dataset,
            label=diag_label,
            seed=expand_cfg["seed"],
            collect_fn=lambda: collect_diagram_store(ckpt_dir, device),
            cache_dir=cache_dir,
            use_cache=use_cache,
        ),
    }


def eval_portfolio(
    store: PredictionStore,
    *,
    topk: int,
    max_days: int | None,
    budget: float,
) -> dict:
    res = run_portfolio_strategy(
        store, name="x", mode="model_topk", topk=topk, budget=budget, max_days=max_days
    )
    out = {k: float(res.metrics[k]) for k in RETURN_METRICS}
    out["n_trading_days"] = int(res.metrics.get("n_trading_days", len(res.dates)))
    out["dates"] = res.dates
    out["budget"] = res.budget
    return out


def compare_dataset(
    dataset: str,
    stores: dict[str, PredictionStore],
    *,
    topk: int,
    max_days: int | None,
    budget: float,
) -> dict:
    diag_label = diagram_label(dataset)
    seeds = PEN_STOCKNET_BEST[dataset]
    expand_cfg = EXPAND_BEST[dataset]

    ckpts = {
        "PEN": str(_pen_ckpt(dataset, seeds["pen"])),
        "StockNet": str(_stocknet_ckpt(dataset, seeds["stocknet"])),
        diag_label: str(expand_ckpt_dir(dataset)),
    }
    seed_map = {"PEN": seeds["pen"], "StockNet": seeds["stocknet"], diag_label: expand_cfg["seed"]}

    methods = {}
    for name, store in stores.items():
        metrics = eval_portfolio(store, topk=topk, max_days=max_days, budget=budget)
        methods[name] = {
            "label": name,
            "seed": seed_map[name],
            "ckpt": ckpts[name],
            **metrics,
        }

    pen = methods["PEN"]
    sn = methods["StockNet"]
    diag = methods[diag_label]
    beat = {
        m: diag[m] > pen[m] and diag[m] > sn[m]
        for m in RETURN_METRICS
    }
    return {
        "dataset": dataset,
        "diagram_label": diag_label,
        "max_days": max_days,
        "topk": topk,
        "methods": methods,
        "diagram_beats_both": beat,
        "diagram_beats_both_on_all_metrics": all(beat.values()),
    }


def compare_all_datasets(
    *,
    datasets: tuple[str, ...] = DATASETS,
    topk: int,
    max_days: int | None,
    budget: float,
    device: torch.device,
    cache_dir: Path = CACHE_DIR,
    use_cache: bool = True,
) -> dict:
    rows = []
    for dataset in datasets:
        stores = ensure_dataset_stores(dataset, device, cache_dir=cache_dir, use_cache=use_cache)
        rows.append(compare_dataset(dataset, stores, topk=topk, max_days=max_days, budget=budget))
    return {
        "config": {"max_days": max_days, "topk": topk, "budget": budget, "metrics": list(RETURN_METRICS)},
        "datasets": rows,
    }


def print_compare_table(result: dict) -> None:
    cfg = result["config"]
    print(f"\nPortfolio: test first {cfg['max_days']} calendar days, top-{cfg['topk']}, budget={cfg['budget']:.0f}")
    print(f"Metrics: {', '.join(cfg['metrics'])}")
    hdr = (
        f"{'dataset':<10} {'method':<14} {'seed':>4}  "
        f"{'ARR':>8} {'SR':>8}  {'vs PEN+SN':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for row in result["datasets"]:
        ds = row["dataset"]
        for name, m in row["methods"].items():
            flag = ""
            if name == row["diagram_label"]:
                flag = "both" if row["diagram_beats_both_on_all_metrics"] else "partial/no"
            print(f"{ds:<10} {name:<14} {m['seed']:>4}  {m['arr']:8.3f} {m['sr']:8.3f}  {flag:>9}")
        ds = ""
