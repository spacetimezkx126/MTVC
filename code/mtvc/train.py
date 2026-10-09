#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MTVC train / validate / test loop and run orchestration.

Role
----
- Build datasets & dataloaders, construct ``Model``, optimize BCE (+ optional L_pair).
- Early-stop on val Acc+MCC; write checkpoints via ``checkpoint.py``.
- Supported mode: ``train`` only.

Paper contrast is enabled with ``--contrast_mode mtvc*`` and ``--contrast_aux_weight > 0``.
"""
from __future__ import annotations

import copy
import csv
import math
import os
import random
from typing import Any

import numpy as np
import torch
try:
    from tool.cuda_guard import cuda_device_lock, cuda_sync_and_clear, forward_with_cuda_retry
except ImportError:
    from .tool.cuda_guard import cuda_device_lock, cuda_sync_and_clear, forward_with_cuda_retry
import torch.optim as optim
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from checkpoint import (
    CheckpointManager,
    apply_checkpoint_to_modules,
    load_checkpoint,
    load_model_state_for_eval,
    maybe_resume_from_args,
)
from data import (
    DatasetProfile,
    default_news_node_emb_dir,
    precompute_news_node_embeddings,
    MarketWindowDataset,
    WindowSample,
    get_profile,
    load_cmin_us_all_data,
    load_csmd_all_data,
    precompute_news_padding_embeddings,
    resolve_vocab_path,
    NUM_VIRTUAL_NEWS_TYPES,
    NUM_RULE_TREND,
)
from tool.oversample_utils import load_oversample_extra_counts, summarize_oversample_counts

# --- misc / demo ---


def _train_oversample_weights(
    batch,
    companies: list[str],
    extra_counts: dict[tuple[str, str], int] | None,
    *,
    device: torch.device,
    n: int,
) -> torch.Tensor:
    """Per-row weights: 1 + extra_copies for (company, target_date)."""
    w = torch.ones(n, device=device, dtype=torch.float32)
    if not extra_counts or n <= 0:
        return w
    dates = getattr(batch["price"], "date", None)
    if not isinstance(dates, (list, tuple)) or len(dates) < n:
        return w
    cids = batch["price"].company_id.view(-1)
    for i in range(n):
        ci = int(cids[i].item())
        if ci < 0 or ci >= len(companies):
            continue
        key = (companies[ci], str(dates[i]))
        extra = extra_counts.get(key)
        if extra:
            w[i] = 1.0 + float(extra)
    return w


def _strong_pos_rate_from_loader(loader) -> tuple[float, int, int]:
    """Train strong&valid label pos-rate (same mask as BCE)."""
    pos = 0
    neg = 0
    for batch in loader:
        if batch["label"].x.numel() == 0:
            continue
        labels = batch["label"].x.view(-1)
        strong = batch["label"].strong_mask.view(-1)
        if hasattr(batch["label"], "valid_mask"):
            valid = batch["label"].valid_mask.view(-1)
        else:
            valid = torch.ones_like(strong, dtype=torch.bool)
        m = strong & valid
        if int(m.sum().item()) == 0:
            continue
        y = labels[m]
        pos += int((y >= 0.5).sum().item())
        neg += int((y < 0.5).sum().item())
    n = pos + neg
    return (pos / n if n else 0.0), pos, neg


def _resolve_bce_pos_weight(args, train_loader) -> float:
    """Optional inverse-frequency BCE: pos_weight=(1-p)/p from train strong labels."""
    _pos_w = float(getattr(args, "bce_pos_weight", 1.0) or 1.0)
    if _pos_w <= 0:
        _pos_w = 1.0
    if not bool(getattr(args, "bce_class_weight", False)):
        return _pos_w
    p, n_pos, n_neg = _strong_pos_rate_from_loader(train_loader)
    if 0.0 < p < 1.0:
        _pos_w = (1.0 - p) / p
        print(
            f"[TRAIN] bce_class_weight ON strong_pos_rate={p:.4f} "
            f"n_pos={n_pos} n_neg={n_neg} pos_weight={_pos_w:.4f}",
            flush=True,
        )
    else:
        print(
            f"[TRAIN] bce_class_weight skipped (pos_rate={p:.4f}); "
            f"keep pos_weight={_pos_w:.4f}",
            flush=True,
        )
    return _pos_w


def resolve_emb_dim(args) -> int:
    """Model width; default 128 so CSMD300/CMIN-CN fit a 24GB card."""
    v = getattr(args, "emb_dim", None)
    if v is None:
        return 128
    return max(32, int(v))


def model_size_kwargs(args) -> dict:
    d = resolve_emb_dim(args)
    nd = getattr(args, "news_emb_dim", None)
    if nd is None:
        nd = 32
    out = {"price_emb_dim": d, "news_emb_dim": max(8, int(nd))}
    _ntl = getattr(args, "news_text_tf_layers", None)
    if _ntl is not None:
        out["news_text_tf_layers"] = max(1, int(_ntl))
    return out


def edge_op_dims(args, model=None) -> dict:
    m = getattr(model, "module", model) if model is not None else None
    d = int(getattr(m, "price_emb_dim", None) or resolve_emb_dim(args))
    return {
        "u_dim": d,
        "i_dim": d,
        "hidden_dim": max(32, d // 2),
    }


def amp_enabled(args) -> bool:
    # off by default: HeteroData / index_add_ paths mix poorly with bf16/fp16
    if bool(getattr(args, "no_amp", False)):
        return False
    return bool(getattr(args, "amp", False))


def autocast_ctx(args, device):
    """bf16 autocast on CUDA (Ampere+); no GradScaler needed for stability."""
    if not amp_enabled(args) or device.type != "cuda":
        from contextlib import nullcontext
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def wrap_model_devices(model, args, device):
    """Optional DataParallel when --devices lists 2+ GPUs (limited help at batch_size=1)."""
    raw = str(getattr(args, "devices", "") or "").strip()
    if not raw:
        return model.to(device)
    ids = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("cuda:"):
            ids.append(int(part.split(":")[-1]))
        else:
            ids.append(int(part))
    if len(ids) <= 1:
        return model.to(device)
    model = model.to(torch.device(f"cuda:{ids[0]}"))
    model = torch.nn.DataParallel(model, device_ids=ids)
    print(f"[INFO] DataParallel device_ids={ids}")
    return model




def _builder_dataset(ds):
    return getattr(ds, "_inner", ds)


def load_all_data_for_profile(
    profile: DatasetProfile,
    root: str,
    news_type_subdir: str = "news_type",
    *,
    require_wind_news_source: bool = True,
    csmd_news_source: str = "raw",
):
    if profile.key == "massive":
        return load_cmin_us_all_data(root, news_type_subdir=news_type_subdir)
    return load_csmd_all_data(
        root,
        news_type_subdir=news_type_subdir,
        news_source=csmd_news_source,
    )


def calculate_mcc(y_true, y_pred, mode="b"):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    num = (tp * tn) - (fp * fn)
    den = np.sqrt(np.float128(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)))
    return float(num / den) if den != 0 else 0.0




# Legacy manifest names → current CLI contrast_mode values.
_CONTRAST_MODE_ALIASES = {
    "method13": "mtvc",
    "method13_gate_only": "mtvc_gate_only",
    "method13_nogate": "mtvc_nogate",
    # temporal = historical no-contrast path (weight 0 / nocontrast runs)
    "temporal": "temporal",
}


def _normalize_contrast_mode(mode: str) -> str:
    m = str(mode or "").strip()
    return _CONTRAST_MODE_ALIASES.get(m, m)


def _is_mtvc_mode(args) -> bool:
    return _normalize_contrast_mode(getattr(args, "contrast_mode", "")) in (
        "mtvc",
        "mtvc_news_peer",
        "mtvc_price_peer",
        "mtvc_gate_only",
        "mtvc_nogate",
    )


def _mtvc_pair_peer(mode: str) -> str:
    m = _normalize_contrast_mode(mode)
    if m == "mtvc_news_peer":
        return "news"
    if m == "mtvc_price_peer":
        return "price"
    return "label"


def _mtvc_enable_gate(mode: str) -> bool:
    """mtvc_nogate = L_pair only; mtvc_gate_only / others keep gates."""
    return _normalize_contrast_mode(mode) != "mtvc_nogate"



def _eval_acc_mcc_on_loader(
    loader,
    model,
    device,
    *,
    loss_fn=None,
    move_batch=None,
    split_name: str = "eval",
    data_split: str = "",
    pred_threshold: float = 0.5,
):
    """在指定 split 上统计 strong 样本的 task_loss / acc / MCC。"""
    model.eval()
    if getattr(device, "type", None) == "cuda":
        try:
            from tool.cuda_guard import cuda_sync_and_clear
            cuda_sync_and_clear(device)
        except Exception:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    all_logits = []
    all_labels = []
    total_loss = 0.0
    n_loss_batches = 0
    with torch.no_grad():
        for batch in loader:
            if batch["price"].x.numel() == 0 or batch["label"].x.numel() == 0:
                continue
            if move_batch is not None:
                move_batch(batch)
            else:
                for nt in batch.node_types:
                    for key, val in batch[nt].__dict__.items():
                        if isinstance(val, torch.Tensor):
                            batch[nt][key] = val.to(device)
            def _fwd():
                return model(batch)

            with cuda_device_lock(device):
                logits = forward_with_cuda_retry(_fwd, device=device, tag=f"{split_name}-eval")
            if logits.numel() == 0:
                continue
            labels = batch["label"].x.view(-1).to(device)
            strong_mask = batch["label"].strong_mask.view(-1).to(device)
            valid_mask = (
                batch["label"].valid_mask.view(-1).to(device)
                if hasattr(batch["label"], "valid_mask")
                else torch.ones_like(strong_mask, dtype=torch.bool)
            )
            eval_mask = strong_mask & valid_mask
            if eval_mask.sum() == 0:
                continue
            logits = logits[eval_mask]
            labels = labels[eval_mask]
            if loss_fn is not None:
                try:
                    from tool.numeric import sanitize_logits_for_bce
                except ImportError:
                    from .tool.numeric import sanitize_logits_for_bce
                _bl = float(loss_fn(sanitize_logits_for_bce(logits), labels).item())
                if math.isfinite(_bl):
                    total_loss += _bl
                    n_loss_batches += 1
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
    if not all_logits:
        print(f"[{split_name}] no strong samples.")
        return None, None, None
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    preds_bin = (torch.sigmoid(logits_cat) > pred_threshold).long().numpy()
    labels_bin = labels_cat.long().numpy()
    acc = float(accuracy_score(labels_bin, preds_bin))
    mcc = float(calculate_mcc(labels_bin, preds_bin))
    avg_loss = (total_loss / n_loss_batches) if (loss_fn is not None and n_loss_batches > 0) else None
    if avg_loss is not None:
        print(f"[{split_name}] task_loss = {avg_loss:.6f}, acc = {acc:.4f}, MCC = {mcc:.4f}")
    else:
        print(f"[{split_name}] acc = {acc:.4f}, MCC = {mcc:.4f}")
    return acc, mcc, avg_loss


def set_all_seeds(seed: int = 42) -> None:
    """Seed RNGs. Set MTVC_DETERMINISTIC=1 for bit-stable training (repro checks)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    want_det = str(os.environ.get("MTVC_DETERMINISTIC", "0")).strip().lower() in (
        "1", "true", "yes", "on",
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        if want_det:
            # Required by some cuBLAS ops under deterministic algorithms.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.deterministic = True
            # Flash/mem-efficient SDPA backward is nondeterministic; force math.
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
            torch.use_deterministic_algorithms(True, warn_only=False)
        else:
            torch.backends.cudnn.deterministic = False
            torch.use_deterministic_algorithms(False)


def resolve_training_device(device_spec=None) -> torch.device:
    if device_spec is None:
        device_spec = "gpu"
    if isinstance(device_spec, (int, float)):
        device_spec = str(int(device_spec))
    if isinstance(device_spec, str):
        s = device_spec.strip().lower()
        if s in ("cpu", "none", "-1"):
            return torch.device("cpu")
        if s in ("gpu", "cuda"):
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            return torch.device("cpu")
        if s.startswith("cuda:"):
            return torch.device(s) if torch.cuda.is_available() else torch.device("cpu")
        try:
            idx = int(s)
            if idx < 0:
                return torch.device("cpu")
            if torch.cuda.is_available():
                return torch.device(f"cuda:{idx}")
            return torch.device("cpu")
        except ValueError:
            return torch.device("cpu")
    if int(device_spec) < 0:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(device_spec)}")
    return torch.device("cpu")


def device_from_args(args) -> torch.device:
    spec = getattr(args, "device", None)
    if spec is None:
        spec = "gpu"
    return resolve_training_device(spec)


def _cli_path(args, name: str) -> str:
    v = getattr(args, name, None)
    return "" if v is None else str(v).strip()


def make_model_dataloader(dataset, batch_size: int = 1, **kwargs):
    def _collate(xs):
        sample = xs[0]
        if isinstance(sample, WindowSample):
            return sample.for_model()
        return sample
    if batch_size == 1 and "collate_fn" not in kwargs:
        kwargs["collate_fn"] = _collate
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, **kwargs)


def move_model_batch_to_device(batch, device: torch.device):
    for nt in batch.node_types:
        store = batch[nt]
        for key, val in list(store._fields.items()):
            if isinstance(val, torch.Tensor):
                store._fields[key] = val.to(device)
    return batch



def dataset_kwargs_from_args(args, profile: DatasetProfile) -> dict:
    kw = dict(
        root=args.dataset_root,
        seg_length=profile.seg_length,
        price_window=profile.price_window,
        news_cluster_k=10,
        price_type_source="rule",  # paper: fixed rule map; no kmeans/dbscan CLI
        num_virtual_window=profile.num_virtual_window,
        news_padding_k=int(getattr(args, "news_padding_k", 5)),
        use_cached_news_padding_emb=False,
        news_padding_emb_cache_path=None,
        skip_finbert_tokenizer=True,
    )
    _vp = str(getattr(args, "vocab_path", "") or "").strip()
    if _vp:
        kw["vocab_path"] = _vp
    if profile.key in ("csmd50", "csmd300", "massive"):
        # Optional CLI override of profile.split_dates.
        _sd = dict(profile.split_dates)
        for k in ("train_start", "train_end", "val_start", "val_end", "test_start", "test_end"):
            v = str(getattr(args, k, "") or "").strip()
            if v:
                _sd[k] = v
        kw["split_dates"] = _sd
        kw["split_mode"] = "contiguous"  # paper: only contiguous phase-aligned ranges
        kw["train_lookback_purge"] = not bool(getattr(args, "no_train_lookback_purge", False))
    if profile.key in ("csmd50", "csmd300"):
        kw["csmd_news_source"] = "raw"  # paper: news/*.csv only
    if profile.key == "massive":
        kw["seg_keep_tail"] = bool(getattr(args, "seg_keep_tail", profile.seg_keep_tail))
    return kw


def normalize_run_args(args, profile: DatasetProfile):
    """填充 main.py 未显式传入的参数为 legacy 默认值。"""
    defaults = {
        "seed": 42,
        "mode": "train",  # paper package: train only
        "news_encode_mode": "vocab",  # paper: vocab Embedding + NewsTransformerEncoder only
        "news_ablation_mode": "full",
        "pred_fusion_ablation_mode": "full",
        "industry_fuse_site": "pred_fusion",
        "news_process_mode": "default",
        "news_word_emb": "random",
        "news_padding_k": 5,
        "epochs": 100,
        "no_price_token": False,
        "grad_clip": 1.0,
        "es_start_epoch": 1,
        "es_patience": 10,
        "es_metric": "acc_mcc",
        "price_type_top_news_csv_k": 5,
        "ckpt": "",
        "ckpt_dir": "",
        "no_save_ckpt": False,
        "save_ckpt_every_epoch": False,
        "resume_ckpt": "",
        "restore_rng_on_resume": True,
        "news_emb_cache_path": "",
        "graph_cache_dir": "",
    }
    for k, v in defaults.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    args.mode = "train"
    args.news_word_emb = "random"
    args.news_process_mode = "default"
    if args.dataset_root is None:
        args.dataset_root = profile.default_root
    if getattr(args, "no_restore_rng_on_resume", False):
        args.restore_rng_on_resume = False
    # Map legacy contrast_mode names from older run_manifest.json files.
    _cm_raw = getattr(args, "contrast_mode", None)
    if _cm_raw is not None:
        _cm = _normalize_contrast_mode(_cm_raw)
        if _cm != str(_cm_raw):
            args.contrast_mode = _cm
        elif _cm == "temporal":
            # Keep as temporal (non-mtvc); ensure contrast path stays off when weight unset.
            args.contrast_mode = "temporal"
            if getattr(args, "contrast_aux_weight", None) is None:
                args.contrast_aux_weight = 0.0
    if getattr(args, "mtvc_case_layer", None) is None:
        args.mtvc_case_layer = 6
    # Legacy CLI / manifest name → PredFusion VN ablation.
    if getattr(args, "pred_fusion_ablation_mode", None) is None and getattr(
        args, "main_mlp_ablation_mode", None
    ) is not None:
        args.pred_fusion_ablation_mode = args.main_mlp_ablation_mode
    return args


def run_pipeline(args, profile: DatasetProfile) -> None:
    import random
    import copy
    import torch.optim as optim
    from torch.nn import functional as F
    from sklearn.metrics import accuracy_score, classification_report

    args = normalize_run_args(args, profile)

    def _main_model_high_lr_param_list(model) -> list:
        """global / industry 虚拟结点及投影层用较高 lr。"""
        params = []
        params.append(model.global_parameter)
        params.extend(model.reduce_dim.parameters())
        if getattr(model, "pred_fusion_global_score_proj", None) is not None:
            params.extend(model.pred_fusion_global_score_proj.parameters())
        if getattr(model, "pred_fusion_global_proj", None) is not None:
            params.extend(model.pred_fusion_global_proj.parameters())
        params.append(model.virtual_industry_node)
        return params

    def _build_main_model_optimizer(
        model,
        *,
        lr_base: float = 1e-4,
        lr_global: float = 1e-4,
        wd: float = 5e-6,
    ):
        high_lr_params = _main_model_high_lr_param_list(model)
        special_ids = {id(p) for p in high_lr_params}
        base_lr_params = [p for p in model.parameters() if id(p) not in special_ids]
        return optim.Adam(
            [
                {"params": base_lr_params, "lr": lr_base, "weight_decay": wd},
                {"params": high_lr_params, "lr": lr_global, "weight_decay": wd},
            ]
        )

    set_all_seeds(int(args.seed))
    _det = str(os.environ.get("MTVC_DETERMINISTIC", "0")).strip().lower() in (
        "1", "true", "yes", "on",
    )
    print(f"[INFO] Random seed: {args.seed} deterministic={int(_det)}")

    ds_root = args.dataset_root
    print(f"[INFO] Dataset root: {ds_root}")
    if str(profile.key).startswith("csmd"):
        print("[INFO] csmd_news_source=raw (news/*.csv)")

    # Paper path: vocab Embedding + NewsTransformerEncoder only.
    _MODEL_NEWS_MODE = {"vocab": "vocab_padding"}
    _enc = str(getattr(args, "news_encode_mode", "vocab") or "vocab")
    print(
        f"[INFO] news_encode_mode={_enc} "
        f"-> Model(news_encode_mode={_MODEL_NEWS_MODE.get(_enc, 'vocab_padding')!r}); "
        f"news_ablation_mode={getattr(args, 'news_ablation_mode', 'full')}; "
        f"no_price_token={int(bool(getattr(args, 'no_price_token', False)))}; "
        f"pred_fusion_ablation_mode={getattr(args, 'pred_fusion_ablation_mode', getattr(args, 'main_mlp_ablation_mode', 'full'))}; "
        f"industry_fuse_site={getattr(args, 'industry_fuse_site', 'pred_fusion')}; "
        f"vin_ablation_mode=no_window; "
        f"news_process_mode=default"
    )
    print("[INFO] vocab 模式：Dataset vocab_input_ids → NewsTransformerEncoder（random emb）")
    _ds_common = dataset_kwargs_from_args(args, profile)
    if profile.key in ("csmd50", "csmd300", "massive"):
        _sd = dict(_ds_common.get("split_dates") or profile.split_dates)
        print(
            "[INFO] split_mode=contiguous "
            f"train=[{_sd.get('train_start')},{_sd.get('train_end')}] "
            f"val=[{_sd.get('val_start')},{_sd.get('val_end')}] "
            f"test=[{_sd.get('test_start')},{_sd.get('test_end')}]",
            flush=True,
        )
    print("[INFO] graph_cache=off：在线构图，不使用 hetero_graph_cache。")

    # Paper package: train only
    # 统一建模：全量 news->price 边，无 volatility / 筛边分支
    from model import Model as SimpleModel

    train_dataset = MarketWindowDataset(profile, mode="train", **_ds_common)
    _ds_eval = dict(_ds_common)
    val_dataset = MarketWindowDataset(profile, mode="val", **_ds_eval)
    test_dataset = MarketWindowDataset(profile, mode="test", **_ds_eval)
    _dl_kw = {}
    if str(os.environ.get("MTVC_DETERMINISTIC", "0")).strip().lower() in (
        "1", "true", "yes", "on",
    ):
        _dl_kw["generator"] = torch.Generator().manual_seed(int(args.seed))
    # Optional: force train window order (JSON list of indices) for archive replay.
    _fixed_order_path = str(os.environ.get("MTVC_FIXED_TRAIN_ORDER", "") or "").strip()
    if _fixed_order_path:
        import json as _json
        from torch.utils.data import Sampler as _Sampler

        class _FixedOrderSampler(_Sampler[int]):
            def __init__(self, order):
                self._order = [int(x) for x in order]

            def __iter__(self):
                return iter(self._order)

            def __len__(self):
                return len(self._order)

        with open(_fixed_order_path, "r", encoding="utf-8") as _f:
            _order = _json.load(_f)
        if len(_order) != len(train_dataset):
            raise RuntimeError(
                f"MTVC_FIXED_TRAIN_ORDER length {len(_order)} != "
                f"train windows {len(train_dataset)}"
            )
        print(
            f"[TRAIN] fixed train order from {_fixed_order_path} "
            f"(n={len(_order)} first={_order[:5]})",
            flush=True,
        )
        train_loader = make_model_dataloader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            sampler=_FixedOrderSampler(_order),
        )
    else:
        train_loader = make_model_dataloader(
            train_dataset, batch_size=1, shuffle=True, **_dl_kw
        )
    val_loader = make_model_dataloader(val_dataset, batch_size=1, shuffle=False)
    test_loader = make_model_dataloader(test_dataset, batch_size=1, shuffle=False)
    device = device_from_args(args)
    _vocab_path = str(getattr(args, "vocab_path", "") or "").strip() or resolve_vocab_path(
        profile.vocab_path,
        "dict_csmd.pkl" if profile.key.startswith("csmd") else "dict_massive.pkl",
    )
    _grad_clip = float(getattr(args, "grad_clip", 1.0))
    _companies = list(getattr(_builder_dataset(train_dataset), "companies", []) or [])
    _n_stocks = len(_companies) if _companies else int(profile.grid_num_stocks)
    _nam = str(getattr(args, "news_ablation_mode", "full") or "full").lower()
    if _nam not in ("full", "no_news"):
        raise ValueError(f"news_ablation_mode must be full|no_news, got {_nam!r}")
    model = SimpleModel(
        news_encode_mode=_MODEL_NEWS_MODE.get(_enc, "vocab_padding"),
        vocab_path=_vocab_path,
        grid_num_stocks=_n_stocks,
        price_input_dim=3,
        no_price_token=bool(getattr(args, "no_price_token", False)),
        industry_no_mean=False,  # paper: always Proj([mean‖v_k])
        news_ablation_mode=_nam,
        pred_fusion_ablation_mode=str(getattr(args, "pred_fusion_ablation_mode", None) or getattr(args, "main_mlp_ablation_mode", "full") or "full"),
        industry_fuse_site=str(getattr(args, "industry_fuse_site", "pred_fusion") or "pred_fusion"),
        vin_ablation_mode="no_window",  # paper fixed: no dual_tf virtual window nodes
        vin_connect_mode=str(getattr(args, "vin_connect_mode", "default") or "default"),
        news_word_emb="random",
        news_process_mode="default",
        dual_own_prop_news=False,
        own_news_only=False,
        prop_same_trend_only=False,
        companies=_companies,
        price_encoder=str(getattr(args, "price_encoder", "partial_unified")),
        news_top_k=int(getattr(args, "news_padding_k", 5) or 5),
        use_pred_fusion_global_score=bool(
            getattr(args, "use_pred_fusion_global_score", False)
        ),
        **model_size_kwargs(args),
    )
    model = wrap_model_devices(model, args, device)
    _contrast_aux_w = float(getattr(args, "contrast_aux_weight", 0.0) or 0.0)
    _contrast_mode = str(getattr(args, "contrast_mode", "temporal") or "temporal")
    # gate-only ablation: enable news gates without L_pair aux
    _mtvc_gate_only = _contrast_mode == "mtvc_gate_only"
    if _contrast_aux_w > 0 or _mtvc_gate_only:
        _core_c = model.module if hasattr(model, "module") else model
        if _contrast_aux_w > 0:
            _proj_dim = int(
                getattr(args, "contrast_proj_dim", None)
                or getattr(args, "ssl_proj_dim", 64)
                or 64
            )
            _core_c.ensure_contrast_heads(proj_dim=_proj_dim)
            for _n in ("contrast_proj_p", "contrast_proj_n"):
                _mod = getattr(_core_c, _n, None)
                if _mod is not None:
                    _mod.to(device)
            print(
                f"[TRAIN] contrast_aux=ON weight={_contrast_aux_w} mode={_contrast_mode} "
                f"temp_pair={float(getattr(args, 'mtvc_temp_pair', 0.07) or 0.07)} "
                f"anchors≤{int(getattr(args, 'contrast_max_anchors', 48) or 48)}",
                flush=True,
            )
        if _is_mtvc_mode(args) and _mtvc_enable_gate(_contrast_mode):
            from contrast_aux import enable_contrast_gates

            _case_L = int(getattr(args, "mtvc_case_layer", 6) or 6)
            enable_contrast_gates(_core_c, case_mine_layer=_case_L)
            _ut = getattr(_core_c, "unitrans", None)
            _ng = getattr(_ut, "news_gates", None) if _ut is not None else None
            _n_g = len(_ng) if _ng is not None else 0
            _cnm = str(
                getattr(args, "mtvc_contrast_news_mode", "nbag") or "nbag"
            )
            if _mtvc_gate_only:
                _peer_msg = "gates only (no L_pair)"
            elif _mtvc_pair_peer(_contrast_mode) == "news":
                _peer_msg = "L_pair among news-similar peers only"
            elif _mtvc_pair_peer(_contrast_mode) == "price":
                _peer_msg = "L_pair among price-similar peers only"
            else:
                _peer_msg = "L_pair (label peers)"
            print(
                f"[TRAIN] MTVC news_gate=ON layers=1..{_n_g} "
                f"case_mine_layer={_case_L} "
                f"contrast_news_mode={_cnm} "
                f"(every layer before last; {_peer_msg})",
                flush=True,
            )
        elif _contrast_mode == "mtvc_nogate":
            print(
                "[TRAIN] MTVC news_gate=OFF (L_pair only ablation)",
                flush=True,
            )
    # L8+ / crowded L3: optional CPU offload for news text TF (DUAL_TF_NEWS_TF_CPU=1).
    import os as _os

    _ntl = int(getattr(args, "news_text_tf_layers", 1) or 1)
    _tf = getattr(model, "unitrans_news_text_tf", None)
    _force_cpu = str(_os.environ.get("DUAL_TF_NEWS_TF_CPU", "0")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if (_ntl >= 4 or _force_cpu) and _tf is not None:
        _tf.to("cpu")
        print(
            f"[TRAIN] unitrans_news_text_tf → CPU (layers={_ntl}, finetune"
            f"{', forced' if _force_cpu and _ntl < 4 else ''})",
            flush=True,
        )
    print(
        f"[TRAIN] price_encoder={getattr(args, 'price_encoder', 'lstm')} "
        f"emb_dim={resolve_emb_dim(args)} news_emb_dim={int(getattr(args, 'news_emb_dim', 32) or 32)} "
        f"news_ablation={_nam} news_text_tf_layers={_ntl}",
        flush=True,
    )
    print(
        f"[TRAIN] main_mlp slots: news={bool(getattr(model, 'include_news_in_main_mlp', False))} "
        f"news_global={bool(getattr(model, 'include_news_global_in_main_mlp', False))} "
        f"market_global={bool(getattr(model, 'include_global_in_main_mlp', False))} "
        f"industry={bool(getattr(model, 'include_industry_in_main_mlp', False))}",
        flush=True,
    )
    if not bool(getattr(model, "include_price_token", True)):
        print("[TRAIN] price_token=OFF (fixed zero; --no_price_token)", flush=True)
    if str(getattr(model, "news_ablation_mode", "full") or "full") == "no_news":
        print("[TRAIN] news_ablation=no_news (news tower off)", flush=True)
    _core = model.module if hasattr(model, "module") else model
    _price_tok = type(getattr(_core, "unitrans_price_tok", None)).__name__
    _price_mlp = type(getattr(_core, "unitrans_price_mlp", None)).__name__
    _news_tf = type(getattr(_core, "unitrans_news_text_tf", None)).__name__
    _news_enc = type(
        getattr(_core, "news_transformer_encoder", None)
        or getattr(_core, "bert_tok_transformer", None)
    ).__name__
    _news_mlp = type(getattr(_core, "unitrans_news_mlp", None)).__name__
    _news_pool = None
    _unitrans = getattr(_core, "unitrans", None)
    if _unitrans is not None:
        _nsp = getattr(_unitrans, "news_slot_pool", None)
        _news_pool = getattr(_nsp, "pool", None) if _nsp is not None else None
    print(
        f"[ARCH] encoder={getattr(_core, 'price_encoder', None)} "
        f"price_tok={_price_tok} price_mlp={_price_mlp} "
        f"news_word_emb={getattr(_core, 'news_word_emb', None)} "
        f"news_text_tf={_news_tf} news_enc={_news_enc} news_mlp={_news_mlp} "
        f"news_pool={_news_pool} "
        f"price_input_dim={int(getattr(_core, 'price_input_dim', 3))}",
        flush=True,
    )
    optimizer = _build_main_model_optimizer(model, lr_base=1e-5, lr_global=1e-5, wd=5e-6)
    _pos_w = _resolve_bce_pos_weight(args, train_loader)
    _bce_pos = torch.tensor([_pos_w], device=device, dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=_bce_pos)
    loss_fn_none = torch.nn.BCEWithLogitsLoss(pos_weight=_bce_pos, reduction="none")
    print(f"[TRAIN] BCEWithLogitsLoss pos_weight={_pos_w}", flush=True)
    _os_path = str(getattr(args, "train_oversample_extra", "") or "").strip()
    _os_counts: dict[tuple[str, str], int] | None = None
    if _os_path:
        _os_counts = load_oversample_extra_counts(_os_path)
        _os_sum = summarize_oversample_counts(_os_counts)
        print(
            f"[TRAIN] train oversample extra={_os_path} "
            f"unique={_os_sum['n_unique_keys']} copies={_os_sum['n_extra_copies']} "
            f"(loss weight=1+extra; val/test unchanged)",
            flush=True,
        )
    # 早停：--es_start_epoch / --es_patience / --es_metric
    ES_START_EPOCH = max(1, int(args.es_start_epoch))
    ES_PATIENCE = max(1, int(args.es_patience))
    ES_METRIC = str(getattr(args, "es_metric", "acc_mcc") or "acc_mcc")
    best_val_acc_plus_mcc = float("-inf")
    best_val_loss = float("inf")
    patience_cnt = 0
    best_test_acc = None
    best_test_mcc = None
    best_test_epoch = None
    print(f"[TRAIN] #train windows = {len(train_dataset)}")
    if ES_METRIC == "val_loss":
        print(
            f"[TRAIN] early_stop: monitor val_task_loss (lower is better) "
            f"from epoch>={ES_START_EPOCH}, patience={ES_PATIENCE}"
        )
    else:
        print(
            f"[TRAIN] early_stop: monitor val_acc+val_MCC (higher is better) "
            f"from epoch>={ES_START_EPOCH}, patience={ES_PATIENCE}"
        )
    ckpt_mgr = CheckpointManager(args, profile)
    _hard_ft = False
    start_epoch, _resume_info = maybe_resume_from_args(
        args,
        profile,
        model=model,
        model_optimizer=optimizer,
        ckpt_manager=ckpt_mgr,
    )
    _fixed_order_active = bool(str(os.environ.get("MTVC_FIXED_TRAIN_ORDER", "") or "").strip())
    for epoch in range(start_epoch, args.epochs + 1):
        if _fixed_order_active:
            _ = int(torch.empty((), dtype=torch.int64).random_().item())
        # ============= Train（只用 strong 样本）=============
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            if batch["price"].x.numel() == 0 or batch["label"].x.numel() == 0:
                continue
            # 搬到 device
            for nt in batch.node_types:
                for key, val in batch[nt].__dict__.items():
                    if isinstance(val, torch.Tensor):
                        batch[nt][key] = val.to(device)
            contrast_aux_v = 0.0
            contrast_bsz = 0.0
            _c_stats: dict = {}
            def _fwd_train():
                return model(batch)

            with cuda_device_lock(device):
                logits_out = forward_with_cuda_retry(_fwd_train, device=device, tag="train-fwd")
                logits = logits_out
                if logits.numel() == 0:
                    continue
                labels = batch["label"].x.view(-1).to(device)               # 0/1
                strong_mask = batch["label"].strong_mask.view(-1).to(device)  # 只用 |y|>0.0055 的样本
                valid_mask = batch["label"].valid_mask.view(-1).to(device) if hasattr(batch["label"], "valid_mask") else torch.ones_like(strong_mask, dtype=torch.bool)
                train_mask = strong_mask & valid_mask
                if train_mask.sum() == 0:
                    continue
                if _os_counts:
                    _w_all = _train_oversample_weights(
                        batch, _companies, _os_counts, device=device, n=int(labels.numel())
                    )
                    _w = _w_all[train_mask]
                    try:
                        from tool.numeric import sanitize_logits_for_bce
                    except ImportError:
                        from .tool.numeric import sanitize_logits_for_bce
                    _logits_t = sanitize_logits_for_bce(logits[train_mask])
                    _per = loss_fn_none(_logits_t, labels[train_mask])
                    task_loss = (_per * _w).sum() / _w.sum().clamp_min(1e-8)
                else:
                    try:
                        from tool.numeric import sanitize_logits_for_bce
                    except ImportError:
                        from .tool.numeric import sanitize_logits_for_bce
                    task_loss = loss_fn(
                        sanitize_logits_for_bce(logits[train_mask]), labels[train_mask]
                    )
                loss = task_loss
                _m = getattr(model, "module", model)
                if _contrast_aux_w > 0:
                    contrast_aux_v = 0.0
                    contrast_bsz = 0.0
                    _c_stats = {}
                    _cmode = str(getattr(args, "contrast_mode", "mtvc"))
                    if not _is_mtvc_mode(args):
                        raise RuntimeError(
                            f"unsupported contrast_mode={_cmode!r}; "
                            "paper package supports mtvc / mtvc_gate_only / "
                            "mtvc_nogate / mtvc_news_peer / mtvc_price_peer "
                            "(or contrast_aux_weight=0 for no-contrast ablations)"
                        )
                    from contrast_aux import compute_contrast_aux

                    _mtvc_peer = _mtvc_pair_peer(_cmode)
                    _mtvc_gate = _mtvc_enable_gate(_cmode)
                    _c_loss, _c_stats = compute_contrast_aux(
                        _m,
                        batch,
                        temperature_pair=float(
                            getattr(args, "mtvc_temp_pair", 0.07) or 0.07
                        ),
                        max_anchors=int(
                            getattr(args, "contrast_max_anchors", 48) or 48
                        ),
                        max_news_per_bag=int(
                            getattr(args, "contrast_max_news_per_bag", 6) or 6
                        ),
                        day_mode=str(getattr(args, "contrast_day_mode", "all")),
                        sim_quantile=float(
                            getattr(args, "sim_quantile", 0.75) or 0.75
                        ),
                        tau_y=float(getattr(args, "mtvc_tau_y", 0.5) or 0.5),
                        beta_hard=float(
                            getattr(args, "mtvc_beta_hard", 0.5) or 0.5
                        ),
                        lambda_pair=float(
                            getattr(args, "mtvc_lambda_pair", 0.2) or 0.2
                        ),
                        pair_peer=_mtvc_peer,
                        device=device,
                        enable_gate=_mtvc_gate,
                        case_mine_layer=int(
                            getattr(args, "mtvc_case_layer", 6) or 6
                        ),
                        contrast_news_mode=str(
                            getattr(args, "mtvc_contrast_news_mode", "nbag")
                            or "nbag"
                        ),
                    )
                    contrast_bsz = float(_c_stats.get("bsz", 0.0) or 0.0)
                    _skipped = float(_c_stats.get("skipped", 1.0))
                    if _skipped < 0.5:
                        loss = loss + _contrast_aux_w * _c_loss
                        contrast_aux_v = float(_c_stats.get("loss", 0.0) or 0.0)
                optimizer.zero_grad()
                loss.backward()
                if _grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), _grad_clip)
                optimizer.step()
            epoch_loss += float(loss.item())
            num_batches += 1
            _aux_msg = ""
            if _contrast_aux_w > 0:
                _aux_msg += (
                    f" contrast={contrast_aux_v:.6f} c_bsz={contrast_bsz:.0f}"
                    f" c_pos={float(_c_stats.get('pos_sim', 0.0) or 0.0):.3f}"
                    f" c_neg={float(_c_stats.get('neg_sim', 0.0) or 0.0):.3f}"
                )
                if _is_mtvc_mode(args):
                    _aux_msg += (
                        f" Lclip={float(_c_stats.get('L_clip', 0.0) or 0.0):.3f}"
                        f" Lpair={float(_c_stats.get('L_pair', 0.0) or 0.0):.3f}"
                        f" lam1={float(_c_stats.get('lam1', 0.0) or 0.0):.3f}"
                        f" lam2={float(_c_stats.get('lam2', 0.0) or 0.0):.3f}"
                        f" g={float(_c_stats.get('gamma_mean', 0.0) or 0.0):.2f}"
                        f" w={float(_c_stats.get('w_mean', 0.0) or 0.0):.2f}"
                    )
                    _pp = float(_c_stats.get("pair_peer", 0.0) or 0.0)
                    if _pp > 0.5:
                        _aux_msg += (
                            f" peer={'news' if _pp < 1.5 else 'price'}"
                            f" n_peer={float(_c_stats.get('n_peer', 0.0) or 0.0):.0f}"
                        )
            print(f"[train] epoch={epoch} batch={num_batches} "
                f"N_price={batch['price'].x.size(0)} "
                f"N_news={batch['news'].x.size(0) if 'news' in batch.node_types else 0} "
                f"loss={loss.item():.6f}{_aux_msg}")
        if num_batches > 0:
            epoch_loss /= num_batches
        print(f"[Epoch {epoch:03d}] train train_loss = {epoch_loss:.6f}")
        cuda_sync_and_clear(device)
        # Optional: skip post-epoch train-set metrics (diagnostic only).
        # Seed/GPU contention often dies here with invalid PC; val/test still drive ES.
        _skip_train_eval = str(os.environ.get("DUAL_TF_SKIP_TRAIN_EVAL", "0")).strip().lower() in (
            "1", "true", "yes", "on",
        )
        if _skip_train_eval:
            print(f"[TRAIN] skip train-eval (DUAL_TF_SKIP_TRAIN_EVAL=1)", flush=True)
        else:
            _eval_acc_mcc_on_loader(
                train_loader,
                model,
                device,
                loss_fn=loss_fn,
                split_name=f"Epoch {epoch:03d} TRAIN",
                                )
        # ============= Val（accuracy & MCC，只用 strong 样本）=============
        model.eval()
        all_logits = []
        all_labels = []
        total_val_task_loss = 0.0
        total_val_batches = 0
        cuda_sync_and_clear(device)
        with torch.no_grad():
            for batch in val_loader:
                if batch["price"].x.numel() == 0 or batch["label"].x.numel() == 0:
                    continue
                for nt in batch.node_types:
                    for key, val in batch[nt].__dict__.items():
                        if isinstance(val, torch.Tensor):
                            batch[nt][key] = val.to(device)
                def _fwd_val():
                    return model(batch)
                with cuda_device_lock(device):
                    logits_out = forward_with_cuda_retry(_fwd_val, device=device, tag="val-fwd")
                logits = logits_out
                if logits.numel() == 0:
                    continue
                labels = batch["label"].x.view(-1).to(device)
                strong_mask = batch["label"].strong_mask.view(-1).to(device)
                valid_mask = batch["label"].valid_mask.view(-1).to(device) if hasattr(batch["label"], "valid_mask") else torch.ones_like(strong_mask, dtype=torch.bool)
                eval_mask = strong_mask & valid_mask
                if eval_mask.sum() == 0:
                    continue
                logits_m = logits[eval_mask]
                labels_m = labels[eval_mask]
                try:
                    from tool.numeric import sanitize_logits_for_bce
                except ImportError:
                    from .tool.numeric import sanitize_logits_for_bce
                logits_m = sanitize_logits_for_bce(logits_m)
                _vloss = float(loss_fn(logits_m, labels_m).item())
                # Skip nan/inf batch losses so one bad batch does not poison the epoch mean.
                if math.isfinite(_vloss):
                    total_val_task_loss += _vloss
                    total_val_batches += 1
                all_logits.append(logits_m.detach().cpu())
                all_labels.append(labels_m.detach().cpu())
        if all_logits:
            logits_cat = torch.cat(all_logits, dim=0)
            labels_cat = torch.cat(all_labels, dim=0)
            probs = torch.sigmoid(logits_cat)
            preds_bin = (probs > 0.5).long().numpy()
            labels_bin = labels_cat.long().numpy()
            acc = accuracy_score(labels_bin, preds_bin)
            mcc = calculate_mcc(labels_bin, preds_bin)
            val_acc_epoch = float(acc)
            val_mcc_epoch = float(mcc)
            if total_val_batches > 0:
                val_task_loss_epoch = total_val_task_loss / total_val_batches
                print(
                    f"[Epoch {epoch:03d}] val_task_loss = {val_task_loss_epoch:.6f}, "
                    f"val_acc = {acc:.4f}, val_MCC = {mcc:.4f}"
                )
            else:
                val_task_loss_epoch = None
                print(
                    f"[Epoch {epoch:03d}] val_task_loss = nan "
                    f"(no finite batch losses), val_acc = {acc:.4f}, val_MCC = {mcc:.4f}"
                )
        else:
            val_task_loss_epoch = None
            val_acc_epoch = None
            val_mcc_epoch = None
            print(f"[Epoch {epoch:03d}] val set has no strong samples.")

        # ============= Test（每个 epoch；ES 不依赖；可 skip 降 CUDA 争用）=============
        test_acc_epoch = None
        test_mcc_epoch = None
        _skip_test = str(os.environ.get("DUAL_TF_SKIP_TEST_DURING_TRAIN", "0")).strip().lower() in (
            "1", "true", "yes", "on",
        )
        if _skip_test:
            print(f"[TRAIN] skip per-epoch test (DUAL_TF_SKIP_TEST_DURING_TRAIN=1)", flush=True)
        else:
            model.eval()
            cuda_sync_and_clear(device)
            all_logits = []
            all_labels = []
            with torch.no_grad():
                for batch in test_loader:
                    if batch["price"].x.numel() == 0 or batch["label"].x.numel() == 0:
                        continue
                    for nt in batch.node_types:
                        for key, val in batch[nt].__dict__.items():
                            if isinstance(val, torch.Tensor):
                                batch[nt][key] = val.to(device)
                    def _fwd_test():
                        return model(batch)
                    with cuda_device_lock(device):
                        logits_out = forward_with_cuda_retry(_fwd_test, device=device, tag="test-fwd")
                    logits = logits_out
                    if logits.numel() == 0:
                        continue
                    labels = batch["label"].x.view(-1).to(device)
                    strong_mask = batch["label"].strong_mask.view(-1).to(device)
                    valid_mask = batch["label"].valid_mask.view(-1).to(device) if hasattr(batch["label"], "valid_mask") else torch.ones_like(strong_mask, dtype=torch.bool)
                    test_mask = strong_mask & valid_mask
                    if test_mask.sum() == 0:
                        continue
                    all_logits.append(logits[test_mask].detach().cpu())
                    all_labels.append(labels[test_mask].detach().cpu())
            if all_logits:
                logits_cat = torch.cat(all_logits, dim=0)
                labels_cat = torch.cat(all_labels, dim=0)
                probs = torch.sigmoid(logits_cat)
                preds_bin = (probs > 0.5).long().numpy()
                labels_bin = labels_cat.long().numpy()
                acc = accuracy_score(labels_bin, preds_bin)
                mcc = calculate_mcc(labels_bin, preds_bin)
                test_acc_epoch = float(acc)
                test_mcc_epoch = float(mcc)
                print(f"[Epoch {epoch:03d}] test_acc = {acc:.4f}, test_MCC = {mcc:.4f}")
            else:
                print(f"[Epoch {epoch:03d}] test set has no strong samples.")

        if epoch >= ES_START_EPOCH:
            if ES_METRIC == "val_loss":
                if val_task_loss_epoch is None or not math.isfinite(float(val_task_loss_epoch)):
                    patience_cnt += 1
                    print(
                        f"[TRAIN] early_stop: no finite val_task_loss this epoch, "
                        f"patience {patience_cnt}/{ES_PATIENCE}"
                    )
                elif val_task_loss_epoch < best_val_loss - 1e-8:
                    best_val_loss = float(val_task_loss_epoch)
                    patience_cnt = 0
                    if test_acc_epoch is not None and test_mcc_epoch is not None:
                        best_test_acc = test_acc_epoch
                        best_test_mcc = test_mcc_epoch
                        best_test_epoch = epoch
                    _va = "None" if val_acc_epoch is None else f"{val_acc_epoch:.4f}"
                    _vm = "None" if val_mcc_epoch is None else f"{val_mcc_epoch:.4f}"
                    print(
                        f"[TRAIN] early_stop: val_task_loss improved -> "
                        f"best_loss={best_val_loss:.6f} (acc={_va}, MCC={_vm})"
                    )
                else:
                    patience_cnt += 1
                    print(
                        f"[TRAIN] early_stop: no val_task_loss improvement, "
                        f"patience {patience_cnt}/{ES_PATIENCE} (best_loss={best_val_loss:.6f})"
                    )
            else:
                val_sum = (
                    (val_acc_epoch + val_mcc_epoch)
                    if (val_acc_epoch is not None and val_mcc_epoch is not None)
                    else None
                )
                if val_sum is None:
                    patience_cnt += 1
                    print(f"[TRAIN] early_stop: no val acc/MCC this epoch, patience {patience_cnt}/{ES_PATIENCE}")
                elif val_sum > best_val_acc_plus_mcc + 1e-8:
                    best_val_acc_plus_mcc = val_sum
                    patience_cnt = 0
                    if test_acc_epoch is not None and test_mcc_epoch is not None:
                        best_test_acc = test_acc_epoch
                        best_test_mcc = test_mcc_epoch
                        best_test_epoch = epoch
                    print(
                        f"[TRAIN] early_stop: val_acc+val_MCC improved -> "
                        f"best_sum={best_val_acc_plus_mcc:.6f} "
                        f"(acc={val_acc_epoch:.4f}, MCC={val_mcc_epoch:.4f}, val_task_loss={val_task_loss_epoch:.6f})"
                    )
                else:
                    patience_cnt += 1
                    print(
                        f"[TRAIN] early_stop: no val_acc+val_MCC improvement, "
                        f"patience {patience_cnt}/{ES_PATIENCE} (best_sum={best_val_acc_plus_mcc:.6f})"
                    )
            ckpt_mgr.maybe_save_best(
                epoch=epoch,
                val_acc=val_acc_epoch,
                val_mcc=val_mcc_epoch,
                val_loss=val_task_loss_epoch,
                es_metric=ES_METRIC,
                model=model,
                model_optimizer=optimizer,
                test_acc=test_acc_epoch,
                test_mcc=test_mcc_epoch,
            )
            if patience_cnt >= ES_PATIENCE:
                mon = "val_task_loss" if ES_METRIC == "val_loss" else "val_acc+val_MCC"
                print(
                    f"[TRAIN] early stopping at epoch {epoch} "
                    f"(no {mon} gain for {ES_PATIENCE} epochs since monitoring started)."
                )
                break
        ckpt_mgr.maybe_save_epoch(epoch=epoch, model=model, model_optimizer=optimizer)

    last_epoch = epoch if "epoch" in locals() else max(0, start_epoch - 1)
    if ckpt_mgr.best_epoch > 0 and os.path.isfile(os.path.join(ckpt_mgr.ckpt_dir, "best.pt")):
        load_model_state_for_eval(model, os.path.join(ckpt_mgr.ckpt_dir, "best.pt"), device)
        print(f"[TRAIN] restored best checkpoint from epoch={ckpt_mgr.best_epoch}")
    ckpt_mgr.save_last(epoch=last_epoch, model=model, model_optimizer=optimizer)
    ckpt_mgr.write_run_manifest()

    print("[TRAIN] Training finished. Final test evaluation...")
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            if batch["price"].x.numel() == 0 or batch["label"].x.numel() == 0:
                continue
            for nt in batch.node_types:
                for key, val in batch[nt].__dict__.items():
                    if isinstance(val, torch.Tensor):
                        batch[nt][key] = val.to(device)
            logits = model(batch)
            if logits.numel() == 0:
                continue
            labels = batch["label"].x.view(-1).to(device)
            strong_mask = batch["label"].strong_mask.view(-1).to(device)
            valid_mask = batch["label"].valid_mask.view(-1).to(device) if hasattr(batch["label"], "valid_mask") else torch.ones_like(strong_mask, dtype=torch.bool)
            test_mask = strong_mask & valid_mask
            if test_mask.sum() == 0:
                continue
            logits = logits[test_mask]
            labels = labels[test_mask]
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
    if all_logits:
        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)
        probs = torch.sigmoid(logits_cat)
        preds_bin = (probs > 0.5).long().numpy()
        labels_bin = labels_cat.long().numpy()
        acc = accuracy_score(labels_bin, preds_bin)
        mcc = calculate_mcc(labels_bin, preds_bin)
        print(f"[FINAL TEST] acc = {acc:.4f}, MCC = {mcc:.4f}")
    else:
        print("[FINAL TEST] test set has no strong samples.")
    if best_test_acc is not None and best_test_mcc is not None:
        tag = "BEST TEST@BEST VAL_LOSS" if ES_METRIC == "val_loss" else "BEST TEST@BEST VAL"
        print(f"[{tag}] epoch={best_test_epoch} acc = {best_test_acc:.4f}, MCC = {best_test_mcc:.4f}")
    else:
        tag = "BEST TEST@BEST VAL_LOSS" if ES_METRIC == "val_loss" else "BEST TEST@BEST VAL"
        print(f"[{tag}] no available test metrics when val improved.")



def run(profile_key: str, args) -> None:
    """统一运行入口。"""
    profile = get_profile(profile_key) if isinstance(profile_key, str) else profile_key
    print(f"[RUN] dataset={profile.key} mode=train")
    run_pipeline(args, profile)


run_legacy = run  # 兼容旧 import

