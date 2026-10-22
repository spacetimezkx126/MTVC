from __future__ import annotations
from data import DatasetProfile, default_news_node_emb_dir

_DEFAULT_FINBERT_BY_DATASET = {
    "csmd50": "models/finbert-tone-chinese",
    "csmd300": "models/finbert-tone-chinese",
    "cmin_cn": "models/finbert-tone-chinese",
    "cmin_us": "models/finbert-pretrain",
    "massive": "models/finbert-pretrain",
    "stocknet": "models/finbert-pretrain",
}


def resolve_emb_dim(args) -> int:
    """Model width; default 128 so CSMD300/CMIN-CN fit a 24GB card."""
    v = getattr(args, "emb_dim", None)
    if v is None:
        return 128
    return max(32, int(v))


def model_size_kwargs(args) -> dict:
    """Price and news embedding widths, plus optional news-transformer depth."""
    d = resolve_emb_dim(args)
    nd = getattr(args, "news_emb_dim", None)
    if nd is None:
        nd = 32
    out = {"price_emb_dim": d, "news_emb_dim": max(8, int(nd))}
    _ntl = getattr(args, "news_text_tf_layers", None)
    if _ntl is not None:
        out["news_text_tf_layers"] = max(1, int(_ntl))
    for name in ("init_news_vocab_size", "init_news_embedding_dim"):
        value = getattr(args, name, None)
        if value is not None:
            out[name] = int(value)
    return out


def edge_op_dims(args, model=None) -> dict:
    """Hidden sizes formerly used by the edge operator. Kept for older callers."""
    m = getattr(model, "module", model) if model is not None else None
    d = int(getattr(m, "price_emb_dim", None) or resolve_emb_dim(args))
    return {"u_dim": d, "i_dim": d, "hidden_dim": max(32, d // 2)}


def _cli_path(args, name: str) -> str:
    """String form of a path argument, or empty."""
    v = getattr(args, name, None)
    return "" if v is None else str(v).strip()


def resolve_finbert_model_path(args, profile: DatasetProfile | None = None) -> str:
    """CLI path if set, otherwise the Chinese or English FinBERT for this dataset."""
    explicit = str(getattr(args, "finbert_model_path", "") or "").strip()
    if explicit:
        return explicit
    key = str(getattr(profile, "key", "") or getattr(args, "dataset", "") or "").strip().lower()
    return _DEFAULT_FINBERT_BY_DATASET.get(key, "models/finbert-tone-chinese")


def model_finbert_kwargs(args, profile: DatasetProfile | None = None) -> dict:
    """FinBERT path kwargs. Empty for the vocab encoder."""
    mode = str(getattr(args, "news_encode_mode", "vocab") or "vocab")
    if mode not in ("finbert", "finbert_embedding", "finbert_tok"):
        return {}
    return dict(
        finbert_model_path=resolve_finbert_model_path(args, profile),
        finbert_freeze=bool(getattr(args, "finbert_freeze", True)),
        finbert_chunk_size=int(getattr(args, "finbert_chunk_size", 32)),
    )


def dataset_kwargs_from_args(args, profile: DatasetProfile) -> dict:
    """Keyword arguments for MarketWindowDataset."""
    _mode = str(getattr(args, "news_encode_mode", "vocab") or "vocab")
    use_cached = bool(getattr(args, "use_cached_news_emb", False)) or _mode == "finbert_embedding"
    if _mode in ("vocab", "finbert", "finbert_tok"):
        use_cached = False
    cache_path = str(getattr(args, "news_emb_cache_path", "") or "").strip() or None
    news_type_subdir = str(getattr(args, "news_type_subdir", "news_type") or "news_type")
    if use_cached and cache_path is None:
        cache_path = default_news_node_emb_dir(
            args.dataset_root, news_type_subdir, max_seq_len=128, pooling="pooler"
        )
    kw = dict(
        root=args.dataset_root,
        seg_length=profile.seg_length,
        price_window=profile.price_window,
        news_cluster_k=10,
        price_type_source=getattr(args, "price_type_source", "rule"),
        num_virtual_window=profile.num_virtual_window,
        news_padding_k=int(getattr(args, "news_padding_k", 5)),
        use_cached_news_padding_emb=use_cached,
        news_padding_emb_cache_path=cache_path,
        skip_finbert_tokenizer=_mode in ("vocab", "finbert_embedding"),
        finbert_model_path=resolve_finbert_model_path(args, profile),
    )
    _vp = str(getattr(args, "vocab_path", "") or "").strip()
    if _vp:
        kw["vocab_path"] = _vp
    if profile.key in ("csmd50", "csmd300", "sse50", "cmin_cn", "cmin_us", "massive", "stocknet"):
        kw["news_type_subdir"] = news_type_subdir
        _sd = dict(profile.split_dates)
        for k in ("train_start", "train_end", "val_start", "val_end", "test_start", "test_end"):
            v = str(getattr(args, k, "") or "").strip()
            if v:
                _sd[k] = v
        kw["split_dates"] = _sd
        kw["split_mode"] = (
            str(getattr(args, "split_mode", "contiguous") or "contiguous").strip().lower()
        )
        kw["train_lookback_purge"] = not bool(getattr(args, "no_train_lookback_purge", False))
    if profile.key in ("csmd50", "csmd300", "sse50"):
        kw["csmd_news_source"] = (
            str(getattr(args, "csmd_news_source", "llm_extract") or "llm_extract").strip().lower()
        )
    if profile.key in ("cmin_cn", "cmin_us", "massive", "stocknet"):
        kw["seg_keep_tail"] = bool(getattr(args, "seg_keep_tail", profile.seg_keep_tail))
    if profile.key == "cmin_cn":
        kw["require_wind_news_source"] = not bool(getattr(args, "cmin_cn_all_news", False))
        kw["require_window_news"] = bool(getattr(args, "require_window_news", False))
        kw["subset_top_companies"] = int(getattr(args, "subset_top_companies", 0) or 0)
    return kw


def normalize_run_args(args, profile: DatasetProfile):
    """Fill arguments that main.py did not set."""
    defaults = {
        "seed": 42,
        "mode": "train",
        "news_encode_mode": "vocab",
        "news_window_agg_mode": "sum",
        "news_ablation_mode": "full",
        "main_mlp_ablation_mode": "full",
        "industry_fuse_site": "pred_fusion",
        "news_process_mode": "default",
        "news_padding_k": 5,
        "price_type_source": "rule",
        "num_score_news_types": profile.num_score_news_types,
        "epochs": 100,
        "no_price_token": False,
        "no_price_token_pad": False,
        "no_news_token": False,
        "no_news_token_pad": False,
        "modality_ablation": "full",
        "industry_no_mean": False,
        "grad_clip": 1.0,
        "es_start_epoch": 1,
        "es_patience": 10,
        "es_metric": "acc_mcc",
        "ckpt": "",
        "ckpt_dir": "",
        "no_save_ckpt": False,
        "save_ckpt_every_epoch": False,
        "resume_ckpt": "",
        "news_emb_cache_path": "",
    }
    for k, v in defaults.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    if args.dataset_root is None:
        args.dataset_root = profile.default_root
    from naming import (
        normalize_price_encoder,
        normalize_contrast_mode,
        normalize_vin_connect,
        normalize_news_price_ablation,
    )

    if getattr(args, "price_encoder", None) is not None:
        args.price_encoder = normalize_price_encoder(args.price_encoder)
    if getattr(args, "contrast_mode", None) is not None:
        args.contrast_mode = normalize_contrast_mode(args.contrast_mode)
    if getattr(args, "vin_connect_mode", None) is not None:
        args.vin_connect_mode = normalize_vin_connect(args.vin_connect_mode)
    args.news_ablation_mode = normalize_news_price_ablation(
        getattr(args, "news_ablation_mode", "full"), bool(getattr(args, "no_price_token", False))
    )
    if args.news_ablation_mode == "no_price":
        args.news_ablation_mode = "full"
        args.no_price_token = True
    _mtvc_defaults = {
        "mtvc_lambda_pair": 0.2,
        "mtvc_beta_hard": 0.5,
        "mtvc_tau_y": 0.5,
        "mtvc_temp_pair": 0.07,
        "mtvc_lambda_clip_start": 0.3,
        "mtvc_lambda_clip_end": 0.05,
        "mtvc_alpha_bad": 0.3,
        "mtvc_tau_pos": 0.3,
        "mtvc_window_k": 3,
        "mtvc_case_layer": 6,
        "mtvc_case_feat": "joint",
    }
    for k, v in _mtvc_defaults.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    return args


import os
import numpy as np
import torch
from data import DatasetProfile, WindowSample, load_cmin_cn_all_data, load_csmd_all_data


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
    return (pos / n if n else 0.0, pos, neg)


def _resolve_bce_pos_weight(args, train_loader) -> float:
    """Optional inverse-frequency BCE: pos_weight=(1-p)/p from train strong labels."""
    _pos_w = float(getattr(args, "bce_pos_weight", 1.0) or 1.0)
    if _pos_w <= 0:
        _pos_w = 1.0
    if not bool(getattr(args, "bce_class_weight", False)):
        return _pos_w
    (p, n_pos, n_neg) = _strong_pos_rate_from_loader(train_loader)
    if 0.0 < p < 1.0:
        _pos_w = (1.0 - p) / p
        print(
            f"[TRAIN] bce_class_weight ON strong_pos_rate={p:.4f} n_pos={n_pos} n_neg={n_neg} pos_weight={_pos_w:.4f}",
            flush=True,
        )
    else:
        print(
            f"[TRAIN] bce_class_weight skipped (pos_rate={p:.4f}); keep pos_weight={_pos_w:.4f}",
            flush=True,
        )
    return _pos_w


def amp_enabled(args) -> bool:
    """Mixed precision is off unless --amp is set. --no_amp forces it off."""
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
    """Unwrap a window dataset to the builder that holds companies and dates."""
    return getattr(ds, "_inner", ds)


def load_all_data_for_profile(
    profile: DatasetProfile,
    root: str,
    news_type_subdir: str = "news_type",
    *,
    require_wind_news_source: bool = True,
    csmd_news_source: str = "llm_extract",
):
    """Load the raw price and news tables for one dataset profile."""
    if profile.key == "cmin_cn":
        return load_cmin_cn_all_data(
            root,
            news_type_subdir=news_type_subdir,
            require_wind_news_source=require_wind_news_source,
        )
    if profile.key in ("cmin_us", "massive", "stocknet"):
        return load_cmin_us_all_data(root, news_type_subdir=news_type_subdir)
    return load_csmd_all_data(root, news_type_subdir=news_type_subdir, news_source=csmd_news_source)


def _is_mtvc_contrast_mode(args) -> bool:
    """True for mtvc and mtvc_nogate."""
    return str(getattr(args, "contrast_mode", "") or "") in ("mtvc", "mtvc_nogate")


def _mtvc_enable_gate(mode: str) -> bool:
    """News gate is on for mtvc and off for mtvc_nogate."""
    return str(mode or "") != "mtvc_nogate"


def set_all_seeds(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch. cuDNN stays non-deterministic."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(False)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_training_device(device_spec=None) -> torch.device:
    """Parse cpu / gpu / cuda:N into a torch device."""
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
    """Training device from --filtered_cuda or --device."""
    spec = getattr(args, "filtered_cuda", None)
    if spec is None:
        spec = getattr(args, "device", None)
    if spec is None:
        spec = "gpu"
    return resolve_training_device(spec)


def make_model_dataloader(dataset, batch_size: int = 1, **kwargs):
    """DataLoader. Batch size 1 unwraps a WindowSample into the model batch."""

    def _collate(xs):
        sample = xs[0]
        if isinstance(sample, WindowSample):
            return sample.for_model()
        return sample

    if batch_size == 1 and "collate_fn" not in kwargs:
        kwargs["collate_fn"] = _collate
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, **kwargs)


def move_model_batch_to_device(batch, device: torch.device):
    """Move every tensor field of a heterogeneous batch onto ``device``."""
    for nt in batch.node_types:
        store = batch[nt]
        for key, val in list(store._fields.items()):
            if isinstance(val, torch.Tensor):
                store._fields[key] = val.to(device)
    return batch


import math
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from tool.cuda_guard import cuda_device_lock, forward_with_cuda_retry


def calculate_mcc(y_true, y_pred, mode="b"):
    """Matthews correlation on binary labels."""
    from sklearn.metrics import confusion_matrix

    (tn, fp, fn, tp) = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    num = tp * tn - fp * fn
    den = np.sqrt(np.float128(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)))
    return float(num / den) if den != 0 else 0.0


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
    """Task loss, accuracy, and MCC on strong samples of one split."""
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
                    from tool.numeric import sanitize_logits_for_bce
                _bl = float(loss_fn(sanitize_logits_for_bce(logits), labels).item())
                if math.isfinite(_bl):
                    total_loss += _bl
                    n_loss_batches += 1
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
    if not all_logits:
        print(f"[{split_name}] no strong samples.")
        return (None, None, None)
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    preds_bin = (torch.sigmoid(logits_cat) > pred_threshold).long().numpy()
    labels_bin = labels_cat.long().numpy()
    acc = float(accuracy_score(labels_bin, preds_bin))
    mcc = float(calculate_mcc(labels_bin, preds_bin))
    avg_loss = total_loss / n_loss_batches if loss_fn is not None and n_loss_batches > 0 else None
    if avg_loss is not None:
        print(f"[{split_name}] task_loss = {avg_loss:.6f}, acc = {acc:.4f}, MCC = {mcc:.4f}")
    else:
        print(f"[{split_name}] acc = {acc:.4f}, MCC = {mcc:.4f}")
    return (acc, mcc, avg_loss)


import torch.optim as optim


def _main_model_high_lr_param_list(model) -> list:
    """Market-global and industry PredFusion tokens use the higher lr."""
    params = [model.global_parameter, model.virtual_industry_node]
    if getattr(model, "pred_fusion_global_proj", None) is not None:
        params.extend(model.pred_fusion_global_proj.parameters())
    if getattr(model, "pred_fusion_industry_proj", None) is not None:
        params.extend(model.pred_fusion_industry_proj.parameters())
    return params


def _main_model_qk_mlp_param_list(model) -> list:
    return []


def _build_main_model_optimizer(
    model,
    *,
    lr_base: float = 0.0001,
    lr_global: float = 0.0001,
    wd: float = 5e-06,
    qk_mlp_wd: float = 0.0,
):
    qk_mlp_params = _main_model_qk_mlp_param_list(model)
    high_lr_params = _main_model_high_lr_param_list(model)
    special_ids = {id(p) for p in high_lr_params + qk_mlp_params}
    base_lr_params = [p for p in model.parameters() if id(p) not in special_ids]
    groups = [
        {"params": base_lr_params, "lr": lr_base, "weight_decay": wd},
        {"params": high_lr_params, "lr": lr_global, "weight_decay": wd},
    ]
    if qk_mlp_params:
        groups.append({"params": qk_mlp_params, "lr": lr_global, "weight_decay": qk_mlp_wd})
    return optim.Adam(groups)


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
    from tool.cuda_guard import cuda_device_lock, cuda_sync_and_clear, forward_with_cuda_retry
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
    MarketWindowDataset,
    WindowSample,
    get_profile,
    load_cmin_cn_all_data,
    load_csmd_all_data,
    resolve_vocab_path,
    NUM_VIRTUAL_NEWS_TYPES,
    NUM_RULE_TREND,
)
from tool.oversample_utils import load_oversample_extra_counts, summarize_oversample_counts


def run_pipeline(args, profile: DatasetProfile) -> None:
    """Build data and model, then train or evaluate."""
    import random
    import copy
    import torch.optim as optim
    from torch.nn import functional as F
    from sklearn.metrics import accuracy_score, classification_report

    args = normalize_run_args(args, profile)
    _nem = str(getattr(args, "news_encode_mode", "vocab") or "vocab")
    if _nem not in ("vocab", "vocab_padding") and (not _nem.startswith("vocab")):
        raise RuntimeError(f"FinBERT / non-vocab news_encode_mode removed, got {_nem!r}; use vocab")
    if str(getattr(args, "mode", "train") or "train") not in ("train", "train_simple"):
        raise RuntimeError(
            f"mode={getattr(args, 'mode', None)!r} is unsupported; use train or train_simple"
        )
    set_all_seeds(int(args.seed))
    print(f"[INFO] Random seed: {args.seed}")
    ds_root = args.dataset_root
    print(f"[INFO] Dataset root: {ds_root}")
    print(f"[INFO] price_type_source={args.price_type_source}")
    _MODEL_NEWS_MODE = {
        "finbert": "finbert_padding",
        "vocab": "vocab_padding",
        "finbert_embedding": "finbert_embedding",
        "finbert_tok": "finbert_tok",
    }
    use_cached_news_padding_ds = (
        bool(args.use_cached_news_emb) or args.news_encode_mode == "finbert_embedding"
    )
    if args.news_encode_mode in ("vocab", "finbert", "finbert_tok") and use_cached_news_padding_ds:
        print(
            f"[INFO] news_encode_mode={args.news_encode_mode!r} -> force disable cached .pt embedding load."
        )
        use_cached_news_padding_ds = False
    _finbert_path = resolve_finbert_model_path(args, profile)
    args.finbert_model_path = _finbert_path
    _news_sub = str(getattr(args, "news_type_subdir", "news_type") or "news_type")
    _emb_cache = str(getattr(args, "news_emb_cache_path", "") or "").strip() or None
    if use_cached_news_padding_ds and (not _emb_cache):
        _emb_cache = default_news_node_emb_dir(ds_root, _news_sub, max_seq_len=128)
        args.news_emb_cache_path = _emb_cache
    _num_score_news_types = max(
        3, int(getattr(args, "num_score_news_types", profile.virtual_news_type_count))
    )
    print(
        f"[INFO] news_encode_mode={args.news_encode_mode} -> Model(news_encode_mode={_MODEL_NEWS_MODE[args.news_encode_mode]!r}); finbert_model_path={_finbert_path!r}; finbert_freeze={bool(getattr(args, 'finbert_freeze', True))}; num_score_news_types={_num_score_news_types}; news_ablation_mode={getattr(args, 'news_ablation_mode', 'full')}; modality_ablation={getattr(args, 'modality_ablation', 'full')}; main_mlp_ablation_mode={getattr(args, 'main_mlp_ablation_mode', 'full')}; industry_fuse_site={getattr(args, 'industry_fuse_site', 'pred_fusion')}; industry_no_mean={bool(getattr(args, 'industry_no_mean', False))}; vin_ablation_mode={getattr(args, 'vin_ablation_mode', 'full')}; news_process_mode={getattr(args, 'news_process_mode', 'default')}; Dataset use_cached_news_emb={use_cached_news_padding_ds}"
        + (f" cache={_emb_cache!r}" if use_cached_news_padding_ds else "")
    )
    if args.news_encode_mode == "vocab":
        print("[INFO] vocab mode: dataset skips the FinBERT tokenizer.")
    elif args.news_encode_mode == "finbert":
        print(f"[INFO] finbert: dataset tokenizes to input_ids; online FinBERT @ {_finbert_path}")
    elif args.news_encode_mode == "finbert_tok":
        print(
            f"[INFO] finbert_tok: dataset uses the FinBERT tokenizer; the model uses its own embedding and transformer @ {_finbert_path}"
        )
    elif args.news_encode_mode == "finbert_embedding":
        print(
            f"[INFO] finbert_embedding: training reads offline news.emb and does not run FinBERT; cache={_emb_cache} (build it with --mode precompute_news_emb if missing)"
        )
    _ds_common = dataset_kwargs_from_args(args, profile)
    if profile.key == "cmin_cn":
        _all = bool(getattr(args, "cmin_cn_all_news", False))
        print(
            f"[INFO] CMIN-CN news source filter: {('ALL sources (--cmin_cn_all_news)' if _all else 'Wind-only (default)')}"
        )
        if bool(getattr(args, "require_window_news", False)):
            print(
                f"[INFO] require_window_news=ON: keep (day,stock) only if all {int(profile.price_window)} price-window days have news",
                flush=True,
            )
        else:
            print(
                "[INFO] require_window_news=OFF: a five-day window may have no news (empty token)",
                flush=True,
            )
    if profile.key in ("csmd50", "csmd300", "sse50", "cmin_cn", "cmin_us", "massive", "stocknet"):
        _sm = str(
            _ds_common.get("split_mode")
            or getattr(args, "split_mode", "contiguous")
            or "contiguous"
        )
        if _sm == "stagger_4y":
            print(
                "[INFO] split_mode=stagger_4y: Y1 train1-9/val10-12; Y2 test1-3/train4-12; Y3 val1-3/test4-6/train7-12; Y4 train1-3/val4-7/test8-12 (year0=first calendar year in data)",
                flush=True,
            )
        elif _sm == "seasonal_h1q3q4":
            print(
                "[INFO] split_mode=seasonal_h1q3q4: train=months[1-6] val=months[7-9] test=months[10-12] (all years pooled)",
                flush=True,
            )
        elif _sm == "week_roundrobin":
            print(
                "[INFO] split_mode=week_roundrobin: ISO weeks 6:2:2 (train/val/test); train samples whose price_window lookback overlaps val/test are purged",
                flush=True,
            )
        elif _sm == "month_roundrobin":
            print(
                "[INFO] split_mode=month_roundrobin: calendar months 4:1:1 (train/val/test); train samples whose price_window lookback overlaps val/test are purged",
                flush=True,
            )
        elif _sm == "timefrac_622":
            print(
                "[INFO] split_mode=timefrac_622: chronological trading-day split 60% train / 20% val / 20% test",
                flush=True,
            )
        elif _sm in ("day_rr_1022", "day_rr_411"):
            print(
                "[INFO] split_mode=day_rr_1022: NEW trading-day 10:2:2 round-robin + lookback purge",
                flush=True,
            )
        elif _sm == "quarter_rr_211":
            print(
                "[INFO] split_mode=quarter_rr_211: NEW quarter 2:1:1 round-robin + lookback purge",
                flush=True,
            )
        elif _sm == "quarter_rr_311":
            _purge = not bool(getattr(args, "no_train_lookback_purge", False))
            print(
                f"[INFO] split_mode=quarter_rr_311: calendar 3:1:1 block (Y1 1-9T/10-12V; Y2 1-3Te/4-12T; Y3 1-3V/4-6Te/7-12T; Y4 1-3T/4-6V/7-12Te) [Y4 differs from stagger_4y: val4-6 test7-12]; train_lookback_purge={_purge}",
                flush=True,
            )
        elif _sm == "year_gap":
            print(
                "[INFO] split_mode=year_gap: NEW non-contiguous years train={Y0,Y2} val=Y1 test=Y3 + lookback purge",
                flush=True,
            )
        else:
            _sd = dict(_ds_common.get("split_dates") or profile.split_dates)
            print(
                f"[INFO] split_dates: train=[{_sd.get('train_start')},{_sd.get('train_end')}] val=[{_sd.get('val_start')},{_sd.get('val_end')}] test=[{_sd.get('test_start')},{_sd.get('test_end')}]",
                flush=True,
            )
    print("[INFO] graph_cache=off: graphs are built online.")
    if args.mode in ("train", "train_simple"):
        from model import Model as SimpleModel

        train_dataset = MarketWindowDataset(profile, mode="train", **_ds_common)
        _subset_n = int(getattr(args, "subset_top_companies", 0) or 0)
        _ds_eval = dict(_ds_common)
        if _subset_n > 0 and profile.key == "cmin_cn":
            _keep = list(getattr(_builder_dataset(train_dataset), "companies", []) or [])
            _ds_eval["subset_top_companies"] = 0
            _ds_eval["companies_whitelist"] = _keep
        val_dataset = MarketWindowDataset(profile, mode="val", **_ds_eval)
        test_dataset = MarketWindowDataset(profile, mode="test", **_ds_eval)
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
                    f"MTVC_FIXED_TRAIN_ORDER length {len(_order)} != train windows {len(train_dataset)}"
                )
            print(
                f"[TRAIN] fixed train order from {_fixed_order_path} (n={len(_order)} first={_order[:5]})",
                flush=True,
            )
            train_loader = make_model_dataloader(
                train_dataset, batch_size=1, shuffle=False, sampler=_FixedOrderSampler(_order)
            )
        else:
            train_loader = make_model_dataloader(train_dataset, batch_size=1, shuffle=True)
        val_loader = make_model_dataloader(val_dataset, batch_size=1, shuffle=False)
        test_loader = make_model_dataloader(test_dataset, batch_size=1, shuffle=False)
        device = device_from_args(args)
        _vocab_path = str(getattr(args, "vocab_path", "") or "").strip() or resolve_vocab_path(
            profile.vocab_path, "dict_csmd.pkl", "dict_cn.pkl"
        )
        _grad_clip = float(getattr(args, "grad_clip", 1.0))
        _companies = list(getattr(_builder_dataset(train_dataset), "companies", []) or [])
        _n_stocks = len(_companies) if _companies else int(profile.grid_num_stocks)
        _nam = str(getattr(args, "news_ablation_mode", "full") or "full").lower()
        if _nam in (
            "add_volatility",
            "vol_screen_trend_attn",
            "vol_screen_shared_pool",
            "vol_screen_trend_attn_aux",
        ):
            raise ValueError(
                f"news_ablation_mode={_nam!r} was removed with the volatility branch; use full / trend_attn / no_news"
            )
        model = SimpleModel(
            news_encode_mode=_MODEL_NEWS_MODE[args.news_encode_mode],
            news_window_agg_mode=args.news_window_agg_mode,
            vocab_path=_vocab_path,
            grid_num_stocks=_n_stocks,
            price_input_dim=3,
            no_price_token=bool(getattr(args, "no_price_token", False)),
            no_price_token_pad=bool(getattr(args, "no_price_token_pad", False)),
            no_news_token=bool(getattr(args, "no_news_token", False)),
            no_news_token_pad=bool(getattr(args, "no_news_token_pad", False)),
            modality_ablation=str(getattr(args, "modality_ablation", "full") or "full"),
            industry_no_mean=bool(getattr(args, "industry_no_mean", False)),
            num_score_news_types=_num_score_news_types,
            news_ablation_mode=_nam,
            main_mlp_ablation_mode=getattr(args, "main_mlp_ablation_mode", "full"),
            industry_fuse_site=str(
                getattr(args, "industry_fuse_site", "pred_fusion") or "pred_fusion"
            ),
            vin_ablation_mode=getattr(args, "vin_ablation_mode", "full"),
            news_word_emb=str(getattr(args, "news_word_emb", "random") or "random"),
            news_process_mode=getattr(args, "news_process_mode", "default"),
            price_encoder=str(getattr(args, "price_encoder", "lstm")),
            vin_connect_mode=str(getattr(args, "vin_connect_mode", "default") or "default"),
            news_top_k=int(getattr(args, "news_padding_k", 5) or 5),
            **model_finbert_kwargs(args, profile),
            **model_size_kwargs(args),
        )
        model = wrap_model_devices(model, args, device)
        _news_pt = str(getattr(args, "news_pretrain_ckpt", "") or "").strip()
        if _news_pt:
            raise RuntimeError("news_pretrain_ckpt unsupported in (pretrain helpers removed)")
        _contrast_aux_w = float(getattr(args, "contrast_aux_weight", 0.0) or 0.0)
        _contrast_mode = str(getattr(args, "contrast_mode", "nocontrast") or "nocontrast")
        if _contrast_aux_w > 0:
            if _contrast_mode not in ("mtvc", "mtvc_nogate"):
                raise ValueError(
                    f"contrast_mode={_contrast_mode!r} is not an official ablation; use mtvc or mtvc_nogate"
                )
            _core_c = model.module if hasattr(model, "module") else model
            _core_c.ensure_contrast_heads(proj_dim=int(getattr(args, "ssl_proj_dim", 64) or 64))
            for _n in (
                "contrast_proj_p",
                "contrast_proj_n",
                "contrast_state_proj",
                "contrast_pair_proj",
            ):
                _mod = getattr(_core_c, _n, None)
                if _mod is not None:
                    _mod.to(device)
            print(
                f"[TRAIN] contrast_aux=ON weight={_contrast_aux_w} mode={_contrast_mode} temp={float(getattr(args, 'contrast_temperature', 0.07) or 0.07)} neg={int(getattr(args, 'contrast_num_neg', 4) or 4)} anchors≤{int(getattr(args, 'contrast_max_anchors', 48) or 48)}",
                flush=True,
            )
            if _mtvc_enable_gate(_contrast_mode):
                from contrast_aux import enable_contrast_gates

                _case_L = int(getattr(args, "mtvc_case_layer", 6) or 6)
                _case_feat = str(getattr(args, "mtvc_case_feat", "joint") or "joint")
                enable_contrast_gates(_core_c, case_mine_layer=_case_L)
                _ut = getattr(_core_c, "unitrans", None)
                _ng = getattr(_ut, "news_gates", None) if _ut is not None else None
                _n_g = len(_ng) if _ng is not None else 0
                print(
                    f"[TRAIN] mtvc news_gate=ON layers=1..{_n_g} case_mine_layer={_case_L} case_feat={_case_feat} contrast_news_mode={getattr(args, 'mtvc_contrast_news_mode', 'nbag')} (every layer before last; L_pair on label peers)",
                    flush=True,
                )
            else:
                print("[TRAIN] mtvc news_gate=OFF (L_pair only)", flush=True)
        if bool(getattr(args, "freeze_news_text_tf", False)):
            raise RuntimeError(
                "freeze_news_text_tf unsupported in (pretrain_massive_news helper removed)"
            )
        else:
            print("[TRAIN] freeze_news_text_tf=OFF (text TF + word_embedding finetune)", flush=True)
        if bool(getattr(args, "freeze_price_news_encoders", False)):
            _fz_enc: list[str] = []
            _core_m = model.module if hasattr(model, "module") else model
            for _name, _p in _core_m.named_parameters():
                _freeze = False
                if _name == "unitrans" or _name.startswith("unitrans."):
                    _freeze = True
                elif any(
                    (
                        _name == pref or _name.startswith(pref + ".")
                        for pref in (
                            "unitrans_price_tok",
                            "bert_tok_transformer",
                            "unitrans_news_text_tf",
                            "unified_news_proj",
                        )
                    )
                ):
                    _freeze = True
                if _freeze:
                    _p.requires_grad = False
                    _fz_enc.append(_name)
            n_train = sum((1 for _p in _core_m.parameters() if _p.requires_grad))
            n_all = sum((1 for _ in _core_m.parameters()))
            print(
                f"[TRAIN] freeze_price_news_encoders=ON frozen={len(_fz_enc)} trainable_tensors={n_train}/{n_all}",
                flush=True,
            )
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
            _mode = "freeze" if bool(getattr(args, "freeze_news_text_tf", False)) else "finetune"
            print(
                f"[TRAIN] unitrans_news_text_tf → CPU (layers={_ntl}, {_mode}{(', forced' if _force_cpu and _ntl < 4 else '')})",
                flush=True,
            )
        print(
            f"[TRAIN] price_encoder={getattr(args, 'price_encoder', 'lstm')} emb_dim={resolve_emb_dim(args)} news_emb_dim={int(getattr(args, 'news_emb_dim', 32) or 32)} news_ablation={_nam} news_text_tf_layers={_ntl}",
            flush=True,
        )
        print(
            f"[TRAIN] main_mlp slots: news={bool(getattr(model, 'include_news_in_main_mlp', False))} news_global={bool(getattr(model, 'include_news_global_in_main_mlp', False))} market_global={bool(getattr(model, 'include_global_in_main_mlp', False))} industry={bool(getattr(model, 'include_industry_in_main_mlp', False))}",
            flush=True,
        )
        if not bool(getattr(model, "include_price_token", True)):
            _pfill = str(getattr(model, "price_token_fill", "zero") or "zero")
            if _pfill == "pad":
                print("[TRAIN] price_token=OFF (learnable pad; no-price ablation)", flush=True)
            else:
                print("[TRAIN] price_token=OFF (fixed zero; no-price ablation)", flush=True)
        if not bool(getattr(model, "include_news_token", True)):
            _fill = str(getattr(model, "news_token_fill", "zero") or "zero")
            if _fill == "pad":
                print("[TRAIN] news_token=OFF (learnable pad; no-news ablation)", flush=True)
            else:
                print("[TRAIN] news_token=OFF (fixed zero; no-news ablation)", flush=True)
        _core = model.module if hasattr(model, "module") else model
        _price_tok = type(getattr(_core, "unitrans_price_tok", None)).__name__
        _price_mlp = type(getattr(_core, "unitrans_price_mlp", None)).__name__
        _news_tf = type(getattr(_core, "unitrans_news_text_tf", None)).__name__
        _bert_tok = type(getattr(_core, "bert_tok_transformer", None)).__name__
        _news_mlp = type(getattr(_core, "unitrans_news_mlp", None)).__name__
        _news_pool = None
        _unitrans = getattr(_core, "unitrans", None)
        if _unitrans is not None:
            _nsp = getattr(_unitrans, "news_slot_pool", None)
            _news_pool = getattr(_nsp, "pool", None) if _nsp is not None else None
        print(
            f"[ARCH] encoder={getattr(_core, 'price_encoder', None)} price_tok={_price_tok} price_mlp={_price_mlp} news_word_emb={getattr(_core, 'news_word_emb', None)} news_text_tf={_news_tf} bert_tok={_bert_tok} news_mlp={_news_mlp} news_pool={_news_pool} vol_token={bool(getattr(_core, 'include_vol_token', False))} vol_in_price={bool(getattr(_core, 'include_vol_in_price', False))} price_input_dim={int(getattr(_core, 'price_input_dim', 3))}",
            flush=True,
        )
        optimizer = _build_main_model_optimizer(model, lr_base=1e-05, lr_global=1e-05, wd=5e-06)
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
                f"[TRAIN] train oversample extra={_os_path} unique={_os_sum['n_unique_keys']} copies={_os_sum['n_extra_copies']} (loss weight=1+extra; val/test unchanged)",
                flush=True,
            )
        ES_START_EPOCH = max(1, int(args.es_start_epoch))
        ES_PATIENCE = max(1, int(args.es_patience))
        ES_METRIC = str(getattr(args, "es_metric", "acc_mcc") or "acc_mcc")
        best_val_acc_plus_mcc = float("-inf")
        best_val_loss = float("inf")
        patience_cnt = 0
        best_test_acc = None
        best_test_mcc = None
        best_test_epoch = None
        best_vol_test_acc = None
        best_vol_test_mcc = None
        print(f"[TRAIN] #train windows = {len(train_dataset)}")
        print(f"[TRAIN] news_window_agg_mode={args.news_window_agg_mode}")
        print("[TRAIN] unified modeling: full news->price edges; no volatility screen")
        if ES_METRIC == "val_loss":
            print(
                f"[TRAIN] early_stop: monitor val_task_loss (lower is better) from epoch>={ES_START_EPOCH}, patience={ES_PATIENCE}"
            )
        else:
            print(
                f"[TRAIN] early_stop: monitor val_acc+val_MCC (higher is better) from epoch>={ES_START_EPOCH}, patience={ES_PATIENCE}"
            )
        ckpt_mgr = CheckpointManager(args, profile)
        (start_epoch, _resume_info) = maybe_resume_from_args(
            args, profile, model=model, model_optimizer=optimizer, ckpt_manager=ckpt_mgr
        )
        _xburn = str(os.environ.get("MTVC_CROSS_ATTN_POST_RANDN", "") or "").strip()
        if _xburn and str(getattr(args, "price_encoder", "") or "") in ("cross_attn", "cross_attn"):
            _xn = max(0, int(_xburn))
            if _xn:
                _ = torch.randn(_xn)
                print(
                    f"[TRAIN] cross_attn extra_randn={_xn} (post-init shuffle calibrate)",
                    flush=True,
                )
        _fixed_order_active = bool(str(os.environ.get("MTVC_FIXED_TRAIN_ORDER", "") or "").strip())
        for epoch in range(start_epoch, args.epochs + 1):
            if _fixed_order_active:
                _ = int(torch.empty((), dtype=torch.int64).random_().item())
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            for batch in train_loader:
                if batch["price"].x.numel() == 0 or batch["label"].x.numel() == 0:
                    continue
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
                    logits = forward_with_cuda_retry(_fwd_train, device=device, tag="train-fwd")
                    if logits.numel() == 0:
                        continue
                    labels = batch["label"].x.view(-1).to(device)
                    strong_mask = batch["label"].strong_mask.view(-1).to(device)
                    valid_mask = (
                        batch["label"].valid_mask.view(-1).to(device)
                        if hasattr(batch["label"], "valid_mask")
                        else torch.ones_like(strong_mask, dtype=torch.bool)
                    )
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
                            from tool.numeric import sanitize_logits_for_bce
                        _logits_t = sanitize_logits_for_bce(logits[train_mask])
                        _per = loss_fn_none(_logits_t, labels[train_mask])
                        task_loss = (_per * _w).sum() / _w.sum().clamp_min(1e-08)
                    else:
                        try:
                            from tool.numeric import sanitize_logits_for_bce
                        except ImportError:
                            from tool.numeric import sanitize_logits_for_bce
                        task_loss = loss_fn(
                            sanitize_logits_for_bce(logits[train_mask]), labels[train_mask]
                        )
                    loss = task_loss
                    _m = getattr(model, "module", model)
                    if _contrast_aux_w > 0:
                        contrast_aux_v = 0.0
                        contrast_bsz = 0.0
                        _c_stats = {}
                        _cmode = str(getattr(args, "contrast_mode", "nocontrast"))
                        if _cmode not in ("mtvc", "mtvc_nogate"):
                            raise ValueError(
                                f"contrast_mode={_cmode!r} is not an official ablation"
                            )
                        from contrast_aux import compute_mtvc_aux

                        (_c_loss, _c_stats) = compute_mtvc_aux(
                            _m,
                            batch,
                            temperature_pair=float(getattr(args, "mtvc_temp_pair", 0.07) or 0.07),
                            max_anchors=int(getattr(args, "contrast_max_anchors", 48) or 48),
                            max_news_per_bag=int(
                                getattr(args, "contrast_max_news_per_bag", 6) or 6
                            ),
                            day_mode=str(getattr(args, "contrast_day_mode", "all")),
                            sim_quantile=float(getattr(args, "sim_quantile", 0.75) or 0.75),
                            tau_y=float(getattr(args, "mtvc_tau_y", 0.5) or 0.5),
                            beta_hard=float(getattr(args, "mtvc_beta_hard", 0.5) or 0.5),
                            lambda_pair=float(getattr(args, "mtvc_lambda_pair", 0.2) or 0.2),
                            pair_peer="label",
                            use_clip=False,
                            state=None,
                            epoch=int(epoch),
                            max_epochs=int(getattr(args, "epochs", 50) or 50),
                            temperature=float(getattr(args, "contrast_temperature", 0.07) or 0.07),
                            alpha_bad=float(getattr(args, "mtvc_alpha_bad", 0.3) or 0.3),
                            lambda_clip_start=float(
                                getattr(args, "mtvc_lambda_clip_start", 0.3) or 0.3
                            ),
                            lambda_clip_end=float(
                                getattr(args, "mtvc_lambda_clip_end", 0.05) or 0.05
                            ),
                            device=device,
                            enable_gate=_mtvc_enable_gate(_cmode),
                            contrast_news_mode=str(
                                getattr(args, "mtvc_contrast_news_mode", "nbag") or "nbag"
                            ),
                            case_mine_layer=int(getattr(args, "mtvc_case_layer", 6) or 6),
                            case_feat=str(getattr(args, "mtvc_case_feat", "joint") or "joint"),
                        )
                        contrast_bsz = float(_c_stats.get("bsz", 0.0) or 0.0)
                        _skipped = float(_c_stats.get("skipped", 1.0))
                        if _skipped < 0.5:
                            loss = loss + _c_loss
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
                    _aux_msg += f" contrast={contrast_aux_v:.6f} c_bsz={contrast_bsz:.0f} c_pos={float(_c_stats.get('pos_sim', 0.0) or 0.0):.3f} c_neg={float(_c_stats.get('neg_sim', 0.0) or 0.0):.3f}"
                    if _is_mtvc_contrast_mode(args):
                        _aux_msg += f" Lclip={float(_c_stats.get('L_clip', 0.0) or 0.0):.3f} Lpair={float(_c_stats.get('L_pair', 0.0) or 0.0):.3f} lam1={float(_c_stats.get('lam1', 0.0) or 0.0):.3f} lam2={float(_c_stats.get('lam2', 0.0) or 0.0):.3f} g={float(_c_stats.get('gamma_mean', 0.0) or 0.0):.2f} w={float(_c_stats.get('w_mean', 0.0) or 0.0):.2f}"
                        _pp = float(_c_stats.get("pair_peer", 0.0) or 0.0)
                        if _pp > 0.5:
                            _aux_msg += f" peer={('news' if _pp < 1.5 else 'price')} n_peer={float(_c_stats.get('n_peer', 0.0) or 0.0):.0f}"
                print(
                    f"[train] epoch={epoch} batch={num_batches} N_price={batch['price'].x.size(0)} N_news={(batch['news'].x.size(0) if 'news' in batch.node_types else 0)} loss={loss.item():.6f}{_aux_msg}"
                )
            if num_batches > 0:
                epoch_loss /= num_batches
            print(f"[Epoch {epoch:03d}] train train_loss = {epoch_loss:.6f}")
            cuda_sync_and_clear(device)
            _skip_train_eval = str(
                os.environ.get("DUAL_TF_SKIP_TRAIN_EVAL", "0")
            ).strip().lower() in ("1", "true", "yes", "on")
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
                        logits = forward_with_cuda_retry(_fwd_val, device=device, tag="val-fwd")
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
                    logits_m = logits[eval_mask]
                    labels_m = labels[eval_mask]
                    try:
                        from tool.numeric import sanitize_logits_for_bce
                    except ImportError:
                        from tool.numeric import sanitize_logits_for_bce
                    logits_m = sanitize_logits_for_bce(logits_m)
                    _vloss = float(loss_fn(logits_m, labels_m).item())
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
                        f"[Epoch {epoch:03d}] val_task_loss = {val_task_loss_epoch:.6f}, val_acc = {acc:.4f}, val_MCC = {mcc:.4f}"
                    )
                else:
                    val_task_loss_epoch = None
                    print(
                        f"[Epoch {epoch:03d}] val_task_loss = nan (no finite batch losses), val_acc = {acc:.4f}, val_MCC = {mcc:.4f}"
                    )
            else:
                val_task_loss_epoch = None
                val_acc_epoch = None
                val_mcc_epoch = None
                print(f"[Epoch {epoch:03d}] val set has no strong samples.")
            test_acc_epoch = None
            test_mcc_epoch = None
            vol_test_acc_epoch = None
            vol_test_mcc_epoch = None
            _skip_test = str(
                os.environ.get("DUAL_TF_SKIP_TEST_DURING_TRAIN", "0")
            ).strip().lower() in ("1", "true", "yes", "on")
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
                            logits = forward_with_cuda_retry(
                                _fwd_test, device=device, tag="test-fwd"
                            )
                        if logits.numel() == 0:
                            continue
                        labels = batch["label"].x.view(-1).to(device)
                        strong_mask = batch["label"].strong_mask.view(-1).to(device)
                        valid_mask = (
                            batch["label"].valid_mask.view(-1).to(device)
                            if hasattr(batch["label"], "valid_mask")
                            else torch.ones_like(strong_mask, dtype=torch.bool)
                        )
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
                            f"[TRAIN] early_stop: no finite val_task_loss this epoch, patience {patience_cnt}/{ES_PATIENCE}"
                        )
                    elif val_task_loss_epoch < best_val_loss - 1e-08:
                        best_val_loss = float(val_task_loss_epoch)
                        patience_cnt = 0
                        if test_acc_epoch is not None and test_mcc_epoch is not None:
                            best_test_acc = test_acc_epoch
                            best_test_mcc = test_mcc_epoch
                            best_test_epoch = epoch
                        if vol_test_acc_epoch is not None and vol_test_mcc_epoch is not None:
                            best_vol_test_acc = vol_test_acc_epoch
                            best_vol_test_mcc = vol_test_mcc_epoch
                        _va = "None" if val_acc_epoch is None else f"{val_acc_epoch:.4f}"
                        _vm = "None" if val_mcc_epoch is None else f"{val_mcc_epoch:.4f}"
                        print(
                            f"[TRAIN] early_stop: val_task_loss improved -> best_loss={best_val_loss:.6f} (acc={_va}, MCC={_vm})"
                        )
                    else:
                        patience_cnt += 1
                        print(
                            f"[TRAIN] early_stop: no val_task_loss improvement, patience {patience_cnt}/{ES_PATIENCE} (best_loss={best_val_loss:.6f})"
                        )
                else:
                    val_sum = (
                        val_acc_epoch + val_mcc_epoch
                        if val_acc_epoch is not None and val_mcc_epoch is not None
                        else None
                    )
                    if val_sum is None:
                        patience_cnt += 1
                        print(
                            f"[TRAIN] early_stop: no val acc/MCC this epoch, patience {patience_cnt}/{ES_PATIENCE}"
                        )
                    elif val_sum > best_val_acc_plus_mcc + 1e-08:
                        best_val_acc_plus_mcc = val_sum
                        patience_cnt = 0
                        if test_acc_epoch is not None and test_mcc_epoch is not None:
                            best_test_acc = test_acc_epoch
                            best_test_mcc = test_mcc_epoch
                            best_test_epoch = epoch
                        if vol_test_acc_epoch is not None and vol_test_mcc_epoch is not None:
                            best_vol_test_acc = vol_test_acc_epoch
                            best_vol_test_mcc = vol_test_mcc_epoch
                        print(
                            f"[TRAIN] early_stop: val_acc+val_MCC improved -> best_sum={best_val_acc_plus_mcc:.6f} (acc={val_acc_epoch:.4f}, MCC={val_mcc_epoch:.4f}, val_task_loss={val_task_loss_epoch:.6f})"
                        )
                    else:
                        patience_cnt += 1
                        print(
                            f"[TRAIN] early_stop: no val_acc+val_MCC improvement, patience {patience_cnt}/{ES_PATIENCE} (best_sum={best_val_acc_plus_mcc:.6f})"
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
                        f"[TRAIN] early stopping at epoch {epoch} (no {mon} gain for {ES_PATIENCE} epochs since monitoring started)."
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
                valid_mask = (
                    batch["label"].valid_mask.view(-1).to(device)
                    if hasattr(batch["label"], "valid_mask")
                    else torch.ones_like(strong_mask, dtype=torch.bool)
                )
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
            print(
                f"[{tag}] epoch={best_test_epoch} acc = {best_test_acc:.4f}, MCC = {best_test_mcc:.4f}"
            )
            if best_vol_test_acc is not None and best_vol_test_mcc is not None:
                print(
                    f"[{tag}] vol_acc = {best_vol_test_acc:.4f}, vol_MCC = {best_vol_test_mcc:.4f}"
                )
        else:
            tag = "BEST TEST@BEST VAL_LOSS" if ES_METRIC == "val_loss" else "BEST TEST@BEST VAL"
            print(f"[{tag}] no available test metrics when val improved.")


def run(profile_key: str, args) -> None:
    """Dispatch train / eval for one dataset profile."""
    profile = get_profile(profile_key) if isinstance(profile_key, str) else profile_key
    print(f"[RUN] dataset={profile.key} mode={args.mode}")
    run_pipeline(args, profile)


run_legacy = run
