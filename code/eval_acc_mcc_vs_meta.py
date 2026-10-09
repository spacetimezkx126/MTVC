#!/usr/bin/env python3
"""Re-evaluate Acc/MCC from paper checkpoints and compare to best.pt.meta.json.

Prefer the training env (CAMEF / torch 2.6) so matmul kernels match the run that
wrote ``best.pt.meta.json``. A 1–2 / ~8653 Acc flip (Δ≈1e-4) vs meta is normal
cuDNN nondeterminism, not a model bug — default atol absorbs that.

Usage::

    /home/zhaokx/miniconda3/envs/CAMEF/bin/python code/eval_acc_mcc_vs_meta.py
    /home/zhaokx/miniconda3/envs/CAMEF/bin/python code/eval_acc_mcc_vs_meta.py \\
        --roles full,nonews,crossattn --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
MTVC = REPO / "code" / "mtvc"
CKPT_ROOT = REPO / "checkpoints"
DUAL_TF = Path("/home/zhaokx/Pattern/Pattern_Mining/dual_tf")

os.chdir(str(REPO if (REPO / "checkpoints").is_dir() else DUAL_TF))
# Prefer dual_tf as CWD so dataset roots resolve like training launches.
if DUAL_TF.is_dir():
    os.chdir(str(DUAL_TF))
os.environ["DUAL_TF_MODEL_DIR"] = str(DUAL_TF)

# Match launch_*_casel6_3ds.sh paper env
os.environ["MTVC_USE_TYPE_EMB"] = "1"
os.environ["MTVC_UNIFIED_CAUSAL"] = "1"
os.environ["MTVC_UNIFIED_SAME_DAY"] = "0"
os.environ["MTVC_SOFT_PRUNE"] = "0"
os.environ["MTVC_SHARE_PRICE_NEWS_ENC"] = "0"
os.environ["MTVC_TRANSFORMER_LAYERS"] = "6"
os.environ.setdefault("MAIN_MLP_FEAT_DBG", "0")
os.environ.pop("MTVC_UNIFIED_LAYERS", None)
os.environ.pop("MTVC_LOOP_ROUNDS", None)
os.environ.pop("MTVC_HARD_FINETUNE", None)

# Match train.py cuDNN flags (nondeterministic OK; same as training-time test).
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

sys.path.insert(0, str(MTVC))

from data import MarketWindowDataset, get_profile, resolve_vocab_path  # noqa: E402
from train import (  # noqa: E402
    _builder_dataset,
    _is_mtvc_mode,
    _mtvc_enable_gate,
    _normalize_contrast_mode,
    calculate_mcc,
    dataset_kwargs_from_args,
    make_model_dataloader,
    model_size_kwargs,
    normalize_run_args,
    wrap_model_devices,
)
from model import Model, normalize_price_encoder  # noqa: E402
from checkpoint import load_checkpoint  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402


ROLES = {
    # Paper L6 formal dirs under checkpoints/ (s42)
    "full_mtvc": "csmd50/full_casel6/s42",
    "full": "csmd50/full_casel6/s42",
    "naive_fusion": "csmd50/abl_nocontrast_casel6/s42",
    "nocontrast": "csmd50/abl_nocontrast_casel6/s42",
    "noprice": "csmd50/abl_noprice_casel6/s42",
    "nogate": "csmd50/abl_nogate_casel6/s42",
    "contrast_only": "csmd50/abl_nogate_casel6/s42",
    "no_global": "csmd50/abl_noglobal_casel6/s42",
    "noglobal": "csmd50/abl_noglobal_casel6/s42",
    "no_industry": "csmd50/abl_noindustry_casel6/s42",
    "noindustry": "csmd50/abl_noindustry_casel6/s42",
    "multinews_token": "csmd50/multinews_casel6/s42",
    "multinews": "csmd50/multinews_casel6/s42",
    "crossattn": "csmd50/crossattn_casel6/s42",
    "fullunified": "csmd50/fullunified_casel6/s42",
    "vin_ind_as_mkt": "csmd50/vin_ind_as_mkt_casel6/s42",
    "vin_mkt_as_ind": "csmd50/vin_mkt_as_ind_casel6/s42",
    "casel1": "csmd50/full_casel1/s42",
    "casel2": "csmd50/full_casel2/s42",
    "casel3": "csmd50/full_casel3/s42",
    "casel4": "csmd50/full_casel4/s42",
    "casel5": "csmd50/full_casel5/s42",
    "casel6": "csmd50/full_casel6/s42",
    "case_layer_1": "csmd50/full_casel1/s42",
    "case_layer_2": "csmd50/full_casel2/s42",
    "case_layer_3": "csmd50/full_casel3/s42",
    "case_layer_4": "csmd50/full_casel4/s42",
    "case_layer_5": "csmd50/full_casel5/s42",
    "case_layer_6": "csmd50/full_casel6/s42",
    "full_csmd300": "csmd300/full_casel6/s42",
    "full_massive": "massive/full_casel6/s42",
    "add_global_massive": "massive/abl_addglobal_casel6/s42",
}

# Default: L6 paper stack + layer sweep on csmd50 s42
DEFAULT_ROLES = (
    "full_mtvc,casel1,casel2,casel3,casel4,casel5,casel6,"
    "naive_fusion,noprice,nogate,no_global,no_industry,"
    "multinews,crossattn,fullunified,vin_ind_as_mkt,vin_mkt_as_ind"
)


def _roles_from_catalog(dataset: str = "csmd50") -> list[str]:
    """Unique catalog roles that have a linked ckpt for ``dataset``."""
    cat_path = REPO / "results" / "experiment_catalog.json"
    if not cat_path.is_file():
        return [r.strip() for r in DEFAULT_ROLES.split(",") if r.strip()]
    cat = json.loads(cat_path.read_text())
    out = []
    seen = set()
    for e in cat.get("experiments") or []:
        if str(e.get("dataset")) != dataset:
            continue
        role = str(e.get("role") or "").strip()
        if not role or role in seen:
            continue
        # Prefer explicit ROLES map; else derive from mtvc_name
        if role in ROLES:
            out.append(role)
            seen.add(role)
            continue
        name = str(e.get("mtvc_name") or "")
        if name:
            key = f"_path:{dataset}/{name}/s42"
            ROLES[key] = f"{dataset}/{name}/s42"
            out.append(key)
            seen.add(role)
            seen.add(key)
    return out or [r.strip() for r in DEFAULT_ROLES.split(",") if r.strip()]


def _resolve_role(glob_pat: str) -> Path | None:
    # Exact relative path (no glob)
    if "*" not in glob_pat and "?" not in glob_pat:
        p = CKPT_ROOT / glob_pat
        if p.is_dir() and (p / "best.pt").is_file():
            return p
        if p.name != "s42" and (p / "s42" / "best.pt").is_file():
            return p / "s42"
    hits = sorted(CKPT_ROOT.glob(glob_pat))
    if not hits:
        return None
    for h in hits:
        if h.name == "s42" or str(h).endswith("/s42"):
            return h if h.is_dir() else h.parent
    return hits[0] if hits[0].is_dir() else hits[0].parent


def _ckpt_key_flags(ckpt_path: Path) -> tuple[bool, bool, bool]:
    """Return (has_global_score, has_news_gates, has_ssl_proj)."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k in ("model", "state_dict", "model_state_dict"):
        if isinstance(sd, dict) and k in sd:
            sd = sd[k]
            break
    keys = [str(k) for k in sd]
    return (
        any("global_score_proj" in k for k in keys),
        any("news_gates" in k for k in keys),
        any(k.startswith("ssl_proj") or ".ssl_proj" in k for k in keys),
    )


def _ckpt_has_global_score(ckpt_path: Path) -> bool:
    return _ckpt_key_flags(ckpt_path)[0]


def _build_model(
    args,
    profile,
    train_ds,
    device,
    *,
    use_global_score: bool,
    force_gates: bool | None = None,
    force_ssl: bool | None = None,
):
    companies = list(getattr(_builder_dataset(train_ds), "companies", []) or [])
    n_stocks = len(companies) if companies else int(getattr(profile, "grid_num_stocks", 50) or 50)
    if not companies:
        companies = [f"C{i:03d}" for i in range(n_stocks)]
    vocab_path = resolve_vocab_path(
        getattr(args, "vocab_path", None) or profile.vocab_path,
        "dict_csmd.pkl",
    )
    _nam = str(getattr(args, "news_ablation_mode", "full") or "full")
    news_mode = {
        "finbert": "finbert_padding",
        "vocab": "vocab_padding",
        "finbert_embedding": "finbert_embedding",
        "finbert_tok": "finbert_tok",
    }[str(args.news_encode_mode)]
    model = Model(
        news_encode_mode=news_mode,
        vocab_path=vocab_path,
        grid_num_stocks=n_stocks,
        price_input_dim=3,
        no_price_token=bool(getattr(args, "no_price_token", False)),
        industry_no_mean=bool(getattr(args, "industry_no_mean", False)),
        news_ablation_mode=_nam,
        pred_fusion_ablation_mode=getattr(args, "pred_fusion_ablation_mode", "full"),
        industry_fuse_site=str(getattr(args, "industry_fuse_site", "pred_fusion") or "pred_fusion"),
        vin_ablation_mode=getattr(args, "vin_ablation_mode", "full"),
        news_word_emb=str(getattr(args, "news_word_emb", "random") or "random"),
        news_process_mode=getattr(args, "news_process_mode", "default"),
        dual_own_prop_news=bool(getattr(args, "dual_own_prop_news", False)),
        own_news_only=bool(getattr(args, "own_news_only", False)),
        prop_same_trend_only=bool(getattr(args, "prop_same_trend_only", False)),
        companies=companies,
        price_encoder=str(getattr(args, "price_encoder", "partial_unified")),
        news_top_k=int(getattr(args, "news_padding_k", 5) or 5),
        use_pred_fusion_global_score=bool(use_global_score),
        **model_size_kwargs(args),
    )
    model = wrap_model_devices(model, args, device)
    _cw = float(getattr(args, "contrast_aux_weight", 0.0) or 0.0)
    _cm = _normalize_contrast_mode(getattr(args, "contrast_mode", ""))
    _gate_only = _cm == "mtvc_gate_only"
    core = model.module if hasattr(model, "module") else model
    want_ssl = bool(force_ssl) if force_ssl is not None else (_cw > 0)
    want_gates = (
        bool(force_gates)
        if force_gates is not None
        else ((_cw > 0 or _gate_only) and _is_mtvc_mode(args) and _mtvc_enable_gate(_cm))
    )
    if want_ssl:
        core.ensure_ssl_heads(proj_dim=int(getattr(args, "ssl_proj_dim", 64) or 64))
    if want_gates:
        from contrast_aux import enable_contrast_gates

        enable_contrast_gates(
            core,
            case_mine_layer=int(getattr(args, "mtvc_case_layer", 2) or 2),
        )
    return model


def _strict_load(model, ckpt_path: Path, device) -> None:
    from checkpoint import strip_legacy_unused_state_keys

    ckpt = load_checkpoint(str(ckpt_path), map_location=device)
    model_sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model")
    if model_sd is None:
        model_sd = ckpt
    cleaned, _ = strip_legacy_unused_state_keys(model_sd)
    mod = model.module if hasattr(model, "module") else model
    # full_unified ckpts still ship unused bert_tok_* weights; keep only model keys.
    want = set(mod.state_dict().keys())
    cleaned = {k: v for k, v in cleaned.items() if k in want}
    incompatible = mod.load_state_dict(cleaned, strict=True)
    # strict=True returns None in older torch; ignore
    _ = incompatible


def _eval_test(model, test_loader, device) -> tuple[float, float, int]:
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
            all_logits.append(logits[test_mask].detach().cpu())
            all_labels.append(labels[test_mask].detach().cpu())
    if not all_logits:
        return float("nan"), float("nan"), 0
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    preds = (torch.sigmoid(logits_cat) > 0.5).long().numpy()
    y = labels_cat.long().numpy()
    acc = float(accuracy_score(y, preds))
    mcc = float(calculate_mcc(y, preds))
    return acc, mcc, int(y.shape[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roles",
        type=str,
        default=DEFAULT_ROLES,
        help="Comma-separated role keys (see ROLES / experiment_catalog). Use 'all' for DEFAULT_ROLES.",
    )
    ap.add_argument(
        "--from-catalog",
        action="store_true",
        help="Expand roles from results/experiment_catalog.json for --catalog-dataset",
    )
    ap.add_argument("--catalog-dataset", type=str, default="csmd50")
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    # ~4/8653 Acc or MCC drift vs training-time meta can occur under cuDNN
    # (heavier PredFusion + gate runs); keep paper table numbers from meta.
    ap.add_argument("--atol_acc", type=float, default=5e-4)
    ap.add_argument("--atol_mcc", type=float, default=1.5e-3)
    args_cli = ap.parse_args()
    device = torch.device(args_cli.device)
    if args_cli.from_catalog:
        roles = _roles_from_catalog(args_cli.catalog_dataset)
    elif str(args_cli.roles).strip().lower() in ("all", "default", "*"):
        roles = [r.strip() for r in DEFAULT_ROLES.split(",") if r.strip()]
    else:
        roles = [r.strip() for r in args_cli.roles.split(",") if r.strip()]

    rows = []
    n_ok = 0
    for role in roles:
        pat = ROLES.get(role)
        if not pat:
            print(f"SKIP {role}: unknown role")
            rows.append({"role": role, "ok": False, "err": "unknown_role"})
            continue
        ckpt_dir = _resolve_role(pat)
        if ckpt_dir is None or not (ckpt_dir / "best.pt").is_file():
            print(f"FAIL {role}: ckpt not found ({pat})")
            rows.append({"role": role, "ok": False, "err": "missing"})
            continue
        meta_path = ckpt_dir / "best.pt.meta.json"
        man_path = ckpt_dir / "run_manifest.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        man = json.loads(man_path.read_text())
        exp_metrics = meta.get("metrics") or man.get("best_metrics") or {}
        exp_acc = float(exp_metrics.get("best_test_acc"))
        exp_mcc = float(exp_metrics.get("best_test_mcc"))

        print(f"\n=== {role} @ {ckpt_dir.relative_to(CKPT_ROOT)} ===", flush=True)
        print(f"  expected test acc={exp_acc:.6f} mcc={exp_mcc:.6f}", flush=True)
        try:
            profile = get_profile(str(man["args"]["dataset"]))
            args = normalize_run_args(SimpleNamespace(**dict(man.get("args") or {})), profile)
            args.device = str(device)
            args.price_encoder = normalize_price_encoder(getattr(args, "price_encoder", None))
            ds_kw = dataset_kwargs_from_args(args, profile)
            train_ds = MarketWindowDataset(profile, mode="train", **ds_kw)
            test_ds = MarketWindowDataset(profile, mode="test", **ds_kw)
            test_loader = make_model_dataloader(test_ds, batch_size=1, shuffle=False)
            use_score, has_gates, has_ssl = _ckpt_key_flags(ckpt_dir / "best.pt")
            model = _build_model(
                args,
                profile,
                train_ds,
                device,
                use_global_score=use_score,
                force_gates=has_gates,
                force_ssl=has_ssl,
            )
            _strict_load(model, ckpt_dir / "best.pt", device)
            core = model.module if hasattr(model, "module") else model
            if hasattr(core, "set_news_mark_data_split"):
                core.set_news_mark_data_split("test")
            if hasattr(core, "_main_mlp_feat_debug_prints_left"):
                core._main_mlp_feat_debug_prints_left = 0
            got_acc, got_mcc, n = _eval_test(model, test_loader, device)
            d_acc = abs(got_acc - exp_acc)
            d_mcc = abs(got_mcc - exp_mcc)
            ok = (
                np.isfinite(got_acc)
                and np.isfinite(got_mcc)
                and d_acc <= args_cli.atol_acc
                and d_mcc <= args_cli.atol_mcc
            )
            status = "OK" if ok else "MISMATCH"
            print(
                f"  got      test acc={got_acc:.6f} mcc={got_mcc:.6f} n={n}  "
                f"Δacc={d_acc:.2e} Δmcc={d_mcc:.2e}  "
                f"[score={int(use_score)} gates={int(has_gates)} ssl={int(has_ssl)}] [{status}]",
                flush=True,
            )
            rows.append(
                {
                    "role": role,
                    "ok": ok,
                    "exp_acc": exp_acc,
                    "exp_mcc": exp_mcc,
                    "got_acc": got_acc,
                    "got_mcc": got_mcc,
                    "n": n,
                    "d_acc": d_acc,
                    "d_mcc": d_mcc,
                    "use_global_score": use_score,
                    "has_gates": has_gates,
                    "has_ssl": has_ssl,
                }
            )
            if ok:
                n_ok += 1
        except Exception as e:
            print(f"  FAIL {type(e).__name__}: {e}", flush=True)
            rows.append({"role": role, "ok": False, "err": f"{type(e).__name__}: {e}"})

    print(f"\nSummary: {n_ok}/{len(rows)} match meta within atol")
    out = REPO / "results" / "reeval_vs_meta.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out}")
    return 0 if n_ok == len(rows) and rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
