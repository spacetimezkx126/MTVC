#!/usr/bin/env python3
"""Strict-load smoke: build Model from run_manifest args and load best.pt strict=True."""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

def _stub_missing(mod_name: str, **attrs):
    if mod_name in sys.modules:
        return
    try:
        __import__(mod_name)
    except ModuleNotFoundError:
        m = types.ModuleType(mod_name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[mod_name] = m


# Base miniconda may lack training extras; paper smoke only needs Model construction.
class _TokStub:
    def __init__(self, *a, **k):
        pass

    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()

    def __call__(self, *a, **k):
        return {
            "input_ids": torch.zeros(1, 8, dtype=torch.long),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
        }

    def __len__(self):
        return 1000

    pad_token_id = 0


_stub_missing(
    "transformers",
    AutoModel=_TokStub,
    AutoTokenizer=_TokStub,
    BertTokenizer=_TokStub,
)
_stub_missing("sklearn")
_stub_missing(
    "sklearn.metrics",
    accuracy_score=lambda *a, **k: 0.0,
    classification_report=lambda *a, **k: "",
    confusion_matrix=lambda *a, **k: None,
)

REPO = Path(__file__).resolve().parents[1]
MTVC = Path(__file__).resolve().parent / "mtvc"
CKPT_ROOT = REPO / "checkpoints"

os.chdir(str(REPO))
_dual_tf = Path("/home/zhaokx/Pattern/Pattern_Mining/dual_tf")
os.environ["DUAL_TF_MODEL_DIR"] = str(_dual_tf if _dual_tf.is_dir() else MTVC)
os.environ["MTVC_USE_TYPE_EMB"] = "1"
os.environ["MTVC_UNIFIED_CAUSAL"] = "1"
os.environ["MTVC_UNIFIED_SAME_DAY"] = "0"
os.environ["MTVC_SOFT_PRUNE"] = "0"
os.environ["MTVC_SHARE_PRICE_NEWS_ENC"] = "0"
os.environ["MTVC_TRANSFORMER_LAYERS"] = "6"
os.environ.pop("MTVC_UNIFIED_LAYERS", None)
os.environ.pop("MTVC_LOOP_ROUNDS", None)

sys.path.insert(0, str(MTVC))

from data import get_profile, resolve_vocab_path  # noqa: E402
from train import (  # noqa: E402
    _builder_dataset,
    _is_mtvc_mode,
    _mtvc_enable_gate,
    _normalize_contrast_mode,
    model_size_kwargs,
    normalize_run_args,
    wrap_model_devices,
)
from model import Model  # noqa: E402
from checkpoint import load_checkpoint  # noqa: E402


def _resolve_role(glob_pat: str) -> Path | None:
    hits = sorted(CKPT_ROOT.glob(glob_pat))
    if not hits:
        return None
    # prefer s42
    for h in hits:
        if h.name == "s42" or "/s42" in str(h):
            return h if h.is_dir() else h.parent
    return hits[0] if hits[0].is_dir() else hits[0].parent


ROLES = [
    ("full", "csmd50/mtvc_full_*/s42"),
    ("crossattn", "csmd50/mtvc_crossattn_*/s42"),
    ("fullunified", "csmd50/mtvc_fullunified_*/s42"),
    ("price_only", "csmd50/mtvc_price_only_*/s42"),
    ("nonews", "csmd50/mtvc_price_only_*/s42"),
    ("naive_fusion", "csmd50/mtvc_naive_fusion_*/s42"),
    ("noprice", "csmd50/mtvc_noprice_*/s42"),
    ("noglobal", "csmd50/mtvc_noglobal_*/s42"),
    ("noindustry", "csmd50/mtvc_noindustry_*/s42"),
    ("gate_only", "csmd50/mtvc_gate_only_*/s42"),
]


def _ckpt_has_global_score(ckpt_path: Path) -> bool:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k in ("model", "state_dict", "model_state_dict"):
        if isinstance(sd, dict) and k in sd:
            sd = sd[k]
            break
    return any("global_score_proj" in str(k) for k in sd)


def _build_model(args, profile, device, *, use_global_score: bool):
    # Prefer companies from train dataset; fall back to grid size placeholder names.
    companies: list[str] = []
    try:
        from data import MarketWindowDataset
        from train import dataset_kwargs_from_args

        ds_kw = dataset_kwargs_from_args(args, profile)
        train_ds = MarketWindowDataset(profile, mode="train", **ds_kw)
        companies = list(getattr(_builder_dataset(train_ds), "companies", []) or [])
    except Exception as e:
        print(f"  [warn] dataset build skipped ({type(e).__name__}: {e})")
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
    if _cw > 0:
        core.ensure_ssl_heads(proj_dim=int(getattr(args, "ssl_proj_dim", 64) or 64))
    if (_cw > 0 or _gate_only) and _is_mtvc_mode(args) and _mtvc_enable_gate(_cm):
        from contrast_aux import enable_contrast_gates

        enable_contrast_gates(core, case_mine_layer=int(getattr(args, "mtvc_case_layer", 2) or 2))
    return model


def _strict_load(model, ckpt_path: Path, device) -> None:
    from checkpoint import strip_legacy_unused_state_keys

    ckpt = load_checkpoint(str(ckpt_path), map_location=device)
    model_sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model")
    if model_sd is None:
        model_sd = ckpt
    cleaned, _ = strip_legacy_unused_state_keys(model_sd)
    mod = model.module if hasattr(model, "module") else model
    want = set(mod.state_dict().keys())
    cleaned = {k: v for k, v in cleaned.items() if k in want}
    mod.load_state_dict(cleaned, strict=True)


def main() -> int:
    device = torch.device("cpu")
    results = []
    for role, pat in ROLES:
        ckpt_dir = _resolve_role(pat)
        if ckpt_dir is None or not (ckpt_dir / "best.pt").exists():
            print(f"FAIL  {role}: checkpoint not found ({pat})")
            results.append(False)
            continue
        try:
            man = json.loads((ckpt_dir / "run_manifest.json").read_text())
            profile = get_profile(str(man["args"]["dataset"]))
            args = normalize_run_args(SimpleNamespace(**dict(man.get("args") or {})), profile)
            args.device = "cpu"
            best = ckpt_dir / "best.pt"
            use_score = _ckpt_has_global_score(best)
            print(
                f"[{role}] {ckpt_dir.relative_to(CKPT_ROOT)} "
                f"pe={args.price_encoder} cm={args.contrast_mode} "
                f"cw={getattr(args, 'contrast_aux_weight', None)} "
                f"global_score={int(use_score)}"
            )
            model = _build_model(args, profile, device, use_global_score=use_score)
            _strict_load(model, best, device)
            print(f"OK    {role}")
            results.append(True)
        except Exception as e:
            print(f"FAIL  {role}: {type(e).__name__}: {e}")
            results.append(False)

    n_ok = sum(1 for x in results if x)
    print(f"\nSummary: {n_ok}/{len(results)} OK")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
