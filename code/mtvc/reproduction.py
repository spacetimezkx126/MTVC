"""Public recipe loading and explicit historical state-dictionary translation."""

from __future__ import annotations
import json
import os
import pickle
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
DATA_NAMES = {"csmd50": "CSMD50", "csmd300": "CSMD300", "massive": "massive_data"}


def configure_environment():
    os.chdir(ROOT)
    os.environ["DUAL_TF_MODEL_DIR"] = str(ROOT)
    for k, v in {
        "USE_TYPE_EMB": "1",
        "UNIFIED_CAUSAL": "1",
        "UNIFIED_SAME_DAY": "0",
        "TRANSFORMER_LAYERS": "6",
        "SOFT_PRUNE": "0",
        "SHARE_PRICE_NEWS_ENC": "0",
    }.items():
        os.environ["MTVC_" + k] = v
    for k in (
        "MTVC_FIXED_TRAIN_ORDER",
        "MTVC_UNIFIED_LAYERS",
        "MTVC_HARD_FINETUNE",
        "MTVC_LOOP_ROUNDS",
    ):
        os.environ.pop(k, None)
    from tool.cuda_guard import harden_cuda_for_multiproc

    harden_cuda_for_multiproc()


def recipe_args(recipe, device, data_root=None):
    from main import build_parser
    from data import get_profile
    from train import normalize_run_args

    args = build_parser().parse_args(["--dataset", recipe["dataset"]])
    defaults = json.loads((ROOT / "configs/paper_recipes.json").read_text()).get("defaults", {})
    for k, v in {**defaults, **recipe["args"]}.items():
        if hasattr(args, k):
            setattr(args, k, v)
    args.dataset = recipe["dataset"]
    args.dataset_root = str(
        Path(data_root).resolve() if data_root else ROOT / "data" / DATA_NAMES[args.dataset]
    )
    vocab = "dict_massive.pkl" if args.dataset == "massive" else "dict_csmd.pkl"
    args.vocab_path = str(ROOT / "dict" / vocab)
    args.device = device
    args.devices = ""
    args.resume_ckpt = ""
    args.ckpt = ""
    args.seed = recipe["seed"]
    if args.train_oversample_extra:
        args.train_oversample_extra = str(
            ROOT / "assets/massive_oversample/train_oversample_extra.jsonl"
        )
    profile = get_profile(args.dataset)
    return (normalize_run_args(args, profile), profile)


def build_model(args, n_stocks):
    from model import Model
    from train import model_size_kwargs, model_finbert_kwargs
    from data import get_profile
    from contrast_aux import enable_contrast_gates

    model = Model(
        news_encode_mode="vocab_padding",
        news_window_agg_mode=args.news_window_agg_mode,
        vocab_path=args.vocab_path,
        grid_num_stocks=n_stocks,
        price_input_dim=3,
        no_price_token=args.no_price_token,
        no_price_token_pad=args.no_price_token_pad,
        no_news_token=args.no_news_token,
        no_news_token_pad=args.no_news_token_pad,
        modality_ablation=args.modality_ablation,
        industry_no_mean=args.industry_no_mean,
        num_score_news_types=args.num_score_news_types,
        news_ablation_mode=args.news_ablation_mode,
        main_mlp_ablation_mode=args.main_mlp_ablation_mode,
        industry_fuse_site=args.industry_fuse_site,
        vin_ablation_mode=args.vin_ablation_mode,
        news_word_emb=args.news_word_emb,
        news_process_mode=args.news_process_mode,
        price_encoder=args.price_encoder,
        vin_connect_mode=args.vin_connect_mode,
        news_top_k=args.news_padding_k,
        **model_finbert_kwargs(args, get_profile(args.dataset)),
        **model_size_kwargs(args),
    )
    if args.contrast_aux_weight > 0:
        model.ensure_contrast_heads(proj_dim=args.ssl_proj_dim)
        if args.contrast_mode == "mtvc":
            enable_contrast_gates(model, case_mine_layer=args.mtvc_case_layer)
    return model


class NumpyArchiveUnpickler(pickle.Unpickler):
    """Read trusted NumPy 2 archives in the existing NumPy 1 environment."""

    def find_class(self, module, name):
        import numpy

        if module.startswith("numpy._core") and int(numpy.__version__.split(".")[0]) < 2:
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_archive_weights(model, path):
    import torch

    compat = types.ModuleType("mtvc_numpy_archive_pickle")
    compat.Unpickler = NumpyArchiveUnpickler
    compat.load = pickle.load
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, pickle_module=compat)
    rename = {
        "news_transformer_encoder": "bert_tok_transformer",
        "ssl_proj_p": "contrast_proj_p",
        "ssl_proj_n": "contrast_proj_n",
        "ssl_recon_n2p": "contrast_state_proj",
        "ssl_recon_mlm": "contrast_pair_proj",
        "time_fuse_mlp": "pred_fusion_day_mix",
        "time_score": "pred_fusion_day_score",
    }
    retired = {
        "reduce_dim",
        "attention_mlp",
        "attn_mlp",
        "attn_score",
        "competition_scorer",
        "edge_feedback_logit_scale",
        "edge_feedback_mlp",
        "edge_feedback_mlp_high",
        "edge_feedback_mlp_low",
        "mlp_news_global",
        "mlp_virtual_industry",
        "mlp_virtual_trend",
        "mlp_virtual_trend_type",
        "mlp_virtual_trend_window",
        "news_day_attn_norm",
        "news_day_weight_mlp",
        "news_gate_proj_reduce",
        "news_global_parameter",
        "news_main_feat_proj",
        "news_type_msg_mlp",
        "sougou_embedding",
        "stock_cross_attn",
        "stock_cross_norm",
        "trend_competition_scorer",
        "type_score_mlp",
        "virtual_trend_emb",
        "vnt_fuse_mlp",
        "vnw_fuse_mlp",
        "window_msg_mlp",
        "window_score_mlp",
    }
    current = model.state_dict()
    (state, ignored) = ({}, [])
    for name, value in checkpoint["model_state_dict"].items():
        (prefix, dot, suffix) = name.partition(".")
        if name not in current and prefix in retired:
            ignored.append(name)
            continue
        target = rename.get(prefix, prefix) + dot + suffix
        if target in state:
            raise ValueError(f"Duplicate translated key: {target}")
        state[target] = value
    for name, value in list(state.items()):
        if name.startswith("unitrans_price_day_mlp."):
            alias = name.replace("unitrans_price_day_mlp.", "unitrans.price_day_mlp.", 1)
            if alias in current:
                if alias in state and (not torch.equal(state[alias], value)):
                    raise ValueError(f"Conflicting price MLP aliases: {name}")
                state[alias] = value
    incompatible = model.load_state_dict(state, strict=False)
    parameters = dict(model.named_parameters())
    missing_active = [
        name
        for name in incompatible.missing_keys
        if name not in parameters or parameters[name].requires_grad
    ]
    if missing_active or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Archive mapping incomplete: missing={missing_active}, unexpected={incompatible.unexpected_keys}"
        )
    return {"missing_frozen": incompatible.missing_keys, "ignored_obsolete": ignored}
