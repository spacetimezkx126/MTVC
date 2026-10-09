"""Shared checkpoint inference + backtest engine for vol_screen_trend_attn models."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from checkpoint import apply_checkpoint_to_modules, load_checkpoint
from count_kept_news_csmd300_0p5 import (
    _cfg,
    _edge_op_kwargs,
    _resolve_news_sum_init_from_edge_tail,
    _type_add_noise,
    _type_multistep_corrupt,
    _type_score_metric,
)
from data import MarketWindowDataset, get_profile, resolve_vocab_path
from model import EdgeTypeVolatilityOperator, EdgeVolatilityOperator, Model as SimpleModel
from train import dataset_kwargs_from_args, make_model_dataloader

_MODEL_NEWS_MODE = {
    "finbert": "finbert_padding",
    "vocab": "vocab_padding",
    "finbert_embedding": "finbert_embedding",
}


@dataclass
class DayRecord:
    prob_sum: float = 0.0
    count: int = 0
    y_org: float = 0.0
    y_bin: float = 0.0
    strong: bool = False

    @property
    def prob(self) -> float:
        return self.prob_sum / max(1, self.count)


@dataclass
class SampleRecord:
    date: str
    company: str
    prob: float
    y_org: float
    y_bin: float
    strong: bool


@dataclass
class PredictionStore:
    day_table: dict[str, dict[str, DayRecord]] = field(default_factory=lambda: defaultdict(dict))
    series_by_company: dict[str, list[SampleRecord]] = field(default_factory=lambda: defaultdict(list))
    all_probs: list[float] = field(default_factory=list)
    all_labels: list[int] = field(default_factory=list)
    all_strong: list[bool] = field(default_factory=list)


@dataclass
class PortfolioResult:
    name: str
    dates: list[str]
    budget: list[float]
    daily_ret: list[float]
    picks: list[list[str]]
    metrics: dict[str, float]


def resolve_inference_seed(args: SimpleNamespace, explicit: int | None = None) -> int:
    if explicit is not None:
        return int(explicit)
    if getattr(args, "inference_seed", None) is not None:
        return int(args.inference_seed)
    return int(getattr(args, "seed", 42))


def enable_vol_screen_inference_determinism(seed: int | None = None) -> None:
    """Enable reproducible vol_screen inference (edge noise + cuDNN)."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))


def load_vol_screen_checkpoint(ckpt_path: Path | str, device: torch.device):
    ckpt_path = Path(ckpt_path)
    ckpt = load_checkpoint(str(ckpt_path), map_location="cpu")
    raw = dict(ckpt.get("args") or {})
    args = SimpleNamespace(**raw)
    enable_vol_screen_inference_determinism(resolve_inference_seed(args))
    ds_key = str(_cfg(raw, "dataset", "csmd50"))
    profile = get_profile(ds_key)
    l2h_w, h2l_w, l2h_t, h2l_t = _resolve_news_sum_init_from_edge_tail(args)
    vocab_path = resolve_vocab_path(profile.vocab_path, "dict_csmd.pkl", "dict_cn.pkl")
    model = SimpleModel(
        news_encode_mode=_MODEL_NEWS_MODE[str(getattr(args, "news_encode_mode", "vocab"))],
        vocab_path=vocab_path,
        grid_num_stocks=profile.grid_num_stocks,
        price_input_dim=3,
        aux_posterior_news_mode=str(getattr(args, "aux_posterior_news_mode", "hybrid")),
        news_ablation_mode=getattr(args, "news_ablation_mode", "full"),
        pred_fusion_ablation_mode=getattr(args, "pred_fusion_ablation_mode", "full"),
        news_process_mode=getattr(args, "news_process_mode", "default"),
        news_vol_extreme_k=int(getattr(args, "news_vol_extreme_k", 5)),
        add_vol_fusion_mode=str(getattr(args, "add_vol_fusion_mode", "gate_residual")),
        trend_attn_temperature=float(getattr(args, "trend_attn_temperature", 1.0)),
    ).to(device)
    edge_operator = EdgeVolatilityOperator(
        u_dim=512, i_dim=512, hidden_dim=256, noise_scale=float(getattr(args, "edge_noise_scale", 0.3)),
        shared_main_feat=True,
    ).to(device)
    edge_operator_type = EdgeTypeVolatilityOperator(
        u_dim=512,
        i_dim=512,
        hidden_dim=256,
        noise_scale=float(getattr(args, "edge_noise_scale", 0.3)),
        multistep_T=int(getattr(args, "edge_type_multistep_T", 4)),
        rnn_hidden_dim=int(getattr(args, "edge_type_rnn_hidden", 128)),
        noise_bound=str(getattr(args, "edge_type_noise_bound", "sigmoid")),
        gauss_shape_weight=float(getattr(args, "edge_type_gauss_shape_weight", 1.0)),
        gauss_u_last_weight=float(getattr(args, "edge_type_gauss_u_last_weight", 0.25)),
        gauss_u_last_shape_weight=float(getattr(args, "edge_type_gauss_u_last_shape_weight", 1.0)),
        shared_main_feat=True,
    ).to(device)
    apply_checkpoint_to_modules(
        ckpt,
        model=model,
        edge_operator=edge_operator,
        edge_operator_type=edge_operator_type,
        restore_rng=False,
    )
    model.eval()
    edge_operator.eval()
    edge_operator_type.eval()
    return ckpt, args, profile, model, edge_operator, edge_operator_type


def resolve_inference_seed(args: SimpleNamespace, explicit: int | None = None) -> int:
    if explicit is not None:
        return int(explicit)
    if getattr(args, "inference_seed", None) is not None:
        return int(args.inference_seed)
    return int(getattr(args, "seed", 42))


def build_test_loader(args, profile, *, split: str = "test", max_windows: int = 0):
    ds_kw = dataset_kwargs_from_args(args, profile)
    ds = MarketWindowDataset(profile, mode=split, **ds_kw)
    loader = make_model_dataloader(ds, batch_size=1, shuffle=False)
    if max_windows > 0:
        from itertools import islice

        loader = islice(loader, max_windows)
    return ds, loader


class VolScreenPredictor:
    def __init__(
        self,
        model,
        edge_operator,
        edge_operator_type,
        args: SimpleNamespace,
        loader,
        companies: list[str],
        *,
        device: torch.device | None = None,
        strong_only: bool = False,
        inference_seed: int | None = None,
    ):
        self.model = model
        self.edge_operator = edge_operator
        self.edge_operator_type = edge_operator_type
        self.args = args
        self.loader = loader
        self.companies = companies
        self.device = device or torch.device("cpu")
        self.strong_only = bool(strong_only)
        self.inference_seed = resolve_inference_seed(args, inference_seed)
        self._noise_gen = torch.Generator(device="cpu")

        self.vol_screen = str(getattr(args, "news_ablation_mode", "")).lower() in (
            "vol_screen_trend_attn", "vol_screen_shared_pool", "vol_screen_trend_attn_aux",
        )
        self.keep_source = str(getattr(args, "edge_type_keep_source", "window")).lower()
        if self.keep_source == "self":
            self.keep_source = "separate"
        self.score_metric, self.use_multistep = _type_score_metric(vars(args))
        self.skw_w = dict(
            topk_ratio=float(getattr(args, "edge_topk_ratio_window", 1.0)),
            keep_mode=str(getattr(args, "edge_keep_mode_window", "dual")).lower(),
            n_src=None,
            w_off=None,
            noisy_edge=None,
            low_tail_ratio=float(getattr(args, "edge_low_tail_ratio_window", 0.2)),
            high_tail_ratio=float(getattr(args, "edge_high_tail_ratio_window", 0.2)),
        )
        self.skw_t = dict(
            topk_ratio=float(getattr(args, "edge_topk_ratio_type", 1.0)),
            keep_mode=str(getattr(args, "edge_keep_mode_type", "dual")).lower(),
            n_src=None,
            w_off=None,
            noisy_edge=None,
            low_tail_ratio=float(getattr(args, "edge_low_tail_ratio_type", 0.2)),
            high_tail_ratio=float(getattr(args, "edge_high_tail_ratio_type", 0.2)),
            score_metric=self.score_metric,
        )
        self.skw_syn = dict(
            topk_ratio=float(getattr(args, "edge_topk_ratio_synthesis", 1.0)),
            keep_mode=str(getattr(args, "edge_keep_mode_synthesis", "dual")).lower(),
            low_tail_ratio=float(getattr(args, "edge_low_tail_ratio_synthesis", 0.2)),
            high_tail_ratio=float(getattr(args, "edge_high_tail_ratio_synthesis", 0.2)),
        )
        self.edge_screen_group_by = str(getattr(args, "edge_screen_group_by", "") or "").lower()

    def _move_batch(self, batch) -> None:
        for nt in batch.node_types:
            for key, val in batch[nt].__dict__.items():
                if isinstance(val, torch.Tensor):
                    batch[nt][key] = val.to(self.device)

    def _reset_inference_rng(self) -> None:
        self._noise_gen = torch.Generator(device="cpu")
        self._noise_gen.manual_seed(self.inference_seed)
        enable_vol_screen_inference_determinism(self.inference_seed)

    def _infer_keep_mask(self, batch, edge_cache):
        n_src = edge_cache["n_src"]
        p_dst = edge_cache["p_dst"]
        w_off = edge_cache["w_off"]
        n_type = edge_cache.get("n_type")
        u_pair = edge_cache["u_embeds"][p_dst]
        i_pair = edge_cache["i_embeds"][n_src]
        target = torch.ones((u_pair.size(0),), dtype=torch.float32, device=self.device)
        gen = self._noise_gen
        noisy_w = self.edge_operator.add_noise(target, generator=gen)
        logits_w = self.edge_operator(
            u_pair, i_pair, noisy_w, **_edge_op_kwargs(edge_cache, p_dst, n_src, "window", self.vol_screen)
        )
        type_fluct = None
        if self.use_multistep:
            noisy_t, type_fluct = _type_multistep_corrupt(
                self.edge_operator_type,
                u_pair,
                i_pair,
                n_type,
                edge_cache,
                p_dst,
                n_src,
                self.score_metric,
                self.vol_screen,
            )
        else:
            noisy_t = _type_add_noise(self.edge_operator_type, target, generator=gen)
        logits_t = self.edge_operator_type(
            u_pair, i_pair, noisy_t, n_type, **_edge_op_kwargs(edge_cache, p_dst, n_src, "type", self.vol_screen)
        )
        sw = dict(self.skw_w)
        sw["n_src"] = n_src
        sw["w_off"] = w_off
        sw["noisy_edge"] = noisy_w
        st = dict(self.skw_t)
        st["n_src"] = None if self.use_multistep else n_src
        st["w_off"] = None if self.use_multistep else w_off
        st["noisy_edge"] = noisy_t
        if self.use_multistep and type_fluct is not None:
            st["multistep_noise_fluct"] = type_fluct
        _gb = self.edge_screen_group_by
        if _gb in ("p_dst", "price", "prediction_day", "per_price"):
            sw.update({"p_dst": p_dst, "group_by": "p_dst"})
            st.update({"p_dst": p_dst, "group_by": "p_dst"})
        _, freq_w = self.edge_operator.sample_keep_mask(
            logits_w.detach(), return_freq_values=True, **sw
        )
        _, freq_t = self.edge_operator_type.sample_keep_mask(
            logits_t.detach(), return_freq_values=True, **st
        )
        if self.keep_source == "synthesis":
            return EdgeVolatilityOperator.sample_synthesis_keep_mask(freq_w, freq_t, **self.skw_syn)
        if self.keep_source == "window":
            return self.edge_operator.sample_keep_mask(logits_w.detach(), **sw)
        if self.keep_source == "type":
            return self.edge_operator_type.sample_keep_mask(logits_t.detach(), **st)
        return self.edge_operator.sample_keep_mask(logits_w.detach(), **sw)

    def collect(self) -> PredictionStore:
        store = PredictionStore()
        self.model.eval()
        self.edge_operator.eval()
        self.edge_operator_type.eval()

        self._reset_inference_rng()
        with torch.no_grad():
            for wi, batch in enumerate(self.loader):
                if batch["price"].x.numel() == 0 or batch["label"].x.numel() == 0:
                    continue
                self._move_batch(batch)
                # Edge-screen / return_edge_cache path removed from Model.
                logits = self.model(batch)
                if logits.numel() == 0:
                    continue

                price_dates = getattr(batch["price"], "date", None)
                if isinstance(price_dates, (list, tuple)) and len(price_dates) == 1 and isinstance(
                    price_dates[0], (list, tuple)
                ):
                    price_dates = price_dates[0]
                if price_dates is None:
                    continue

                labels = batch["label"].x.view(-1).cpu().numpy()
                label_org = (
                    batch["label"].org.view(-1).cpu().numpy()
                    if hasattr(batch["label"], "org")
                    else labels
                )
                strong = batch["label"].strong_mask.view(-1).cpu().numpy().astype(bool)
                valid = (
                    batch["label"].valid_mask.view(-1).cpu().numpy().astype(bool)
                    if hasattr(batch["label"], "valid_mask")
                    else np.ones_like(strong, dtype=bool)
                )
                comp_ids = batch["price"].company_id.view(-1).cpu().numpy()
                probs = torch.sigmoid(logits.view(-1)).cpu().numpy()

                for i in range(int(probs.shape[0])):
                    if not valid[i]:
                        continue
                    if self.strong_only and not strong[i]:
                        continue
                    if i >= len(price_dates):
                        continue
                    date_str = str(price_dates[i])
                    ci = int(comp_ids[i])
                    comp = self.companies[ci] if 0 <= ci < len(self.companies) else str(ci)
                    rec = store.day_table[date_str].get(comp)
                    if rec is None:
                        rec = DayRecord()
                        store.day_table[date_str][comp] = rec
                    rec.prob_sum += float(probs[i])
                    rec.count += 1
                    rec.y_org = float(label_org[i])
                    rec.y_bin = float(labels[i])
                    rec.strong = bool(strong[i])
                    store.all_probs.append(float(probs[i]))
                    store.all_labels.append(int(labels[i] > 0.5))
                    store.all_strong.append(bool(strong[i]))

                if (wi + 1) % 20 == 0:
                    print(f"[vol_screen] windows={wi + 1} dates={len(store.day_table)}", flush=True)

        for date_str, stocks in store.day_table.items():
            for comp, rec in stocks.items():
                if rec.count <= 0:
                    continue
                store.series_by_company[comp].append(
                    SampleRecord(
                        date=date_str,
                        company=comp,
                        prob=rec.prob,
                        y_org=rec.y_org,
                        y_bin=rec.y_bin,
                        strong=rec.strong,
                    )
                )
        for comp in store.series_by_company:
            store.series_by_company[comp].sort(key=lambda x: x.date)
        return store


def _calc_fees(transaction_vol: float, fees_pct: float) -> float:
    return transaction_vol * fees_pct * 2.0 / 100.0


def _portfolio_metrics(budget: list[float]) -> dict[str, float]:
    from backtest.metrics import calculate_ARR, calculate_Calmar_Ratio, calculate_IR, calculate_MDD, calculate_SR

    arr = float(calculate_ARR(budget))
    mdd = float(calculate_MDD(budget))
    return {
        "arr": arr,
        "sr": float(calculate_SR(budget)),
        "mdd": mdd,
        "cr": float(calculate_Calmar_Ratio(arr, mdd)),
        "ir": float(calculate_IR(budget)),
        "final_budget": float(budget[-1]),
    }


def run_portfolio_strategy(
    store: PredictionStore,
    *,
    name: str,
    mode: str,
    budget: float = 10_000.0,
    fees: float = 0.0,
    topk: int = 5,
    prob_threshold: float = 0.0,
    model_ew_threshold: float = 0.5,
) -> PortfolioResult:
    dates = sorted(store.day_table.keys())
    asset = [float(budget)]
    daily_ret: list[float] = []
    picks: list[list[str]] = []
    out_dates: list[str] = []

    for date_str in dates:
        stocks = store.day_table[date_str]
        items = [
            (comp, rec.prob, rec.y_org)
            for comp, rec in stocks.items()
            if rec.count > 0 and rec.prob >= prob_threshold
        ]
        if not items:
            continue

        if mode == "label_ew":
            day_ret = float(np.mean([x[2] for x in items]))
            day_picks = [x[0] for x in items]
        elif mode == "model_ew":
            sel = [x for x in items if x[1] >= model_ew_threshold]
            day_ret = float(np.mean([x[2] for x in sel])) if sel else 0.0
            day_picks = [x[0] for x in sel]
        elif mode == "model_topk":
            sel = sorted(items, key=lambda x: x[1], reverse=True)[:topk]
            day_ret = float(np.mean([x[2] for x in sel]))
            day_picks = [x[0] for x in sel]
        else:
            raise ValueError(f"unknown portfolio mode: {mode}")

        cur = asset[-1]
        asset.append(cur * (1.0 + day_ret) - _calc_fees(cur, fees))
        daily_ret.append(day_ret)
        picks.append(day_picks)
        out_dates.append(date_str)

    if len(asset) < 2:
        raise RuntimeError(f"portfolio mode={mode}: not enough trading days")

    metrics = _portfolio_metrics(asset)
    metrics["n_trading_days"] = len(out_dates)
    return PortfolioResult(
        name=name,
        dates=out_dates,
        budget=asset[1:],
        daily_ret=daily_ret,
        picks=picks,
        metrics=metrics,
    )


def run_single_stock_backtest(
    store: PredictionStore,
    *,
    budget: float = 10_000.0,
    fees: float = 0.0,
    allow_short: bool = True,
    prob_threshold: float = 0.5,
) -> tuple[list[dict], dict[str, float]]:
    from backtest.metrics import calculate_ACC, calculate_ARR, calculate_MDD, calculate_SR

    per_stock: list[dict] = []
    for comp, series in sorted(store.series_by_company.items()):
        if len(series) < 2:
            continue
        asset = [float(budget)]
        preds: list[int] = []
        labels: list[int] = []
        position = 0
        for rec in series:
            pred_up = int(rec.prob >= prob_threshold)
            preds.append(pred_up)
            labels.append(int(rec.y_bin > 0.5))
            if pred_up and position <= 0:
                position = 1
            elif (not pred_up) and allow_short and position >= 0:
                position = -1
            elif (not pred_up) and (not allow_short) and position > 0:
                position = 0
            day_pnl = position * rec.y_org
            cur = asset[-1]
            asset.append(cur * (1.0 + day_pnl) - _calc_fees(cur, fees))

        per_stock.append(
            {
                "company": comp,
                "acc": float(calculate_ACC(labels, preds)),
                "arr": float(calculate_ARR(asset)),
                "sr": float(calculate_SR(asset)),
                "mdd": float(calculate_MDD(asset)),
                "final_budget": float(asset[-1]),
                "n_days": len(series),
            }
        )

    if not per_stock:
        raise RuntimeError("single_stock: no company series")

    def _mean(key: str) -> float:
        vals = [float(x[key]) for x in per_stock if not np.isnan(float(x[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "mean_acc": _mean("acc"),
        "mean_arr": _mean("arr"),
        "mean_sr": _mean("sr"),
        "mean_mdd": _mean("mdd"),
        "n_stocks": len(per_stock),
    }
    return per_stock, summary


def run_classification_eval(
    store: PredictionStore,
    *,
    prob_threshold: float = 0.5,
    strong_only: bool = False,
) -> dict[str, float]:
    from backtest.metrics import calculate_ACC
    from train import calculate_mcc

    probs = np.asarray(store.all_probs, dtype=np.float64)
    labels = np.asarray(store.all_labels, dtype=np.int64)
    strong = np.asarray(store.all_strong, dtype=bool)
    if strong_only:
        mask = strong
        probs = probs[mask]
        labels = labels[mask]
    if probs.size == 0:
        raise RuntimeError("classification: no samples")
    preds = (probs >= prob_threshold).astype(np.int64)
    return {
        "acc": float(calculate_ACC(labels.tolist(), preds.tolist())),
        "mcc": float(calculate_mcc(labels.tolist(), preds.tolist())),
        "n_samples": int(probs.size),
        "strong_only": int(strong_only),
    }
