#!/usr/bin/env python3
"""
LightQuant price-only baselines (LSTM / ALSTM / BiLSTM / Adv-ALSTM / DTML)
on CSMD50/300 & CMIN-CN/US.

Reuses four-dataset date splits from main_stocknet. Sample valid/strong scope
matches beifen graph Dataset (inlined below, no beifen_0602).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from main_stocknet import (
    DATASET_SPLITS,
    MV_NEUTRAL_LOW,
    MV_STRONG_HIGH,
    binary_mv_label,
    calculate_mcc,
    detect_dataset_kind,
    is_strong_mv,
    load_all_data,
    set_seed,
    split_range_for_mode,
)
from pen1_data import (
    ROLE_SPLIT_MODES,
    date_in_mode,
    pack_valid_dates_flat,
    quarter_rr_311_defer_months,
)
from model_single import SINGLE_MODEL_CHOICES, build_single_model
from oversample_utils import (
    expand_tuple_samples,
    extra_counts_from_list,
    load_oversample_extra_list,
    summarize_oversample_counts,
)

LOGGER = logging.getLogger("single")


def setup_logger(log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("single")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info("log file: %s", log_file)
    return logger


def log_info(msg: str, *args) -> None:
    LOGGER.info(msg, *args)


def _split_range(mode: str, kind: str) -> tuple[str, str]:
    return split_range_for_mode(mode, kind)


def _global_trading_dates(stock_data: dict) -> list[str]:
    all_dates: set[str] = set()
    for day_dict in stock_data.values():
        all_dates.update(day_dict.keys())
    return sorted(all_dates)


def _extract_graph_aligned_samples(
    stock_data: dict,
    label: dict,
    kind: str,
    mode: str,
    price_window: int = 5,
    seg_length: int = 10,
    split_mode: str = "contiguous",
) -> tuple[list[tuple[str, str, float]], set[tuple[str, str]], set[tuple[str, str]]]:
    """
    Same (company, target_date) set as beifen graph valid/strong price nodes:
      global calendar + seg_length segments, inclusive split bounds (start <= d <= end)
      valid: pos >= price_window, label parseable
      strong: mv > MV_STRONG_HIGH or mv < MV_NEUTRAL_LOW
    """
    if kind not in DATASET_SPLITS:
        raise ValueError(f"unsupported kind {kind!r}")

    start, end = _split_range(mode, kind)
    sm = str(split_mode or "contiguous").strip().lower()
    if sm in ROLE_SPLIT_MODES:
        valid_dates = [
            d for d in _global_trading_dates(stock_data) if date_in_mode(d, mode, kind, sm)
        ]
        defer = quarter_rr_311_defer_months(kind) if sm == "quarter_rr_311" else None
        # Drop incomplete last flat chunk (legacy); July deferred for qrr311.
        valid_dates = pack_valid_dates_flat(
            valid_dates, seg_length, keep_tail=False, defer_months=defer
        )
    else:
        valid_dates = [d for d in _global_trading_dates(stock_data) if start <= d <= end]
        n_seg = len(valid_dates) // seg_length
        valid_dates = valid_dates[: n_seg * seg_length]
    if not valid_dates:
        return [], set(), set()

    companies = sorted(stock_data.keys())
    comp_dates: dict[str, list[str]] = {}
    comp_pos: dict[str, dict[str, int]] = {}
    for comp in companies:
        cds = sorted(stock_data.get(comp, {}).keys())
        comp_dates[comp] = cds
        comp_pos[comp] = {d: i for i, d in enumerate(cds)}

    samples: list[tuple[str, str, float]] = []
    valid_pairs: set[tuple[str, str]] = set()
    strong_pairs: set[tuple[str, str]] = set()
    seen: set[tuple[str, str]] = set()

    for target_date in valid_dates:
        for comp in companies:
            pos_map = comp_pos.get(comp, {})
            if target_date not in pos_map:
                continue
            if pos_map[target_date] < price_window:
                continue
            raw_label = label.get(comp, {}).get(target_date)
            try:
                y = float(raw_label) if raw_label is not None else None
            except (TypeError, ValueError):
                y = None
            if y is None:
                continue
            key = (comp, target_date)
            if key in seen:
                continue
            seen.add(key)
            valid_pairs.add(key)
            if is_strong_mv(y):
                strong_pairs.add(key)
            samples.append((comp, target_date, float(y)))

    samples.sort(key=lambda x: (x[1], x[0]))
    return samples, valid_pairs, strong_pairs


def _price_feat_dim(stock_data: dict) -> int:
    for day_dict in stock_data.values():
        for v in day_dict.values():
            return len(v)
    return 3


def _fit_stock_scalers(stock_data: dict, train_start: str, train_end: str, feat_dim: int | None = None) -> dict:
    if feat_dim is None:
        feat_dim = _price_feat_dim(stock_data)
    scalers = {}
    for comp, day_dict in stock_data.items():
        rows = [
            day_dict[d][:feat_dim]
            for d in sorted(day_dict.keys())
            if train_start <= d <= train_end
        ]
        if len(rows) >= 2:
            scalers[comp] = StandardScaler().fit(np.asarray(rows, dtype=np.float32))
    return scalers


def _transform_hlc(scalers: dict, company: str, window: np.ndarray, use_scaler: bool) -> np.ndarray:
    if use_scaler and company in scalers:
        return scalers[company].transform(window).astype(np.float32)
    return window.astype(np.float32)


def _window_before_target(
    stock_data: dict,
    company: str,
    target_date: str,
    look_back: int,
    split_start: str | None = None,
    feat_dim: int | None = None,
) -> np.ndarray | None:
    """Price window ending at prev(target_date). HLC or HLCV.

    split_start is set: only dates in [split_start, target_date] (LightQuant).
    split_start is None: full stock calendar (aligned with beifen CSMD graphs).
    """
    day_dict = stock_data.get(company, {})
    all_dates = sorted(day_dict.keys())
    if target_date not in all_dates:
        return None
    if split_start is not None:
        dates = [d for d in all_dates if split_start <= d <= target_date]
    else:
        dates = all_dates
    if target_date not in dates:
        return None
    i = dates.index(target_date)
    if i < look_back:
        return None
    if feat_dim is None:
        feat_dim = _price_feat_dim(stock_data)
    return np.asarray(
        [day_dict[dates[j]][:feat_dim] for j in range(i - look_back, i)],
        dtype=np.float32,
    )


def _fix_window(window: np.ndarray, look_back: int, feat_dim: int | None = None) -> np.ndarray:
    if feat_dim is None:
        feat_dim = int(window.shape[1]) if window.ndim == 2 and window.size else 3
    if window.shape[0] >= look_back:
        return window[-look_back:].astype(np.float32)
    pad = np.zeros((look_back - window.shape[0], feat_dim), dtype=np.float32)
    return np.vstack([pad, window.astype(np.float32)])


def _training_mask(batch: dict, device: torch.device) -> torch.Tensor:
    mv = batch["mv"].to(device).reshape(-1)
    window_valid = batch["window_valid"].to(device).reshape(-1).bool()
    if "strong" in batch:
        strong = batch["strong"].to(device).reshape(-1).bool()
    else:
        strong = (mv > MV_STRONG_HIGH) | (mv < MV_NEUTRAL_LOW)
    return strong & window_valid


class PriceWindowDataset(TorchDataset):
    """LightQuant Normal_Dataset on preprocessed price txt + shared date splits."""

    def __init__(
        self,
        samples: list[tuple[str, str, float]],
        stock_data: dict,
        scalers: dict,
        look_back: int,
        use_scaler: bool,
        split_start: str | None,
        strong_pairs: set[tuple[str, str]] | None = None,
    ):
        self.samples = samples
        self.stock_data = stock_data
        self.scalers = scalers
        self.look_back = look_back
        self.use_scaler = use_scaler
        self.split_start = split_start
        self.strong_pairs = strong_pairs
        self.feat_dim = _price_feat_dim(stock_data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        comp, target_date, mv = self.samples[idx]
        is_strong = (
            (comp, target_date) in self.strong_pairs
            if self.strong_pairs is not None
            else is_strong_mv(mv)
        )
        window = _window_before_target(
            self.stock_data, comp, target_date, self.look_back,
            split_start=self.split_start,
            feat_dim=self.feat_dim,
        )
        if window is None:
            return (
                torch.zeros(self.look_back, self.feat_dim),
                torch.tensor(binary_mv_label(mv), dtype=torch.float32),
                torch.tensor(mv, dtype=torch.float32),
                torch.tensor(False),
                torch.tensor(is_strong),
            )
        window = _fix_window(window, self.look_back, feat_dim=self.feat_dim)
        window = _transform_hlc(self.scalers, comp, window, self.use_scaler)
        return (
            torch.from_numpy(window),
            torch.tensor(binary_mv_label(mv), dtype=torch.float32),
            torch.tensor(mv, dtype=torch.float32),
            torch.tensor(True),
            torch.tensor(is_strong),
        )


def _parse_mv_value(label_data: dict, company: str, target_date: str) -> float | None:
    raw = label_data.get(company, {}).get(target_date)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _build_dtml_date_groups(
    samples: list[tuple[str, str, float]],
    stock_data: dict,
    label_data: dict,
    scalers: dict,
    look_back: int,
    use_scaler: bool,
    split_start: str | None,
    start: str,
    end: str,
    valid_pairs: set[tuple[str, str]] | None = None,
    strong_pairs: set[tuple[str, str]] | None = None,
    extra_by_date: dict[str, list[str]] | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], int, int]:
    """One sample per trading day: ALL stocks in universe (fixed order).

    Calendar = unique target_date from graph-valid samples,
    NOT news_texts / PEN news-aligned subset.
    Train extras are appended as duplicated stock columns (same loss).
    """
    companies = sorted(stock_data.keys())
    comp_to_idx = {c: i for i, c in enumerate(companies)}
    feat_dim = _price_feat_dim(stock_data)
    target_dates = sorted({target_date for _, target_date, _ in samples})

    groups: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    n_hit = n_miss = 0
    for target_date in target_dates:
        windows, labels, mvs, valids, strongs = [], [], [], [], []
        for comp in companies:
            mv = _parse_mv_value(label_data, comp, target_date)
            in_scope = valid_pairs is None or (comp, target_date) in valid_pairs
            is_strong = (
                (comp, target_date) in strong_pairs
                if strong_pairs is not None
                else (is_strong_mv(mv) if mv is not None else False)
            )
            window = _window_before_target(
                stock_data, comp, target_date, look_back, split_start=split_start,
                feat_dim=feat_dim,
            )
            if not in_scope or window is None or mv is None:
                window = np.zeros((look_back, feat_dim), dtype=np.float32)
                valid = False
                mv = 0.0 if mv is None else mv
                is_strong = False
            else:
                window = _fix_window(window, look_back, feat_dim=feat_dim)
                window = _transform_hlc(scalers, comp, window, use_scaler)
                valid = True
            windows.append(window)
            labels.append(binary_mv_label(mv) if mv is not None else 0.0)
            mvs.append(float(mv) if mv is not None else 0.0)
            valids.append(valid)
            strongs.append(is_strong)
        if extra_by_date:
            for extra_comp in extra_by_date.get(target_date, []):
                i = comp_to_idx.get(extra_comp)
                if i is None or not valids[i]:
                    n_miss += 1
                    continue
                windows.append(windows[i])
                labels.append(labels[i])
                mvs.append(mvs[i])
                valids.append(valids[i])
                strongs.append(strongs[i])
                n_hit += 1
        if not any(valids):
            continue
        feats = np.stack(windows, axis=1)
        groups.append((
            feats.astype(np.float32),
            np.asarray(labels, dtype=np.float32),
            np.asarray(mvs, dtype=np.float32),
            np.asarray(valids, dtype=bool),
            np.asarray(strongs, dtype=bool),
        ))
    return groups, n_hit, n_miss


class DTMLPriceDataset(TorchDataset):
    """DTML: each sample is all stocks on one target_date (cross-section)."""

    def __init__(
        self,
        date_groups: list[tuple[np.ndarray, ...]],
    ):
        self.date_groups = date_groups

    def __len__(self):
        return len(self.date_groups)

    def __getitem__(self, idx):
        feats, labels, mvs, valid, strong = self.date_groups[idx]
        return (
            torch.from_numpy(feats),
            torch.from_numpy(labels),
            torch.from_numpy(mvs),
            torch.from_numpy(valid),
            torch.from_numpy(strong),
        )


def build_price_datasets(
    dataset_root: str,
    mode: str,
    look_back: int,
    filter_neutral: bool,
    use_strong_only: bool,
    seed: int,
    model_name: str,
    scalers: dict | None = None,
    bundle: dict | None = None,
    split_mode: str = "contiguous",
    oversample_extra_path: str = "",
):
    del seed
    del use_strong_only
    kind = detect_dataset_kind(dataset_root)
    if bundle is None:
        bundle = load_all_data(dataset_root)
    stock_data = bundle["stock_data"]
    label_data = bundle["label"]
    # All five models share raw HLC from price txt (already normalized in dataset).
    use_scaler = False
    start, end = _split_range(mode, kind)
    if scalers is None:
        scalers = {}
    valid_pairs: set[tuple[str, str]] | None = None
    strong_pairs: set[tuple[str, str]] | None = None
    window_split_start: str | None = start
    samples, valid_pairs, strong_pairs = _extract_graph_aligned_samples(
        stock_data,
        label_data,
        kind,
        mode,
        price_window=look_back,
        split_mode=split_mode,
    )
    if filter_neutral:
        samples = [(c, d, mv) for c, d, mv in samples if (c, d) in strong_pairs]
        valid_pairs = {(c, d) for c, d, _ in samples}
    window_split_start = None

    extra_by_date: dict[str, list[str]] | None = None
    extra_path = str(oversample_extra_path or "").strip()
    if extra_path and mode == "train":
        extras = load_oversample_extra_list(extra_path)
        os_sum = summarize_oversample_counts(extra_counts_from_list(extras))
        n_before = len(samples)
        samples, n_hit, n_miss = expand_tuple_samples(samples, extras)
        extra_by_date = {}
        for comp, d in extras:
            extra_by_date.setdefault(d, []).append(comp)
        log_info(
            "train oversample extra=%s unique=%d copies=%d hit=%d miss=%d n %d->%d "
            "(val/test unchanged)",
            extra_path,
            os_sum["n_unique_keys"],
            os_sum["n_extra_copies"],
            n_hit,
            n_miss,
            n_before,
            len(samples),
        )

    if model_name == "dtml":
        date_groups, dtml_hit, dtml_miss = _build_dtml_date_groups(
            samples, stock_data, label_data, scalers, look_back, use_scaler,
            split_start=window_split_start, start=start, end=end,
            valid_pairs=valid_pairs, strong_pairs=strong_pairs,
            extra_by_date=extra_by_date if mode == "train" else None,
        )
        if extra_by_date:
            log_info(
                "dtml train oversample columns duplicated hit=%d miss=%d "
                "(same loss; extra stocks appended in the date cross-section)",
                dtml_hit, dtml_miss,
            )
        ds = DTMLPriceDataset(date_groups)
        avg_n_stocks = int(np.mean([g[0].shape[1] for g in date_groups])) if date_groups else 0
        max_n_stocks = int(max((g[0].shape[1] for g in date_groups), default=0))
    else:
        ds = PriceWindowDataset(
            samples, stock_data, scalers, look_back, use_scaler,
            split_start=window_split_start, strong_pairs=strong_pairs,
        )
        avg_n_stocks = 0
        max_n_stocks = 0
    pos_rate, _ = _split_pos_rate(samples, strong_pairs=strong_pairs)
    return ds, scalers, len(ds), pos_rate, avg_n_stocks, max_n_stocks


def _dtml_scope_stats(
    date_groups: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[int, int, int]:
    """Return (window_valid stock-days, strong stock-days, strong&window train scope)."""
    n_window = n_strong = n_scope = 0
    for group in date_groups:
        valid, strong = group[3], group[4]
        w_f = valid.reshape(-1)
        s_f = strong.reshape(-1)
        n_window += int(w_f.sum())
        n_strong += int(s_f.sum())
        n_scope += int((w_f & s_f).sum())
    return n_window, n_strong, n_scope


def collate_single_batch(batch, model_name: str, dtml_max_stocks: int = 0):
    if model_name == "dtml":
        seq_len = batch[0][0].shape[0]
        max_n = dtml_max_stocks or max(item[0].shape[1] for item in batch)
        feat_dim = int(batch[0][0].shape[-1]) if batch else 3
        price = torch.zeros(len(batch), seq_len, max_n, feat_dim)
        y = torch.zeros(len(batch), max_n)
        mv = torch.zeros(len(batch), max_n)
        window_valid = torch.zeros(len(batch), max_n, dtype=torch.bool)
        strong = torch.zeros(len(batch), max_n, dtype=torch.bool)
        for i, item in enumerate(batch):
            n = item[0].shape[1]
            price[i, :, :n, :] = item[0]
            y[i, :n] = item[1]
            mv[i, :n] = item[2]
            window_valid[i, :n] = item[3]
            strong[i, :n] = item[4]
        return {"price": price, "y": y, "mv": mv, "window_valid": window_valid, "strong": strong}

    price = torch.stack([item[0] for item in batch], dim=0)
    y = torch.stack([item[1] for item in batch], dim=0)
    mv = torch.stack([item[2] for item in batch], dim=0)
    window_valid = torch.stack([item[3] for item in batch], dim=0)
    strong = torch.stack([item[4] for item in batch], dim=0)
    return {"price": price, "y": y, "mv": mv, "window_valid": window_valid, "strong": strong}


def _load_ckpt(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _needs_bn_batch(model_name: str) -> bool:
    """LightQuant single_exp/train.py: BN models skip batch_size<=1 in training."""
    return model_name in ("lstm", "alstm", "bi_lstm")


def _flatten_labels(labels: torch.Tensor) -> torch.Tensor:
    return labels.reshape(-1) if labels.dim() > 1 else labels.squeeze()


def _run_model(
    model,
    model_name: str,
    price: torch.Tensor,
    window_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    if model_name == "adv_alstm":
        pred = model.predict(price)
    elif model_name == "dtml":
        pred = model.predict(price, valid_mask=window_valid)
    else:
        pred = model(price)
    if model_name == "dtml" and pred.dim() > 1:
        return pred.reshape(-1)
    return pred.squeeze(-1)


def _prepare_batch(batch: dict, device: torch.device):
    price = batch["price"].to(device)
    labels = _flatten_labels(batch["y"].float().to(device)).reshape(-1)
    mask = _training_mask(batch, device).reshape(-1)
    window_valid = batch["window_valid"].to(device)
    return price, labels, mask, window_valid


def _split_pos_rate(
    samples: list[tuple[str, str, float]],
    strong_pairs: set[tuple[str, str]] | None = None,
) -> tuple[float, int]:
    pos, tot = 0, 0
    for comp, target_date, mv in samples:
        if strong_pairs is not None:
            if (comp, target_date) not in strong_pairs:
                continue
        elif not is_strong_mv(mv):
            continue
        tot += 1
        if mv > 0.0:
            pos += 1
    return pos / max(tot, 1), tot


def _majority_acc(pos_rate: float) -> float:
    return max(pos_rate, 1.0 - pos_rate)


def _weighted_bce(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: float | None,
) -> torch.Tensor:
    if pos_weight is None or pos_weight <= 0:
        return nn.functional.binary_cross_entropy(prediction, labels)
    weight = torch.where(labels >= 0.5, torch.full_like(labels, pos_weight), torch.ones_like(labels))
    return nn.functional.binary_cross_entropy(prediction, labels, weight=weight)


def _compute_loss(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    model_name: str,
    pos_weight: float | None,
) -> torch.Tensor:
    del model_name
    return _weighted_bce(prediction, labels, pos_weight)


def _forward_on_scope(
    model,
    model_name: str,
    price: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    window_valid: torch.Tensor | None = None,
):
    prediction = _run_model(
        model, model_name, price, window_valid=window_valid,
    ).reshape(-1)
    return prediction[mask], labels[mask]


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    model_name: str,
    pos_weight: float | None = None,
    split_name: str = "",
):
    model.eval()
    all_true, all_pred = [], []
    total_loss = 0.0

    for batch in loader:
        price, labels, mask, window_valid = _prepare_batch(batch, device)
        if not mask.any():
            continue
        prediction, labels = _forward_on_scope(
            model, model_name, price, labels, mask, window_valid=window_valid,
        )
        loss = _compute_loss(prediction, labels, model_name, pos_weight)
        total_loss += float(loss.item())
        pred_cls = (prediction > 0.5).float()
        pc = pred_cls.cpu().tolist()
        tc = labels.cpu().tolist()
        if isinstance(pc, float):
            pc, tc = [pc], [tc]
        all_pred.extend(pc)
        all_true.extend(tc)

    acc = accuracy_score(all_true, all_pred) if all_true else 0.0
    mcc = calculate_mcc(all_true, all_pred) if all_true else 0.0
    avg_loss = total_loss / max(len(loader), 1)
    if split_name:
        log_info("[%s] loss=%.6f acc=%.4f mcc=%.4f n=%d", split_name, avg_loss, acc, mcc, len(all_true))
    return avg_loss, acc, mcc


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    model_name: str,
    grad_clip: float,
    pos_weight: float | None,
    epoch: int,
    log_every: int,
):
    model.train()
    total_loss = 0.0
    n_correct, n_samples = 0, 0
    n_steps = len(loader)
    n_steps_done, n_steps_skipped = 0, 0
    log_info("[Epoch %03d] train start, %d batches", epoch, n_steps)

    for batch_idx, batch in enumerate(loader, start=1):
        price, labels, mask, window_valid = _prepare_batch(batch, device)
        if _needs_bn_batch(model_name) and price.shape[0] <= 1:
            n_steps_skipped += 1
            continue
        if not mask.any():
            n_steps_skipped += 1
            continue
        n_steps_done += 1
        if model_name == "adv_alstm":
            price_s = price[mask]
            labels_s = labels[mask]
            loss = model.compute_loss(price_s, labels_s, pos_weight)
            prediction = model.predict(price_s).detach()
        elif model_name == "dtml":
            prediction, labels_s = _forward_on_scope(
                model, model_name, price, labels, mask, window_valid=window_valid,
            )
            loss = _compute_loss(prediction, labels_s, model_name, pos_weight)
        else:
            price_s = price[mask]
            labels_s = labels[mask]
            prediction = _run_model(model, model_name, price_s).reshape(-1)
            loss = _compute_loss(prediction, labels_s, model_name, pos_weight)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        pred_cls = (prediction > 0.5).float()
        n_correct += int((pred_cls == labels_s).sum().item())
        n_samples += int(labels_s.numel())
        batch_loss = float(loss.item())
        total_loss += batch_loss

        if log_every > 0 and (batch_idx % log_every == 0 or batch_idx == n_steps):
            acc = float(n_correct) / float(n_samples) if n_samples else 0.0
            log_info(
                "[Epoch %03d] batch %d/%d loss=%.6f acc=%.4f",
                epoch, batch_idx, n_steps, batch_loss, acc,
            )

    avg_loss = total_loss / max(n_steps_done, 1)
    train_acc = float(n_correct) / float(n_samples) if n_samples else 0.0
    log_info(
        "[Epoch %03d] train done loss=%.6f acc=%.4f steps=%d/%d skipped=%d labels=%d",
        epoch, avg_loss, train_acc, n_steps_done, n_steps, n_steps_skipped, n_samples,
    )
    return avg_loss, train_acc


MODEL_DEFAULT_LR = {
    "dtml": 1e-5,
    "adv_alstm": 3e-4,
    "lstm": 3e-4,
    "alstm": 3e-4,
    "bi_lstm": 3e-4,
}

MODEL_DEFAULT_HPARAMS = {
    "bi_lstm": {"layers": 3, "dropout": 0.3},
    "dtml": {"layers": 1, "max_n_days": 10},
}


def main():
    parser = argparse.ArgumentParser(description="LightQuant price baselines on Pattern_Mining datasets")
    parser.add_argument("--model", type=str, required=True, choices=list(SINGLE_MODEL_CHOICES))
    parser.add_argument("--dataset_root", type=str, default="./dataset/CSMD50")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "stats"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=None, help="default: dtml 1e-5, others 3e-4")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--scheduler_eta_min", type=float, default=1e-6)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--early_stop_patience_epochs", type=int, default=25,
                        help="stop after this many val checks without best_metric improvement")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument(
        "--split_mode",
        type=str,
        default="contiguous",
        choices=["contiguous", "seasonal_h1q3q4", "stagger_4y", "quarter_rr_311"],
        help="contiguous=DATASET_SPLITS; seasonal_h1q3q4=H1/Q3/Q4; stagger_4y / quarter_rr_311=calendar role splits",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="True: cudnn deterministic (reproducible). False: seed only, like main.py train.set_all_seeds",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--max_n_days",
        type=int,
        default=5,
        help="look_back_window: consecutive HLC days before target (LightQuant)",
    )
    parser.add_argument("--use_volume", action="store_true",
                        help="use HLCV (volume-return) instead of HLC")
    parser.add_argument("--input_size", type=int, default=0,
                        help="price feat dim; 0=auto (4 if --use_volume else 3)")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attention_size", type=int, default=128)
    parser.add_argument("--adv_eps", type=float, default=0.01,
                        help="Adv-ALSTM adversarial scale eps on fea_con (default 1e-2)")
    parser.add_argument("--adv_beta", type=float, default=0.01,
                        help="Adv-ALSTM weight on adversarial loss (default 1e-2)")
    parser.add_argument("--adv_l2_alpha", type=float, default=0.01,
                        help="Adv-ALSTM L2 penalty on output layer (default 1e-2)")
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument(
        "--dtml_beta",
        type=float,
        default=0.1,
        help="multi-level context weight beta (DTML-pytorch ipynb default 0.1)",
    )
    parser.add_argument(
        "--dtml_drop_rate",
        type=float,
        default=0.1,
        help="dropout in DataAxisAttention (DTML-pytorch ipynb default 0.1)",
    )
    parser.add_argument(
        "--dtml_market_index",
        type=int,
        default=-1,
        help="index stock for market context (>=0); -1 uses cross-section mean HLC",
    )
    parser.add_argument(
        "--filter_neutral",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="if set, drop neutral days when building dataset; default keeps all days and masks at loss (LightQuant)",
    )
    parser.add_argument("--use_strong_only", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/single")
    parser.add_argument("--log_dir", type=str, default="./logs/single")
    parser.add_argument("--log_file", type=str, default="")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_test_each_epoch", action="store_true")
    parser.add_argument(
        "--class_weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reweight BCE by train strong-label pos rate (helps val/test distribution shift)",
    )
    parser.add_argument(
        "--best_metric",
        type=str,
        default="val_loss",
        choices=("val_acc_mcc", "val_mcc", "val_debiased", "val_loss"),
        help="val_loss=lower is better (default); val_acc_mcc=acc+mcc; val_mcc=mcc only; val_debiased=mcc+(acc-majority_acc)",
    )
    parser.add_argument(
        "--train_oversample_extra",
        type=str,
        default="",
        help="train_oversample_extra.jsonl (company,date copies). Train only; val/test unchanged.",
    )
    args = parser.parse_args()
    if args.lr is None:
        args.lr = MODEL_DEFAULT_LR.get(args.model, 3e-4)
    if not args.input_size:
        args.input_size = 4 if args.use_volume else 3
    if args.model == "dtml" and args.batch_size == 64:
        args.batch_size = 16
    if args.model == "dtml":
        if args.layers == 2:
            args.layers = MODEL_DEFAULT_HPARAMS["dtml"]["layers"]
        if args.max_n_days == 5:
            args.max_n_days = MODEL_DEFAULT_HPARAMS["dtml"]["max_n_days"]
    if args.model == "bi_lstm":
        if args.layers == 2:
            args.layers = MODEL_DEFAULT_HPARAMS["bi_lstm"]["layers"]
        if args.dropout == 0.1:
            args.dropout = MODEL_DEFAULT_HPARAMS["bi_lstm"]["dropout"]

    kind = detect_dataset_kind(args.dataset_root)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = args.log_file or (
        os.path.join(args.log_dir, f"{args.model}_{kind.lower()}_{ts}.log") if args.mode != "stats" else ""
    )
    setup_logger(log_file or None)

    set_seed(args.seed, deterministic=bool(args.deterministic))
    LOGGER.info("seed=%d deterministic=%s", int(args.seed), bool(args.deterministic))
    device = torch.device(args.device)
    log_info("model=%s dataset=%s root=%s device=%s", args.model, kind, args.dataset_root, device)

    look_back = args.max_n_days
    ds_kw = dict(
        dataset_root=args.dataset_root,
        look_back=look_back,
        filter_neutral=args.filter_neutral,
        use_strong_only=args.use_strong_only,
        seed=args.seed,
        model_name=args.model,
        split_mode=str(getattr(args, "split_mode", "contiguous") or "contiguous"),
        oversample_extra_path=str(getattr(args, "train_oversample_extra", "") or ""),
    )

    if args.mode == "stats":
        tr_start, tr_end = _split_range("train", kind)
        va_start, va_end = _split_range("val", kind)
        te_start, te_end = _split_range("test", kind)
        print(f"periods train={tr_start}..{tr_end} val={va_start}..{va_end} test={te_start}..{te_end}")
        for split in ("train", "val", "test"):
            ds, _, n, _, _, _ = build_price_datasets(mode=split, scalers={}, **ds_kw)
            n_strong, n_window, n_train_scope = 0, 0, 0
            pos, neg = 0, 0
            for i in range(len(ds)):
                item = ds[i]
                if len(item) == 5:
                    _, y, mv, window_ok, strong_ok = item
                else:
                    _, y, mv, window_ok = item
                    strong_ok = torch.tensor([
                        is_strong_mv(float(m)) for m in mv.reshape(-1)
                    ], dtype=torch.bool).reshape(mv.shape)
                y_flat = y.reshape(-1)
                mv_flat = mv.reshape(-1)
                w_flat = window_ok.reshape(-1).bool()
                s_flat = strong_ok.reshape(-1).bool()
                for j in range(len(y_flat)):
                    y_v = float(y_flat[j].item())
                    window_ok_v = bool(w_flat[j].item())
                    strong_v = bool(s_flat[j].item())
                    if window_ok_v:
                        n_window += 1
                    if strong_v:
                        n_strong += 1
                    if strong_v and window_ok_v:
                        n_train_scope += 1
                        if y_v >= 0.5:
                            pos += 1
                        else:
                            neg += 1
            pos_rate = pos / max(n_train_scope, 1)
            if args.model == "dtml" and isinstance(ds, DTMLPriceDataset):
                n_window, n_strong, n_scope = _dtml_scope_stats(ds.date_groups)
                print(
                    f"  dtml scope: window_valid={n_window} strong={n_strong} "
                    f"train_scope(strong&window)={n_scope} "
                    f"(=LSTM enumerate stock-days, NOT news filter)"
                )
            print(
                f"[{split}] n={n} window_valid={n_window} strong={n_strong} "
                f"train_scope(strong&window)={n_train_scope} "
                f"pos={pos_rate:.3f} always0_acc={1-pos_rate:.3f}"
            )
        return

    log_info("building datasets (LightQuant price windows + shared splits)...")
    data_bundle = load_all_data(args.dataset_root, use_volume=args.use_volume)
    train_ds, scalers, n_train, train_pos_rate, dtml_avg_stocks, dtml_max_stocks = build_price_datasets(
        mode="train", bundle=data_bundle, **ds_kw,
    )
    val_ds, _, n_val, val_pos_rate, _, _ = build_price_datasets(
        mode="val", scalers=scalers, bundle=data_bundle, **ds_kw,
    )
    test_ds, _, n_test, test_pos_rate, _, _ = build_price_datasets(
        mode="test", scalers=scalers, bundle=data_bundle, **ds_kw,
    )
    if args.model == "dtml":
        if dtml_max_stocks <= 0:
            raise RuntimeError("DTML dataset has no stocks; check dataset_root and splits")
        log_info("dtml n_stocks=%d (auto from dataset universe)", dtml_max_stocks)
    log_info(
        "io: price feats [T,%d] (%s); strong&window_valid mask at loss; "
        "dtml batches [B,T,N,%d]",
        args.input_size,
        "HLCV" if args.use_volume else "HLC",
        args.input_size,
    )
    log_info("samples train=%d val=%d test=%d", n_train, n_val, n_test)
    log_info(
        "strong pos_rate train=%.3f val=%.3f test=%.3f (majority acc val=%.3f test=%.3f)",
        train_pos_rate, val_pos_rate, test_pos_rate,
        _majority_acc(val_pos_rate), _majority_acc(test_pos_rate),
    )
    pos_weight = None
    if args.class_weight and train_pos_rate > 0.0 and train_pos_rate < 1.0:
        pos_weight = (1.0 - train_pos_rate) / train_pos_rate
        log_info("class_weight pos_weight=%.4f", pos_weight)
    if args.model == "dtml":
        n_batches = (n_train + args.batch_size - 1) // args.batch_size
        tr_window, _, tr_scope = _dtml_scope_stats(train_ds.date_groups)
        log_info(
            "dtml calendar: %d price trading-days/group (from graph-valid samples, NOT news); "
            "window_valid stock-days=%d",
            n_train, tr_window,
        )
        log_info(
            "dtml: %d trading-days × %d stocks/day -> %d optimizer steps/epoch "
            "(LSTM ref: ~%d steps); use smaller --batch_size for more steps",
            n_train, dtml_max_stocks, n_batches,
            (tr_window + args.batch_size - 1) // args.batch_size,
        )
        log_info(
            "dtml date-groups train=%d val=%d test=%d "
            "(avg=%d max=%d stocks/date, beta=%.3f market_idx=%d)",
            n_train, n_val, n_test, dtml_avg_stocks, dtml_max_stocks,
            args.dtml_beta, args.dtml_market_index,
        )
        log_info(
            "dtml batches/epoch train=%d val=%d test=%d (batch_size=%d)",
            n_batches,
            (n_val + args.batch_size - 1) // args.batch_size,
            (n_test + args.batch_size - 1) // args.batch_size,
            args.batch_size,
        )
    log_info(
        "hparams model=%s batch=%d epochs=%d lr=%g wd=%g hidden=%d layers=%d "
        "dropout=%g max_n_days=%d val_every=%d es_patience=%d best_metric=%s class_weight=%s",
        args.model, args.batch_size, args.epochs, args.lr, args.weight_decay,
        args.hidden_size, args.layers, args.dropout, args.max_n_days,
        args.val_every, args.early_stop_patience_epochs, args.best_metric, args.class_weight,
    )

    collate_fn = lambda b: collate_single_batch(
        b, args.model, dtml_max_stocks if args.model == "dtml" else 0,
    )
    loader_kw = dict(num_workers=args.num_workers, collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    model = build_single_model(
        args.model,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        dropout=args.dropout,
        batch_first=args.batch_first,
        attention_size=args.attention_size,
        adv_eps=args.adv_eps,
        adv_beta=args.adv_beta,
        adv_l2_alpha=args.adv_l2_alpha,
        n_heads=args.n_heads,
        dtml_layers=args.layers,
        dtml_hidden=64,
        dtml_beta=args.dtml_beta,
        dtml_drop_rate=args.dtml_drop_rate,
        dtml_market_index=args.dtml_market_index,
    ).to(device)

    if args.ckpt and os.path.isfile(args.ckpt):
        model.load_state_dict(_load_ckpt(args.ckpt, device))
        log_info("loaded checkpoint %s", args.ckpt)


    if args.mode == "eval":
        evaluate(model, val_loader, device, args.model, pos_weight, split_name="VAL")
        evaluate(model, test_loader, device, args.model, pos_weight, split_name="TEST")
        return

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.scheduler_eta_min,
    )
    save_dir = os.path.join(args.save_dir, args.model)
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_val_mcc = float("-inf")
    best_val_acc = 0.0
    best_score = float("inf") if args.best_metric == "val_loss" else float("-inf")
    best_epoch = 0
    last_improve_val_step = None
    val_step = 0
    val_every = max(1, args.val_every)
    log_info(
        "training start, save_dir=%s checkpoint/early_stop metric=%s patience=%d val_checks",
        save_dir, args.best_metric, args.early_stop_patience_epochs,
    )

    val_majority = _majority_acc(val_pos_rate)

    def _val_metric(acc: float, mcc: float, loss: float) -> float:
        if args.best_metric == "val_loss":
            return loss
        if args.best_metric == "val_mcc":
            return mcc
        if args.best_metric == "val_debiased":
            return mcc + (acc - val_majority)
        return acc + mcc

    def _metric_improved(current: float, best: float) -> bool:
        if args.best_metric == "val_loss":
            return current < best - args.early_stop_min_delta
        return current > best + args.early_stop_min_delta

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device,
            args.model, args.grad_clip, pos_weight, epoch, args.log_every,
        )
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        log_info(
            "[Epoch %03d] lr=%.2e train_loss=%.6f train_acc=%.4f epoch_time=%.2fs",
            epoch, cur_lr, train_loss, train_acc, time.perf_counter() - t_epoch,
        )

        if epoch % val_every == 0:
            val_step += 1
            val_loss, val_acc, val_mcc = evaluate(
                model, val_loader, device, args.model, pos_weight, split_name="VAL",
            )
            model.train()
            if args.eval_test_each_epoch:
                evaluate(model, test_loader, device, args.model, pos_weight, split_name="TEST")
            val_score = _val_metric(val_acc, val_mcc, val_loss)
            log_info(
                "[Epoch %03d] val_loss=%.6f val_acc=%.4f val_mcc=%.4f %s=%.4f",
                epoch, val_loss, val_acc, val_mcc, args.best_metric, val_score,
            )
            if _metric_improved(val_score, best_score):
                best_score = val_score
                best_val_loss = val_loss
                best_val_mcc = val_mcc
                best_val_acc = val_acc
                best_epoch = epoch
                last_improve_val_step = val_step
                ckpt = os.path.join(save_dir, f"best_{kind.lower()}.pt")
                torch.save(model.state_dict(), ckpt)
                log_info(
                    "%s new best -> %.4f (acc=%.4f mcc=%.4f), saved %s",
                    args.best_metric, best_score, best_val_acc, best_val_mcc, ckpt,
                )
            elif (
                last_improve_val_step is not None
                and (val_step - last_improve_val_step) >= args.early_stop_patience_epochs
            ):
                log_info(
                    "[EARLY STOP] %s no improvement for %d val checks (last best epoch %d)",
                    args.best_metric, args.early_stop_patience_epochs, best_epoch,
                )
                break

    best_ckpt = os.path.join(save_dir, f"best_{kind.lower()}.pt")
    if os.path.isfile(best_ckpt):
        model.load_state_dict(_load_ckpt(best_ckpt, device))
        log_info(
            "loaded best checkpoint from epoch %d (%s=%.4f val_acc=%.4f val_mcc=%.4f)",
            best_epoch, args.best_metric, best_score, best_val_acc, best_val_mcc,
        )
    evaluate(model, test_loader, device, args.model, pos_weight, split_name="FINAL_TEST")
    log_info(
        "[FINAL] best_epoch=%d %s=%.4f val_acc=%.4f val_mcc=%.4f test_pos_rate=%.3f",
        best_epoch, args.best_metric, best_score, best_val_acc, best_val_mcc, test_pos_rate,
    )


if __name__ == "__main__":
    main()
