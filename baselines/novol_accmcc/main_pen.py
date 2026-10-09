#!/usr/bin/env python3
"""
Train / eval PEN on Pattern_Mining datasets.

Split-lookback flat tensors (CminCnPenDataset) for CMIN-CN / CSMD50 /
CSMD300 / CMIN-US; auto-detected from ``--dataset_root``.

Model: model_pen.PENModel (PEN-main MEL -> MSIN -> VMD -> ATA).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle as pkl
import random
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model_pen import PENModel, build_pen_model
from pen1_data import (
    DATASET_SPLITS,
    Pen1Dataset,
    ROLE_SPLIT_MODES,
    __pen1_getitem__,
    _parse_mv,
    date_in_mode,
    detect_dataset_kind,
    get_ss_index,
    infer_tokenize_mode,
    is_strong_mv,
    load_news_and_types_csmd,
    load_price,
    pack_valid_dates_flat,
    quarter_rr_311_defer_months,
    resolve_vocab_path,
    set_seed,
    tokenize_text,
)
from pen1_train import evaluate as evaluate_flat
from pen1_train import kl_lambda_at_step, load_ckpt, save_ckpt, train_one_epoch as train_one_epoch_flat
from oversample_utils import apply_oversample_to_dataset_samples

LOGGER = logging.getLogger("pen")


def setup_logger(log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("pen")
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


# --- Split-lookback flat PEN dataset (CMIN-CN / CSMD / CMIN-US; same sample rules) ---

SEG_LENGTH = 10


def split_lookback_periods(kind: str) -> dict[str, tuple[str, str]]:
    sp = DATASET_SPLITS[kind]
    return {
        "train": (sp["train_start"], sp["train_end"]),
        "val": (sp["val_start"], sp["val_end"]),
        "test": (sp["test_start"], sp["test_end"]),
    }


CMIN_CN_PERIODS = split_lookback_periods("CMIN-CN")

_CMIN_CN_DATA_CACHE: dict[tuple[str, str, str], dict] = {}


def resolve_cmin_cn_news_source(news_source: str) -> str | None:
    """Return None for all sources; 'wind' for Wind-only (CSV col-5 contains 'wind')."""
    ns = (news_source or "wind").strip().lower()
    if ns in ("all", "none", ""):
        return None
    return ns


def _csv_row_passes_news_source(row: list, source_mode: str | None) -> bool:
    if len(row) < 4:
        return False
    if source_mode is None:
        return True
    if len(row) < 5:
        return False
    src = str(row[4]).lower()
    if source_mode == "wind":
        return "wind" in src
    return src.strip() == source_mode


def load_cmin_cn_news(news_dir: str, news_type_dir: str, source_mode: str | None = None):
    news_texts = defaultdict(lambda: defaultdict(dict))
    if not os.path.isdir(news_dir):
        return news_texts
    for company in os.listdir(news_dir):
        type_dir = os.path.join(news_type_dir, company)
        company_news_dir = os.path.join(news_dir, company)
        if not os.path.exists(type_dir) or not os.path.isdir(company_news_dir):
            continue
        for file_name in os.listdir(type_dir):
            if not file_name.endswith(".json"):
                continue
            date_str = file_name.replace(".json", "")
            type_path = os.path.join(type_dir, file_name)
            news_path = os.path.join(company_news_dir, f"{date_str}.csv")
            if not os.path.exists(news_path):
                continue
            try:
                with open(type_path, "r", encoding="utf-8") as f:
                    type_dict = json.load(f)
            except Exception:
                continue
            id_to_text: dict[str, str] = {}
            try:
                with open(news_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for idx, row in enumerate(reader):
                        if _csv_row_passes_news_source(row, source_mode):
                            id_to_text[str(idx)] = row[3]
            except Exception:
                continue
            for news_id in type_dict:
                if news_id in id_to_text:
                    news_texts[company][date_str][news_id] = id_to_text[news_id]
    return news_texts


def load_cmin_cn_data(dataset_root: str, news_source: str = "wind", use_volume: bool = False) -> dict:
    """Load price + news for split-lookback sampling (auto-detect dataset kind)."""
    root = os.path.abspath(dataset_root)
    kind = detect_dataset_kind(root)
    if kind in ("CSMD50", "CSMD300"):
        src_key = "all"
        source_mode = None
    else:
        source_mode = resolve_cmin_cn_news_source(news_source)
        src_key = source_mode or "all"
    cache_key = (root, kind, src_key, bool(use_volume))
    if cache_key in _CMIN_CN_DATA_CACHE:
        return _CMIN_CN_DATA_CACHE[cache_key]

    price_dir = os.path.join(root, "price", "preprocessed")
    stock_data, label, _raw_line = load_price(price_dir, use_volume=use_volume)
    if kind in ("CSMD50", "CSMD300"):
        llm_dir = os.path.join(root, "llm_extract", "news_type")
        news_texts, _news_types = load_news_and_types_csmd(llm_dir)
    else:
        news_dir = os.path.join(root, "news")
        news_type_dir = os.path.join(root, "llm_extract", "news_type")
        news_texts = load_cmin_cn_news(news_dir, news_type_dir, source_mode=source_mode)
    bundle = {
        "stock_data": stock_data,
        "label": label,
        "news_texts": news_texts,
        "news_source": src_key,
        "dataset_kind": kind,
    }
    _CMIN_CN_DATA_CACHE[cache_key] = bundle
    return bundle


def binary_label_cmin(mv: float) -> list[float]:
    return [0.0, 1.0] if mv > 0.0 else [1.0, 0.0]


def _global_trading_dates(stock_data: dict) -> list[str]:
    all_dates: set[str] = set()
    for day_dict in stock_data.values():
        all_dates.update(day_dict.keys())
    return sorted(all_dates)


def enumerate_cmin_cn_samples(
    stock_data: dict,
    label: dict,
    mode: str,
    price_window: int = 5,
    seg_length: int = SEG_LENGTH,
    *,
    kind: str = "CMIN-CN",
    periods: dict[str, tuple[str, str]] | None = None,
    split_mode: str = "contiguous",
) -> list[tuple[str, str, float]]:
    if periods is None:
        periods = split_lookback_periods(kind)
    start, end = periods[mode]
    sm = str(split_mode or "contiguous").strip().lower()
    if sm in ROLE_SPLIT_MODES:
        valid_dates = [d for d in _global_trading_dates(stock_data) if date_in_mode(d, mode, kind, sm)]
        defer = quarter_rr_311_defer_months(kind) if sm == "quarter_rr_311" else None
        # keep_tail=True matches prior pen packing; July deferred for qrr311 order.
        valid_dates = pack_valid_dates_flat(
            valid_dates, seg_length, keep_tail=True, defer_months=defer
        )
    else:
        valid_dates = [d for d in _global_trading_dates(stock_data) if start <= d <= end]
    if not valid_dates:
        return []

    companies = sorted(stock_data.keys())
    comp_pos = {
        c: {d: i for i, d in enumerate(sorted(stock_data.get(c, {}).keys()))}
        for c in companies
    }
    samples: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()

    for target_date in valid_dates:
        for comp in companies:
            pm = comp_pos.get(comp, {})
            if target_date not in pm or pm[target_date] < price_window:
                continue
            mv = _parse_mv(label.get(comp, {}).get(target_date))
            if mv is None:
                continue
            key = (comp, target_date)
            if key in seen:
                continue
            seen.add(key)
            samples.append((comp, target_date, float(mv)))
    samples.sort(key=lambda x: (x[1], x[0]))
    return samples


class CminCnPenDataset(Pen1Dataset):
    """One sample = one valid (stock, target_date) with split-internal price lookback."""

    def __init__(
        self,
        dataset_root: str,
        mode: str = "train",
        price_window: int = 5,
        news_padding_k: int = 5,
        max_n_msgs: int | None = None,
        max_n_words: int = 30,
        y_size: int = 2,
        use_strong_only: bool = False,
        shuffle_build: bool = False,
        seed: int = 37,
        news_source: str = "wind",
        use_volume: bool = False,
        split_mode: str = "contiguous",
    ):
        assert mode in {"train", "val", "test"}
        self.dataset_root = os.path.abspath(dataset_root)
        self.dataset_kind = detect_dataset_kind(self.dataset_root)
        self.mode = mode
        self.split_mode = str(split_mode or "contiguous").strip().lower() or "contiguous"
        self.use_volume = bool(use_volume)
        self.price_feat_dim = 4 if self.use_volume else 3
        self.price_window = price_window
        self.news_padding_k = int(news_padding_k)
        self.max_n_msgs = int(max_n_msgs) if max_n_msgs is not None else self.news_padding_k
        self.max_n_words = max_n_words
        self.max_n_days = price_window
        self.y_size = y_size
        self.use_strong_only = use_strong_only
        self.news_source = resolve_cmin_cn_news_source(news_source) or "all"
        periods = split_lookback_periods(self.dataset_kind)
        self.start_date, self.end_date = periods[mode]

        all_data = load_cmin_cn_data(self.dataset_root, news_source=news_source, use_volume=self.use_volume)
        self.stock_data = all_data["stock_data"]
        self.label = all_data["label"]
        self.news_texts = all_data["news_texts"]
        self.companies = sorted(self.stock_data.keys())
        self.company2id = {c: i for i, c in enumerate(self.companies)}

        vocab_path = resolve_vocab_path(self.dataset_root)
        self.vocab = pkl.load(open(vocab_path, "rb"))
        self.vocab_pad_id = int(self.vocab.get("<pad>", 0))
        self.vocab_unk_id = int(self.vocab.get("<unk>", 0))
        self.tokenize_mode = infer_tokenize_mode(self.dataset_root)
        self.use_char_tokenize = self.dataset_kind == "CMIN-CN"

        self.prev_date: dict[str, dict[str, str]] = {}
        for comp, day_dict in self.stock_data.items():
            dates_sorted = sorted(day_dict.keys())
            prev_map = {}
            for i in range(1, len(dates_sorted)):
                prev_map[dates_sorted[i]] = dates_sorted[i - 1]
            self.prev_date[comp] = prev_map

        raw_samples = enumerate_cmin_cn_samples(
            self.stock_data,
            self.label,
            mode,
            price_window=price_window,
            kind=self.dataset_kind,
            split_mode=self.split_mode,
        )
        self.samples: list[dict] = []
        rng = random.Random(seed)
        build_list = list(raw_samples)
        if shuffle_build:
            rng.shuffle(build_list)
        for comp, td, mv in build_list:
            sample = self._build_sample(comp, td, mv)
            if sample is None:
                continue
            if use_strong_only and not sample["strong"]:
                continue
            self.samples.append(sample)
        if shuffle_build and mode == "train":
            rng.shuffle(self.samples)
        n_strong = sum(1 for s in self.samples if s["strong"])
        log_info(
            "CminCnPenDataset %s kind=%s split_mode=%s n=%d strong=%d pw=%d news_k=%d news_source=%s vocab=%s",
            mode,
            self.dataset_kind,
            self.split_mode,
            len(self.samples),
            n_strong,
            price_window,
            self.news_padding_k,
            all_data.get("news_source", self.news_source),
            vocab_path,
        )

    def _prev_day_texts(self, comp: str, target_date: str) -> list[str]:
        prev_d = self.prev_date.get(comp, {}).get(target_date)
        if not prev_d:
            return []
        by_id = self.news_texts.get(comp, {}).get(prev_d, {})
        out: list[str] = []
        for _nid, text in by_id.items():
            if text:
                out.append(str(text))
            if len(out) >= self.news_padding_k:
                break
        return out

    def _char_ids(self, text: str) -> list[int]:
        s = "" if text is None else str(text)
        return [int(self.vocab.get(ch, self.vocab_unk_id)) for ch in s][: self.max_n_words]

    def _text_to_ids(self, text: str, comp: str) -> tuple[list[int], int]:
        if self.use_char_tokenize:
            ids = self._char_ids(text)
            return ids, (len(ids) - 1 if ids else 0)
        tokens = tokenize_text(text, self.tokenize_mode)[: self.max_n_words]
        if not tokens:
            return [], 0
        ids = [int(self.vocab.get(t, self.vocab_unk_id)) for t in tokens]
        return ids, get_ss_index(tokens, comp)

    def _encode_prev_news(self, texts: list[str], comp: str):
        word_mat = np.zeros((self.max_n_msgs, self.max_n_words), dtype=np.int32)
        n_word_vec = np.zeros((self.max_n_msgs,), dtype=np.int32)
        ss_index_vec = np.zeros((self.max_n_msgs,), dtype=np.int32)
        msg_id = 0
        for text in texts[: self.max_n_msgs]:
            ids, ss_idx = self._text_to_ids(text, comp)
            if not ids:
                continue
            n = len(ids)
            n_word_vec[msg_id] = n
            word_mat[msg_id, :n] = ids
            ss_index_vec[msg_id] = ss_idx
            msg_id += 1
        return word_mat[:msg_id], ss_index_vec[:msg_id], n_word_vec[:msg_id], msg_id

    def _build_sample(self, comp: str, target_date_str: str, mv: float):
        day_dict = self.stock_data.get(comp)
        if not day_dict:
            return None
        dates_sorted = sorted(day_dict.keys())
        if target_date_str not in dates_sorted:
            return None
        split_dates = [d for d in dates_sorted if self.start_date <= d <= target_date_str]
        if target_date_str not in split_dates:
            return None
        pos = split_dates.index(target_date_str)
        if pos < self.price_window:
            return None
        last_k = split_dates[pos - self.price_window : pos]

        T = self.price_window
        prices = np.zeros((T, self.price_feat_dim), dtype=np.float32)
        ys = np.zeros((T, self.y_size), dtype=np.float32)
        word_ids = np.zeros((T, self.max_n_msgs, self.max_n_words), dtype=np.int32)
        ss_index = np.zeros((T, self.max_n_msgs), dtype=np.int32)
        n_words = np.zeros((T, self.max_n_msgs), dtype=np.int32)
        n_msgs = np.zeros((T,), dtype=np.int32)

        for i, d_str in enumerate(last_k):
            raw = day_dict.get(d_str)
            if raw is None:
                return None
            prices[i] = np.array(list(raw)[: self.price_feat_dim], dtype=np.float32)
            if i < T - 1:
                mv_i = _parse_mv(self.label.get(comp, {}).get(d_str))
                if mv_i is None:
                    return None
                ys[i] = np.array(binary_label_cmin(mv_i), dtype=np.float32)
            news_key = target_date_str if i == T - 1 else d_str
            day_texts = self._prev_day_texts(comp, news_key)
            if day_texts:
                wm, ss, nw, nm = self._encode_prev_news(day_texts, comp)
                if nm > 0:
                    n_msgs[i] = nm
                    word_ids[i, :nm] = wm
                    ss_index[i, :nm] = ss
                    n_words[i, :nm] = nw

        ys[T - 1] = np.array(binary_label_cmin(mv), dtype=np.float32)

        return {
            "stock_id": self.company2id[comp],
            "company": comp,
            "target_date": target_date_str,
            "T": T,
            "word_ids": word_ids,
            "ss_index": ss_index,
            "n_words": n_words,
            "n_msgs": n_msgs,
            "price": prices,
            "y": ys,
            "strong": bool(is_strong_mv(mv)),
            "main_mv": float(mv),
        }


CminCnPenDataset.__getitem__ = lambda self, idx: {  # type: ignore[method-assign,return-value]
    **__pen1_getitem__(self, idx),
    "strong": bool(self.samples[idx]["strong"]),
}
CminCnPenDataset.__len__ = lambda self: len(self.samples)  # type: ignore[method-assign]


def _resolve_news_source(dataset_root: str, news_source: str | None) -> str:
    if news_source is not None:
        return news_source
    kind = detect_dataset_kind(dataset_root)
    return "wind" if kind == "CMIN-CN" else "all"


def _flat_ds_kwargs(args) -> dict:
    return dict(
        dataset_root=args.dataset_root,
        price_window=args.max_n_days,
        news_padding_k=args.news_padding_k,
        max_n_msgs=args.max_n_msgs,
        max_n_words=args.max_n_words,
        use_strong_only=args.use_strong_only,
        seed=args.seed,
        news_source=args.news_source,
        use_volume=bool(getattr(args, "use_volume", False)),
        split_mode=str(getattr(args, "split_mode", "contiguous") or "contiguous"),
    )


def _make_flat_dataset(mode: str, args, shuffle_build: bool) -> CminCnPenDataset:
    return CminCnPenDataset(mode=mode, shuffle_build=shuffle_build, **_flat_ds_kwargs(args))


def _vocab_info(dataset_root: str):
    vocab_path = resolve_vocab_path(dataset_root)
    vocab = pkl.load(open(vocab_path, "rb"))
    pad_id = int(vocab.get("<pad>", 0))
    vocab_vals = [int(v) for v in vocab.values() if isinstance(v, (int, np.integer))]
    vocab_size = max(len(vocab), max(vocab_vals) + 1 if vocab_vals else 1, pad_id + 1)
    return vocab_path, vocab_size, pad_id


def _build_model(vocab_size: int, pad_id: int, args) -> PENModel:
    price_input_size = getattr(args, "price_input_size", 0) or (4 if getattr(args, "use_volume", False) else 3)
    return build_pen_model(
        vocab_size,
        pad_id=pad_id,
        word_embed_size=args.word_embed_size,
        mel_h_size=args.mel_h_size,
        msin_h_size=args.msin_h_size,
        h_size=args.h_size,
        g_size=args.g_size,
        max_n_days=args.max_n_days,
        max_n_msgs=args.max_n_msgs,
        max_n_words=args.max_n_words,
        variant_type=args.variant_type,
        vmd_rec=args.vmd_rec,
        mel_cell_type=args.mel_cell_type,
        vmd_cell_type=args.vmd_cell_type,
        daily_att=args.daily_att,
        alpha=args.alpha,
        dropout_mel_in=args.dropout_mel_in,
        dropout_mel=args.dropout_mel,
        dropout_vmd_in=args.dropout_vmd_in,
        dropout_vmd=args.dropout_vmd,
        price_input_size=price_input_size,
    )


def _run_flat(args, kind: str, device: torch.device, log_file: str) -> None:
    if args.mode == "stats":
        for split in ("train", "val", "test"):
            ds = _make_flat_dataset(split, args, shuffle_build=False)
            start, end = split_lookback_periods(kind)[split]
            strong_n = sum(1 for s in ds.samples if s.get("strong"))
            log_info("[%s] n=%d strong=%d dates %s..%s", split, len(ds), strong_n, start, end)
        return

    log_info("building datasets (split-lookback)...")
    train_ds = _make_flat_dataset("train", args, shuffle_build=True)
    apply_oversample_to_dataset_samples(
        train_ds, str(getattr(args, "train_oversample_extra", "") or ""), log_fn=log_info,
    )
    val_ds = _make_flat_dataset("val", args, shuffle_build=False)
    test_ds = _make_flat_dataset("test", args, shuffle_build=False)
    log_info("samples train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))

    vocab_path, vocab_size, pad_id = _vocab_info(args.dataset_root)
    log_info("vocab=%s size=%d", vocab_path, vocab_size)

    loader_kw = dict(num_workers=args.num_workers, collate_fn=lambda b: b)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    model = _build_model(vocab_size, pad_id, args).to(device)

    if args.mode == "eval":
        ckpt_path = args.ckpt or os.path.join(args.save_dir, f"best_{kind.lower()}.pt")
        if os.path.isfile(ckpt_path):
            state, meta = load_ckpt(ckpt_path, device)
            model.load_state_dict(state)
            if "kl_lambda" in meta:
                model.set_kl_lambda(meta["kl_lambda"])
            log_info("loaded %s", ckpt_path)
        evaluate_flat(model, val_loader, device, args.max_n_days, "VAL", LOGGER, args.strong_eval)
        evaluate_flat(model, test_loader, device, args.max_n_days, "TEST", LOGGER, args.strong_eval)
        return

    if args.ckpt and os.path.isfile(args.ckpt):
        state, meta = load_ckpt(args.ckpt, device)
        model.load_state_dict(state)
        log_info("loaded init checkpoint %s", args.ckpt)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(args.save_dir, exist_ok=True)
    global_step = 0
    minimize = args.best_metric == "val_loss"
    best_score = float("inf") if minimize else float("-inf")
    best_epoch, stale = 0, 0
    log_info("training start best_metric=%s save_dir=%s", args.best_metric, args.save_dir)

    for epoch in range(1, args.epochs + 1):
        if args.variant_type != "discriminative":
            model.set_kl_lambda(
                kl_lambda_at_step(
                    global_step, args.kl_anneal_rate, args.kl_start_step, args.constant_kl_lambda
                )
            )
        train_one_epoch_flat(
            model, train_loader, optimizer, device, args.max_n_days,
            args.grad_clip, epoch, args.log_every, LOGGER, args.strong_eval,
        )
        global_step += max(len(train_loader), 1)
        val_loss, val_acc, val_mcc, _ = evaluate_flat(
            model, val_loader, device, args.max_n_days, "VAL", LOGGER, args.strong_eval
        )
        if args.best_metric == "val_loss":
            score = val_loss
        elif args.best_metric == "val_mcc":
            score = val_mcc
        else:
            score = val_acc + val_mcc
        improved = score < best_score if minimize else score > best_score
        log_info(
            "[Epoch %03d] val_loss=%.6f val_acc=%.4f val_mcc=%.4f %s=%.4f",
            epoch, val_loss, val_acc, val_mcc, args.best_metric, score,
        )
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            ckpt_path = os.path.join(args.save_dir, f"best_{kind.lower()}.pt")
            test_loss, test_acc, test_mcc, test_n = evaluate_flat(
                model, test_loader, device, args.max_n_days, "TEST", LOGGER, args.strong_eval
            )
            save_ckpt(
                ckpt_path,
                model,
                best_epoch=epoch,
                best_val_score=score,
                kl_lambda=getattr(model, "kl_lambda", 0.0),
                seed=args.seed,
                test_loss=test_loss,
                test_acc=test_acc,
                test_mcc=test_mcc,
                test_n=test_n,
            )
            log_info(
                "saved best checkpoint epoch=%d %s=%.4f -> %s",
                epoch, args.best_metric, score, ckpt_path,
            )
        else:
            stale += 1
            log_info("[Epoch %03d] no improve stale=%d/%d", epoch, stale, args.es_patience)
            if stale >= args.es_patience:
                log_info("early stop at epoch %d", epoch)
                break

    log_info("done best_epoch=%d best_%s=%.4f", best_epoch, args.best_metric, best_score)


def main():
    parser = argparse.ArgumentParser(
        description="PEN training (split-lookback flat tensors, all datasets)"
    )
    parser.add_argument("--dataset_root", type=str, default="./dataset/CMIN-CN")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "stats"])
    parser.add_argument("--batch_size", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=666)
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
    parser.add_argument("--max_n_days", type=int, default=5)
    parser.add_argument("--use_volume", action="store_true")
    parser.add_argument("--price_input_size", type=int, default=0)
    parser.add_argument("--max_n_msgs", type=int, default=5)
    parser.add_argument("--max_n_words", type=int, default=30)
    parser.add_argument("--news_padding_k", type=int, default=5)
    parser.add_argument("--word_embed_size", type=int, default=50)
    parser.add_argument("--mel_h_size", type=int, default=100)
    parser.add_argument("--msin_h_size", type=int, default=100)
    parser.add_argument("--h_size", type=int, default=150)
    parser.add_argument("--g_size", type=int, default=50)
    parser.add_argument("--variant_type", type=str, default="hedge", choices=["hedge", "fund", "tech", "discriminative"])
    parser.add_argument("--vmd_rec", type=str, default="zh", choices=["zh", "h"])
    parser.add_argument("--mel_cell_type", type=str, default="gru", choices=["gru", "basic", "ln-lstm"])
    parser.add_argument("--vmd_cell_type", type=str, default="gru", choices=["gru", "ln-lstm"])
    parser.add_argument("--daily_att", type=str, default="y", choices=["y", "g", "none"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--kl_anneal_rate", type=float, default=0.005)
    parser.add_argument("--kl_start_step", type=int, default=0)
    parser.add_argument("--constant_kl_lambda", type=float, default=None)
    parser.add_argument("--dropout_mel_in", type=float, default=0.3)
    parser.add_argument("--dropout_mel", type=float, default=0.0)
    parser.add_argument("--dropout_vmd_in", type=float, default=0.3)
    parser.add_argument("--dropout_vmd", type=float, default=0.0)
    parser.add_argument("--use_strong_only", action="store_true")
    parser.add_argument(
        "--strong_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="metrics on strong samples only",
    )
    parser.add_argument(
        "--news_source",
        type=str,
        default=None,
        choices=["wind", "all"],
        help="CMIN-CN default wind; CSMD/CMIN-US default all",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/pen1")
    parser.add_argument("--log_dir", type=str, default="./logs/pen1")
    parser.add_argument("--log_file", type=str, default="")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--es_patience", type=int, default=10)
    parser.add_argument(
        "--best_metric",
        type=str,
        default="val_loss",
        choices=("val_loss", "val_acc_mcc", "val_mcc"),
        help="checkpoint / early-stop metric; val_loss=lower is better (default)",
    )
    parser.add_argument(
        "--train_oversample_extra",
        type=str,
        default="",
        help="train_oversample_extra.jsonl (company,date copies). Train only; val/test unchanged.",
    )
    args = parser.parse_args()
    args.news_source = _resolve_news_source(args.dataset_root, args.news_source)

    kind = detect_dataset_kind(args.dataset_root)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = args.log_file or (os.path.join(args.log_dir, f"pen_{kind.lower()}_{ts}.log") if args.mode != "stats" else "")
    setup_logger(log_file or None)
    set_seed(args.seed, deterministic=bool(args.deterministic))
    LOGGER.info("seed=%d deterministic=%s", int(args.seed), bool(args.deterministic))
    device = torch.device(args.device)
    log_info(
        "pen dataset=%s root=%s device=%s news_source=%s strong_eval=%s",
        kind, args.dataset_root, device, args.news_source, args.strong_eval,
    )
    _run_flat(args, kind, device, log_file)


if __name__ == "__main__":
    main()
