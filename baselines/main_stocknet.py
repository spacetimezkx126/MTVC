#!/usr/bin/env python3
"""
Train / eval StockNet on CSMD50/300 & CMIN-CN/US.

Data alignment matches main_pen.py: split-lookback flat tensors (CminCnPenDataset),
prev-trading-day news padding, strong_eval metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle as pkl
import random
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from main_pen import CminCnPenDataset, _resolve_news_source, split_lookback_periods
from model_stocknet import StockNetModel
from pen1_train import (
    evaluate as evaluate_flat,
    load_ckpt,
    save_ckpt,
    train_one_epoch as train_one_epoch_flat,
)
from oversample_utils import apply_oversample_to_dataset_samples

LOGGER = logging.getLogger("stocknet")

_DATA_BUNDLE_CACHE: dict[tuple[str, str], dict] = {}


def _pen_precomputed_dir(dataset_root: str) -> str:
    d = os.path.join(dataset_root, "precomputed")
    os.makedirs(d, exist_ok=True)
    return d


def _pen_news_cache_tag(dataset_root: str, news_source: str) -> str:
    kind = detect_dataset_kind(dataset_root)
    ns = (news_source or "auto").strip().lower()
    if ns == "auto":
        src = "wind" if kind == "CMIN-CN" else "all"
    elif ns in ("all", "none", ""):
        src = "all"
    else:
        src = ns
    return f"{kind.lower()}_{src}"


def _pen_samples_cache_tag(
    dataset_root: str,
    news_source: str,
    data_mode: str,
    filter_neutral: bool,
    use_strong_only: bool,
    max_n_days: int,
    use_volume: bool = False,
) -> str:
    base = _pen_news_cache_tag(dataset_root, news_source)
    return (
        f"{base}_{data_mode}_d{max_n_days}_fn{int(filter_neutral)}_st{int(use_strong_only)}"
        f"_vol{int(bool(use_volume))}_excl"
    )


def date_in_pen_split(d: str, start: str, end: str) -> bool:
    """Match PEN-main DataPipe: start <= target_date < end (end exclusive)."""
    return start <= d < end


def _pen_news_cache_path(dataset_root: str, news_source: str) -> str:
    tag = _pen_news_cache_tag(dataset_root, news_source)
    return os.path.join(_pen_precomputed_dir(dataset_root), f"pen_news_{tag}.pkl")


def _pen_samples_cache_path(
    dataset_root: str,
    mode: str,
    news_source: str,
    data_mode: str,
    filter_neutral: bool,
    use_strong_only: bool,
    max_n_days: int,
    use_volume: bool = False,
) -> str:
    tag = _pen_samples_cache_tag(
        dataset_root,
        news_source,
        data_mode,
        filter_neutral,
        use_strong_only,
        max_n_days,
        use_volume=use_volume,
    )
    return os.path.join(_pen_precomputed_dir(dataset_root), f"pen_samples_{tag}_{mode}.pkl")


def _bundle_to_plain_dict(bundle: dict) -> dict:
    """Convert defaultdict news maps to plain dict for stable pickle caches."""
    out = dict(bundle)
    news_texts = {}
    for comp, by_date in bundle.get("news_texts", {}).items():
        news_texts[comp] = {d: dict(msgs) for d, msgs in by_date.items()}
    news_types = {}
    for comp, by_date in bundle.get("news_types", {}).items():
        news_types[comp] = {d: dict(msgs) for d, msgs in by_date.items()}
    out["news_texts"] = news_texts
    out["news_types"] = news_types
    return out


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pkl.load(f)


def _save_pickle(obj, path: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def setup_logger(log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("stocknet")
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


def log_batch_acc(n_correct: int, n_total: int) -> float:
    return float(n_correct) / float(n_total) if n_total > 0 else 0.0

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_ASCII_WORDCHAR_RE = re.compile(r"[A-Za-z0-9_@$']")
TOKEN_RE_EN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z0-9_@$']+|[^\w\s]",
    re.UNICODE,
)

DATASET_SPLITS = {
    "CSMD50": {
          "train_start": "2021-01-01",
                "train_end": "2023-01-01",
                "val_start": "2023-01-02",
                "val_end": "2024-01-02",
                "test_start": "2024-01-03",
                "test_end": "2024-12-31",
    },
            "CSMD300": {
                "train_start": "2021-01-01",
                "train_end": "2023-01-01",
                "val_start": "2023-01-02",
                "val_end": "2024-01-02",
                "test_start": "2024-01-03",
                "test_end": "2024-12-31",
            },
    "CMIN-CN": {
        "train_start": "2018-01-01",
        "train_end": "2021-04-30",
        "val_start": "2021-05-01",
        "val_end": "2021-08-31",
        "test_start": "2021-09-01",
        "test_end": "2021-12-31",
    },
    "CMIN-US": {
        "train_start": "2018-01-01",
        "train_end": "2021-04-30",
        "val_start": "2021-05-01",
        "val_end": "2021-08-31",
        "test_start": "2021-09-01",
        "test_end": "2021-12-31",
    },
    # Align with dual_tf/data.py DatasetProfile "massive" (inclusive calendar ends).
    "MASSIVE": {
        "train_start": "2022-01-01",
        "train_end": "2023-12-31",
        "val_start": "2024-01-01",
        "val_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-12-31",
    },
}


def calculate_mcc(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    num = (tp * tn) - (fp * fn)
    den = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    return float(num / den) if den != 0 else 0.0


def detect_dataset_kind(dataset_root: str) -> str:
    name = os.path.basename(os.path.normpath(dataset_root)).upper()
    if "CMIN-CN" in name or name == "CMINCN":
        return "CMIN-CN"
    if "CMIN-US" in name or "CMINUS" in name:
        return "CMIN-US"
    if "MASSIVE" in name:
        return "MASSIVE"
    if "300" in name:
        return "CSMD300"
    return "CSMD50"


def load_price(directory_path, use_volume: bool = False):
    """Load preprocessed price txt.

    Columns: date, close_ret(label), open_ret, high_ret, low_ret, close_ret, volume[_ret].
    Default features = H/L/C returns (parts[3:6]).
    If use_volume=True, append volume return (parts[6]) → HLCV [4].
    CSMD50/300 volume col is converted to return vs prev volume (same scale as HLC).
    """
    stock_data, label, raw_line = {}, {}, {}
    feat_end = 7 if use_volume else 6
    for filename in os.listdir(directory_path):
        if not filename.endswith(".txt"):
            continue
        company_name = filename.replace(".txt", "")
        file_path = os.path.join(directory_path, filename)
        stock_data[company_name] = {}
        label[company_name] = {}
        raw_line[company_name] = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                time = parts[0]
                feats = [float(x) for x in parts[3:min(feat_end, len(parts))]]
                if use_volume and len(feats) < 4:
                    feats = (feats + [0.0])[:4]
                features = feats[: 4 if use_volume else 3]
                stock_data[company_name][time] = features
                label[company_name][time] = parts[1]
                raw_line[company_name][time] = line.strip()
    return stock_data, label, raw_line


def _get_content_and_type_csmd(item):
    if not isinstance(item, dict):
        return None, None
    content = item.get("内容") or item.get("content") or ""
    dtype = item.get("类型") or item.get("type") or ""
    return (content.strip() if content else None), (dtype.strip() if dtype else None)


def load_news_and_types_from_llm_extract(llm_extract_news_type_dir):
    news_texts = defaultdict(lambda: defaultdict(dict))
    news_types = defaultdict(lambda: defaultdict(dict))
    if not os.path.exists(llm_extract_news_type_dir):
        return news_texts, news_types
    for company in os.listdir(llm_extract_news_type_dir):
        company_dir = os.path.join(llm_extract_news_type_dir, company)
        if not os.path.isdir(company_dir):
            continue
        for file_name in os.listdir(company_dir):
            if not file_name.endswith(".json"):
                continue
            date_str = file_name.replace(".json", "")
            file_path = os.path.join(company_dir, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            for news_id, item in data.items():
                content, dtype = _get_content_and_type_csmd(item)
                if not content:
                    continue
                news_texts[company][date_str][news_id] = content
                news_types[company][date_str][news_id] = dtype if dtype else ""
    return news_texts, news_types


def load_news_and_types(news_dir, news_type_dir, source_filter: str | None = None):
    news_texts = defaultdict(lambda: defaultdict(dict))
    news_types = defaultdict(lambda: defaultdict(dict))
    if not os.path.isdir(news_dir):
        return news_texts, news_types
    src_need = source_filter.strip().lower() if source_filter else None
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
            id_to_text = {}
            try:
                with open(news_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for idx, row in enumerate(reader):
                        if len(row) < 4:
                            continue
                        if src_need is not None:
                            if len(row) < 5:
                                continue
                            if row[4].strip().lower() != src_need:
                                continue
                        id_to_text[str(idx)] = row[3]
            except Exception:
                continue
            for news_id, news_type in type_dict.items():
                if news_id in id_to_text:
                    news_texts[company][date_str][news_id] = id_to_text[news_id]
                    news_types[company][date_str][news_id] = news_type
    return news_texts, news_types


def resolve_news_source(dataset_root: str, news_source: str = "auto") -> str | None:
    """
    Return a lower-case source name to keep, or None for all sources.
    auto: CMIN-CN -> wind only; other datasets -> all sources.
    """
    ns = (news_source or "auto").strip().lower()
    if ns in ("all", "none", ""):
        return None
    if ns == "auto":
        return "wind" if detect_dataset_kind(dataset_root) == "CMIN-CN" else None
    return ns


def load_all_data(
    dataset_root: str,
    news_source: str = "auto",
    use_cache: bool = True,
    rebuild_cache: bool = False,
    use_volume: bool = False,
) -> dict:
    root = os.path.abspath(dataset_root)
    source_filter = resolve_news_source(dataset_root, news_source)
    src_key = source_filter or "all"
    mem_key = (root, src_key, bool(use_volume))
    if use_cache and not rebuild_cache and mem_key in _DATA_BUNDLE_CACHE:
        return _DATA_BUNDLE_CACHE[mem_key]

    # News pickle cache is independent of price feat dim; reload price with use_volume.
    cache_path = _pen_news_cache_path(dataset_root, news_source)
    kind = detect_dataset_kind(dataset_root)
    price_dir = os.path.join(dataset_root, "price", "preprocessed")
    stock_data, label, raw_line = load_price(price_dir, use_volume=use_volume)
    if use_cache and not rebuild_cache and os.path.isfile(cache_path):
        bundle = _load_pickle(cache_path)
        bundle = dict(bundle)
        bundle["stock_data"] = stock_data
        bundle["label"] = label
        bundle["raw_line"] = raw_line
        bundle["use_volume"] = bool(use_volume)
        bundle["price_feat_dim"] = 4 if use_volume else 3
        _DATA_BUNDLE_CACHE[mem_key] = bundle
        log_info("loaded news bundle cache: %s (use_volume=%s)", cache_path, use_volume)
        return bundle

    if kind in ("CSMD50", "CSMD300"):
        llm_dir = os.path.join(dataset_root, "llm_extract", "news_type")
        news_texts, news_types = load_news_and_types_from_llm_extract(llm_dir)
    else:
        news_dir = os.path.join(dataset_root, "news")
        news_type_dir = os.path.join(dataset_root, "llm_extract", "news_type")
        news_texts, news_types = load_news_and_types(
            news_dir, news_type_dir, source_filter=source_filter
        )
    bundle = {
        "stock_data": stock_data,
        "label": label,
        "raw_line": raw_line,
        "news_texts": news_texts,
        "news_types": news_types,
        "news_source_filter": source_filter,
        "use_volume": bool(use_volume),
        "price_feat_dim": 4 if use_volume else 3,
    }
    bundle = _bundle_to_plain_dict(bundle)
    _DATA_BUNDLE_CACHE[mem_key] = bundle
    if use_cache:
        _save_pickle(bundle, cache_path)
        log_info("saved news bundle cache: %s", cache_path)
    return bundle

def resolve_vocab_path(dataset_root: str) -> str:
    kind = detect_dataset_kind(dataset_root)
    _here = os.path.dirname(os.path.abspath(__file__))
    # MTVC/baselines → MTVC/dict
    _pkg_dict = os.path.normpath(os.path.join(_here, "../dict"))
    candidates = {
        "CSMD50": [
            os.path.join(_pkg_dict, "dict_csmd.pkl"),
            "./dict/dict_csmd.pkl",
        ],
        "CSMD300": [
            os.path.join(_pkg_dict, "dict_csmd.pkl"),
            "./dict/dict_csmd.pkl",
        ],
        "CMIN-CN": [
            os.path.join(_pkg_dict, "dict_cn.pkl"),
            "./dict/dict_cn.pkl",
        ],
        "CMIN-US": [
            os.path.join(_pkg_dict, "dict_us.pkl"),
            "./dict/dict_us.pkl",
        ],
        "MASSIVE": [
            os.path.join(_pkg_dict, "dict_massive.pkl"),
            "./dict/dict_massive.pkl",
            os.path.join(_pkg_dict, "dict_us.pkl"),
            "./dict/dict_us.pkl",
        ],
    }
    for p in candidates.get(kind, candidates["CSMD50"]):
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"vocab not found for {kind}")


def tokenize_en(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE_EN.findall(text or "")]


def tokenize_cn(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text or "")
    out: list[str] = []
    for ch in text:
        if ch.isspace():
            continue
        if _CJK_RE.fullmatch(ch):
            out.append(ch)
        elif _ASCII_WORDCHAR_RE.fullmatch(ch):
            out.append(ch.lower() if ch.isalpha() else ch)
        else:
            out.append(ch)
    return out


def tokenize_text(text: str, mode: str) -> list[str]:
    return tokenize_en(text) if mode == "en" else tokenize_cn(text)


def infer_tokenize_mode(dataset_root: str) -> str:
    kind = detect_dataset_kind(dataset_root)
    return "en" if kind in ("CMIN-US", "MASSIVE") else "cn"


def _parse_mv(label_raw) -> float | None:
    if label_raw is None:
        return None
    s = str(label_raw).strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return None


MV_NEUTRAL_LOW = -0.005
MV_STRONG_HIGH = 0.0055


def is_strong_mv(mv: float) -> bool:
    """Strong-move mask aligned with main_csmd_50 / LightQuant valid labels."""
    return mv > MV_STRONG_HIGH or mv < MV_NEUTRAL_LOW


def binary_mv_label(mv: float) -> float:
    return 1.0 if mv > 0.0 else 0.0


def split_range_for_mode(mode: str, kind: str) -> tuple[str, str]:
    splits = DATASET_SPLITS[kind]
    if mode == "train":
        return splits["train_start"], splits["train_end"]
    if mode == "val":
        return splits["val_start"], splits["val_end"]
    return splits["test_start"], splits["test_end"]


def enumerate_price_samples(
    stock_data: dict,
    label_data: dict,
    start_date: str,
    end_date: str,
    look_back: int = 0,
    require_strong: bool = False,
) -> list[tuple[str, str, float]]:
    """
    LightQuant Normal_Dataset-style sample list: every (stock, target_date) in split
    with enough price history. Neutral days are kept unless require_strong=True.
    """
    samples: list[tuple[str, str, float]] = []
    for comp in sorted(stock_data.keys()):
        day_dict = stock_data[comp]
        lab_dict = label_data.get(comp, {})
        dates = [d for d in sorted(day_dict.keys()) if date_in_pen_split(d, start_date, end_date)]
        for i, target_date in enumerate(dates):
            if i < look_back:
                continue
            mv = _parse_mv(lab_dict.get(target_date))
            if mv is None:
                continue
            if require_strong and not is_strong_mv(mv):
                continue
            samples.append((comp, target_date, float(mv)))
    return samples


def _mv_to_onehot(mv: float, y_size: int = 2) -> list[float]:
    if y_size == 2:
        if mv <= 1e-7:
            return [1.0, 0.0]
        return [0.0, 1.0]
    thr1, thr2 = -0.004, 0.005
    if mv < thr1:
        return [1.0, 0.0, 0.0]
    if mv < thr2:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def _mv_to_class(mv: float, y_size: int = 2) -> int:
    oh = _mv_to_onehot(mv, y_size)
    return int(np.argmax(oh))


def get_ss_index(tokens: list[str], stock_symbol: str) -> int:
    ss = stock_symbol.lower()
    ss_index = len(tokens) - 1
    if not tokens:
        return 0
    if ss in tokens:
        return tokens.index(ss)
    if "$" in tokens:
        dollar_index = tokens.index("$")
        if dollar_index != len(tokens) - 1 and ss in tokens[dollar_index + 1 :]:
            return tokens.index(ss, dollar_index + 1)
        for index in range(dollar_index + 1, len(tokens)):
            if ss in tokens[index]:
                return index
    return ss_index


class Dataset(TorchDataset):
    """
    One sample = one (stock, target_date), aligned with PEN DataPipe.
    """

    def __init__(
        self,
        dataset_root: str,
        mode: str = "train",
        max_n_days: int = 5,
        max_n_msgs: int = 20,
        max_n_words: int = 100,
        y_size: int = 2,
        filter_neutral: bool = True,
        use_strong_only: bool = False,
        shuffle_build: bool = False,
        seed: int = 37,
        news_source: str = "all",
        data_mode: str = "strict",
        use_pen_cache: bool = True,
        rebuild_pen_cache: bool = False,
        skip_sample_build: bool = False,
        target_dates_filter: list[str] | None = None,
        use_volume: bool = False,
    ):
        assert mode in {"train", "val", "test"}
        if skip_sample_build or target_dates_filter is not None:
            use_pen_cache = False
        self.dataset_root = dataset_root
        self.mode = mode
        self.max_n_days = max_n_days
        self.max_n_msgs = max_n_msgs
        self.max_n_words = max_n_words
        self.y_size = y_size
        self.filter_neutral = filter_neutral
        self.use_strong_only = use_strong_only
        self.news_source = news_source
        self.data_mode = data_mode
        self.use_volume = bool(use_volume)
        self.price_feat_dim = 4 if self.use_volume else 3

        kind = detect_dataset_kind(dataset_root)
        splits = DATASET_SPLITS[kind]
        if mode == "train":
            self.start_date, self.end_date = splits["train_start"], splits["train_end"]
        elif mode == "val":
            self.start_date, self.end_date = splits["val_start"], splits["val_end"]
        else:
            self.start_date, self.end_date = splits["test_start"], splits["test_end"]

        samples_cache = _pen_samples_cache_path(
            dataset_root,
            mode,
            news_source,
            data_mode,
            filter_neutral,
            use_strong_only,
            max_n_days,
            use_volume=self.use_volume,
        )
        if (
            not skip_sample_build
            and target_dates_filter is None
            and use_pen_cache
            and not rebuild_pen_cache
            and os.path.isfile(samples_cache)
        ):
            cached = _load_pickle(samples_cache)
            self.companies = cached["companies"]
            self.company2id = cached["company2id"]
            self.samples = cached["samples"]
            bundle = load_all_data(
                dataset_root,
                news_source=news_source,
                use_cache=use_pen_cache,
                rebuild_cache=False,
                use_volume=self.use_volume,
            )
            self.stock_data = bundle["stock_data"]
            self.label = bundle["label"]
            self.news_texts = bundle["news_texts"]
            vocab_path = resolve_vocab_path(dataset_root)
            self.vocab = pkl.load(open(vocab_path, "rb"))
            self.vocab_pad_id = int(self.vocab.get("<pad>", 0))
            self.vocab_unk_id = int(self.vocab.get("<unk>", 0))
            self.tokenize_mode = infer_tokenize_mode(dataset_root)
            log_info(
                "loaded %s samples from cache (%s, n=%d)",
                mode,
                samples_cache,
                len(self.samples),
            )
            return

        bundle = load_all_data(
            dataset_root,
            news_source=news_source,
            use_cache=use_pen_cache,
            rebuild_cache=rebuild_pen_cache,
            use_volume=self.use_volume,
        )
        self.stock_data = bundle["stock_data"]
        self.label = bundle["label"]
        self.news_texts = bundle["news_texts"]
        self.companies = sorted(self.stock_data.keys())
        self.company2id = {c: i for i, c in enumerate(self.companies)}

        vocab_path = resolve_vocab_path(dataset_root)
        self.vocab = pkl.load(open(vocab_path, "rb"))
        self.vocab_pad_id = int(self.vocab.get("<pad>", 0))
        self.vocab_unk_id = int(self.vocab.get("<unk>", 0))
        self.tokenize_mode = infer_tokenize_mode(dataset_root)

        self.samples: list[dict] = []
        if skip_sample_build:
            log_info(
                "skip_sample_build: loaded bundle only (%s, companies=%d)",
                mode,
                len(self.companies),
            )
            return

        date_filter = set(target_dates_filter) if target_dates_filter else None
        rng = random.Random(seed)
        specs = self._sample_target_specs()
        if shuffle_build:
            rng.shuffle(specs)
        for comp, td in specs:
            if date_filter is not None and td not in date_filter:
                continue
            sample = self._build_sample(comp, td)
            if sample is not None:
                self.samples.append(sample)

        if shuffle_build and mode == "train":
            rng.shuffle(self.samples)

        if use_pen_cache and target_dates_filter is None:
            _save_pickle(
                {
                    "companies": self.companies,
                    "company2id": self.company2id,
                    "samples": self.samples,
                },
                samples_cache,
            )
            log_info("saved %s samples cache: %s (n=%d)", mode, samples_cache, len(self.samples))

    def _target_dates_for_stock(self, comp: str) -> list[str]:
        day_dict = self.stock_data.get(comp, {})
        out = []
        for d in sorted(day_dict.keys()):
            if date_in_pen_split(d, self.start_date, self.end_date):
                out.append(d)
        return out

    def _sample_target_specs(self) -> list[tuple[str, str]]:
        specs: list[tuple[str, str]] = []
        for comp in self.companies:
            for td in self._target_dates_for_stock(comp):
                specs.append((comp, td))
        return specs

    def _convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        out = []
        for t in tokens:
            tid = self.vocab.get(t, self.vocab_unk_id)
            out.append(int(tid))
        return out

    def _get_prices_and_ts(self, comp: str, main_target_date: datetime.date):
        day_dict = self.stock_data.get(comp)
        if not day_dict:
            return None
        lab_dict = self.label.get(comp, {})
        dates_desc = sorted(day_dict.keys(), reverse=True)
        d_t_min = main_target_date - timedelta(days=self.max_n_days - 1)

        ts, ys, prices, mv_percents = [], [], [], []
        main_mv = None

        for d_str in dates_desc:
            d = datetime.strptime(d_str, "%Y-%m-%d").date()
            mv = _parse_mv(lab_dict.get(d_str))
            if mv is None:
                continue
            if d == main_target_date:
                if self.filter_neutral and (-0.005 <= mv < 0.0055):
                    return None
                main_mv = mv
                ts.append(d)
                ys.append(_mv_to_onehot(mv, self.y_size))
            elif d_t_min <= d < main_target_date:
                ts.append(d)
                ys.append(_mv_to_onehot(mv, self.y_size))
                prices.append(list(day_dict[d_str])[: self.price_feat_dim])
                mv_percents.append(_mv_to_class(mv, self.y_size))
            elif d < d_t_min:
                prices.append(list(day_dict[d_str])[: self.price_feat_dim])
                mv_percents.append(_mv_to_class(mv, self.y_size))
                break

        t_len = len(ts)
        if t_len == 0 or len(ys) != t_len or len(prices) != t_len or len(mv_percents) != t_len:
            return None

        for item in (ts, ys, mv_percents, prices):
            item.reverse()

        if self.use_strong_only and main_mv is not None:
            if not (main_mv > 0.0055 or main_mv < -0.005):
                return None

        return {
            "T": t_len,
            "ts": ts,
            "ys": ys,
            "main_mv_percent": main_mv,
            "mv_percents": mv_percents,
            "prices": prices,
        }

    def _get_unaligned_corpora(self, comp: str, main_target_date: datetime.date):
        unaligned = []
        d_max = main_target_date - timedelta(days=1)
        d_min = main_target_date - timedelta(days=self.max_n_days)
        d = d_max
        news_by_date = self.news_texts.get(comp, {})
        while d >= d_min:
            d_str = d.isoformat()
            texts = list(news_by_date.get(d_str, {}).values())
            word_mat = np.zeros((self.max_n_msgs, self.max_n_words), dtype=np.int32)
            n_word_vec = np.zeros((self.max_n_msgs,), dtype=np.int32)
            ss_index_vec = np.zeros((self.max_n_msgs,), dtype=np.int32)
            msg_id = 0
            for text in texts:
                tokens = tokenize_text(text, self.tokenize_mode)[: self.max_n_words]
                if not tokens:
                    continue
                word_ids = self._convert_tokens_to_ids(tokens)
                n_words = len(word_ids)
                n_word_vec[msg_id] = n_words
                word_mat[msg_id, :n_words] = word_ids
                ss_index_vec[msg_id] = get_ss_index(tokens, comp)
                msg_id += 1
                if msg_id >= self.max_n_msgs:
                    break
            if msg_id > 0:
                unaligned.append(
                    [
                        d,
                        word_mat[:msg_id],
                        ss_index_vec[:msg_id],
                        n_word_vec[:msg_id],
                        msg_id,
                    ]
                )
            d -= timedelta(days=1)
        unaligned.reverse()
        return unaligned

    def _trading_day_alignment(self, ts, t_len, unaligned_corpora):
        aligned_word = np.zeros((t_len, self.max_n_msgs, self.max_n_words), dtype=np.int32)
        aligned_ss = np.zeros((t_len, self.max_n_msgs), dtype=np.int32)
        aligned_n_words = np.zeros((t_len, self.max_n_msgs), dtype=np.int32)
        aligned_n_msgs = np.zeros((t_len,), dtype=np.int32)

        aligned_msgs = [[] for _ in range(t_len)]
        aligned_ss_list = [[] for _ in range(t_len)]
        aligned_n_words_lists = [[] for _ in range(t_len)]
        aligned_n_msgs_lists = [[] for _ in range(t_len)]

        corpus_t_indices = []
        for corpus in unaligned_corpora:
            d = corpus[0]
            assigned = False
            for t in range(t_len):
                if d < ts[t]:
                    corpus_t_indices.append(t)
                    assigned = True
                    break
            if not assigned:
                return None

        if len(corpus_t_indices) != len(unaligned_corpora):
            return None

        for i, corpus in enumerate(unaligned_corpora):
            t = corpus_t_indices[i]
            word_mat, ss_index_vec, n_word_vec, n_msgs = corpus[1:]
            aligned_msgs[t].append(word_mat)
            aligned_ss_list[t].append(ss_index_vec)
            aligned_n_words_lists[t].append(n_word_vec)
            aligned_n_msgs_lists[t].append(n_msgs)

        n_fails = sum(1 for n_list in aligned_n_msgs_lists if sum(n_list) == 0)
        if n_fails > 0:
            return None

        for t in range(t_len):
            if not aligned_msgs[t]:
                continue
            msgs = np.vstack(aligned_msgs[t])
            ss_indices = np.hstack(aligned_ss_list[t])
            n_word = np.hstack(aligned_n_words_lists[t])
            n_msgs = min(sum(aligned_n_msgs_lists[t]), self.max_n_msgs)
            aligned_n_msgs[t] = n_msgs
            aligned_word[t, :n_msgs] = msgs[:n_msgs]
            aligned_ss[t, :n_msgs] = ss_indices[:n_msgs]
            aligned_n_words[t, :n_msgs] = n_word[:n_msgs]

        return {
            "msgs": aligned_word,
            "ss_indices": aligned_ss,
            "n_words": aligned_n_words,
            "n_msgs": aligned_n_msgs,
        }

    def _build_sample(self, comp: str, target_date_str: str):
        main_target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        prices_and_ts = self._get_prices_and_ts(comp, main_target_date)
        if not prices_and_ts:
            return None
        unaligned = self._get_unaligned_corpora(comp, main_target_date)
        aligned = self._trading_day_alignment(
            prices_and_ts["ts"], prices_and_ts["T"], unaligned
        )
        if not aligned:
            return None
        t_len = prices_and_ts["T"]
        return {
            "stock_id": self.company2id[comp],
            "company": comp,
            "target_date": target_date_str,
            "T": t_len,
            "word_ids": aligned["msgs"][:t_len],
            "ss_index": aligned["ss_indices"][:t_len],
            "n_words": aligned["n_words"][:t_len],
            "n_msgs": aligned["n_msgs"][:t_len],
            "price": np.array(prices_and_ts["prices"][:t_len], dtype=np.float32),
            "y": np.array(prices_and_ts["ys"][:t_len], dtype=np.float32),
            "main_mv": prices_and_ts["main_mv_percent"],
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        item = {
            "stock_id": s["stock_id"],
            "company": s["company"],
            "target_date": s["target_date"],
            "T": s["T"],
            "word_ids": torch.tensor(s["word_ids"], dtype=torch.long),
            "ss_index": torch.tensor(s["ss_index"], dtype=torch.long),
            "n_words": torch.tensor(s["n_words"], dtype=torch.long),
            "n_msgs": torch.tensor(s["n_msgs"], dtype=torch.long),
            "price": torch.tensor(s["price"], dtype=torch.float32),
            "y": torch.tensor(s["y"], dtype=torch.float32),
        }
        if "main_mv" in s and s["main_mv"] is not None:
            item["main_mv"] = float(s["main_mv"])
        return item


class PENDataset(Dataset):
    """Default PEN/StockNet loader: all news sources, strict alignment, no sample cache."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("news_source", "all")
        kwargs.setdefault("use_pen_cache", False)
        kwargs.setdefault("data_mode", "strict")
        super().__init__(*args, **kwargs)


class RelaxedPenDataset(Dataset):
    """
    Relaxed PEN samples: same strict price window as Dataset, but keep the sample when
    at least one lookback day has news (strict requires news on every window day).
    Missing news days are zero-padded in alignment.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("data_mode", "relaxed")
        super().__init__(*args, **kwargs)

    def _build_sample(self, comp: str, target_date_str: str):
        main_target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        prices_and_ts = self._get_prices_and_ts(comp, main_target_date)
        if not prices_and_ts:
            return None
        unaligned = self._get_unaligned_corpora(comp, main_target_date)
        if not unaligned:
            return None
        aligned = self._trading_day_alignment(
            prices_and_ts["ts"], prices_and_ts["T"], unaligned
        )
        if aligned is None or int(np.sum(aligned["n_msgs"])) == 0:
            return None
        t_len = prices_and_ts["T"]
        return {
            "stock_id": self.company2id[comp],
            "company": comp,
            "target_date": target_date_str,
            "T": t_len,
            "word_ids": aligned["msgs"][:t_len],
            "ss_index": aligned["ss_indices"][:t_len],
            "n_words": aligned["n_words"][:t_len],
            "n_msgs": aligned["n_msgs"][:t_len],
            "price": np.array(prices_and_ts["prices"][:t_len], dtype=np.float32),
            "y": np.array(prices_and_ts["ys"][:t_len], dtype=np.float32),
            "main_mv": prices_and_ts["main_mv_percent"],
        }

    def _trading_day_alignment(self, ts, t_len, unaligned_corpora):
        aligned_word = np.zeros((t_len, self.max_n_msgs, self.max_n_words), dtype=np.int32)
        aligned_ss = np.zeros((t_len, self.max_n_msgs), dtype=np.int32)
        aligned_n_words = np.zeros((t_len, self.max_n_msgs), dtype=np.int32)
        aligned_n_msgs = np.zeros((t_len,), dtype=np.int32)

        aligned_msgs = [[] for _ in range(t_len)]
        aligned_ss_list = [[] for _ in range(t_len)]
        aligned_n_words_lists = [[] for _ in range(t_len)]
        aligned_n_msgs_lists = [[] for _ in range(t_len)]

        for corpus in unaligned_corpora:
            d = corpus[0]
            for t in range(t_len):
                if d < ts[t]:
                    word_mat, ss_index_vec, n_word_vec, n_msgs = corpus[1:]
                    aligned_msgs[t].append(word_mat)
                    aligned_ss_list[t].append(ss_index_vec)
                    aligned_n_words_lists[t].append(n_word_vec)
                    aligned_n_msgs_lists[t].append(n_msgs)
                    break

        for t in range(t_len):
            if not aligned_msgs[t]:
                continue
            msgs = np.vstack(aligned_msgs[t])
            ss_indices = np.hstack(aligned_ss_list[t])
            n_word = np.hstack(aligned_n_words_lists[t])
            n_msgs = min(sum(aligned_n_msgs_lists[t]), self.max_n_msgs)
            aligned_n_msgs[t] = n_msgs
            aligned_word[t, :n_msgs] = msgs[:n_msgs]
            aligned_ss[t, :n_msgs] = ss_indices[:n_msgs]
            aligned_n_words[t, :n_msgs] = n_word[:n_msgs]

        return {
            "msgs": aligned_word,
            "ss_indices": aligned_ss,
            "n_words": aligned_n_words,
            "n_msgs": aligned_n_msgs,
        }


class AllPenDataset(RelaxedPenDataset):
    """
    All price-window samples: same target-date calendar as main_single.py
    (``enumerate_price_samples``), regardless of news (zero-padded when missing).
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("data_mode", "all")
        super().__init__(*args, **kwargs)

    def _sample_target_specs(self) -> list[tuple[str, str]]:
        require_strong = bool(self.filter_neutral or self.use_strong_only)
        pairs = enumerate_price_samples(
            self.stock_data,
            self.label,
            self.start_date,
            self.end_date,
            look_back=self.max_n_days,
            require_strong=require_strong,
        )
        return [(comp, target_date) for comp, target_date, _ in pairs]

    def _empty_aligned(self, t_len: int) -> dict:
        return {
            "msgs": np.zeros((t_len, self.max_n_msgs, self.max_n_words), dtype=np.int32),
            "ss_indices": np.zeros((t_len, self.max_n_msgs), dtype=np.int32),
            "n_words": np.zeros((t_len, self.max_n_msgs), dtype=np.int32),
            "n_msgs": np.zeros((t_len,), dtype=np.int32),
        }

    def _build_sample(self, comp: str, target_date_str: str):
        main_target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        prices_and_ts = self._get_prices_and_ts(comp, main_target_date)
        if not prices_and_ts:
            return None
        unaligned = self._get_unaligned_corpora(comp, main_target_date)
        if unaligned:
            aligned = self._trading_day_alignment(
                prices_and_ts["ts"], prices_and_ts["T"], unaligned
            )
            if aligned is None:
                aligned = self._empty_aligned(prices_and_ts["T"])
        else:
            aligned = self._empty_aligned(prices_and_ts["T"])
        t_len = prices_and_ts["T"]
        return {
            "stock_id": self.company2id[comp],
            "company": comp,
            "target_date": target_date_str,
            "T": t_len,
            "word_ids": aligned["msgs"][:t_len],
            "ss_index": aligned["ss_indices"][:t_len],
            "n_words": aligned["n_words"][:t_len],
            "n_msgs": aligned["n_msgs"][:t_len],
            "price": np.array(prices_and_ts["prices"][:t_len], dtype=np.float32),
            "y": np.array(prices_and_ts["ys"][:t_len], dtype=np.float32),
            "main_mv": prices_and_ts["main_mv_percent"],
        }


def pen_dataset_cls(data_mode: str):
    if data_mode == "relaxed":
        return RelaxedPenDataset
    if data_mode == "all":
        return AllPenDataset
    if data_mode == "strict":
        return Dataset
    raise ValueError(f"unknown data_mode: {data_mode!r}")


def filter_neutral_explicit_in_argv(argv: list[str] | None = None) -> bool:
    argv = argv if argv is not None else sys.argv
    flags = ("--filter-neutral", "--no-filter-neutral", "--filter_neutral", "--no-filter_neutral")
    return any(flag in argv for flag in flags)


def resolve_filter_neutral(data_mode: str, filter_neutral: bool, *, explicit: bool) -> bool:
    """``all`` mode defaults to keeping neutral days (same calendar as main_single.py)."""
    if explicit:
        return bool(filter_neutral)
    if data_mode == "all":
        return False
    return bool(filter_neutral)


def collate_pen_batch(batch, max_n_days: int):
    b = len(batch)
    m = batch[0]["word_ids"].size(1)
    w = batch[0]["word_ids"].size(2)
    y_size = batch[0]["y"].size(-1)

    word_ids = torch.zeros(b, max_n_days, m, w, dtype=torch.long)
    ss_index = torch.zeros(b, max_n_days, m, dtype=torch.long)
    n_words = torch.zeros(b, max_n_days, m, dtype=torch.long)
    n_msgs = torch.zeros(b, max_n_days, dtype=torch.long)
    p0 = batch[0]["price"]
    if torch.is_tensor(p0):
        price_dim = int(p0.size(-1))
    else:
        price_dim = int(np.asarray(p0).shape[-1])
    price = torch.zeros(b, max_n_days, price_dim, dtype=torch.float32)
    y = torch.zeros(b, max_n_days, y_size, dtype=torch.float32)
    t_idx = torch.zeros(b, dtype=torch.long)
    stock_id = torch.zeros(b, dtype=torch.long)
    main_mv = torch.full((b,), float("nan"), dtype=torch.float32)
    companies, dates = [], []

    for i, item in enumerate(batch):
        t = int(item["T"])
        word_ids[i, :t] = item["word_ids"]
        ss_index[i, :t] = item["ss_index"]
        n_words[i, :t] = item["n_words"]
        n_msgs[i, :t] = item["n_msgs"]
        price[i, :t] = item["price"]
        y[i, :t] = item["y"]
        t_idx[i] = t
        stock_id[i] = item["stock_id"]
        if "main_mv" in item:
            main_mv[i] = float(item["main_mv"])
        companies.append(item["company"])
        dates.append(item["target_date"])

    return {
        "word_ids": word_ids,
        "ss_index": ss_index,
        "n_words": n_words,
        "n_msgs": n_msgs,
        "price": price,
        "y": y,
        "t_idx": t_idx,
        "stock_id": stock_id,
        "main_mv": main_mv,
        "companies": companies,
        "dates": dates,
    }


def pen_strong_eval_mask(main_mv: torch.Tensor) -> torch.Tensor:
    """Strong-move mask aligned with main_single ``_training_mask`` (target-day mv)."""
    mv = main_mv.reshape(-1).float()
    return (mv > MV_STRONG_HIGH) | (mv < MV_NEUTRAL_LOW)


def majority_acc_from_classes(all_true: list[int]) -> float:
    if not all_true:
        return 0.0
    pos_rate = float(sum(all_true)) / float(len(all_true))
    return max(pos_rate, 1.0 - pos_rate)


def filter_pen_predictions_by_strong_mask(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    main_mv: torch.Tensor,
) -> tuple[list[int], list[int]]:
    mask = pen_strong_eval_mask(main_mv).cpu()
    yt = y_true.detach().cpu().reshape(-1)
    yp = y_pred.detach().cpu().reshape(-1)
    return yt[mask].tolist(), yp[mask].tolist()


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python/NumPy/PyTorch RNGs; optional cuDNN deterministic mode.

    deterministic=True  -> bit-reproducible (cudnn deterministic + CUBLAS workspace)
    deterministic=False -> same as Pattern_Mining/normal/train.set_all_seeds (seed only)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        if deterministic:
            torch.backends.cudnn.deterministic = True
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except (AttributeError, TypeError):
                pass
        else:
            torch.backends.cudnn.deterministic = False
            try:
                torch.use_deterministic_algorithms(False)
            except (AttributeError, TypeError):
                pass
    elif deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, TypeError):
            pass


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _load_ckpt(path: str, device):
    del device
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


_CKPT_META_KEYS = (
    "best_epoch",
    "best_val_score",
    "test_loss",
    "test_acc",
    "test_mcc",
    "test_n",
    "kl_lambda",
    "seed",
)


def _parse_ckpt(raw):
    if isinstance(raw, dict) and "state_dict" in raw:
        meta = {k: raw[k] for k in _CKPT_META_KEYS if k in raw}
        if "test_rng_state" in raw:
            meta["test_rng_state"] = _normalize_rng_state_blob(raw["test_rng_state"])
        return raw["state_dict"], meta
    return raw, {}


def _load_ckpt_bundle(path: str, device):
    return _parse_ckpt(_load_ckpt(path, device))


def _test_metrics_from_meta(meta: dict) -> tuple[float, float, float, int] | None:
    if not meta or "test_loss" not in meta:
        return None
    return (
        float(meta["test_loss"]),
        float(meta["test_acc"]),
        float(meta["test_mcc"]),
        int(meta["test_n"]),
    )


def _as_byte_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(dtype=torch.uint8).contiguous()
    return torch.as_tensor(x, dtype=torch.uint8).cpu().contiguous()


def _normalize_rng_state_blob(blob: dict | None) -> dict | None:
    if not blob or not isinstance(blob, dict):
        return blob
    out: dict = {}
    if "torch_cpu" in blob:
        out["torch_cpu"] = _as_byte_tensor(blob["torch_cpu"])
    if "torch_cuda" in blob:
        raw_cuda = blob["torch_cuda"]
        if isinstance(raw_cuda, (list, tuple)):
            out["torch_cuda"] = [_as_byte_tensor(s) for s in raw_cuda]
        else:
            out["torch_cuda"] = [_as_byte_tensor(raw_cuda)]
    return out


def _capture_rng_state() -> dict:
    state = {"torch_cpu": torch.get_rng_state().cpu().to(dtype=torch.uint8)}
    if torch.cuda.is_available():
        state["torch_cuda"] = [
            s.cpu().to(dtype=torch.uint8) for s in torch.cuda.get_rng_state_all()
        ]
    return state


def _restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    state = _normalize_rng_state_blob(state) or {}
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        cuda_states = list(state["torch_cuda"])
        if len(cuda_states) < n_devices:
            cuda_states.extend(
                _as_byte_tensor(s) for s in torch.cuda.get_rng_state_all()[len(cuda_states):]
            )
        torch.cuda.set_rng_state_all(cuda_states[:n_devices])


def _log_final_test_check(
    ref: tuple[float, float, float, int] | None,
    final_loss: float,
    final_acc: float,
    final_mcc: float,
    final_n: int,
) -> None:
    if ref is None:
        return
    ref_loss, ref_acc, ref_mcc, ref_n = ref
    match = (
        abs(final_loss - ref_loss) < 1e-4
        and abs(final_acc - ref_acc) < 1e-5
        and abs(final_mcc - ref_mcc) < 1e-5
        and final_n == ref_n
    )
    log_info(
        "[FINAL_TEST check] match=%s | epoch TEST loss=%.6f acc=%.4f mcc=%.4f n=%d | "
        "re-run FINAL_TEST loss=%.6f acc=%.4f mcc=%.4f n=%d",
        "YES" if match else "NO",
        ref_loss,
        ref_acc,
        ref_mcc,
        ref_n,
        final_loss,
        final_acc,
        final_mcc,
        final_n,
    )


def _run_final_test(
    model,
    test_loader,
    device,
    max_n_days: int,
    *,
    ckpt_meta: dict | None,
    ref_test_metrics: tuple[float, float, float, int] | None,
    best_epoch: int,
    strong_only: bool = True,
):
    rng = (ckpt_meta or {}).get("test_rng_state")
    if rng:
        _restore_rng_state(rng)
        log_info("restored test RNG state before FINAL_TEST (best_epoch=%d)", best_epoch)
    else:
        log_info(
            "no test RNG state in checkpoint; FINAL_TEST re-run may differ from epoch TEST"
        )
    final_loss, final_acc, final_mcc, final_n = evaluate_flat(
        model, test_loader, device, max_n_days, "FINAL_TEST", LOGGER, strong_only
    )
    ref = ref_test_metrics or _test_metrics_from_meta(ckpt_meta or {})
    _log_final_test_check(ref, final_loss, final_acc, final_mcc, final_n)
    return final_loss, final_acc, final_mcc, final_n


def _save_best_ckpt(
    path: str,
    model,
    *,
    best_epoch: int,
    best_val_score: float,
    kl_lambda: float,
    seed: int,
    test_metrics: tuple[float, float, float, int] | None = None,
    test_rng_state: dict | None = None,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val_score),
        "kl_lambda": float(kl_lambda),
        "seed": int(seed),
    }
    if test_metrics is not None:
        t_loss, t_acc, t_mcc, t_n = test_metrics
        payload["test_loss"] = float(t_loss)
        payload["test_acc"] = float(t_acc)
        payload["test_mcc"] = float(t_mcc)
        payload["test_n"] = int(t_n)
    if test_rng_state is not None:
        payload["test_rng_state"] = test_rng_state
    torch.save(payload, path)


def _load_checkpoint_into_model(
    model,
    path: str,
    device,
    args=None,
    log_label: str = "checkpoint",
) -> dict:
    state_dict, meta = _load_ckpt_bundle(path, device)
    model.load_state_dict(state_dict)
    log_info("loaded %s %s", log_label, path)
    if "kl_lambda" in meta:
        model.set_kl_lambda(meta["kl_lambda"])
        log_info("restored kl_lambda=%.4f from checkpoint", meta["kl_lambda"])
    if args is not None and "seed" in meta:
        ckpt_seed = int(meta["seed"])
        log_info("checkpoint seed=%d", ckpt_seed)
        if int(args.seed) != ckpt_seed:
            log_info(
                "align --seed to checkpoint seed=%d (was %d)",
                ckpt_seed,
                int(args.seed),
            )
            args.seed = ckpt_seed
            set_seed(args.seed, deterministic=bool(getattr(args, "deterministic", True)))
    return meta


@torch.no_grad()
def evaluate(model, loader, device, max_n_days: int, split_name: str = ""):
    model.eval()
    all_true, all_pred = [], []
    total_loss = 0.0
    n_batches = 0
    n_all = 0
    for raw in loader:
        batch = collate_pen_batch(raw, max_n_days)
        out = model(
            batch["word_ids"].to(device),
            batch["n_words"].to(device),
            batch["ss_index"].to(device),
            batch["n_msgs"].to(device),
            batch["price"].to(device),
            batch["y"].to(device),
            batch["t_idx"].to(device),
            compute_loss=True,
        )
        if out["loss"] is not None:
            total_loss += float(out["loss"].item())
            n_batches += 1
        n_all += int(out["y_true_class"].numel())
        yt, yp = filter_pen_predictions_by_strong_mask(
            out["y_true_class"], out["y_pred_class"], batch["main_mv"]
        )
        all_true.extend(yt)
        all_pred.extend(yp)
    acc = accuracy_score(all_true, all_pred) if all_true else 0.0
    mcc = calculate_mcc(all_true, all_pred) if all_true else 0.0
    avg_loss = total_loss / max(n_batches, 1)
    n_samples = len(all_true)
    majority = majority_acc_from_classes(all_true)
    if split_name:
        log_info(
            "[%s] loss=%.6f acc=%.4f mcc=%.4f n=%d strong_n=%d majority_acc=%.4f",
            split_name,
            avg_loss,
            acc,
            mcc,
            n_all,
            n_samples,
            majority,
        )
    return avg_loss, acc, mcc, n_samples


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    max_n_days: int,
    grad_clip: float,
    epoch: int,
    log_every: int = 10,
):
    model.train()
    total_loss = 0.0
    n_batches = 0
    n_correct = 0
    n_samples = 0
    n_steps = len(loader)
    log_info("[Epoch %03d] train start, %d batches", epoch, n_steps)

    for batch_idx, raw in enumerate(loader, start=1):
        batch = collate_pen_batch(raw, max_n_days)
        optimizer.zero_grad(set_to_none=True)
        out = model(
            batch["word_ids"].to(device),
            batch["n_words"].to(device),
            batch["ss_index"].to(device),
            batch["n_msgs"].to(device),
            batch["price"].to(device),
            batch["y"].to(device),
            batch["t_idx"].to(device),
            compute_loss=True,
        )
        loss = out["loss"]
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = int(out["y_true_class"].numel())
        yt, yp = filter_pen_predictions_by_strong_mask(
            out["y_true_class"], out["y_pred_class"], batch["main_mv"]
        )
        n_correct += sum(int(t == p) for t, p in zip(yt, yp))
        n_samples += len(yt)
        batch_loss = float(loss.item())
        total_loss += batch_loss
        n_batches += 1

        if log_every > 0 and (batch_idx % log_every == 0 or batch_idx == n_steps):
            log_info(
                "[Epoch %03d] batch %d/%d loss=%.6f acc=%.4f",
                epoch,
                batch_idx,
                n_steps,
                batch_loss,
                log_batch_acc(n_correct, n_samples),
            )

    avg_loss = total_loss / max(n_batches, 1)
    train_acc = log_batch_acc(n_correct, n_samples)
    log_info(
        "[Epoch %03d] train done loss=%.6f acc=%.4f",
        epoch,
        avg_loss,
        train_acc,
    )
    return avg_loss, train_acc


def kl_lambda_at_step(step: int, anneal_rate: float, start_step: int, constant: float | None) -> float:
    if step < start_step:
        return 0.0
    if constant is not None:
        return constant
    return min(anneal_rate * step, 1.0)


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
        use_volume=args.use_volume,
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


def main():
    parser = argparse.ArgumentParser(
        description="StockNet on Pattern_Mining datasets (split-lookback flat tensors, same as main_pen)"
    )
    parser.add_argument("--dataset_root", type=str, default="./dataset/CSMD50")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "stats"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=789)
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
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_n_days", type=int, default=5)
    parser.add_argument("--use_volume", action="store_true",
                        help="append volume-return channel (HLCV) as price input")
    parser.add_argument("--price_input_size", type=int, default=0,
                        help="price feat dim; 0=auto (4 if --use_volume else 3)")
    parser.add_argument("--max_n_msgs", type=int, default=5)
    parser.add_argument("--max_n_words", type=int, default=30)
    parser.add_argument("--news_padding_k", type=int, default=5)
    parser.add_argument("--word_embed_size", type=int, default=50)
    parser.add_argument("--mel_h_size", type=int, default=100)
    parser.add_argument("--h_size", type=int, default=150)
    parser.add_argument("--g_size", type=int, default=50)
    parser.add_argument("--variant_type", type=str, default="hedge", choices=["hedge", "fund", "tech"])
    parser.add_argument("--vmd_rec", type=str, default="zh", choices=["zh", "h"])
    parser.add_argument("--daily_att", type=str, default="y", choices=["y", "g"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--kl_anneal_rate", type=float, default=0.005)
    parser.add_argument("--kl_start_step", type=int, default=0)
    parser.add_argument("--constant_kl_lambda", type=float, default=None)
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
    parser.add_argument("--save_dir", type=str, default="./checkpoints/stocknet")
    parser.add_argument("--log_dir", type=str, default="./logs/stocknet", help="训练日志目录")
    parser.add_argument("--log_file", type=str, default="", help="指定日志文件；默认自动生成到 log_dir")
    parser.add_argument("--log_every", type=int, default=10, help="每 N 个 batch 打印一次训练进度，0=仅 epoch 末")
    parser.add_argument("--eval_test_each_epoch", action="store_true", help="每个 epoch 结束后在 test 上评估并写日志")
    parser.add_argument("--es_patience", type=int, default=5)
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
    log_file = args.log_file
    if not log_file and args.mode in ("train", "eval"):
        log_file = os.path.join(args.log_dir, f"stocknet_{kind.lower()}_{ts}.log")
    setup_logger(log_file if args.mode != "stats" else None)

    set_seed(args.seed, deterministic=bool(args.deterministic))
    log_info("seed=%d deterministic=%s", int(args.seed), bool(args.deterministic))
    device = torch.device(args.device)
    log_info(
        "stocknet dataset=%s root=%s device=%s news_source=%s strong_eval=%s",
        kind,
        args.dataset_root,
        device,
        args.news_source,
        args.strong_eval,
    )

    if args.mode == "stats":
        for split in ("train", "val", "test"):
            ds = _make_flat_dataset(split, args, shuffle_build=False)
            start, end = split_lookback_periods(kind)[split]
            strong_n = sum(1 for s in ds.samples if s.get("strong"))
            log_info("[%s] n=%d strong=%d dates %s..%s", split, len(ds), strong_n, start, end)
        return

    log_info("building datasets (split-lookback, same as main_pen)...")
    train_ds = _make_flat_dataset("train", args, shuffle_build=True)
    apply_oversample_to_dataset_samples(
        train_ds, str(getattr(args, "train_oversample_extra", "") or ""), log_fn=log_info,
    )
    val_ds = _make_flat_dataset("val", args, shuffle_build=False)
    test_ds = _make_flat_dataset("test", args, shuffle_build=False)
    log_info("samples train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))

    vocab_path, vocab_size, pad_id = _vocab_info(args.dataset_root)
    log_info("vocab=%s size=%d", vocab_path, vocab_size)
    log_info(
        "hparams batch_size=%d epochs=%d lr=%g variant=%s vmd_rec=%s daily_att=%s alpha=%g",
        args.batch_size,
        args.epochs,
        args.lr,
        args.variant_type,
        args.vmd_rec,
        args.daily_att,
        args.alpha,
    )

    loader_kw = dict(num_workers=args.num_workers, collate_fn=lambda b: b)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    price_input_size = args.price_input_size or (4 if args.use_volume else 3)
    model = StockNetModel(
        vocab_size=vocab_size,
        pad_id=pad_id,
        word_embed_size=args.word_embed_size,
        mel_h_size=args.mel_h_size,
        h_size=args.h_size,
        g_size=args.g_size,
        max_n_days=args.max_n_days,
        max_n_msgs=args.max_n_msgs,
        max_n_words=args.max_n_words,
        variant_type=args.variant_type,
        vmd_rec=args.vmd_rec,
        daily_att=args.daily_att,
        alpha=args.alpha,
        price_input_size=price_input_size,
    ).to(device)

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
        if "kl_lambda" in meta:
            model.set_kl_lambda(meta["kl_lambda"])
        log_info("loaded init checkpoint %s", args.ckpt)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(args.save_dir, exist_ok=True)
    global_step = 0
    minimize = args.best_metric == "val_loss"
    best_score = float("inf") if minimize else float("-inf")
    best_epoch = 0
    best_test_metrics = None
    stale = 0
    log_info("training start best_metric=%s save_dir=%s", args.best_metric, args.save_dir)

    for epoch in range(1, args.epochs + 1):
        kl_l = kl_lambda_at_step(
            global_step, args.kl_anneal_rate, args.kl_start_step, args.constant_kl_lambda
        )
        model.set_kl_lambda(kl_l)
        train_loss, train_acc = train_one_epoch_flat(
            model,
            train_loader,
            optimizer,
            device,
            args.max_n_days,
            args.grad_clip,
            epoch,
            args.log_every,
            LOGGER,
            args.strong_eval,
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
        test_rng_state = None
        if args.eval_test_each_epoch:
            test_rng_state = _capture_rng_state()
            test_loss, test_acc, test_mcc, test_n = evaluate_flat(
                model, test_loader, device, args.max_n_days, "TEST", LOGGER, args.strong_eval
            )
            log_info(
                "[Epoch %03d] summary train_loss=%.6f train_acc=%.4f val_loss=%.6f "
                "val_acc=%.4f val_mcc=%.4f %s=%.4f test_loss=%.6f "
                "test_acc=%.4f test_mcc=%.4f kl_lambda=%.4f",
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                val_mcc,
                args.best_metric,
                score,
                test_loss,
                test_acc,
                test_mcc,
                kl_l,
            )
        else:
            log_info(
                "[Epoch %03d] summary train_loss=%.6f train_acc=%.4f val_loss=%.6f "
                "val_acc=%.4f val_mcc=%.4f %s=%.4f kl_lambda=%.4f",
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                val_mcc,
                args.best_metric,
                score,
                kl_l,
            )
        if improved:
            best_score = score
            best_epoch = epoch
            stale = 0
            if args.eval_test_each_epoch:
                best_test_metrics = (test_loss, test_acc, test_mcc, test_n)
            else:
                test_loss, test_acc, test_mcc, test_n = evaluate_flat(
                    model, test_loader, device, args.max_n_days, "TEST", LOGGER, args.strong_eval
                )
                best_test_metrics = (test_loss, test_acc, test_mcc, test_n)
            ckpt_path = os.path.join(args.save_dir, f"best_{kind.lower()}.pt")
            save_ckpt(
                ckpt_path,
                model,
                best_epoch=epoch,
                best_val_score=score,
                kl_lambda=kl_l,
                seed=args.seed,
                test_loss=test_loss,
                test_acc=test_acc,
                test_mcc=test_mcc,
                test_n=test_n,
                test_rng_state=test_rng_state,
            )
            log_info(
                "saved best checkpoint (%s=%.4f) -> %s",
                args.best_metric, score, ckpt_path,
            )
        else:
            stale += 1
            log_info("no val improvement, stale=%d/%d", stale, args.es_patience)
            if stale >= args.es_patience:
                log_info(
                    "[EARLY STOP] no %s improvement for %d epochs",
                    args.best_metric, args.es_patience,
                )
                break

    best_ckpt = os.path.join(args.save_dir, f"best_{kind.lower()}.pt")
    ckpt_meta: dict = {}
    if os.path.isfile(best_ckpt):
        ckpt_meta = _load_checkpoint_into_model(
            model, best_ckpt, device, args, log_label=f"best checkpoint from epoch {best_epoch}"
        )
        if ckpt_meta.get("best_epoch"):
            best_epoch = int(ckpt_meta["best_epoch"])
        if best_test_metrics is None:
            best_test_metrics = _test_metrics_from_meta(ckpt_meta)
        if "kl_lambda" not in ckpt_meta:
            best_step = max(0, (best_epoch - 1) * len(train_loader))
            model.set_kl_lambda(
                kl_lambda_at_step(
                    best_step,
                    args.kl_anneal_rate,
                    args.kl_start_step,
                    args.constant_kl_lambda,
                )
            )
            log_info(
                "FINAL eval kl_lambda=%.4f (matched to best_epoch=%d)",
                model.kl_lambda,
                best_epoch,
            )
    test_loss, test_acc, test_mcc, _ = _run_final_test(
        model,
        test_loader,
        device,
        args.max_n_days,
        ckpt_meta=ckpt_meta or None,
        ref_test_metrics=best_test_metrics,
        best_epoch=best_epoch,
        strong_only=args.strong_eval,
    )
    log_info(
        "[FINAL TEST @ best epoch %d] loss=%.6f acc=%.4f mcc=%.4f best_%s=%.4f",
        best_epoch,
        test_loss,
        test_acc,
        test_mcc,
        args.best_metric,
        best_score,
    )


if __name__ == "__main__":
    main()
