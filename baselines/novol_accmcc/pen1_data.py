"""PEN data helpers (extracted from Pattern_Mining/main_pen.py)."""
from __future__ import annotations

import csv
import json
import logging
import os
import pickle as pkl
import random
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import Dataset as TorchDataset

LOGGER = logging.getLogger("pen1_data")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_ASCII_WORDCHAR_RE = re.compile(r"[A-Za-z0-9_@$']")
TOKEN_RE_EN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z0-9_@$']+|[^\w\s]",
    re.UNICODE,
)

MV_NEUTRAL_LOW = -0.005
MV_STRONG_HIGH = 0.0055

# Seasonal pooled split (align with dual_tf --split_mode seasonal_h1q3q4)
SEASONAL_H1Q3Q4_MONTHS = {
    "train": {1, 2, 3, 4, 5, 6},
    "val": {7, 8, 9},
    "test": {10, 11, 12},
}

# First calendar year for stagger_4y (align with dataset coverage)
STAGGER_4Y_YEAR0 = {
    "CSMD50": 2021,
    "CSMD300": 2021,
    "MASSIVE": 2022,
    "CMIN-CN": 2018,
    "CMIN-US": 2018,
}

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


def log_info(msg: str, *args) -> None:
    LOGGER.info(msg, *args)


def calculate_mcc(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    num = (tp * tn) - (fp * fn)
    den = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    return float(num / den) if den != 0 else 0.0


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Match main_stocknet.set_seed / train.set_all_seeds when deterministic=False."""
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


def date_in_split(d: str, start: str, end: str) -> bool:
    """PEN DataPipe: start <= target_date < end."""
    return start <= d < end


def date_in_seasonal_h1q3q4(d: str, mode: str) -> bool:
    try:
        m = int(str(d)[5:7])
    except (TypeError, ValueError, IndexError):
        return False
    return m in SEASONAL_H1Q3Q4_MONTHS.get(str(mode), set())


def date_in_stagger_4y(d: str, mode: str, year0: int) -> bool:
    """Same as dual_tf.data.date_in_stagger_4y."""
    try:
        y = int(str(d)[:4])
        m = int(str(d)[5:7])
    except (TypeError, ValueError, IndexError):
        return False
    r = y - int(year0)
    md = str(mode)
    if md == "train":
        return (
            (r == 0 and 1 <= m <= 9)
            or (r == 1 and 4 <= m <= 12)
            or (r == 2 and 7 <= m <= 12)
            or (r == 3 and 1 <= m <= 3)
        )
    if md == "val":
        return (
            (r == 0 and 10 <= m <= 12)
            or (r == 2 and 1 <= m <= 3)
            or (r == 3 and 4 <= m <= 7)
        )
    if md == "test":
        return (
            (r == 1 and 1 <= m <= 3)
            or (r == 2 and 4 <= m <= 6)
            or (r == 3 and 8 <= m <= 12)
        )
    return False


def date_in_quarter_rr_311(d: str, mode: str, year0: int) -> bool:
    """Same as dual_tf.data.build_quarter_rr_311_roles for one date.

    Differs from stagger_4y only on Y4: val 4-6 (not 4-7), test 7-12 (not 8-12).
    """
    try:
        y = int(str(d)[:4])
        m = int(str(d)[5:7])
    except (TypeError, ValueError, IndexError):
        return False
    r = y - int(year0)
    md = str(mode)
    if md == "train":
        return (
            (r == 0 and 1 <= m <= 9)
            or (r == 1 and 4 <= m <= 12)
            or (r == 2 and 7 <= m <= 12)
            or (r == 3 and 1 <= m <= 3)
        )
    if md == "val":
        return (
            (r == 0 and 10 <= m <= 12)
            or (r == 2 and 1 <= m <= 3)
            or (r == 3 and 4 <= m <= 6)
        )
    if md == "test":
        return (
            (r == 1 and 1 <= m <= 3)
            or (r == 2 and 4 <= m <= 6)
            or (r == 3 and 7 <= m <= 12)
        )
    return False


def stagger_4y_year0_for_kind(kind: str) -> int:
    if kind in STAGGER_4Y_YEAR0:
        return int(STAGGER_4Y_YEAR0[kind])
    # fallback: earliest train_start year in DATASET_SPLITS
    return int(DATASET_SPLITS[kind]["train_start"][:4])


def quarter_rr_311_defer_months(kind: str, year0: int | None = None) -> set[str]:
    """Y4 July (year0+3-07): flat-chunk separately so Aug–Dec mates match stagger."""
    y0 = int(year0) if year0 is not None else stagger_4y_year0_for_kind(kind)
    return {f"{y0 + 3}-07"}


def pack_valid_dates_flat(
    valid_dates: list[str],
    seg_length: int,
    *,
    keep_tail: bool,
    defer_months: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Legacy flat seg packing; optionally defer months then append (July jdefer)."""
    seg = max(1, int(seg_length))

    def _flat(dates: list[str]) -> list[str]:
        n = len(dates)
        if n <= 0:
            return []
        n_win = (n + seg - 1) // seg if keep_tail else n // seg
        out: list[str] = []
        for i in range(n_win):
            start = i * seg
            end = min(start + seg, n) if keep_tail else start + seg
            out.extend(dates[start:end])
        return out

    dates = list(valid_dates)
    if not defer_months:
        return _flat(dates)
    defer_set = {str(m)[:7] for m in defer_months}
    head = [d for d in dates if str(d)[:7] not in defer_set]
    tail = [d for d in dates if str(d)[:7] in defer_set]
    return _flat(head) + _flat(tail)


ROLE_SPLIT_MODES = frozenset({"seasonal_h1q3q4", "stagger_4y", "quarter_rr_311"})


def date_in_mode(
    d: str,
    mode: str,
    kind: str,
    split_mode: str = "contiguous",
    *,
    inclusive_end: bool = True,
    year0: int | None = None,
) -> bool:
    """Filter target dates by contiguous DATASET_SPLITS or seasonal / stagger modes."""
    sm = str(split_mode or "contiguous").strip().lower()
    if sm == "seasonal_h1q3q4":
        return date_in_seasonal_h1q3q4(d, mode)
    if sm == "stagger_4y":
        y0 = int(year0) if year0 is not None else stagger_4y_year0_for_kind(kind)
        return date_in_stagger_4y(d, mode, y0)
    if sm == "quarter_rr_311":
        y0 = int(year0) if year0 is not None else stagger_4y_year0_for_kind(kind)
        return date_in_quarter_rr_311(d, mode, y0)
    start, end = split_range_for_mode(mode, kind)
    if inclusive_end:
        return start <= d <= end
    return start <= d < end


def split_range_for_mode(mode: str, kind: str) -> tuple[str, str]:
    splits = DATASET_SPLITS[kind]
    if mode == "train":
        return splits["train_start"], splits["train_end"]
    if mode == "val":
        return splits["val_start"], splits["val_end"]
    return splits["test_start"], splits["test_end"]


def is_strong_mv(mv: float) -> bool:
    return mv > MV_STRONG_HIGH or mv < MV_NEUTRAL_LOW


def binary_mv_label(mv: float) -> float:
    return 1.0 if mv > 0.0 else 0.0


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
    return int(np.argmax(_mv_to_onehot(mv, y_size)))


def load_price(directory_path: str, use_volume: bool = False):
    """beifen_0602/main_cmin_cn.py load_price (+ optional volume return channel)."""
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


def load_all_data(dataset_root: str, use_volume: bool = False, **_) -> dict:
    """Price-only bundle for shared scripts (e.g. main_single price baselines)."""
    price_dir = os.path.join(os.path.abspath(dataset_root), "price", "preprocessed")
    stock_data, label, raw_line = load_price(price_dir, use_volume=use_volume)
    return {
        "stock_data": stock_data,
        "label": label,
        "raw_line": raw_line,
        "use_volume": bool(use_volume),
        "price_feat_dim": 4 if use_volume else 3,
    }


def _csv_row_source_contains_wind(row: list) -> bool:
    if len(row) < 5:
        return False
    return "wind" in str(row[4]).lower()


def load_news_and_types_cmin(news_dir: str, news_type_dir: str):
    """beifen_0602: CMIN CSV news + llm_extract types, Wind source only."""
    news_texts = defaultdict(lambda: defaultdict(dict))
    news_types = defaultdict(lambda: defaultdict(dict))
    if not os.path.isdir(news_dir):
        return news_texts, news_types
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
                        if len(row) >= 4 and _csv_row_source_contains_wind(row):
                            id_to_text[str(idx)] = row[3]
            except Exception:
                continue
            for news_id, news_type in type_dict.items():
                if news_id in id_to_text:
                    news_texts[company][date_str][news_id] = id_to_text[news_id]
                    news_types[company][date_str][news_id] = news_type
    return news_texts, news_types


def _get_content_and_type_csmd(item):
    if not isinstance(item, dict):
        return None, None
    content = item.get("内容") or item.get("content") or ""
    dtype = item.get("类型") or item.get("type") or ""
    return (content.strip() if content else None), (dtype.strip() if dtype else None)


def load_news_and_types_csmd(llm_extract_news_type_dir: str):
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


def load_dataset_bundle(dataset_root: str) -> dict:
    """beifen_0602 load_cmin_cn_all_data 的 PEN 所需子集（price + news）。"""
    kind = detect_dataset_kind(dataset_root)
    price_dir = os.path.join(dataset_root, "price", "preprocessed")
    stock_data, label, raw_line = load_price(price_dir)
    if kind in ("CSMD50", "CSMD300"):
        llm_dir = os.path.join(dataset_root, "llm_extract", "news_type")
        news_texts, news_types = load_news_and_types_csmd(llm_dir)
    else:
        news_dir = os.path.join(dataset_root, "news")
        news_type_dir = os.path.join(dataset_root, "llm_extract", "news_type")
        news_texts, news_types = load_news_and_types_cmin(news_dir, news_type_dir)
    news_texts = {c: {d: dict(msgs) for d, msgs in by_date.items()} for c, by_date in news_texts.items()}
    news_types = {c: {d: dict(msgs) for d, msgs in by_date.items()} for c, by_date in news_types.items()}
    return {
        "stock_data": stock_data,
        "label": label,
        "raw_line": raw_line,
        "news_texts": news_texts,
        "news_types": news_types,
        "dataset_kind": kind,
    }


def resolve_vocab_path(dataset_root: str) -> str:
    kind = detect_dataset_kind(dataset_root)
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo_dict = os.path.normpath(os.path.join(_here, "../../dict"))
    _home_dict = "/home/zhaokx/Pattern/Pattern_Mining/dict"
    candidates = {
        "CSMD50": [
            os.path.join(_home_dict, "dict_csmd.pkl"),
            os.path.join(_repo_dict, "dict_csmd.pkl"),
            "./dict/dict_csmd.pkl",
        ],
        "CSMD300": [
            os.path.join(_home_dict, "dict_csmd.pkl"),
            os.path.join(_repo_dict, "dict_csmd.pkl"),
            "./dict/dict_csmd.pkl",
        ],
        "CMIN-CN": [
            os.path.join(_home_dict, "dict_cn.pkl"),
            os.path.join(_repo_dict, "dict_cn.pkl"),
            "./dict/dict_cn.pkl",
        ],
        "CMIN-US": [
            os.path.join(_home_dict, "dict_us.pkl"),
            os.path.join(_repo_dict, "dict_us.pkl"),
            "./dict/dict_us.pkl",
        ],
        "MASSIVE": [
            os.path.join(_home_dict, "dict_massive.pkl"),
            os.path.join(_repo_dict, "dict_massive.pkl"),
            "./dict/dict_massive.pkl",
            os.path.join(_home_dict, "dict_us.pkl"),
            os.path.join(_repo_dict, "dict_us.pkl"),
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
    return "en" if detect_dataset_kind(dataset_root) in ("CMIN-US", "MASSIVE") else "cn"


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


class Pen1Dataset(TorchDataset):
    """
    One sample = one (stock, target_date).
    Aligns with PEN-main DataPipe; max_news_fail_days controls news strictness.
    """

    def __init__(
        self,
        dataset_root: str,
        mode: str = "train",
        max_n_days: int = 5,
        max_n_msgs: int = 20,
        max_n_words: int = 30,
        y_size: int = 2,
        filter_neutral: bool = True,
        use_strong_only: bool = False,
        max_news_fail_days: int = 5,
        shuffle_build: bool = False,
        seed: int = 37,
        require_any_news: bool = False,
        split_mode: str = "contiguous",
    ):
        assert mode in {"train", "val", "test"}
        self.dataset_root = dataset_root
        self.mode = mode
        self.max_n_days = max_n_days
        self.max_n_msgs = max_n_msgs
        self.max_n_words = max_n_words
        self.y_size = y_size
        self.filter_neutral = filter_neutral
        self.use_strong_only = use_strong_only
        self.max_news_fail_days = max_news_fail_days
        self.require_any_news = require_any_news
        self.split_mode = str(split_mode or "contiguous").strip().lower() or "contiguous"

        kind = detect_dataset_kind(dataset_root)
        self.dataset_kind = kind
        self.start_date, self.end_date = split_range_for_mode(mode, kind)

        bundle = load_dataset_bundle(dataset_root)
        self.stock_data = bundle["stock_data"]
        self.label = bundle["label"]
        self.news_texts = bundle["news_texts"]
        self.companies = sorted(self.stock_data.keys())
        self.company2id = {c: i for i, c in enumerate(self.companies)}

        vocab_path = resolve_vocab_path(dataset_root)
        self.vocab = pkl.load(open(vocab_path, "rb"))
        self.vocab_unk_id = int(self.vocab.get("<unk>", 0))
        self.tokenize_mode = infer_tokenize_mode(dataset_root)

        self.samples: list[dict] = []
        rng = random.Random(seed)
        for comp in self.companies:
            targets = self._target_dates_for_stock(comp)
            if shuffle_build:
                rng.shuffle(targets)
            for td in targets:
                sample = self._build_sample(comp, td)
                if sample is not None:
                    self.samples.append(sample)
        if shuffle_build and mode == "train":
            rng.shuffle(self.samples)
        log_info(
            "Pen1Dataset %s: kind=%s n=%d max_news_fail_days=%d",
            mode,
            kind,
            len(self.samples),
            max_news_fail_days,
        )

    def _target_dates_for_stock(self, comp: str) -> list[str]:
        day_dict = self.stock_data.get(comp, {})
        kind = getattr(self, "dataset_kind", None) or detect_dataset_kind(self.dataset_root)
        return [
            d
            for d in sorted(day_dict.keys())
            if date_in_mode(
                d,
                self.mode,
                kind,
                getattr(self, "split_mode", "contiguous"),
                inclusive_end=False,
            )
        ]

    def _convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [int(self.vocab.get(t, self.vocab_unk_id)) for t in tokens]

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
                prices.append(list(day_dict[d_str])[:3])
                mv_percents.append(_mv_to_class(mv, self.y_size))
            elif d < d_t_min:
                prices.append(list(day_dict[d_str])[:3])
                mv_percents.append(_mv_to_class(mv, self.y_size))
                break

        t_len = len(ts)
        if t_len == 0 or len(ys) != t_len or len(prices) != t_len or len(mv_percents) != t_len:
            return None
        for item in (ts, ys, mv_percents, prices):
            item.reverse()
        if self.use_strong_only and main_mv is not None and not is_strong_mv(main_mv):
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
                unaligned.append([d, word_mat[:msg_id], ss_index_vec[:msg_id], n_word_vec[:msg_id], msg_id])
            d -= timedelta(days=1)
        unaligned.reverse()
        return unaligned


def _pen1_align_news(ts, t_len, unaligned_corpora, max_n_msgs, max_n_words, max_news_fail_days):
    aligned_word = np.zeros((t_len, max_n_msgs, max_n_words), dtype=np.int32)
    aligned_ss = np.zeros((t_len, max_n_msgs), dtype=np.int32)
    aligned_n_words = np.zeros((t_len, max_n_msgs), dtype=np.int32)
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

    n_fails = sum(1 for n_list in aligned_n_msgs_lists if sum(n_list) == 0)
    if n_fails > max_news_fail_days:
        return None

    for t in range(t_len):
        if not aligned_msgs[t]:
            continue
        msgs = np.vstack(aligned_msgs[t])
        ss_indices = np.hstack(aligned_ss_list[t])
        n_word = np.hstack(aligned_n_words_lists[t])
        n_msgs = min(sum(aligned_n_msgs_lists[t]), max_n_msgs)
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


def _pen1_build_sample_impl(self: Pen1Dataset, comp: str, target_date_str: str):
    main_target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    prices_and_ts = self._get_prices_and_ts(comp, main_target_date)
    if not prices_and_ts:
        return None
    unaligned = self._get_unaligned_corpora(comp, main_target_date)
    if self.require_any_news and not unaligned:
        return None
    aligned = _pen1_align_news(
        prices_and_ts["ts"],
        prices_and_ts["T"],
        unaligned,
        self.max_n_msgs,
        self.max_n_words,
        self.max_news_fail_days,
    )
    if aligned is None:
        return None
    if self.require_any_news and int(np.sum(aligned["n_msgs"])) == 0:
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


Pen1Dataset._build_sample = _pen1_build_sample_impl  # type: ignore[method-assign]


def collate_pen1_batch(batch, max_n_days: int):
    b = len(batch)
    m = batch[0]["word_ids"].shape[1]
    w = batch[0]["word_ids"].shape[2]
    y_size = batch[0]["y"].shape[-1]

    word_ids = torch.zeros(b, max_n_days, m, w, dtype=torch.long)
    ss_index = torch.zeros(b, max_n_days, m, dtype=torch.long)
    n_words = torch.zeros(b, max_n_days, m, dtype=torch.long)
    n_msgs = torch.zeros(b, max_n_days, dtype=torch.long)
    price_dim = int(np.asarray(batch[0]["price"]).shape[-1])
    price = torch.zeros(b, max_n_days, price_dim, dtype=torch.float32)
    y = torch.zeros(b, max_n_days, y_size, dtype=torch.float32)
    t_idx = torch.zeros(b, dtype=torch.long)
    stock_id = torch.zeros(b, dtype=torch.long)
    companies, dates = [], []

    for i, item in enumerate(batch):
        t = int(item["T"])
        word_ids[i, :t] = torch.as_tensor(item["word_ids"], dtype=torch.long)
        ss_index[i, :t] = torch.as_tensor(item["ss_index"], dtype=torch.long)
        n_words[i, :t] = torch.as_tensor(item["n_words"], dtype=torch.long)
        n_msgs[i, :t] = torch.as_tensor(item["n_msgs"], dtype=torch.long)
        price[i, :t] = torch.as_tensor(item["price"], dtype=torch.float32)
        y[i, :t] = torch.as_tensor(item["y"], dtype=torch.float32)
        t_idx[i] = t
        stock_id[i] = item["stock_id"]
        companies.append(item["company"])
        dates.append(item["target_date"])

    out = {
        "word_ids": word_ids,
        "ss_index": ss_index,
        "n_words": n_words,
        "n_msgs": n_msgs,
        "price": price,
        "y": y,
        "t_idx": t_idx,
        "stock_id": stock_id,
        "companies": companies,
        "dates": dates,
    }
    if batch and "strong" in batch[0]:
        out["strong"] = torch.tensor([bool(x["strong"]) for x in batch], dtype=torch.bool)
    return out


Pen1Dataset.__len__ = lambda self: len(self.samples)  # type: ignore[method-assign]


def __pen1_getitem__(self, idx):
    s = self.samples[idx]
    return {
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


Pen1Dataset.__getitem__ = __pen1_getitem__  # type: ignore[method-assign]



