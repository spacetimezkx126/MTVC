"""MTVC experiment utilities."""

from __future__ import annotations
import argparse
import calendar
import csv
import hashlib
import json
import os
import pickle
import pickle as pkl
import re
import sys
import types
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, BertTokenizer

_INPUT_ORDER_ROOTS = {}
_INPUT_ORDER_TABLES = {}


def _list_input_directory(path):
    from pathlib import Path

    folder = Path(path).resolve()
    actual = [name for name in os.listdir(folder) if name != ".mtvc_file_order.json"]
    if folder not in _INPUT_ORDER_ROOTS:
        root = next(
            (p for p in (folder, *folder.parents) if (p / ".mtvc_file_order.json").is_file()), None
        )
        _INPUT_ORDER_ROOTS[folder] = root
    root = _INPUT_ORDER_ROOTS[folder]
    if root is None:
        return actual
    if root not in _INPUT_ORDER_TABLES:
        _INPUT_ORDER_TABLES[root] = json.loads((root / ".mtvc_file_order.json").read_text())[
            "directories"
        ]
    relative = folder.relative_to(root).as_posix()
    order = _INPUT_ORDER_TABLES[root].get(relative)
    if order is None:
        return actual
    if set(order) != set(actual):
        raise RuntimeError(
            "Dataset files differ from the recorded input-order manifest: " + relative
        )
    return list(order)


_DIR = os.path.dirname(os.path.abspath(__file__))
_PATTERN_ROOT = os.environ.get("DUAL_TF_MODEL_DIR", os.path.dirname(os.path.dirname(_DIR)))


def _dict_search_dirs() -> list[str]:
    dirs: list[str] = []
    for d in (
        os.path.join(_PATTERN_ROOT, "dict"),
        os.path.join(_DIR, "dict"),
        "dict",
        "dict",
        os.path.join(_PATTERN_ROOT, "dict"),
    ):
        if d not in dirs:
            dirs.append(d)
    return dirs


def _primary_dict_dir() -> str:
    for d in _dict_search_dirs():
        if os.path.isdir(d):
            return d
    return os.path.join(_PATTERN_ROOT, "dict")


DICT_DIR = _primary_dict_dir()


def resolve_vocab_path(path: str | None, *fallback_names: str) -> str:
    """Resolve vocab path."""
    candidates: list[str] = []
    if path:
        candidates.append(path)
        if not os.path.isabs(path):
            candidates.append(os.path.join(_DIR, path))
        for root in _dict_search_dirs():
            candidates.append(os.path.join(root, os.path.basename(path)))
    for name in fallback_names:
        for root in _dict_search_dirs():
            candidates.append(os.path.join(root, name))
    seen: set[str] = set()
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"vocab not found, tried: {list(seen)}")


def is_us_vocab_path(vocab_path: str) -> bool:
    """Is us vocab path."""
    base = os.path.basename(vocab_path).lower()
    return "dict_us" in base or "dict_massive" in base or "dict_stocknet" in base


def is_cn_word_vocab_path(vocab_path: str) -> bool:
    """Is cn word vocab path."""
    base = os.path.basename(vocab_path).lower()
    return "jieba" in base or "sogou_word" in base or "csmd_word" in base


def is_word_vocab_path(vocab_path: str) -> bool:
    return is_us_vocab_path(vocab_path) or is_cn_word_vocab_path(vocab_path)


def _lookup_vocab_word(vocab: dict, word: str, unk_id: int) -> int:
    if word in vocab:
        return int(vocab[word])
    lower = word.lower()
    if lower in vocab:
        return int(vocab[lower])
    stripped = word.strip(".,;:!?\"'()[]{}")
    if stripped in vocab:
        return int(vocab[stripped])
    stripped_lower = stripped.lower()
    if stripped_lower in vocab:
        return int(vocab[stripped_lower])
    return int(unk_id)


def text_to_vocab_ids(
    text: str | None,
    vocab: dict,
    *,
    unk_id: int,
    max_len: int,
    word_level: bool = False,
    use_jieba: bool = False,
) -> list[int]:
    s = "" if text is None else str(text)
    if not word_level:
        return [int(vocab.get(ch, unk_id)) for ch in s][:max_len]
    ids: list[int] = []
    if use_jieba:
        import jieba

        tokens = jieba.lcut(s)
    else:
        tokens = []
        for raw in s.split():
            word = raw.strip(".,;:!?\"'()[]{}")
            if word:
                tokens.append(word)
    for word in tokens:
        if not word or word.isspace():
            continue
        ids.append(_lookup_vocab_word(vocab, word, unk_id))
        if len(ids) >= max_len:
            break
    return ids


def pad_vocab_ids(ids: list[int], pad_id: int, max_len: int) -> tuple[list[int], list[int]]:
    attn = [1] * len(ids)
    pad_len = max_len - len(ids)
    if pad_len > 0:
        ids = ids + [pad_id] * pad_len
        attn = attn + [0] * pad_len
    return (ids, attn)


NUM_VIRTUAL_NEWS_TYPES = 13
_INDIVIDUAL_SUBTYPE_NAMES = {
    1: "公司经营",
    2: "资本运作",
    3: "股东与股权变动",
    4: "交易与市场行为",
    5: "机构与资金动向",
    6: "战略合作与对外关系",
    7: "风险负面",
    8: "分红与利润分配",
    9: "监管与合规",
    10: "行业与政策影响",
    11: "其他",
}
NUM_RULE_TREND = 8
RULE_TREND_CN_TO_ID = {
    "冲高回落": 0,
    "宽幅震荡": 1,
    "弱势上涨": 2,
    "弱势下跌": 3,
    "强势上涨": 4,
    "强势下跌": 5,
    "探底回升": 6,
    "窄幅横盘": 7,
}
RULE_TREND_ID_TO_CN = {v: k for (k, v) in RULE_TREND_CN_TO_ID.items()}
RULE_TREND_CN_TO_EN = {
    "冲高回落": "Peak and Pullback",
    "宽幅震荡": "Wide Range Fluctuation",
    "弱势上涨": "Weak Uptrend",
    "弱势下跌": "Weak Downtrend",
    "强势上涨": "Strong Uptrend",
    "强势下跌": "Strong Downtrend",
    "探底回升": "Bottom and Rebound",
    "窄幅横盘": "Narrow Range",
}
RULE_TREND_ID_TO_EN = {tid: RULE_TREND_CN_TO_EN[cn] for (cn, tid) in RULE_TREND_CN_TO_ID.items()}
SEASONAL_H1Q3Q4_MONTHS: dict[str, set[int]] = {
    "train": {1, 2, 3, 4, 5, 6},
    "val": {7, 8, 9},
    "test": {10, 11, 12},
}


def date_in_seasonal_h1q3q4(d: str, mode: str) -> bool:
    """True if YYYY-MM-DD falls in seasonal months for mode."""
    try:
        m = int(str(d)[5:7])
    except (TypeError, ValueError, IndexError):
        return False
    return m in SEASONAL_H1Q3Q4_MONTHS.get(str(mode), set())


def date_in_stagger_4y(d: str, mode: str, year0: int) -> bool:
    """
    4-year staggered calendar split (pooled), relative to year0 (= first year):
      Y1: train 1-9,  val 10-12,            test → Y2 1-3
      Y2: train 4-12,                       test 1-3; val → Y3 1-3
      Y3: train 7-12, val 1-3,              test 4-6; train continues → Y4 1-3
      Y4: train 1-3,  val 4-7,              test 8-12
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
            r == 0
            and 1 <= m <= 9
            or (r == 1 and 4 <= m <= 12)
            or (r == 2 and 7 <= m <= 12)
            or (r == 3 and 1 <= m <= 3)
        )
    if md == "val":
        return r == 0 and 10 <= m <= 12 or (r == 2 and 1 <= m <= 3) or (r == 3 and 4 <= m <= 7)
    if md == "test":
        return r == 1 and 1 <= m <= 3 or (r == 2 and 4 <= m <= 6) or (r == 3 and 8 <= m <= 12)
    return False


def _iso_week_key(d: str) -> tuple[int, int]:
    """(iso_year, iso_week) for YYYY-MM-DD."""
    dt = datetime.strptime(str(d)[:10], "%Y-%m-%d")
    iso = dt.isocalendar()
    return (int(iso[0]), int(iso[1]))


def build_week_roundrobin_roles(global_dates: list[str]) -> dict[str, str]:
    """
    ISO-week round-robin roles with ratio train:val:test = 6:2:2.

    Ordered unique ISO weeks w0,w1,... get:
      week_index % 10 in {0..5} -> train
      week_index % 10 in {6,7}  -> val
      week_index % 10 in {8,9}  -> test

    Timeline: T×6 V V Te Te | T×6 V V Te Te | ...
    so train blocks often sit *after* a val/test block.
    """
    if not global_dates:
        return {}
    week_of: dict[str, tuple[int, int]] = {}
    weeks_ordered: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for d in global_dates:
        k = _iso_week_key(d)
        week_of[d] = k
        if k not in seen:
            seen.add(k)
            weeks_ordered.append(k)
    week_idx = {k: i for (i, k) in enumerate(weeks_ordered)}
    roles: dict[str, str] = {}
    for d in global_dates:
        i = week_idx[week_of[d]] % 10
        if i <= 5:
            roles[d] = "train"
        elif i <= 7:
            roles[d] = "val"
        else:
            roles[d] = "test"
    return roles


def date_in_week_roundrobin(
    d: str,
    mode: str,
    global_dates: list[str],
    *,
    price_window: int = 5,
    roles: dict[str, str] | None = None,
    date_to_pos: dict[str, int] | None = None,
) -> bool:
    """
    Label-date membership for week_roundrobin (6:2:2), with train lookback purge.

    Rules
    -----
    1) Label day d must have role(d)==mode (never train on val/test labels).
    2) Features for day d use the previous ``price_window`` trading days on the
       global calendar (same convention as dataset price/news lookback: bars
       are in [prev(d) ...], not including d).
    3) For mode in {val, test}: allow lookback into any past calendar days
       (including earlier train / val). This matches contiguous splits where
       test may look into the end of val.
    4) For mode==train when a train week sits *after* val/test:
       if any lookback day has role in {val, test}, **drop** this train sample
       (purge / embargo of length ``price_window``).

    Consequence: right after each V/Te block, the first ~price_window trading
    days that fall in the next train weeks are removed from train.
    """
    md = str(mode)
    if not global_dates:
        return False
    if roles is None:
        roles = build_week_roundrobin_roles(global_dates)
    if roles.get(str(d)[:10]) != md:
        return False
    if md != "train":
        return True
    pw = max(1, int(price_window))
    if date_to_pos is None:
        date_to_pos = {x: i for (i, x) in enumerate(global_dates)}
    pos = date_to_pos.get(str(d)[:10])
    if pos is None or pos < 1:
        return False
    lo = max(0, pos - pw)
    for lb in global_dates[lo:pos]:
        if roles.get(lb) in ("val", "test"):
            return False
    return True


def week_roundrobin_split_stats(
    global_dates: list[str], *, price_window: int = 5
) -> dict[str, Any]:
    """Counts for logging: raw roles vs train-after-purge."""
    roles = build_week_roundrobin_roles(global_dates)
    date_to_pos = {x: i for (i, x) in enumerate(global_dates)}
    raw = {"train": 0, "val": 0, "test": 0}
    for d in global_dates:
        raw[roles[d]] = raw.get(roles[d], 0) + 1
    train_kept = [
        d
        for d in global_dates
        if date_in_week_roundrobin(
            d,
            "train",
            global_dates,
            price_window=price_window,
            roles=roles,
            date_to_pos=date_to_pos,
        )
    ]
    val_n = sum((1 for d in global_dates if roles[d] == "val"))
    test_n = sum((1 for d in global_dates if roles[d] == "test"))
    pw = max(1, int(price_window))
    return {
        "n_dates": len(global_dates),
        "n_iso_weeks": len({_iso_week_key(d) for d in global_dates}),
        "raw_role_days": raw,
        "train_after_purge": len(train_kept),
        "train_purged": raw["train"] - len(train_kept),
        "val": val_n,
        "test": test_n,
        "price_window": pw,
    }


def timefrac_622_cut_indices(n: int) -> tuple[int, int]:
    """Return (train_end, val_end) exclusive indices for 60%/20%/20% on n days."""
    n = int(n)
    if n <= 0:
        return (0, 0)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    train_end = max(1, min(train_end, n - 2)) if n >= 3 else max(0, n - 2)
    val_end = max(train_end + 1, min(val_end, n - 1)) if n >= 3 else max(train_end, n - 1)
    return (train_end, val_end)


def date_in_timefrac_622(
    d: str,
    mode: str,
    global_dates: list[str],
    *,
    date_to_pos: dict[str, int] | None = None,
    train_end: int | None = None,
    val_end: int | None = None,
) -> bool:
    """
    Chronological whole-dataset split by trading-day index: 60% train / 20% val / 20% test.

    No interleaved train-after-holdout, so no lookback purge on train.
    Val/test may still look back into earlier calendar days (cross-split lookback).
    """
    if not global_dates:
        return False
    if date_to_pos is None:
        date_to_pos = {x: i for (i, x) in enumerate(global_dates)}
    pos = date_to_pos.get(str(d)[:10])
    if pos is None:
        return False
    if train_end is None or val_end is None:
        (train_end, val_end) = timefrac_622_cut_indices(len(global_dates))
    md = str(mode)
    if md == "train":
        return 0 <= pos < int(train_end)
    if md == "val":
        return int(train_end) <= pos < int(val_end)
    if md == "test":
        return int(val_end) <= pos < len(global_dates)
    return False


def timefrac_622_split_stats(global_dates: list[str]) -> dict[str, Any]:
    (train_end, val_end) = timefrac_622_cut_indices(len(global_dates))
    n = len(global_dates)
    return {
        "n_dates": n,
        "train_end_idx": train_end,
        "val_end_idx": val_end,
        "train_range": (global_dates[0], global_dates[train_end - 1]) if train_end > 0 else None,
        "val_range": (
            (global_dates[train_end], global_dates[val_end - 1]) if val_end > train_end else None
        ),
        "test_range": (global_dates[val_end], global_dates[-1]) if val_end < n else None,
        "counts": {
            "train": train_end,
            "val": max(0, val_end - train_end),
            "test": max(0, n - val_end),
        },
    }


def _year_month_key(d: str) -> tuple[int, int]:
    """(year, month) for YYYY-MM-DD."""
    s = str(d)[:10]
    return (int(s[:4]), int(s[5:7]))


def build_month_roundrobin_roles(global_dates: list[str]) -> dict[str, str]:
    """
    Calendar-month round-robin roles with ratio train:val:test = 4:1:1.

    Ordered unique (year, month) m0,m1,... get:
      month_index % 6 in {0,1,2,3} -> train
      month_index % 6 == 4         -> val
      month_index % 6 == 5         -> test

    Timeline: T T T T V Te | T T T T V Te | ... (each letter = one calendar month).
    """
    if not global_dates:
        return {}
    month_of: dict[str, tuple[int, int]] = {}
    months_ordered: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for d in global_dates:
        k = _year_month_key(d)
        month_of[d] = k
        if k not in seen:
            seen.add(k)
            months_ordered.append(k)
    month_idx = {k: i for (i, k) in enumerate(months_ordered)}
    roles: dict[str, str] = {}
    for d in global_dates:
        i = month_idx[month_of[d]] % 6
        if i <= 3:
            roles[d] = "train"
        elif i == 4:
            roles[d] = "val"
        else:
            roles[d] = "test"
    return roles


def date_in_month_roundrobin(
    d: str,
    mode: str,
    global_dates: list[str],
    *,
    price_window: int = 5,
    roles: dict[str, str] | None = None,
    date_to_pos: dict[str, int] | None = None,
) -> bool:
    """
    Label-date membership for month_roundrobin (4mo train / 1mo val / 1mo test),
    with the same train lookback purge as week_roundrobin.

    After each val+test month pair, train days whose ``price_window`` lookback
    still overlaps those holdout months are dropped.
    """
    md = str(mode)
    if not global_dates:
        return False
    if roles is None:
        roles = build_month_roundrobin_roles(global_dates)
    if roles.get(str(d)[:10]) != md:
        return False
    if md != "train":
        return True
    pw = max(1, int(price_window))
    if date_to_pos is None:
        date_to_pos = {x: i for (i, x) in enumerate(global_dates)}
    pos = date_to_pos.get(str(d)[:10])
    if pos is None or pos < 1:
        return False
    lo = max(0, pos - pw)
    for lb in global_dates[lo:pos]:
        if roles.get(lb) in ("val", "test"):
            return False
    return True


def month_roundrobin_split_stats(
    global_dates: list[str], *, price_window: int = 5
) -> dict[str, Any]:
    """Counts for logging: raw month roles vs train-after-purge."""
    roles = build_month_roundrobin_roles(global_dates)
    date_to_pos = {x: i for (i, x) in enumerate(global_dates)}
    raw = {"train": 0, "val": 0, "test": 0}
    for d in global_dates:
        raw[roles[d]] = raw.get(roles[d], 0) + 1
    train_kept = [
        d
        for d in global_dates
        if date_in_month_roundrobin(
            d,
            "train",
            global_dates,
            price_window=price_window,
            roles=roles,
            date_to_pos=date_to_pos,
        )
    ]
    val_n = sum((1 for d in global_dates if roles[d] == "val"))
    test_n = sum((1 for d in global_dates if roles[d] == "test"))
    pw = max(1, int(price_window))
    return {
        "n_dates": len(global_dates),
        "n_months": len({_year_month_key(d) for d in global_dates}),
        "raw_role_days": raw,
        "train_after_purge": len(train_kept),
        "train_purged": raw["train"] - len(train_kept),
        "val": val_n,
        "test": test_n,
        "price_window": pw,
    }


def date_in_roles_with_purge(
    d: str,
    mode: str,
    global_dates: list[str],
    roles: dict[str, str],
    *,
    price_window: int = 5,
    date_to_pos: dict[str, int] | None = None,
    train_lookback_purge: bool = True,
) -> bool:
    """Generic: label-day role match + optional train lookback purge vs val/test."""
    md = str(mode)
    key = str(d)[:10]
    if roles.get(key) != md:
        return False
    if md != "train" or not train_lookback_purge:
        return True
    pw = max(1, int(price_window))
    if date_to_pos is None:
        date_to_pos = {x: i for (i, x) in enumerate(global_dates)}
    pos = date_to_pos.get(key)
    if pos is None or pos < 1:
        return False
    lo = max(0, pos - pw)
    for lb in global_dates[lo:pos]:
        if roles.get(lb) in ("val", "test"):
            return False
    return True


def build_day_rr_411_roles(global_dates: list[str]) -> dict[str, str]:
    """Deprecated alias — use day_rr_1022 (4:1:1 day cadence is incompatible with pw=5 purge)."""
    return build_day_rr_1022_roles(global_dates)


def build_day_rr_1022_roles(global_dates: list[str]) -> dict[str, str]:
    """
    NEW: trading-day round-robin 10:2:2 (finer than weeks, long enough for lookback purge).

    day_index % 14 in {0..9} -> train; {10,11} -> val; {12,13} -> test.
    After each V+Te (4 days), ~price_window train days are purged; remaining train stays.
    """
    roles: dict[str, str] = {}
    for i, d in enumerate(global_dates):
        r = i % 14
        if r <= 9:
            roles[d] = "train"
        elif r <= 11:
            roles[d] = "val"
        else:
            roles[d] = "test"
    return roles


def _year_quarter_key(d: str) -> tuple[int, int]:
    s = str(d)[:10]
    (y, m) = (int(s[:4]), int(s[5:7]))
    return (y, (m - 1) // 3 + 1)


def build_quarter_rr_211_roles(global_dates: list[str]) -> dict[str, str]:
    """
    NEW: calendar-quarter round-robin 2:1:1 (unlike fixed seasonal H1/Q3/Q4).

    Ordered unique (year, quarter) % 4: {0,1}->train, 2->val, 3->test.
    Timeline: T T V Te | T T V Te | ...
    """
    if not global_dates:
        return {}
    q_of: dict[str, tuple[int, int]] = {}
    ordered: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for d in global_dates:
        k = _year_quarter_key(d)
        q_of[d] = k
        if k not in seen:
            seen.add(k)
            ordered.append(k)
    idx = {k: i for (i, k) in enumerate(ordered)}
    roles: dict[str, str] = {}
    for d in global_dates:
        i = idx[q_of[d]] % 4
        if i <= 1:
            roles[d] = "train"
        elif i == 2:
            roles[d] = "val"
        else:
            roles[d] = "test"
    return roles


def build_quarter_rr_311_roles(global_dates: list[str], year0: int) -> dict[str, str]:
    """
    Calendar 3:1:1-style block relative to year0 (stagger_4y with Y4 tweak).

      Y1: train 1-9, val 10-12
      Y2: test 1-3, train 4-12
      Y3: val 1-3, test 4-6, train 7-12
      Y4: train 1-3, val 4-6, test 7-12

    Differs from stagger_4y only on Y4: val 4-6 (not 4-7), test 7-12 (not 8-12).
    """
    roles: dict[str, str] = {}
    y0 = int(year0)
    for d in global_dates:
        try:
            y = int(str(d)[:4])
            m = int(str(d)[5:7])
        except (TypeError, ValueError, IndexError):
            continue
        r = y - y0
        if (
            r == 0
            and 1 <= m <= 9
            or (r == 1 and 4 <= m <= 12)
            or (r == 2 and 7 <= m <= 12)
            or (r == 3 and 1 <= m <= 3)
        ):
            roles[d] = "train"
        elif r == 0 and 10 <= m <= 12 or (r == 2 and 1 <= m <= 3) or (r == 3 and 4 <= m <= 6):
            roles[d] = "val"
        elif r == 1 and 1 <= m <= 3 or (r == 2 and 4 <= m <= 6) or (r == 3 and 7 <= m <= 12):
            roles[d] = "test"
    return roles


def build_year_gap_roles(global_dates: list[str], year0: int) -> dict[str, str]:
    """
    NEW: non-contiguous calendar years (gap years).

    Relative year r = y - year0:
      r % 4 in {0, 2} -> train   (e.g. 2022, 2024)
      r % 4 == 1      -> val     (e.g. 2023)
      r % 4 == 3      -> test    (e.g. 2025)

    Train years sit both before and after val -> lookback purge applies.
    """
    roles: dict[str, str] = {}
    y0 = int(year0)
    for d in global_dates:
        try:
            r = int(str(d)[:4]) - y0
        except (TypeError, ValueError):
            continue
        m = r % 4
        if m in (0, 2):
            roles[d] = "train"
        elif m == 1:
            roles[d] = "val"
        else:
            roles[d] = "test"
    return roles


def role_split_stats(
    global_dates: list[str],
    roles: dict[str, str],
    *,
    price_window: int = 5,
    train_lookback_purge: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    date_to_pos = {x: i for (i, x) in enumerate(global_dates)}
    raw = {"train": 0, "val": 0, "test": 0}
    for d in global_dates:
        raw[roles[d]] = raw.get(roles[d], 0) + 1
    train_kept = sum(
        (
            1
            for d in global_dates
            if date_in_roles_with_purge(
                d,
                "train",
                global_dates,
                roles,
                price_window=price_window,
                date_to_pos=date_to_pos,
                train_lookback_purge=train_lookback_purge,
            )
        )
    )
    out = {
        "n_dates": len(global_dates),
        "raw_role_days": raw,
        "train_after_purge": train_kept,
        "train_purged": raw["train"] - train_kept,
        "val": raw["val"],
        "test": raw["test"],
        "price_window": max(1, int(price_window)),
        "train_lookback_purge": bool(train_lookback_purge),
    }
    if extra:
        out.update(extra)
    return out


NEW_ROLE_SPLIT_MODES = frozenset(
    {"day_rr_1022", "day_rr_411", "quarter_rr_211", "quarter_rr_311", "year_gap"}
)


def build_seg_windows(
    valid_dates: list[str],
    global_dates: list[str],
    seg_length: int,
    *,
    keep_tail: bool = True,
    break_on_month: bool = False,
    defer_months: set[str] | frozenset[str] | None = None,
) -> list[list[str]]:
    """Legacy flat chunk of ``valid_dates`` by ``seg_length`` (ignores calendar gaps).

    Role-based splits leave holes; flat slicing glues distant months into one window
    (day_id / virtual_window mates depend on list position). That packing is what
    produced the historical stagger ~0.05–0.07 test MCC.

    ``defer_months`` (YYYY-MM): those days are flat-chunked *separately* and appended
    as extra windows. Used by ``quarter_rr_311`` so Y4 July can sit in test without
    shifting Aug–Dec day_ids relative to stagger_4y (which keeps July in val).

    ``global_dates`` / ``break_on_month`` kept for call-site compatibility; unused.
    """
    del global_dates, break_on_month

    def _flat(dates: list[str]) -> list[list[str]]:
        seg = max(1, int(seg_length))
        n = len(dates)
        if n <= 0 or seg <= 0:
            return []
        n_win = (n + seg - 1) // seg if keep_tail else n // seg
        out: list[list[str]] = []
        for i in range(n_win):
            start = i * seg
            end = min(start + seg, n) if keep_tail else start + seg
            out.append(list(dates[start:end]))
        return out

    dates = list(valid_dates)
    if not defer_months:
        return _flat(dates)
    defer_set = {str(m)[:7] for m in defer_months}
    head = [d for d in dates if str(d)[:7] not in defer_set]
    tail = [d for d in dates if str(d)[:7] in defer_set]
    return _flat(head) + _flat(tail)


def _split_mode_break_on_month(split_mode: str | None) -> bool:
    """Always False: keep legacy flat seg windows (needed for stagger-scale MCC)."""
    del split_mode
    return False


def _defer_months_for_split(
    split_mode: str | None, global_dates: list[str] | None, start_date: str | None = None
) -> set[str] | None:
    """Months to append after flat packing so they don't remap later day_ids.

    quarter_rr_311 puts Y4 July in test; stagger_4y keeps it in val. Deferring
    ``{year0+3}-07`` restores stagger's Aug–Dec window mates while still scoring July.
    """
    sm = str(split_mode or "").strip().lower()
    if sm != "quarter_rr_311":
        return None
    y0 = resolve_split_year0(global_dates, start_date)
    return {f"{y0 + 3}-07"}


def resolve_new_role_split_roles(
    split_mode: str, global_dates: list[str], *, start_date: str | None = None
) -> dict[str, str] | None:
    """Roles for newly introduced split modes (not previously sweeped)."""
    sm = str(split_mode or "").strip().lower()
    if sm in ("day_rr_1022", "day_rr_411"):
        return build_day_rr_1022_roles(global_dates)
    if sm == "quarter_rr_211":
        return build_quarter_rr_211_roles(global_dates)
    if sm == "quarter_rr_311":
        y0 = resolve_split_year0(global_dates, start_date)
        return build_quarter_rr_311_roles(global_dates, y0)
    if sm == "year_gap":
        y0 = resolve_split_year0(global_dates, start_date)
        return build_year_gap_roles(global_dates, y0)
    return None


def resolve_split_year0(dates: list[str] | None, fallback: str | None = None) -> int:
    """First calendar year present in dates (or fallback YYYY-MM-DD / YYYY)."""
    if dates:
        return int(str(dates[0])[:4])
    if fallback:
        return int(str(fallback)[:4])
    raise ValueError("resolve_split_year0: no dates/fallback")


@dataclass(frozen=True)
class DatasetProfile:
    key: str
    description: str
    default_root: str
    vocab_path: str
    grid_num_stocks: int
    num_score_news_types: int
    num_industry: int
    num_rule_trend: int
    price_window: int
    seg_length: int
    seg_keep_tail: bool
    num_virtual_window: int
    split_dates: dict[str, str]
    pattern_subdir: str
    include_calendar_frac: bool
    include_industry: bool
    include_rule_trend_node: bool
    news_subtype: bool
    virtual_news_type_count: int


def _filter_comp_date_map(mapping: dict | None, start: str, end: str) -> dict:
    """Keep only {comp: {date: ...}} entries with start <= date <= end."""
    if not mapping:
        return {}
    out: dict = {}
    for comp, by_date in mapping.items():
        if not isinstance(by_date, dict):
            out[comp] = by_date
            continue
        out[comp] = {d: v for (d, v) in by_date.items() if start <= str(d) <= end}
    return out


def _build_prev_next_maps(stock_data: dict) -> tuple[dict, dict]:
    next_label_date: dict = {}
    prev_date: dict = {}
    for comp, day_dict in stock_data.items():
        if not day_dict:
            next_label_date[comp] = {}
            prev_date[comp] = {}
            continue
        dates_sorted = sorted(day_dict.keys())
        nxt_map = {}
        prev_map = {}
        for i in range(len(dates_sorted) - 1):
            nxt_map[dates_sorted[i]] = dates_sorted[i + 1]
        for i in range(1, len(dates_sorted)):
            prev_map[dates_sorted[i]] = dates_sorted[i - 1]
        next_label_date[comp] = nxt_map
        prev_date[comp] = prev_map
    return (next_label_date, prev_date)


def isolate_dataset_to_split_range(ds) -> None:
    """Isolate dataset to split range."""
    (start, end) = (str(ds.start_date), str(ds.end_date))
    ds.stock_data = _filter_comp_date_map(ds.stock_data, start, end)
    ds.label = _filter_comp_date_map(getattr(ds, "label", None), start, end)
    if hasattr(ds, "volume"):
        ds.volume = _filter_comp_date_map(ds.volume, start, end)
    if hasattr(ds, "raw_line"):
        ds.raw_line = _filter_comp_date_map(ds.raw_line, start, end)
    ds.news_texts = _filter_comp_date_map(getattr(ds, "news_texts", None), start, end)
    ds.news_types = _filter_comp_date_map(getattr(ds, "news_types", None), start, end)
    if hasattr(ds, "news_subtypes"):
        ds.news_subtypes = _filter_comp_date_map(ds.news_subtypes, start, end)
    if hasattr(ds, "news_clusters"):
        ds.news_clusters = _filter_comp_date_map(ds.news_clusters, start, end)
    if hasattr(ds, "cluster_trend"):
        ds.cluster_trend = _filter_comp_date_map(ds.cluster_trend, start, end)
    if hasattr(ds, "rule_trend"):
        ds.rule_trend = _filter_comp_date_map(ds.rule_trend, start, end)
    (ds.next_label_date, ds.prev_date) = _build_prev_next_maps(ds.stock_data)
    n_dates = len({d for by_d in ds.stock_data.values() for d in by_d})
    tag = type(ds).__name__.replace("_", "")
    print(
        f"[{tag}] split-isolated mode={getattr(ds, 'mode', '?')} range=[{start},{end}] n_unique_dates={n_dates} (features/lookback stay inside this split)",
        flush=True,
    )


@dataclass
class _Block:
    _fields: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        if name in self._fields:
            return self._fields[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_fields":
            super().__setattr__(name, value)
        else:
            self._fields[name] = value

    def __getitem__(self, key: str) -> Any:
        return self._fields[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._fields[key] = value

    @property
    def __dict__(self):
        return self._fields

    @property
    def num_nodes(self) -> int:
        x = self._fields.get("x")
        if isinstance(x, torch.Tensor) and x.ndim >= 1:
            return int(x.size(0))
        return 0


@dataclass
class _Link:
    _fields: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        if name in self._fields:
            return self._fields[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_fields":
            super().__setattr__(name, value)
        else:
            self._fields[name] = value

    def __getitem__(self, key: str) -> Any:
        return self._fields[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._fields[key] = value


@dataclass
class BatchBuffer:
    """Batchbuffer."""

    def __init__(self):
        self._blocks: dict[str, _Block] = {}
        self._links: dict[tuple[str, str, str], _Link] = {}
        self.node_types: list[str] = []
        self.edge_types: list[tuple[str, str, str]] = []
        self._meta: dict[str, Any] = {}

    def _block(self, name: str) -> _Block:
        if name not in self._blocks:
            self._blocks[name] = _Block()
            if name not in self.node_types:
                self.node_types.append(name)
        return self._blocks[name]

    def _link(self, src: str, rel: str, dst: str) -> _Link:
        key = (src, rel, dst)
        if key not in self._links:
            self._links[key] = _Link()
            if key not in self.edge_types:
                self.edge_types.append(key)
        return self._links[key]

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 3:
            return self._link(key[0], key[1], key[2])
        if isinstance(key, str):
            if key in self._blocks:
                return self._blocks[key]
            if key in self._meta:
                return self._meta[key]
            return self._block(key)
        raise KeyError(key)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self._meta[key] = value

    def __contains__(self, key) -> bool:
        if isinstance(key, str):
            return key in self._blocks or key in self._meta
        if isinstance(key, tuple) and len(key) == 3:
            return key in self._links
        return False

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name in ("node_types", "edge_types"):
            raise AttributeError(name)
        if name in self._meta:
            return self._meta[name]
        raise AttributeError(name)


@dataclass
class WindowSample:
    """Windowsample."""

    range_dates: list[str] = field(default_factory=list)
    price_seq: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 0, 0))
    price_company_id: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    price_day_id: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    price_valid_mask: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, 1, dtype=torch.bool)
    )
    price_target_dates: list[str] = field(default_factory=list)
    price_debug_line: list[str] = field(default_factory=list)
    calendar_frac: torch.Tensor | None = None
    price_vol_seq: torch.Tensor | None = None
    news_count: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 1))
    news_company_id: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    news_day_id: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    news_dates: list[str] = field(default_factory=list)
    news_types: list[str] = field(default_factory=list)
    news_ids: list = field(default_factory=list)
    news_texts: list[str] = field(default_factory=list)
    news_vocab_ids: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, 100, dtype=torch.long)
    )
    news_vocab_mask: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, 100, dtype=torch.long)
    )
    news_input_ids: torch.Tensor | None = None
    news_attention_mask: torch.Tensor | None = None
    news_token_type_ids: torch.Tensor | None = None
    news_emb: torch.Tensor | None = None
    news_impact_y: torch.Tensor | None = None
    news_impact_y_valid: torch.Tensor | None = None
    news_pad_x: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 0, 1))
    news_pad_is_real: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 0))
    news_pad_vocab_ids: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, 0, 100, dtype=torch.long)
    )
    news_pad_vocab_mask: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, 0, 100, dtype=torch.long)
    )
    news_pad_input_ids: torch.Tensor | None = None
    news_pad_attention_mask: torch.Tensor | None = None
    news_pad_token_type_ids: torch.Tensor | None = None
    news_pad_emb: torch.Tensor | None = None
    news_type_feat: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 2))
    price_type_feat: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 3))
    industry_id_feat: torch.Tensor | None = None
    rule_trend_id_feat: torch.Tensor | None = None
    label_bin: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 1))
    label_org: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 1))
    label_strong_mask: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, 1, dtype=torch.bool)
    )
    label_valid_mask: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, 1, dtype=torch.bool)
    )
    label_vol_bin: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 1))
    label_vol_org: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 1))
    label_vol_valid_mask: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, 1, dtype=torch.bool)
    )
    virtual_window_table: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 1))
    virtual_news_table: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 1))
    virtual_news_type_table: torch.Tensor = field(default_factory=lambda: torch.zeros(12, 1))
    virtual_price_type_table: torch.Tensor = field(default_factory=lambda: torch.zeros(10, 1))
    virtual_industry_table: torch.Tensor | None = None
    virtual_trend_table: torch.Tensor | None = None
    virtual_rule_trend_table: torch.Tensor | None = None
    map_news_to_news_type: torch.Tensor = field(
        default_factory=lambda: torch.zeros(2, 0, dtype=torch.long)
    )
    map_price_to_price_type: torch.Tensor = field(
        default_factory=lambda: torch.zeros(2, 0, dtype=torch.long)
    )
    map_price_to_industry: torch.Tensor | None = None
    map_price_to_rule_trend: torch.Tensor | None = None
    map_price_to_virtual_window: torch.Tensor = field(
        default_factory=lambda: torch.zeros(2, 0, dtype=torch.long)
    )
    map_news_to_virtual_window: torch.Tensor = field(
        default_factory=lambda: torch.zeros(2, 0, dtype=torch.long)
    )
    map_news_to_virtual_news: torch.Tensor = field(
        default_factory=lambda: torch.zeros(2, 0, dtype=torch.long)
    )
    map_news_type_to_virtual_news_type: torch.Tensor = field(
        default_factory=lambda: torch.zeros(2, 0, dtype=torch.long)
    )
    map_price_type_to_virtual_price_type: torch.Tensor = field(
        default_factory=lambda: torch.zeros(2, 0, dtype=torch.long)
    )
    map_virtual_price_type_to_price_type: torch.Tensor = field(
        default_factory=lambda: torch.zeros(2, 0, dtype=torch.long)
    )
    map_industry_to_virtual_industry: torch.Tensor | None = None
    map_virtual_industry_to_industry: torch.Tensor | None = None
    map_price_to_virtual_trend: torch.Tensor | None = None
    map_virtual_trend_to_price: torch.Tensor | None = None
    map_rule_trend_to_virtual_rule_trend: torch.Tensor | None = None
    map_virtual_rule_trend_to_rule_trend: torch.Tensor | None = None

    def for_model(self) -> BatchBuffer:
        """For model."""
        return window_sample_to_buffer(self)


class _CSMDBuilderDataset(torch.utils.data.Dataset):

    def _load_all_data(self, root):
        return load_csmd_all_data(
            root,
            news_type_subdir=getattr(self, "news_type_subdir", "news_type"),
            news_source=getattr(self, "csmd_news_source", "raw"),
        )

    def __len__(self):
        return self.len()

    def __getitem__(self, idx):
        return self.get(idx)

    "Market-window dataset construction."

    def __init__(
        self,
        root,
        mode: str = "train",
        seg_length: int = 10,
        price_window: int = 5,
        news_cluster_k: int = 10,
        num_virtual_window: int | None = None,
        news_padding_k: int = 5,
        use_cached_news_padding_emb: bool = False,
        news_padding_emb_cache_path: str | None = None,
        skip_finbert_tokenizer: bool = False,
        vocab_path: str | None = None,
        finbert_model_path: str | None = None,
        transform=None,
        pre_transform=None,
        price_type_source: str = "rule",
        news_type_subdir: str = "news_type",
        csmd_news_source: str = "raw",
        split_dates: dict | None = None,
        split_mode: str = "contiguous",
        train_lookback_purge: bool = True,
        news_impact_oracle_path: str | None = None,
    ):
        super().__init__()
        assert mode in {"train", "val", "test"}
        assert news_cluster_k in {10, 20, 30}
        assert price_type_source in {"rule", "kmeans", "dbscan", "mix"}
        _news_src = str(csmd_news_source or "raw").strip().lower()
        assert _news_src in {"llm_extract", "raw"}
        self.root = root
        self.mode = mode
        self.seg_length = seg_length
        self.price_window = price_window
        self.news_cluster_k = news_cluster_k
        self.num_virtual_window = (
            num_virtual_window if num_virtual_window is not None else seg_length
        )
        self.news_padding_k = int(news_padding_k)
        self.use_cached_news_padding_emb = bool(use_cached_news_padding_emb)
        self.news_padding_emb_cache_path = news_padding_emb_cache_path
        self.skip_finbert_tokenizer = bool(skip_finbert_tokenizer)
        self._root_tag = hashlib.md5(os.path.abspath(str(root)).encode("utf-8")).hexdigest()[:8]
        self.price_type_source = str(price_type_source).lower()
        self.news_type_subdir = str(news_type_subdir or "news_type").strip() or "news_type"
        self.csmd_news_source = _news_src
        self.split_dates = dict(split_dates) if split_dates else None
        self.split_mode = str(split_mode or "contiguous").strip().lower() or "contiguous"
        self.train_lookback_purge = bool(train_lookback_purge)
        self.news_impact_oracle_path = str(news_impact_oracle_path or "").strip() or None
        self._news_impact_cache = None
        if self.news_impact_oracle_path:
            try:
                from tool.news_impact import load_oracle_cache
            except ImportError:
                from .news_impact import load_oracle_cache
            self._news_impact_cache = load_oracle_cache(self.news_impact_oracle_path)
            print(
                f"[IMPACT-ORACLE] loaded {self.news_impact_oracle_path} mode={mode} (labels for pretrain; do not use as test features)",
                flush=True,
            )
        all_data = self._load_all_data(root)
        self.stock_data = all_data["stock_data"]
        self.label = all_data["label"]
        self.volume = all_data.get("volume", {})
        self.raw_line = all_data.get("raw_line", {})
        self.news_texts = all_data["news_texts"]
        self.news_types = all_data["news_types"]
        self.news_subtypes = all_data["news_subtypes"]
        self.cluster_trend = all_data["cluster_trend"]
        self.rule_trend = all_data["rule_trend"]
        self.industry_info = all_data["industry_info"]
        self.code_to_name = all_data["code_to_name"]
        self.name_to_code = all_data["name_to_code"]
        _ind_cn_set = sorted(
            {
                str(v.get("industry_cn", "")).strip()
                for v in self.industry_info.values()
                if v.get("industry_cn")
            }
        )
        self.industry_cn_to_id = {cn: i for (i, cn) in enumerate(_ind_cn_set)}
        self.num_industries_dataset = max(len(self.industry_cn_to_id), 1)
        self.next_label_date = {}
        self.prev_date = {}
        for comp, day_dict in self.stock_data.items():
            if not day_dict:
                self.next_label_date[comp] = {}
                self.prev_date[comp] = {}
                continue
            dates_sorted = sorted(day_dict.keys())
            nxt_map = {}
            prev_map = {}
            for i in range(len(dates_sorted) - 1):
                nxt_map[dates_sorted[i]] = dates_sorted[i + 1]
            for i in range(1, len(dates_sorted)):
                prev_map[dates_sorted[i]] = dates_sorted[i - 1]
            self.next_label_date[comp] = nxt_map
            self.prev_date[comp] = prev_map
        _default_fb = "models/finbert-tone-chinese"
        finbert_model_path = str(finbert_model_path or "").strip() or _default_fb
        self._finbert_model_path = finbert_model_path
        self.news_tokenizer = None
        if not self.skip_finbert_tokenizer:
            self.news_tokenizer = AutoTokenizer.from_pretrained(finbert_model_path)
        self.news_max_len = 128
        vocab_file = resolve_vocab_path(vocab_path, "dict_csmd.pkl", "dict_cn.pkl")
        self.vocab = pkl.load(open(vocab_file, "rb"))
        self.vocab_pad_id = int(self.vocab.get("<pad>", 0))
        self.vocab_unk_id = int(self.vocab.get("<unk>", 0))
        self.vocab_word_level = is_word_vocab_path(vocab_file)
        self.vocab_use_jieba = is_cn_word_vocab_path(vocab_file)
        _vm = (
            "word-level jieba"
            if self.vocab_use_jieba
            else "word-level" if self.vocab_word_level else "char-level"
        )
        print(f"[CSMD_Dataset] vocab={vocab_file} mode={_vm} size={len(self.vocab)}")
        self.news_padding_emb_cache = None
        self.news_padding_emb_cache_root = None
        self.news_node_emb_cache_root = None
        if self.use_cached_news_padding_emb:
            cache_path = self.news_padding_emb_cache_path
            if cache_path is None:
                cache_path = default_news_node_emb_dir(
                    self.root, getattr(self, "news_type_subdir", "news_type"), max_seq_len=128
                )
            if os.path.isdir(cache_path):
                self.news_padding_emb_cache_root = cache_path
                self.news_node_emb_cache_root = cache_path
                print(f"[CSMD_Dataset] using news emb cache dir: {cache_path}")
            elif os.path.exists(cache_path):
                self.news_padding_emb_cache = torch.load(cache_path, map_location="cpu")
                print(f"[CSMD_Dataset] loaded news_padding emb cache file: {cache_path}")
            else:
                print(
                    f"[CSMD_Dataset] cache not found: {cache_path} (run --mode precompute_news_emb)"
                )
        self.companies = sorted(list(self.stock_data.keys()))
        self.company2id = {c: i for (i, c) in enumerate(self.companies)}
        data_period = {
            "CSMD-300": {
                "train_start": "2021-01-01",
                "train_end": "2023-01-01",
                "val_start": "2023-01-02",
                "val_end": "2024-01-02",
                "test_start": "2024-01-03",
                "test_end": "2024-12-31",
            }
        }
        full_period = dict(data_period["CSMD-300"])
        if self.split_dates:
            full_period.update({k: str(v) for (k, v) in self.split_dates.items() if v})
        if mode == "train":
            self.start_date = full_period["train_start"]
            self.end_date = full_period["train_end"]
        elif mode == "val":
            self.start_date = full_period["val_start"]
            self.end_date = full_period["val_end"]
        else:
            self.start_date = full_period["test_start"]
            self.end_date = full_period["test_end"]
        if self.split_mode == "stagger_4y":
            print(
                f"[CSMD_Dataset] split_mode=stagger_4y mode={mode} (year0 from calendar; cross-split lookback on full calendar)",
                flush=True,
            )
        elif self.split_mode == "seasonal_h1q3q4":
            _months = sorted(SEASONAL_H1Q3Q4_MONTHS.get(mode, set()))
            print(
                f"[CSMD_Dataset] split_mode=seasonal_h1q3q4 mode={mode} months={_months} (cross-split lookback on full calendar)",
                flush=True,
            )
        elif self.split_mode == "week_roundrobin":
            print(
                f"[CSMD_Dataset] split_mode=week_roundrobin mode={mode} ISO-week 6:2:2; train lookback-purge vs val/test (price_window={self.price_window})",
                flush=True,
            )
        elif self.split_mode == "month_roundrobin":
            print(
                f"[CSMD_Dataset] split_mode=month_roundrobin mode={mode} calendar-month 4:1:1; train lookback-purge vs val/test (price_window={self.price_window})",
                flush=True,
            )
        elif self.split_mode == "timefrac_622":
            print(
                f"[CSMD_Dataset] split_mode=timefrac_622 mode={mode} chronological 60%/20%/20% by trading days",
                flush=True,
            )
        elif self.split_mode in NEW_ROLE_SPLIT_MODES:
            print(
                f"[CSMD_Dataset] split_mode={self.split_mode} mode={mode} (NEW role-split + train lookback purge, pw={self.price_window})",
                flush=True,
            )
        else:
            print(
                f"[CSMD_Dataset] cross-split lookback mode={mode} predict_range=[{self.start_date},{self.end_date}]",
                flush=True,
            )
        self.global_dates = self._build_global_trading_dates()
        if self.split_mode == "stagger_4y":
            print(
                f"[CSMD_Dataset] stagger_4y year0={resolve_split_year0(self.global_dates, self.start_date)} n_valid_dates pending filter",
                flush=True,
            )
        if self.split_mode == "week_roundrobin":
            _st = week_roundrobin_split_stats(
                self.global_dates, price_window=int(self.price_window)
            )
            print(
                f"[CSMD_Dataset] week_roundrobin stats: weeks={_st['n_iso_weeks']} raw={_st['raw_role_days']} train_kept={_st['train_after_purge']} train_purged={_st['train_purged']} (lookback embargo)",
                flush=True,
            )
        if self.split_mode == "month_roundrobin":
            _st = month_roundrobin_split_stats(
                self.global_dates, price_window=int(self.price_window)
            )
            print(
                f"[CSMD_Dataset] month_roundrobin stats: months={_st['n_months']} raw={_st['raw_role_days']} train_kept={_st['train_after_purge']} train_purged={_st['train_purged']} (lookback embargo)",
                flush=True,
            )
        if self.split_mode == "timefrac_622":
            _st = timefrac_622_split_stats(self.global_dates)
            print(
                f"[CSMD_Dataset] timefrac_622 stats: counts={_st['counts']} train={_st['train_range']} val={_st['val_range']} test={_st['test_range']}",
                flush=True,
            )
        if self.split_mode in NEW_ROLE_SPLIT_MODES:
            _roles = resolve_new_role_split_roles(
                self.split_mode, self.global_dates, start_date=self.start_date
            )
            _purge = bool(getattr(self, "train_lookback_purge", True))
            _st = role_split_stats(
                self.global_dates,
                _roles or {},
                price_window=int(self.price_window),
                train_lookback_purge=_purge,
            )
            print(
                f"[CSMD_Dataset] {self.split_mode} stats: raw={_st['raw_role_days']} train_kept={_st['train_after_purge']} train_purged={_st['train_purged']} lookback_purge={_purge}",
                flush=True,
            )
        self.valid_dates = self._build_valid_dates_for_mode()
        self._rebuild_seg_windows()
        if self.split_mode == "stagger_4y":
            print(
                f"[CSMD_Dataset] stagger_4y mode={mode} n_valid_dates={len(self.valid_dates)} n_windows={len(self._seg_windows)}",
                flush=True,
            )
        elif self.split_mode == "week_roundrobin":
            print(
                f"[CSMD_Dataset] week_roundrobin mode={mode} n_valid_dates={len(self.valid_dates)} n_windows={len(self._seg_windows)}",
                flush=True,
            )
        elif self.split_mode == "month_roundrobin":
            print(
                f"[CSMD_Dataset] month_roundrobin mode={mode} n_valid_dates={len(self.valid_dates)} n_windows={len(self._seg_windows)}",
                flush=True,
            )
        elif self.split_mode == "timefrac_622":
            print(
                f"[CSMD_Dataset] timefrac_622 mode={mode} n_valid_dates={len(self.valid_dates)} n_windows={len(self._seg_windows)}",
                flush=True,
            )
        elif self.split_mode in NEW_ROLE_SPLIT_MODES:
            print(
                f"[CSMD_Dataset] {self.split_mode} mode={mode} n_valid_dates={len(self.valid_dates)} n_windows={len(self._seg_windows)}",
                flush=True,
            )

    def _build_global_trading_dates(self):
        all_dates = set()
        for _, day_dict in self.stock_data.items():
            all_dates.update(day_dict.keys())
        return sorted(all_dates)

    def _build_valid_dates_for_mode(self):
        sm = getattr(self, "split_mode", "contiguous")
        if sm == "stagger_4y":
            y0 = resolve_split_year0(self.global_dates, self.start_date)
            return [d for d in self.global_dates if date_in_stagger_4y(d, self.mode, y0)]
        if sm == "seasonal_h1q3q4":
            return [d for d in self.global_dates if date_in_seasonal_h1q3q4(d, self.mode)]
        if sm == "week_roundrobin":
            roles = build_week_roundrobin_roles(self.global_dates)
            date_to_pos = {x: i for (i, x) in enumerate(self.global_dates)}
            pw = max(1, int(getattr(self, "price_window", 5) or 5))
            return [
                d
                for d in self.global_dates
                if date_in_week_roundrobin(
                    d,
                    self.mode,
                    self.global_dates,
                    price_window=pw,
                    roles=roles,
                    date_to_pos=date_to_pos,
                )
            ]
        if sm == "month_roundrobin":
            roles = build_month_roundrobin_roles(self.global_dates)
            date_to_pos = {x: i for (i, x) in enumerate(self.global_dates)}
            pw = max(1, int(getattr(self, "price_window", 5) or 5))
            return [
                d
                for d in self.global_dates
                if date_in_month_roundrobin(
                    d,
                    self.mode,
                    self.global_dates,
                    price_window=pw,
                    roles=roles,
                    date_to_pos=date_to_pos,
                )
            ]
        if sm == "timefrac_622":
            date_to_pos = {x: i for (i, x) in enumerate(self.global_dates)}
            (te, ve) = timefrac_622_cut_indices(len(self.global_dates))
            return [
                d
                for d in self.global_dates
                if date_in_timefrac_622(
                    d,
                    self.mode,
                    self.global_dates,
                    date_to_pos=date_to_pos,
                    train_end=te,
                    val_end=ve,
                )
            ]
        if sm in NEW_ROLE_SPLIT_MODES:
            roles = resolve_new_role_split_roles(
                sm, self.global_dates, start_date=getattr(self, "start_date", None)
            )
            date_to_pos = {x: i for (i, x) in enumerate(self.global_dates)}
            pw = max(1, int(getattr(self, "price_window", 5) or 5))
            do_purge = bool(getattr(self, "train_lookback_purge", True))
            return [
                d
                for d in self.global_dates
                if date_in_roles_with_purge(
                    d,
                    self.mode,
                    self.global_dates,
                    roles or {},
                    price_window=pw,
                    date_to_pos=date_to_pos,
                    train_lookback_purge=do_purge,
                )
            ]
        return [d for d in self.global_dates if self.start_date <= d <= self.end_date]

    def _rebuild_seg_windows(self) -> None:
        gdates = list(getattr(self, "global_dates", None) or [])
        self._seg_windows = build_seg_windows(
            list(getattr(self, "valid_dates", None) or []),
            gdates,
            int(self.seg_length),
            keep_tail=True,
            break_on_month=_split_mode_break_on_month(getattr(self, "split_mode", None)),
            defer_months=_defer_months_for_split(
                getattr(self, "split_mode", None), gdates, getattr(self, "start_date", None)
            ),
        )

    def len(self):
        wins = getattr(self, "_seg_windows", None)
        if wins is None:
            self._rebuild_seg_windows()
            wins = self._seg_windows
        return len(wins)

    def _range_dates_for_idx(self, idx: int) -> list[str]:
        wins = getattr(self, "_seg_windows", None)
        if wins is None:
            self._rebuild_seg_windows()
            wins = self._seg_windows
        return list(wins[idx])

    def _ensure_price_feature_dim(self, data):
        """Ensure price feature dim."""
        if not hasattr(data, "price") or not hasattr(data["price"], "x"):
            return
        px = data["price"].x
        if not isinstance(px, torch.Tensor) or px.numel() == 0 or px.dim() != 3:
            return
        fdim = int(px.size(-1))
        if fdim == 8:
            return
        if fdim == 9:
            data["price"].x = torch.cat([px[..., 1:4], px[..., 4:9]], dim=-1)
            return
        raise RuntimeError(f"Unsupported price feature dim in cache: {fdim} (expect 8 or legacy 9)")

    def _ensure_labels_target_day(self, data, range_dates):
        """Ensure labels target day."""
        if not hasattr(data, "label") or data["price"].x.size(0) == 0:
            return
        label_bin = []
        label_org = []
        label_strong = []
        vol_bin = []
        vol_org = []
        vol_valid = []
        for p_idx in range(data["price"].x.size(0)):
            comp_idx = int(data["price"].company_id[p_idx].item())
            day_i = int(data["price"].day_id[p_idx].item())
            comp_name = self.companies[comp_idx]
            target_date = range_dates[day_i]
            raw_label = self.label.get(comp_name, {}).get(target_date, None)
            try:
                y = float(raw_label) if raw_label is not None else 0.0
            except Exception:
                y = 0.0
            label_org.append(y)
            label_bin.append(1.0 if y > 0.0 else 0.0)
            label_strong.append(1.0 if y > 0.0055 or y < -0.005 else 0.0)
            prev_d = self.prev_date.get(comp_name, {}).get(target_date)
            (vb, vo, vv) = _vol_change_from_map(
                getattr(self, "volume", None) or {}, comp_name, target_date, prev_d
            )
            vol_bin.append(vb)
            vol_org.append(vo)
            vol_valid.append(vv)
        data["label"].x = torch.tensor(label_bin, dtype=torch.float32).view(-1, 1)
        data["label"].org = torch.tensor(label_org, dtype=torch.float32).view(-1, 1)
        data["label"].strong_mask = torch.tensor(label_strong, dtype=torch.bool).view(-1, 1)
        data["label"].vol_x = torch.tensor(vol_bin, dtype=torch.float32).view(-1, 1)
        data["label"].vol_org = torch.tensor(vol_org, dtype=torch.float32).view(-1, 1)
        data["label"].vol_valid_mask = torch.tensor(vol_valid, dtype=torch.bool).view(-1, 1)

    def _resolve_trend_date(self, comp_name: str, target_date: str) -> str:
        """Resolve trend date."""
        prev_d = self.prev_date.get(comp_name, {}).get(target_date)
        return prev_d if prev_d else target_date

    def _ensure_rule_trend_feature_date(self, data, range_dates):
        """Ensure rule trend feature date."""
        if not range_dates or data["price"].x.size(0) == 0:
            return
        if "rule_trend_type" in data.node_types and data["rule_trend_type"].x.size(0) > 0:
            feats = []
            for p_idx in range(data["price"].x.size(0)):
                comp_idx = int(data["price"].company_id[p_idx].item())
                day_i = int(data["price"].day_id[p_idx].item())
                comp_name = self.companies[comp_idx]
                target_date = range_dates[day_i]
                trend_date = self._resolve_trend_date(comp_name, target_date)
                rt_cn = self.rule_trend.get(comp_name, {}).get(trend_date, None)
                feats.append(torch.tensor([float(rule_trend_cn_to_id(rt_cn))], dtype=torch.float32))
            data["rule_trend_type"].x = torch.stack(feats, dim=0)
        if "price_type" in data.node_types and data["price_type"].x.size(0) > 0:
            pt_feats = []
            for p_idx in range(data["price"].x.size(0)):
                comp_idx = int(data["price"].company_id[p_idx].item())
                day_i = int(data["price"].day_id[p_idx].item())
                comp_name = self.companies[comp_idx]
                target_date = range_dates[day_i]
                trend_date = self._resolve_trend_date(comp_name, target_date)
                ct = self.cluster_trend.get(comp_name, {}).get(trend_date, {})
                kmeans_id = (
                    int(ct["kmeans_cluster"])
                    if "kmeans_cluster" in ct and ct["kmeans_cluster"] is not None
                    else -1
                )
                dbscan_id = (
                    int(ct["dbscan_cluster"])
                    if "dbscan_cluster" in ct and ct["dbscan_cluster"] is not None
                    else -1
                )
                rt_cn = self.rule_trend.get(comp_name, {}).get(trend_date, None)
                rule_id = float(rule_trend_cn_to_id(rt_cn))
                pt_feats.append(torch.tensor([kmeans_id, dbscan_id, rule_id], dtype=torch.float32))
            data["price_type"].x = torch.stack(pt_feats, dim=0)

    def _news_impact_targets(
        self, news_company_ids: list, news_dates: list
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalized oracle [N,3] + valid [N] from publish dates (pretrain labels)."""
        n = len(news_company_ids)
        y = torch.zeros((n, 3), dtype=torch.float32)
        valid = torch.zeros((n,), dtype=torch.bool)
        cache = getattr(self, "_news_impact_cache", None)
        if cache is None or n <= 0:
            return (y, valid)
        try:
            from tool.news_impact import lookup_oracle
        except ImportError:
            from .news_impact import lookup_oracle
        for i, (ci, pub) in enumerate(zip(news_company_ids, news_dates)):
            comp = self.companies[int(ci)] if 0 <= int(ci) < len(self.companies) else None
            if comp is None:
                continue
            trip = lookup_oracle(cache, comp, str(pub), normalized=True)
            if trip is None:
                continue
            y[i, 0] = float(trip[0])
            y[i, 1] = float(trip[1])
            y[i, 2] = float(trip[2])
            valid[i] = True
        return (y, valid)

    def get(self, idx):
        """Get."""
        range_dates = self._range_dates_for_idx(idx)
        data = BatchBuffer()
        num_comp = len(self.companies)
        num_seg = len(range_dates)
        n_price_fixed = num_seg * num_comp
        price_x = torch.zeros((n_price_fixed, self.price_window, 8), dtype=torch.float32)
        vol_seq = torch.zeros((n_price_fixed, self.price_window, 1), dtype=torch.float32)
        price_comp_ids = torch.zeros((n_price_fixed,), dtype=torch.long)
        price_day_ids = torch.zeros((n_price_fixed,), dtype=torch.long)
        price_valid_mask = torch.zeros((n_price_fixed, 1), dtype=torch.bool)
        comp_date_to_pos = {}
        comp_dates_sorted_map = {}
        for comp in self.companies:
            day_dict = self.stock_data.get(comp, {})
            cds = sorted(day_dict.keys()) if day_dict else []
            comp_dates_sorted_map[comp] = cds
            comp_date_to_pos[comp] = {d: i for (i, d) in enumerate(cds)}
        for day_i, d in enumerate(range_dates):
            for comp_idx, comp in enumerate(self.companies):
                p_idx = day_i * num_comp + comp_idx
                price_comp_ids[p_idx] = comp_idx
                price_day_ids[p_idx] = day_i
                day_dict = self.stock_data.get(comp, {})
                date_to_pos = comp_date_to_pos.get(comp, {})
                comp_dates_sorted = comp_dates_sorted_map.get(comp, [])
                if not day_dict or d not in date_to_pos:
                    continue
                pos = date_to_pos[d]
                if pos < self.price_window:
                    continue
                last_k_dates = comp_dates_sorted[pos - self.price_window : pos]
                feat_seq_8 = []
                for dd in last_k_dates:
                    raw = day_dict[dd]
                    if hasattr(raw, "tolist"):
                        raw = raw.tolist()
                    raw_3 = list(raw)[:3]
                    try:
                        wd = datetime.strptime(str(dd), "%Y-%m-%d").weekday()
                    except Exception:
                        wd = 0
                    onehot = [0.0] * 5
                    if wd < 5:
                        onehot[wd] = 1.0
                    feat_seq_8.append(raw_3 + onehot)
                if len(feat_seq_8) != self.price_window:
                    continue
                raw_label_d = self.label.get(comp, {}).get(d, None)
                try:
                    y_d = float(raw_label_d) if raw_label_d is not None else None
                except Exception:
                    y_d = None
                if y_d is None:
                    continue
                price_x[p_idx] = torch.tensor(feat_seq_8, dtype=torch.float32)
                vol_seq[p_idx] = torch.tensor(
                    _vol_return_seq_for_dates(
                        getattr(self, "volume", None) or {},
                        comp,
                        last_k_dates,
                        comp_dates_sorted,
                        date_to_pos,
                    ),
                    dtype=torch.float32,
                )
                price_valid_mask[p_idx] = True
        data["price"].x = price_x
        data["price"].vol_seq = vol_seq
        data["price"].company_id = price_comp_ids
        data["price"].day_id = price_day_ids
        data["price"].valid_mask = price_valid_mask
        data["price"].date = [
            range_dates[int(price_day_ids[p_idx].item())] for p_idx in range(n_price_fixed)
        ]
        news_company_ids = []
        news_dates = []
        news_types_str = []
        news_text_list = []
        news_ids = []
        news_target_day_ids = []
        for comp_idx, comp in enumerate(self.companies):
            if comp not in self.news_texts:
                continue
            comp_news_by_date = self.news_texts[comp]
            comp_type_by_date = self.news_types.get(comp, {})
            comp_dates_sorted = comp_dates_sorted_map.get(comp, [])
            date_to_pos = comp_date_to_pos.get(comp, {})
            for day_i, d in enumerate(range_dates):
                if d not in date_to_pos or date_to_pos[d] < 1:
                    continue
                prev_date = comp_dates_sorted[date_to_pos[d] - 1]
                if prev_date not in comp_news_by_date:
                    continue
                for news_id, text in comp_news_by_date[prev_date].items():
                    news_company_ids.append(comp_idx)
                    news_dates.append(prev_date)
                    news_target_day_ids.append(day_i)
                    news_text_list.append(text)
                    news_ids.append(news_id)
                    ttype = comp_type_by_date.get(prev_date, {}).get(news_id, "")
                    news_types_str.append(ttype)
        if news_text_list:
            Nn = len(news_text_list)
            if self.skip_finbert_tokenizer:
                data["news"].input_ids = torch.zeros((Nn, self.news_max_len), dtype=torch.long)
                data["news"].attention_mask = torch.zeros((Nn, self.news_max_len), dtype=torch.long)
            else:
                encodings = self.news_tokenizer(
                    news_text_list,
                    padding="max_length",
                    truncation=True,
                    max_length=self.news_max_len,
                    return_tensors="pt",
                )
                data["news"].input_ids = encodings["input_ids"]
                data["news"].attention_mask = encodings["attention_mask"]
                if "token_type_ids" in encodings:
                    data["news"].token_type_ids = encodings["token_type_ids"]
            data["news"].x = torch.zeros((Nn, 1), dtype=torch.float32)
            data["news"].company_id = torch.tensor(news_company_ids, dtype=torch.long)
            data["news"].date = news_dates
            data["news"].day_id = torch.tensor(news_target_day_ids, dtype=torch.long)
            data["news"].type = news_types_str
            data["news"].news_id = news_ids
            data["news"].text = news_text_list
            MAX_SEQ_LEN = 100
            vocab_input_ids = []
            vocab_attention = []
            for txt in news_text_list:
                idx = text_to_vocab_ids(
                    txt,
                    self.vocab,
                    unk_id=self.vocab_unk_id,
                    max_len=MAX_SEQ_LEN,
                    word_level=self.vocab_word_level,
                    use_jieba=getattr(self, "vocab_use_jieba", False),
                )
                (idx, attn) = pad_vocab_ids(idx, self.vocab_pad_id, MAX_SEQ_LEN)
                vocab_input_ids.append(idx)
                vocab_attention.append(attn)
            data["news"].vocab_input_ids = torch.tensor(vocab_input_ids, dtype=torch.long)
            data["news"].vocab_attention_mask = torch.tensor(vocab_attention, dtype=torch.long)
            if getattr(self, "news_node_emb_cache_root", None):
                data["news"].emb = load_news_node_emb_rows(
                    self.news_node_emb_cache_root,
                    self.companies,
                    news_company_ids,
                    news_dates,
                    news_ids,
                )
            (data["news"].impact_y, data["news"].impact_y_valid) = self._news_impact_targets(
                news_company_ids, news_dates
            )
        else:
            data["news"].x = torch.zeros((0, 1), dtype=torch.float32)
            data["news"].company_id = torch.zeros((0,), dtype=torch.long)
            data["news"].date = []
            data["news"].day_id = torch.zeros((0,), dtype=torch.long)
            data["news"].type = []
            data["news"].news_id = []
            data["news"].text = []
            data["news"].vocab_input_ids = torch.zeros((0, 100), dtype=torch.long)
            data["news"].vocab_attention_mask = torch.zeros((0, 100), dtype=torch.long)
            if getattr(self, "news_node_emb_cache_root", None):
                data["news"].emb = torch.zeros((0, 768), dtype=torch.float32)
            data["news"].impact_y = torch.zeros((0, 3), dtype=torch.float32)
            data["news"].impact_y_valid = torch.zeros((0,), dtype=torch.bool)
        MAX_NEWS_PER_DAY = int(self.news_padding_k)
        MAX_SEQ_LEN = 100
        n_price = int(data["price"].x.size(0))
        data["news_padding"].x = torch.zeros((n_price, MAX_NEWS_PER_DAY, 1), dtype=torch.float32)
        key2price_idx = {}
        for p_idx in range(n_price):
            c = int(data["price"].company_id[p_idx].item())
            d = int(data["price"].day_id[p_idx].item())
            key2price_idx[c, d] = p_idx
        texts_by_price = [[] for _ in range(n_price)]
        for n_idx, txt in enumerate(news_text_list):
            c = int(news_company_ids[n_idx])
            target_d = int(news_target_day_ids[n_idx])
            p_idx = key2price_idx.get((c, target_d), None)
            if p_idx is None:
                continue
            if len(texts_by_price[p_idx]) < MAX_NEWS_PER_DAY:
                texts_by_price[p_idx].append(txt)
        docs_mat: list[list[str]] = []
        is_real_mat: list[list[float]] = []
        for p_idx in range(n_price):
            day_docs = texts_by_price[p_idx]
            row_docs: list[str] = []
            row_real: list[float] = []
            for slot in range(MAX_NEWS_PER_DAY):
                if slot < len(day_docs):
                    row_docs.append(day_docs[slot])
                    row_real.append(1.0)
                else:
                    row_docs.append("")
                    row_real.append(0.0)
            docs_mat.append(row_docs)
            is_real_mat.append(row_real)
        if n_price > 0:
            data["news_padding"].is_real = torch.tensor(is_real_mat, dtype=torch.float32)
        else:
            data["news_padding"].is_real = torch.zeros((0, MAX_NEWS_PER_DAY), dtype=torch.float32)
        if n_price > 0:
            flat_docs = [d for row in docs_mat for d in row]
            if self.skip_finbert_tokenizer:
                data["news_padding"].input_ids = torch.zeros(
                    (n_price, MAX_NEWS_PER_DAY, MAX_SEQ_LEN), dtype=torch.long
                )
                data["news_padding"].attention_mask = torch.zeros(
                    (n_price, MAX_NEWS_PER_DAY, MAX_SEQ_LEN), dtype=torch.long
                )
            else:
                chunk_size = 4000
                input_ids_list = []
                attention_mask_list = []
                token_type_ids_list = []
                use_token_type_ids = True
                with torch.no_grad():
                    for start in range(0, len(flat_docs), chunk_size):
                        end = min(start + chunk_size, len(flat_docs))
                        enc = self.news_tokenizer(
                            flat_docs[start:end],
                            padding="max_length",
                            truncation=True,
                            max_length=MAX_SEQ_LEN,
                            return_tensors="pt",
                        )
                        input_ids_list.append(enc["input_ids"])
                        attention_mask_list.append(enc["attention_mask"])
                        if "token_type_ids" in enc:
                            token_type_ids_list.append(enc["token_type_ids"])
                        else:
                            use_token_type_ids = False
                _ids = torch.cat(input_ids_list, dim=0)
                _mask = torch.cat(attention_mask_list, dim=0)
                data["news_padding"].input_ids = _ids.view(n_price, MAX_NEWS_PER_DAY, MAX_SEQ_LEN)
                data["news_padding"].attention_mask = _mask.view(
                    n_price, MAX_NEWS_PER_DAY, MAX_SEQ_LEN
                )
                if use_token_type_ids and len(token_type_ids_list) > 0:
                    _tt = torch.cat(token_type_ids_list, dim=0)
                    data["news_padding"].token_type_ids = _tt.view(
                        n_price, MAX_NEWS_PER_DAY, MAX_SEQ_LEN
                    )
            vocab_input_ids = []
            vocab_attention = []
            for doc in flat_docs:
                idx = text_to_vocab_ids(
                    doc,
                    self.vocab,
                    unk_id=self.vocab_unk_id,
                    max_len=MAX_SEQ_LEN,
                    word_level=getattr(self, "vocab_word_level", False),
                    use_jieba=getattr(self, "vocab_use_jieba", False),
                )
                (idx, attn) = pad_vocab_ids(idx, self.vocab_pad_id, MAX_SEQ_LEN)
                vocab_input_ids.append(idx)
                vocab_attention.append(attn)
            _vocab_ids = torch.tensor(vocab_input_ids, dtype=torch.long)
            _vocab_attn = torch.tensor(vocab_attention, dtype=torch.long)
            data["news_padding"].vocab_input_ids = _vocab_ids.view(
                n_price, MAX_NEWS_PER_DAY, MAX_SEQ_LEN
            )
            data["news_padding"].vocab_attention_mask = _vocab_attn.view(
                n_price, MAX_NEWS_PER_DAY, MAX_SEQ_LEN
            )
            _skip_pad_emb = (
                getattr(self, "news_node_emb_cache_root", None)
                and hasattr(data["news"], "emb")
                and isinstance(getattr(data["news"], "emb", None), torch.Tensor)
                and (int(data["news"].emb.numel()) > 0)
            )
            if self.use_cached_news_padding_emb and (not _skip_pad_emb):
                emb_dim = 768
                if (
                    isinstance(self.news_padding_emb_cache_root, str)
                    and self.news_padding_emb_cache_root
                ):
                    emb_out = torch.zeros((n_price, MAX_NEWS_PER_DAY, emb_dim), dtype=torch.float32)
                    is_real_out = data["news_padding"].is_real.clone()
                    for p_idx in range(n_price):
                        comp_idx = int(data["price"].company_id[p_idx].item())
                        day_id = int(data["price"].day_id[p_idx].item())
                        comp_name = self.companies[comp_idx]
                        cur_date = range_dates[day_id]
                        comp_dates = comp_dates_sorted_map.get(comp_name, [])
                        date_to_pos = comp_date_to_pos.get(comp_name, {})
                        prev_date = (
                            comp_dates[date_to_pos[cur_date] - 1]
                            if cur_date in date_to_pos and date_to_pos[cur_date] >= 1
                            else None
                        )
                        date_str = prev_date if prev_date else cur_date
                        fpath = os.path.join(
                            self.news_padding_emb_cache_root, comp_name, f"{date_str}.pt"
                        )
                        if not os.path.exists(fpath):
                            continue
                        obj = torch.load(fpath, map_location="cpu")
                        emb = obj.get("emb", None)
                        ir = obj.get("is_real", None)
                        layout = obj.get("emb_layout", None)
                        n_day = len(texts_by_price[p_idx])
                        if not isinstance(emb, torch.Tensor) or emb.dim() != 2:
                            continue
                        if (
                            layout in ("packed", "by_news_id")
                            or int(emb.size(0)) != MAX_NEWS_PER_DAY
                        ):
                            n_f = int(emb.size(0))
                            n_take = min(n_f, n_day, MAX_NEWS_PER_DAY)
                            if n_take > 0:
                                emb_out[p_idx, :n_take] = emb[:n_take].to(dtype=torch.float32)
                        else:
                            if emb.numel() > 0:
                                emb_out[p_idx] = emb[:MAX_NEWS_PER_DAY].to(dtype=torch.float32)
                            if isinstance(ir, torch.Tensor) and ir.numel() > 0:
                                is_real_out[p_idx] = ir[:MAX_NEWS_PER_DAY].to(dtype=torch.float32)
                    data["news_padding"].emb = emb_out
                    data["news_padding"].is_real = is_real_out
                elif self.news_padding_emb_cache is not None:
                    cache = self.news_padding_emb_cache
                    idx_map = cache.get("index", {})
                    emb_cache = cache.get("emb", None)
                    is_real_cache = cache.get("is_real", None)
                    if isinstance(emb_cache, torch.Tensor) and emb_cache.numel() > 0:
                        Kc = int(cache.get("k", MAX_NEWS_PER_DAY))
                        if Kc == MAX_NEWS_PER_DAY:
                            emb_out = torch.zeros(
                                (n_price, MAX_NEWS_PER_DAY, emb_cache.size(-1)), dtype=torch.float32
                            )
                            is_real_out = data["news_padding"].is_real.clone()
                            for p_idx in range(n_price):
                                comp_idx = int(data["price"].company_id[p_idx].item())
                                day_id = int(data["price"].day_id[p_idx].item())
                                comp_name = self.companies[comp_idx]
                                cur_date = range_dates[day_id]
                                comp_dates = comp_dates_sorted_map.get(comp_name, [])
                                date_to_pos = comp_date_to_pos.get(comp_name, {})
                                prev_date = (
                                    comp_dates[date_to_pos[cur_date] - 1]
                                    if cur_date in date_to_pos and date_to_pos[cur_date] >= 1
                                    else cur_date
                                )
                                date_str = prev_date
                                row = idx_map.get((comp_name, date_str), None)
                                if row is None:
                                    continue
                                emb_out[p_idx] = emb_cache[row]
                                if (
                                    isinstance(is_real_cache, torch.Tensor)
                                    and is_real_cache.numel() > 0
                                ):
                                    is_real_out[p_idx] = is_real_cache[row]
                                else:
                                    is_real_out[p_idx] = (
                                        emb_cache[row].abs().sum(dim=-1) > 0
                                    ).float()
                            data["news_padding"].emb = emb_out
                            data["news_padding"].is_real = is_real_out
                if not hasattr(data["news_padding"], "emb"):
                    data["news_padding"].emb = torch.zeros(
                        (n_price, MAX_NEWS_PER_DAY, emb_dim), dtype=torch.float32
                    )

        def _encode_base_news_type(t: str) -> int:
            if t is None:
                return 2
            if "市场" in t:
                return 0
            if "行业" in t or "板块" in t:
                return 1
            return 2

        news_type_feats = []
        news_to_type_edges = []
        if data["news"].x.size(0) > 0:
            for n_idx in range(data["news"].x.size(0)):
                comp_idx = int(data["news"].company_id[n_idx].item())
                comp_name = self.companies[comp_idx]
                date_str = data["news"].date[n_idx]
                n_type_str = data["news"].type[n_idx]
                n_id = str(data["news"].news_id[n_idx])
                base_tid = _encode_base_news_type(n_type_str)
                subtype_id = -1
                if base_tid == 2:
                    subtype_id = 11
                    if (
                        comp_name in self.news_subtypes
                        and date_str in self.news_subtypes[comp_name]
                    ):
                        day_sub = self.news_subtypes[comp_name][date_str]
                        if n_id in day_sub:
                            sid = int(day_sub[n_id])
                            if 1 <= sid <= 11:
                                subtype_id = sid
                news_type_feats.append(torch.tensor([base_tid, subtype_id], dtype=torch.float32))
                news_to_type_edges.append([n_idx, n_idx])
        if news_type_feats:
            data["news_type"].x = torch.stack(news_type_feats, dim=0)
        else:
            data["news_type"].x = torch.zeros((0, 2), dtype=torch.float32)
        if news_to_type_edges:
            data["news", "to", "news_type"].edge_index = torch.tensor(
                news_to_type_edges, dtype=torch.long
            ).t()
        else:
            data["news", "to", "news_type"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        price_type_feats = []
        price_to_type_edges = []
        if not hasattr(self, "_rule_trend_map"):
            self._rule_trend_map = {}
        if data["price"].x.size(0) > 0 and len(range_dates) > 0:
            for p_idx in range(data["price"].x.size(0)):
                comp_idx = int(data["price"].company_id[p_idx].item())
                day_i = int(data["price"].day_id[p_idx].item())
                comp_name = self.companies[comp_idx]
                target_date = range_dates[day_i]
                trend_date = self._resolve_trend_date(comp_name, target_date)
                ct = self.cluster_trend.get(comp_name, {}).get(trend_date, {})
                kmeans_id = (
                    int(ct["kmeans_cluster"])
                    if "kmeans_cluster" in ct and ct["kmeans_cluster"] is not None
                    else -1
                )
                dbscan_id = (
                    int(ct["dbscan_cluster"])
                    if "dbscan_cluster" in ct and ct["dbscan_cluster"] is not None
                    else -1
                )
                rt = self.rule_trend.get(comp_name, {}).get(trend_date, None)
                rule_id = float(rule_trend_cn_to_id(rt))
                feat = torch.tensor([kmeans_id, dbscan_id, rule_id], dtype=torch.float32)
                price_type_feats.append(feat)
                price_to_type_edges.append([p_idx, p_idx])
        if price_type_feats:
            data["price_type"].x = torch.stack(price_type_feats, dim=0)
        else:
            data["price_type"].x = torch.zeros((0, 3), dtype=torch.float32)
        if price_to_type_edges:
            data["price", "to", "price_type"].edge_index = torch.tensor(
                price_to_type_edges, dtype=torch.long
            ).t()
        else:
            data["price", "to", "price_type"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        industry_type_feats = []
        price_to_industry_edges = []
        default_ind_id = 0
        if data["price"].x.size(0) > 0 and len(range_dates) > 0:
            for p_idx in range(data["price"].x.size(0)):
                comp_idx = int(data["price"].company_id[p_idx].item())
                comp_name = self.companies[comp_idx]
                ind_cn = self.industry_info.get(comp_name, {}).get("industry_cn", "")
                ind_id = self.industry_cn_to_id.get(ind_cn, default_ind_id)
                industry_type_feats.append(torch.tensor([float(ind_id)], dtype=torch.float32))
                price_to_industry_edges.append([p_idx, p_idx])
        if industry_type_feats:
            data["industry_type"].x = torch.stack(industry_type_feats, dim=0)
        else:
            data["industry_type"].x = torch.zeros((0, 1), dtype=torch.float32)
        if price_to_industry_edges:
            data["price", "to", "industry_type"].edge_index = torch.tensor(
                price_to_industry_edges, dtype=torch.long
            ).t()
        else:
            data["price", "to", "industry_type"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        num_industry_vi = max(int(getattr(self, "num_industries_dataset", 27)), 1)
        data["virtual_industry"].x = torch.zeros((num_industry_vi, 1), dtype=torch.float32)
        industry_type_to_vi = []
        if data["industry_type"].x.size(0) > 0:
            for t_idx in range(data["industry_type"].x.size(0)):
                ind_id = int(data["industry_type"].x[t_idx, 0].item())
                vi_idx = int(ind_id) % num_industry_vi
                industry_type_to_vi.append([t_idx, vi_idx])
        data["industry_type", "to", "virtual_industry"].edge_index = (
            torch.tensor(industry_type_to_vi, dtype=torch.long).t()
            if industry_type_to_vi
            else torch.zeros((2, 0), dtype=torch.long)
        )
        data["virtual_industry", "to", "industry_type"].edge_index = (
            data["industry_type", "to", "virtual_industry"].edge_index.flip(0)
            if industry_type_to_vi
            else torch.zeros((2, 0), dtype=torch.long)
        )
        rule_trend_type_feats = []
        price_to_rule_trend_edges = []
        if data["price"].x.size(0) > 0 and len(range_dates) > 0:
            for p_idx in range(data["price"].x.size(0)):
                comp_idx = int(data["price"].company_id[p_idx].item())
                day_i = int(data["price"].day_id[p_idx].item())
                comp_name = self.companies[comp_idx]
                target_date = range_dates[day_i]
                trend_date = self._resolve_trend_date(comp_name, target_date)
                rt_cn = self.rule_trend.get(comp_name, {}).get(trend_date, None)
                trend_id = rule_trend_cn_to_id(rt_cn)
                rule_trend_type_feats.append(torch.tensor([float(trend_id)], dtype=torch.float32))
                price_to_rule_trend_edges.append([p_idx, p_idx])
        if rule_trend_type_feats:
            data["rule_trend_type"].x = torch.stack(rule_trend_type_feats, dim=0)
        else:
            data["rule_trend_type"].x = torch.zeros((0, 1), dtype=torch.float32)
        if price_to_rule_trend_edges:
            data["price", "to", "rule_trend_type"].edge_index = torch.tensor(
                price_to_rule_trend_edges, dtype=torch.long
            ).t()
        else:
            data["price", "to", "rule_trend_type"].edge_index = torch.zeros(
                (2, 0), dtype=torch.long
            )
        label_bin = []
        label_org = []
        label_strong = []
        vol_bin = []
        vol_org = []
        vol_valid = []
        price_raw_line = []
        for p_idx in range(data["price"].x.size(0)):
            comp_idx = int(data["price"].company_id[p_idx].item())
            day_i = int(data["price"].day_id[p_idx].item())
            comp_name = self.companies[comp_idx]
            cur_date = range_dates[day_i]
            raw_label = self.label.get(comp_name, {}).get(cur_date, None)
            comp_dates = comp_dates_sorted_map.get(comp_name, [])
            date_to_pos = comp_date_to_pos.get(comp_name, {})
            prev_date = None
            if cur_date in date_to_pos and date_to_pos[cur_date] >= 1:
                prev_date = comp_dates[date_to_pos[cur_date] - 1]
                day_dict = self.stock_data.get(comp_name, {})
                if prev_date in day_dict:
                    raw_3 = list(day_dict[prev_date])[:3]
                    wd = datetime.strptime(str(prev_date), "%Y-%m-%d").weekday() if prev_date else 0
                    onehot = [0.0] * 5
                    if wd < 5:
                        onehot[wd] = 1.0
                    raw_line_str = f"3cols(HLC):{raw_3} onehot:{onehot}"
                else:
                    last_row = data["price"].x[p_idx, -1, :].tolist()
                    raw_line_str = f"3cols(HLC):{last_row[:3]} onehot:{last_row[3:8]}"
            else:
                last_row = data["price"].x[p_idx, -1, :].tolist()
                raw_line_str = f"3cols(HLC):{last_row[:3]} onehot:{last_row[3:8]}"
            price_raw_line.append(raw_line_str)
            try:
                y = float(raw_label) if raw_label is not None else 0.0
            except Exception:
                y = 0.0
            y_bin = 1.0 if y > 0.0 else 0.0
            is_strong = 1.0 if y > 0.0055 or y < -0.005 else 0.0
            label_org.append(y)
            label_bin.append(y_bin)
            label_strong.append(is_strong)
            (vb, vo, vv) = _vol_change_from_map(
                getattr(self, "volume", None) or {}, comp_name, cur_date, prev_date
            )
            vol_bin.append(vb)
            vol_org.append(vo)
            vol_valid.append(vv)
        data["price"].raw_line = price_raw_line
        if label_bin:
            data["label"].x = torch.tensor(label_bin, dtype=torch.float32).view(-1, 1)
            data["label"].org = torch.tensor(label_org, dtype=torch.float32).view(-1, 1)
            data["label"].strong_mask = torch.tensor(label_strong, dtype=torch.bool).view(-1, 1)
            data["label"].vol_x = torch.tensor(vol_bin, dtype=torch.float32).view(-1, 1)
            data["label"].vol_org = torch.tensor(vol_org, dtype=torch.float32).view(-1, 1)
            data["label"].vol_valid_mask = torch.tensor(vol_valid, dtype=torch.bool).view(-1, 1)
            if hasattr(data["price"], "valid_mask"):
                data["label"].valid_mask = data["price"].valid_mask.clone()
            else:
                data["label"].valid_mask = torch.ones_like(
                    data["label"].strong_mask, dtype=torch.bool
                )
        else:
            data["label"].x = torch.zeros((0, 1), dtype=torch.float32)
            data["label"].org = torch.zeros((0, 1), dtype=torch.float32)
            data["label"].strong_mask = torch.zeros((0, 1), dtype=torch.bool)
            data["label"].valid_mask = torch.zeros((0, 1), dtype=torch.bool)
            data["label"].vol_x = torch.zeros((0, 1), dtype=torch.float32)
            data["label"].vol_org = torch.zeros((0, 1), dtype=torch.float32)
            data["label"].vol_valid_mask = torch.zeros((0, 1), dtype=torch.bool)
        num_comp = len(self.companies)
        num_seg = len(range_dates)
        n_price = data["price"].x.size(0)
        vw_cap = max(1, int(self.num_virtual_window))
        vw_num = min(vw_cap, num_seg)
        data["virtual_window"].x = torch.zeros((vw_num, 1), dtype=torch.float32)
        vw_edges_price = []
        for p_idx in range(n_price):
            day_id = int(data["price"].day_id[p_idx].item())
            if day_id < vw_num:
                vw_edges_price.append([p_idx, day_id])
        vw_edges_news = []
        for n_idx in range(data["news"].x.size(0)):
            day_id = int(data["news"].day_id[n_idx].item())
            if day_id < vw_num:
                vw_edges_news.append([n_idx, day_id])
        data["price", "to", "virtual_window"].edge_index = (
            torch.tensor(vw_edges_price, dtype=torch.long).t()
            if vw_edges_price
            else torch.zeros((2, 0), dtype=torch.long)
        )
        data["news", "to", "virtual_window"].edge_index = (
            torch.tensor(vw_edges_news, dtype=torch.long).t()
            if vw_edges_news
            else torch.zeros((2, 0), dtype=torch.long)
        )
        vn_num = n_price
        data["virtual_news"].x = torch.zeros((vn_num, 1), dtype=torch.float32)
        p_to_vn = [[p_idx, p_idx] for p_idx in range(n_price)]
        key2pidx = {}
        for p_idx in range(n_price):
            comp_idx = int(data["price"].company_id[p_idx].item())
            day_id = int(data["price"].day_id[p_idx].item())
            key2pidx[comp_idx, day_id] = p_idx
        n_to_vn = []
        for n_idx in range(data["news"].x.size(0)):
            comp_idx = int(data["news"].company_id[n_idx].item())
            day_id = int(data["news"].day_id[n_idx].item())
            p_idx = key2pidx.get((comp_idx, day_id))
            if p_idx is not None:
                n_to_vn.append([n_idx, p_idx])
        data["price", "to", "virtual_news"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["news", "to", "virtual_news"].edge_index = (
            torch.tensor(n_to_vn, dtype=torch.long).t()
            if n_to_vn
            else torch.zeros((2, 0), dtype=torch.long)
        )
        NUM_NEWS_TYPE = NUM_VIRTUAL_NEWS_TYPES
        data["virtual_news_type"].x = torch.zeros((NUM_NEWS_TYPE, 1), dtype=torch.float32)
        news_type_to_vnt = []
        if data["news_type"].x.size(0) > 0:
            for t_idx in range(data["news_type"].x.size(0)):
                base_tid = int(data["news_type"].x[t_idx, 0].item())
                subtype_id = int(data["news_type"].x[t_idx, 1].item())
                if base_tid == 0:
                    vnt_idx = 0
                elif base_tid == 1:
                    vnt_idx = 1
                else:
                    sid = subtype_id if 1 <= subtype_id <= 11 else 11
                    vnt_idx = 1 + sid
                vnt_idx = min(vnt_idx, NUM_NEWS_TYPE - 1)
                news_type_to_vnt.append([t_idx, vnt_idx])
        data["news_type", "to", "virtual_news_type"].edge_index = (
            torch.tensor(news_type_to_vnt, dtype=torch.long).t()
            if news_type_to_vnt
            else torch.zeros((2, 0), dtype=torch.long)
        )
        NUM_PRICE_TYPE = 10
        data["virtual_price_type"].x = torch.zeros((NUM_PRICE_TYPE, 1), dtype=torch.float32)
        price_type_to_vpt = []
        if data["price_type"].x.size(0) > 0:
            for t_idx in range(data["price_type"].x.size(0)):
                kmeans_id = (
                    int(data["price_type"].x[t_idx, 0].item())
                    if data["price_type"].x[t_idx, 0].item() >= 0
                    else 0
                )
                dbscan_id = (
                    int(data["price_type"].x[t_idx, 1].item())
                    if data["price_type"].x[t_idx, 1].item() >= 0
                    else 0
                )
                rule_id = (
                    int(data["price_type"].x[t_idx, 2].item())
                    if data["price_type"].x[t_idx, 2].item() >= 0
                    else 0
                )
                src = getattr(self, "price_type_source", "rule")
                if src == "kmeans":
                    src_id = kmeans_id
                elif src == "dbscan":
                    src_id = dbscan_id
                elif src == "mix":
                    src_id = kmeans_id + dbscan_id + rule_id
                else:
                    src_id = rule_id
                vpt_idx = int(src_id) % NUM_PRICE_TYPE
                price_type_to_vpt.append([t_idx, vpt_idx])
        data["price_type", "to", "virtual_price_type"].edge_index = (
            torch.tensor(price_type_to_vpt, dtype=torch.long).t()
            if price_type_to_vpt
            else torch.zeros((2, 0), dtype=torch.long)
        )
        data["virtual_price_type", "to", "price_type"].edge_index = (
            data["price_type", "to", "virtual_price_type"].edge_index.flip(0)
            if price_type_to_vpt
            else torch.zeros((2, 0), dtype=torch.long)
        )
        data["range_dates"] = range_dates
        return buffer_to_window_sample(data)


class _CMINBuilderDataset(torch.utils.data.Dataset):

    def _load_all_data(self, root):
        return load_cmin_cn_all_data(
            root,
            news_type_subdir=getattr(self, "news_type_subdir", "news_type"),
            require_wind_news_source=bool(getattr(self, "require_wind_news_source", True)),
        )

    def __len__(self) -> int:
        return self.len()

    def __getitem__(self, idx: int) -> WindowSample:
        return self.get(idx)

    "Market-window dataset construction."

    def __init__(
        self,
        root,
        mode: str = "train",
        seg_length: int = 10,
        seg_keep_tail: bool = False,
        price_window: int = 10,
        news_cluster_k: int = 10,
        price_type_source: str = "rule",
        num_virtual_window: int | None = None,
        news_padding_k: int = 5,
        use_cached_news_padding_emb: bool = False,
        news_padding_emb_cache_path: str | None = None,
        skip_finbert_tokenizer: bool = False,
        vocab_path: str | None = None,
        finbert_model_path: str | None = None,
        news_type_subdir: str = "news_type",
        require_wind_news_source: bool = True,
        require_window_news: bool = False,
        subset_top_companies: int = 0,
        companies_whitelist: list[str] | None = None,
        split_dates: dict | None = None,
        split_mode: str = "contiguous",
        train_lookback_purge: bool = True,
        transform=None,
        pre_transform=None,
    ):
        super().__init__()
        assert mode in {"train", "val", "test"}
        assert news_cluster_k in {10, 20, 30}
        assert price_type_source in {"rule", "kmeans", "dbscan", "mix"}
        self.root = root
        self.mode = mode
        self.seg_length = seg_length
        self.seg_keep_tail = bool(seg_keep_tail)
        self.price_window = price_window
        self.news_cluster_k = news_cluster_k
        self.price_type_source = str(price_type_source).lower()
        self.num_virtual_window = (
            num_virtual_window if num_virtual_window is not None else seg_length
        )
        self.news_padding_k = int(news_padding_k)
        self.use_cached_news_padding_emb = bool(use_cached_news_padding_emb)
        self.news_padding_emb_cache_path = news_padding_emb_cache_path
        self.skip_finbert_tokenizer = bool(skip_finbert_tokenizer)
        self.news_type_subdir = str(news_type_subdir or "news_type").strip() or "news_type"
        self.require_wind_news_source = bool(require_wind_news_source)
        self.require_window_news = bool(require_window_news)
        self.subset_top_companies = max(0, int(subset_top_companies or 0))
        self.companies_whitelist = (
            [str(c) for c in companies_whitelist] if companies_whitelist else None
        )
        self.split_dates = dict(split_dates) if split_dates else None
        self.split_mode = str(split_mode or "contiguous").strip().lower() or "contiguous"
        self.train_lookback_purge = bool(train_lookback_purge)
        self._root_tag = hashlib.md5(os.path.abspath(str(root)).encode("utf-8")).hexdigest()[:8]
        all_data = self._load_all_data(root)
        self.stock_data = all_data["stock_data"]
        self.label = all_data["label"]
        self.volume = all_data.get("volume", {})
        self.raw_line = all_data.get("raw_line", {})
        self.news_texts = all_data["news_texts"]
        self.news_types = all_data["news_types"]
        self.news_clusters = all_data["news_clusters"]
        self.cluster_trend = all_data["cluster_trend"]
        self.rule_trend = all_data["rule_trend"]
        self.industry_info = all_data["industry_info"]
        self.code_to_name = all_data["code_to_name"]
        self.name_to_code = all_data["name_to_code"]
        self.industry_dict = load_industry_dict()
        self._industry_cn_to_id = {}
        _local_cns = sorted(
            {
                str(v.get("industry_cn", "")).strip()
                for v in (self.industry_info or {}).values()
                if str(v.get("industry_cn", "")).strip()
            }
        )
        if _local_cns:
            self._industry_cn_to_id = {cn: i for (i, cn) in enumerate(_local_cns)}
            self._num_industries_graph = max(len(_local_cns), 1)
        else:
            self._num_industries_graph = 27
        self.num_industries_dataset = int(self._num_industries_graph)
        print(
            f"[CMIN_Dataset] industry local map n={self._num_industries_graph} (dense ids from comp_indus; not industry_dict)",
            flush=True,
        )
        self.next_label_date = {}
        self.prev_date = {}
        for comp, day_dict in self.stock_data.items():
            if not day_dict:
                self.next_label_date[comp] = {}
                self.prev_date[comp] = {}
                continue
            dates_sorted = sorted(day_dict.keys())
            nxt_map = {}
            prev_map = {}
            for i in range(len(dates_sorted) - 1):
                nxt_map[dates_sorted[i]] = dates_sorted[i + 1]
            for i in range(1, len(dates_sorted)):
                prev_map[dates_sorted[i]] = dates_sorted[i - 1]
            self.next_label_date[comp] = nxt_map
            self.prev_date[comp] = prev_map
        _default_fb = "models/finbert-tone-chinese"
        finbert_model_path = str(finbert_model_path or "").strip() or _default_fb
        self._finbert_model_path = finbert_model_path
        self.news_tokenizer = None
        if not self.skip_finbert_tokenizer:
            self.news_tokenizer = AutoTokenizer.from_pretrained(finbert_model_path)
        self.news_max_len = 128
        vocab_file = resolve_vocab_path(vocab_path, "dict_cn.pkl", "dict_us.pkl")
        self.vocab = pkl.load(open(vocab_file, "rb"))
        self.vocab_pad_id = int(self.vocab.get("<pad>", 0))
        self.vocab_unk_id = int(self.vocab.get("<unk>", 0))
        self.vocab_word_level = is_us_vocab_path(vocab_file)
        _vocab_mode = "word-level (dict_us/dict_massive)" if self.vocab_word_level else "char-level"
        _wind = "Wind-only" if self.require_wind_news_source else "all-sources"
        _rw = "ON" if self.require_window_news else "off"
        print(
            f"[CMIN_Dataset] vocab {_vocab_mode}: {vocab_file}; news={_wind}; require_window_news={_rw}",
            flush=True,
        )
        self.news_padding_emb_cache = None
        self.news_padding_emb_cache_root = None
        self.news_node_emb_cache_root = None
        if self.use_cached_news_padding_emb:
            cache_path = self.news_padding_emb_cache_path
            if cache_path is None:
                cache_path = default_news_node_emb_dir(
                    self.root, getattr(self, "news_type_subdir", "news_type"), max_seq_len=128
                )
            if os.path.isdir(cache_path):
                self.news_padding_emb_cache_root = cache_path
                self.news_node_emb_cache_root = cache_path
                print(f"[CMIN_Dataset] using news emb cache dir: {cache_path}")
            elif os.path.exists(cache_path):
                self.news_padding_emb_cache = torch.load(cache_path, map_location="cpu")
                print(f"[CMIN_Dataset] loaded news_padding emb cache file: {cache_path}")
            else:
                print(
                    f"[CMIN_Dataset] cache not found: {cache_path} (run --mode precompute_news_emb)"
                )
        self.companies = sorted(list(self.stock_data.keys()))
        self.company2id = {c: i for (i, c) in enumerate(self.companies)}
        _default_period = {
            "train_start": "2018-01-01",
            "train_end": "2021-04-30",
            "val_start": "2021-05-01",
            "val_end": "2021-08-31",
            "test_start": "2021-09-01",
            "test_end": "2021-12-31",
        }
        full_period = dict(_default_period)
        if self.split_dates:
            full_period.update({k: str(v) for (k, v) in self.split_dates.items() if v})
        if mode == "train":
            self.start_date = full_period["train_start"]
            self.end_date = full_period["train_end"]
        elif mode == "val":
            self.start_date = full_period["val_start"]
            self.end_date = full_period["val_end"]
        else:
            self.start_date = full_period["test_start"]
            self.end_date = full_period["test_end"]
        if self.split_mode == "stagger_4y":
            print(f"[CMINBuilderDataset] split_mode=stagger_4y mode={mode}", flush=True)
        elif self.split_mode == "seasonal_h1q3q4":
            _months = sorted(SEASONAL_H1Q3Q4_MONTHS.get(mode, set()))
            print(
                f"[CMINBuilderDataset] split_mode=seasonal_h1q3q4 mode={mode} months={_months}",
                flush=True,
            )
        elif self.split_mode == "week_roundrobin":
            print(
                f"[CMINBuilderDataset] split_mode=week_roundrobin mode={mode} ISO-week 6:2:2; train lookback-purge (price_window={self.price_window})",
                flush=True,
            )
        elif self.split_mode == "month_roundrobin":
            print(
                f"[CMINBuilderDataset] split_mode=month_roundrobin mode={mode} calendar-month 4:1:1; train lookback-purge (price_window={self.price_window})",
                flush=True,
            )
        elif self.split_mode == "timefrac_622":
            print(
                f"[CMINBuilderDataset] split_mode=timefrac_622 mode={mode} chronological 60%/20%/20% by trading days",
                flush=True,
            )
        elif self.split_mode in NEW_ROLE_SPLIT_MODES:
            print(
                f"[CMINBuilderDataset] split_mode={self.split_mode} mode={mode} (NEW role-split + train lookback purge, pw={self.price_window})",
                flush=True,
            )
        else:
            print(
                f"[CMINBuilderDataset] split mode={mode} range=[{self.start_date},{self.end_date}]",
                flush=True,
            )
        print(
            f"[CMINBuilderDataset] cross-split lookback enabled (predict_range=[{self.start_date},{self.end_date}] split_mode={self.split_mode})",
            flush=True,
        )
        self._apply_company_subset_if_needed()
        self.global_dates = self._build_global_trading_dates()
        if self.split_mode == "week_roundrobin":
            _st = week_roundrobin_split_stats(
                self.global_dates, price_window=int(self.price_window)
            )
            print(
                f"[CMINBuilderDataset] week_roundrobin stats: weeks={_st['n_iso_weeks']} raw={_st['raw_role_days']} train_kept={_st['train_after_purge']} train_purged={_st['train_purged']}",
                flush=True,
            )
        if self.split_mode == "month_roundrobin":
            _st = month_roundrobin_split_stats(
                self.global_dates, price_window=int(self.price_window)
            )
            print(
                f"[CMINBuilderDataset] month_roundrobin stats: months={_st['n_months']} raw={_st['raw_role_days']} train_kept={_st['train_after_purge']} train_purged={_st['train_purged']}",
                flush=True,
            )
        if self.split_mode == "timefrac_622":
            _st = timefrac_622_split_stats(self.global_dates)
            print(
                f"[CMINBuilderDataset] timefrac_622 stats: counts={_st['counts']} train={_st['train_range']} val={_st['val_range']} test={_st['test_range']}",
                flush=True,
            )
        if self.split_mode in NEW_ROLE_SPLIT_MODES:
            _roles = resolve_new_role_split_roles(
                self.split_mode, self.global_dates, start_date=self.start_date
            )
            _purge = bool(getattr(self, "train_lookback_purge", True))
            _st = role_split_stats(
                self.global_dates,
                _roles or {},
                price_window=int(self.price_window),
                train_lookback_purge=_purge,
            )
            print(
                f"[CMINBuilderDataset] {self.split_mode} stats: raw={_st['raw_role_days']} train_kept={_st['train_after_purge']} train_purged={_st['train_purged']} lookback_purge={_purge}",
                flush=True,
            )
        self.valid_dates = self._build_valid_dates_for_mode()
        self._rebuild_seg_windows()
        if self.split_mode == "week_roundrobin":
            print(
                f"[CMINBuilderDataset] week_roundrobin mode={mode} n_valid_dates={len(self.valid_dates)} n_windows={len(self._seg_windows)}",
                flush=True,
            )
        elif self.split_mode == "month_roundrobin":
            print(
                f"[CMINBuilderDataset] month_roundrobin mode={mode} n_valid_dates={len(self.valid_dates)} n_windows={len(self._seg_windows)}",
                flush=True,
            )
        elif self.split_mode == "timefrac_622":
            print(
                f"[CMINBuilderDataset] timefrac_622 mode={mode} n_valid_dates={len(self.valid_dates)} n_windows={len(self._seg_windows)}",
                flush=True,
            )
        elif self.split_mode in NEW_ROLE_SPLIT_MODES:
            print(
                f"[CMINBuilderDataset] {self.split_mode} mode={mode} n_valid_dates={len(self.valid_dates)} n_windows={len(self._seg_windows)}",
                flush=True,
            )
        elif self.split_mode == "stagger_4y":
            print(
                f"[CMINBuilderDataset] stagger_4y mode={mode} n_valid_dates={len(self.valid_dates)} n_windows={len(self._seg_windows)}",
                flush=True,
            )

    def _count_window_news_valid_days(self, comp: str) -> int:
        """Count window news valid days."""
        cds = sorted(self.stock_data.get(comp, {}) or {})
        if not cds:
            return 0
        news = self.news_texts.get(comp, {}) or {}
        pos = {d: i for (i, d) in enumerate(cds)}
        pw = max(1, int(self.price_window))
        n = 0
        for d in getattr(self, "valid_dates", None) or [
            x for x in self._build_global_trading_dates() if self.start_date <= x <= self.end_date
        ]:
            if d not in pos or pos[d] < pw:
                continue
            last = cds[pos[d] - pw : pos[d]]
            if all((dd in news and bool(news[dd]) for dd in last)):
                n += 1
        return n

    def _apply_company_subset_if_needed(self) -> None:
        wl = self.companies_whitelist
        top_n = int(getattr(self, "subset_top_companies", 0) or 0)
        if not wl and top_n <= 0:
            return
        if wl:
            have = set(self.companies)
            keep = [c for c in wl if c in have]
        else:
            self.global_dates = self._build_global_trading_dates()
            self.valid_dates = self._build_valid_dates_for_mode()
            scored = [(c, self._count_window_news_valid_days(c)) for c in self.companies]
            scored.sort(key=lambda x: (-x[1], x[0]))
            keep = [c for (c, _) in scored[:top_n]]
            print(
                f"[CMINBuilderDataset] subset_top_companies={top_n} coverage_top={(scored[0][1] if scored else 0)} coverage_cut={(scored[min(top_n, len(scored)) - 1][1] if scored else 0)}",
                flush=True,
            )
        keep_set = set(keep)
        self.companies = keep
        self.company2id = {c: i for (i, c) in enumerate(self.companies)}
        for attr in (
            "stock_data",
            "label",
            "raw_line",
            "news_texts",
            "news_types",
            "news_clusters",
            "cluster_trend",
            "rule_trend",
            "prev_date",
            "next_label_date",
        ):
            d = getattr(self, attr, None)
            if isinstance(d, dict):
                setattr(self, attr, {c: v for (c, v) in d.items() if c in keep_set})
        print(
            f"[CMINBuilderDataset] company subset mode={self.mode} n={len(self.companies)} ({', '.join(self.companies[:5])}{('...' if len(self.companies) > 5 else '')})",
            flush=True,
        )

    def _build_global_trading_dates(self):
        all_dates = set()
        for _, day_dict in self.stock_data.items():
            all_dates.update(day_dict.keys())
        return sorted(all_dates)

    def _build_valid_dates_for_mode(self):
        sm = getattr(self, "split_mode", "contiguous")
        if sm == "stagger_4y":
            y0 = resolve_split_year0(self.global_dates, self.start_date)
            return [d for d in self.global_dates if date_in_stagger_4y(d, self.mode, y0)]
        if sm == "seasonal_h1q3q4":
            return [d for d in self.global_dates if date_in_seasonal_h1q3q4(d, self.mode)]
        if sm == "week_roundrobin":
            roles = build_week_roundrobin_roles(self.global_dates)
            date_to_pos = {x: i for (i, x) in enumerate(self.global_dates)}
            pw = max(1, int(getattr(self, "price_window", 5) or 5))
            return [
                d
                for d in self.global_dates
                if date_in_week_roundrobin(
                    d,
                    self.mode,
                    self.global_dates,
                    price_window=pw,
                    roles=roles,
                    date_to_pos=date_to_pos,
                )
            ]
        if sm == "month_roundrobin":
            roles = build_month_roundrobin_roles(self.global_dates)
            date_to_pos = {x: i for (i, x) in enumerate(self.global_dates)}
            pw = max(1, int(getattr(self, "price_window", 5) or 5))
            return [
                d
                for d in self.global_dates
                if date_in_month_roundrobin(
                    d,
                    self.mode,
                    self.global_dates,
                    price_window=pw,
                    roles=roles,
                    date_to_pos=date_to_pos,
                )
            ]
        if sm == "timefrac_622":
            date_to_pos = {x: i for (i, x) in enumerate(self.global_dates)}
            (te, ve) = timefrac_622_cut_indices(len(self.global_dates))
            return [
                d
                for d in self.global_dates
                if date_in_timefrac_622(
                    d,
                    self.mode,
                    self.global_dates,
                    date_to_pos=date_to_pos,
                    train_end=te,
                    val_end=ve,
                )
            ]
        if sm in NEW_ROLE_SPLIT_MODES:
            roles = resolve_new_role_split_roles(
                sm, self.global_dates, start_date=getattr(self, "start_date", None)
            )
            date_to_pos = {x: i for (i, x) in enumerate(self.global_dates)}
            pw = max(1, int(getattr(self, "price_window", 5) or 5))
            do_purge = bool(getattr(self, "train_lookback_purge", True))
            return [
                d
                for d in self.global_dates
                if date_in_roles_with_purge(
                    d,
                    self.mode,
                    self.global_dates,
                    roles or {},
                    price_window=pw,
                    date_to_pos=date_to_pos,
                    train_lookback_purge=do_purge,
                )
            ]
        return [d for d in self.global_dates if self.start_date <= d <= self.end_date]

    def _rebuild_seg_windows(self) -> None:
        gdates = list(getattr(self, "global_dates", None) or [])
        self._seg_windows = build_seg_windows(
            list(getattr(self, "valid_dates", None) or []),
            gdates,
            int(self.seg_length),
            keep_tail=bool(getattr(self, "seg_keep_tail", False)),
            break_on_month=_split_mode_break_on_month(getattr(self, "split_mode", None)),
            defer_months=_defer_months_for_split(
                getattr(self, "split_mode", None), gdates, getattr(self, "start_date", None)
            ),
        )

    def len(self):
        wins = getattr(self, "_seg_windows", None)
        if wins is None:
            self._rebuild_seg_windows()
            wins = self._seg_windows
        return len(wins)

    def _range_dates_for_idx(self, idx: int) -> list[str]:
        wins = getattr(self, "_seg_windows", None)
        if wins is None:
            self._rebuild_seg_windows()
            wins = self._seg_windows
        return list(wins[idx])

    def _resolve_trend_date(self, comp_name: str, target_date: str) -> str:
        """Resolve trend date."""
        prev_d = self.prev_date.get(comp_name, {}).get(target_date)
        return prev_d if prev_d else target_date

    def _ensure_trend_prev_date(self, data, range_dates):
        """Ensure trend prev date."""
        if not range_dates or data["price"].x.size(0) == 0:
            return
        if "price_type" not in data.node_types or data["price_type"].x.size(0) == 0:
            return
        pt_feats = []
        for p_idx in range(data["price"].x.size(0)):
            comp_idx = int(data["price"].company_id[p_idx].item())
            day_i = int(data["price"].day_id[p_idx].item())
            comp_name = self.companies[comp_idx]
            target_date = range_dates[day_i]
            trend_date = self._resolve_trend_date(comp_name, target_date)
            ct = self.cluster_trend.get(comp_name, {}).get(trend_date, {})
            kmeans_id = (
                int(ct["kmeans_cluster"])
                if "kmeans_cluster" in ct and ct["kmeans_cluster"] is not None
                else -1
            )
            dbscan_id = (
                int(ct["dbscan_cluster"])
                if "dbscan_cluster" in ct and ct["dbscan_cluster"] is not None
                else -1
            )
            rt_cn = self.rule_trend.get(comp_name, {}).get(trend_date, None)
            rule_id = float(rule_trend_cn_to_id(rt_cn))
            pt_feats.append(torch.tensor([kmeans_id, dbscan_id, rule_id], dtype=torch.float32))
        data["price_type"].x = torch.stack(pt_feats, dim=0)

    def get(self, idx):
        """Get."""
        range_dates = self._range_dates_for_idx(idx)
        data = BatchBuffer()
        num_comp = len(self.companies)
        num_seg = len(range_dates)
        n_price_fixed = num_seg * num_comp
        price_x = torch.zeros((n_price_fixed, self.price_window, 8), dtype=torch.float32)
        date_x = torch.zeros((n_price_fixed, self.price_window, 1), dtype=torch.float32)
        vol_seq = torch.zeros((n_price_fixed, self.price_window, 1), dtype=torch.float32)
        price_comp_ids = torch.zeros((n_price_fixed,), dtype=torch.long)
        price_day_ids = torch.zeros((n_price_fixed,), dtype=torch.long)
        price_valid_mask = torch.zeros((n_price_fixed, 1), dtype=torch.bool)
        comp_date_to_pos = {}
        comp_dates_sorted_map = {}
        for comp in self.companies:
            day_dict = self.stock_data.get(comp, {})
            cds = sorted(day_dict.keys()) if day_dict else []
            comp_dates_sorted_map[comp] = cds
            comp_date_to_pos[comp] = {d: i for (i, d) in enumerate(cds)}
        for day_i, d in enumerate(range_dates):
            for comp_idx, comp in enumerate(self.companies):
                p_idx = day_i * num_comp + comp_idx
                price_comp_ids[p_idx] = comp_idx
                price_day_ids[p_idx] = day_i
                day_dict = self.stock_data.get(comp, {})
                date_to_pos = comp_date_to_pos.get(comp, {})
                comp_dates_sorted = comp_dates_sorted_map.get(comp, [])
                if not day_dict or d not in date_to_pos:
                    continue
                pos = date_to_pos[d]
                if pos < self.price_window:
                    continue
                last_k_dates = comp_dates_sorted[pos - self.price_window : pos]
                feat_seq_8 = []
                date_seq_1 = []
                for dd in last_k_dates:
                    raw = day_dict[dd]
                    if hasattr(raw, "tolist"):
                        raw = raw.tolist()
                    raw_3 = list(raw)[:3]
                    try:
                        wd = datetime.strptime(str(dd), "%Y-%m-%d").weekday()
                    except Exception:
                        wd = 0
                    onehot = [0.0] * 5
                    if wd < 5:
                        onehot[wd] = 1.0
                    feat_seq_8.append(raw_3 + onehot)
                    date_seq_1.append([_day_of_year_frac(dd)])
                if len(feat_seq_8) != self.price_window:
                    continue
                if bool(getattr(self, "require_window_news", False)):
                    comp_news = self.news_texts.get(comp, {})
                    if not all((dd in comp_news and bool(comp_news[dd]) for dd in last_k_dates)):
                        continue
                raw_label_d = self.label.get(comp, {}).get(d, None)
                try:
                    y_d = float(raw_label_d) if raw_label_d is not None else None
                except Exception:
                    y_d = None
                if y_d is None:
                    continue
                price_x[p_idx] = torch.tensor(feat_seq_8, dtype=torch.float32)
                date_x[p_idx] = torch.tensor(date_seq_1, dtype=torch.float32)
                vol_seq[p_idx] = torch.tensor(
                    _vol_return_seq_for_dates(
                        getattr(self, "volume", None) or {},
                        comp,
                        last_k_dates,
                        comp_dates_sorted,
                        date_to_pos,
                    ),
                    dtype=torch.float32,
                )
                price_valid_mask[p_idx] = True
        data["price"].x = price_x
        data["date"].x = date_x
        data["price"].vol_seq = vol_seq
        data["price"].company_id = price_comp_ids
        data["price"].day_id = price_day_ids
        data["price"].valid_mask = price_valid_mask
        data["price"].date = [
            range_dates[int(price_day_ids[p_idx].item())] for p_idx in range(n_price_fixed)
        ]
        need_news_slots: set[tuple[int, int]] | None = None
        if bool(getattr(self, "require_window_news", False)):
            K_news = max(1, int(self.price_window))
            need_news_slots = set()
            for day_i in range(num_seg):
                for comp_idx in range(num_comp):
                    p_idx = day_i * num_comp + comp_idx
                    if not bool(price_valid_mask[p_idx].item()):
                        continue
                    for j in range(max(0, day_i - K_news + 1), day_i + 1):
                        need_news_slots.add((comp_idx, j))
        news_company_ids = []
        news_dates = []
        news_types_str = []
        news_text_list = []
        news_ids = []
        news_target_day_ids = []
        _news_cap = (
            max(1, int(self.news_padding_k))
            if bool(getattr(self, "require_window_news", False))
            else None
        )
        for comp_idx, comp in enumerate(self.companies):
            if comp not in self.news_texts:
                continue
            comp_news_by_date = self.news_texts[comp]
            comp_type_by_date = self.news_types.get(comp, {})
            comp_dates_sorted = comp_dates_sorted_map.get(comp, [])
            date_to_pos = comp_date_to_pos.get(comp, {})
            for day_i, d in enumerate(range_dates):
                if need_news_slots is not None and (comp_idx, day_i) not in need_news_slots:
                    continue
                if d not in date_to_pos or date_to_pos[d] < 1:
                    continue
                prev_date = comp_dates_sorted[date_to_pos[d] - 1]
                if prev_date not in comp_news_by_date:
                    continue
                day_items = list(comp_news_by_date[prev_date].items())
                if _news_cap is not None and len(day_items) > _news_cap:
                    day_items = day_items[:_news_cap]
                for news_id, text in day_items:
                    news_company_ids.append(comp_idx)
                    news_dates.append(prev_date)
                    news_target_day_ids.append(day_i)
                    news_text_list.append(text)
                    news_ids.append(news_id)
                    ttype = comp_type_by_date.get(prev_date, {}).get(news_id, "")
                    news_types_str.append(ttype)
        if news_text_list:
            Nn = len(news_text_list)
            if self.skip_finbert_tokenizer:
                data["news"].input_ids = torch.zeros((Nn, self.news_max_len), dtype=torch.long)
                data["news"].attention_mask = torch.zeros((Nn, self.news_max_len), dtype=torch.long)
            else:
                encodings = self.news_tokenizer(
                    news_text_list,
                    padding="max_length",
                    truncation=True,
                    max_length=self.news_max_len,
                    return_tensors="pt",
                )
                data["news"].input_ids = encodings["input_ids"]
                data["news"].attention_mask = encodings["attention_mask"]
                if "token_type_ids" in encodings:
                    data["news"].token_type_ids = encodings["token_type_ids"]
            data["news"].x = torch.zeros((Nn, 1), dtype=torch.float32)
            data["news"].company_id = torch.tensor(news_company_ids, dtype=torch.long)
            data["news"].date = news_dates
            data["news"].day_id = torch.tensor(news_target_day_ids, dtype=torch.long)
            data["news"].type = news_types_str
            data["news"].news_id = news_ids
            data["news"].text = news_text_list
            MAX_SEQ_LEN = 100
            vocab_input_ids = []
            vocab_attention = []
            for txt in news_text_list:
                idx = text_to_vocab_ids(
                    txt,
                    self.vocab,
                    unk_id=self.vocab_unk_id,
                    max_len=MAX_SEQ_LEN,
                    word_level=self.vocab_word_level,
                    use_jieba=getattr(self, "vocab_use_jieba", False),
                )
                (idx, attn) = pad_vocab_ids(idx, self.vocab_pad_id, MAX_SEQ_LEN)
                vocab_input_ids.append(idx)
                vocab_attention.append(attn)
            data["news"].vocab_input_ids = torch.tensor(vocab_input_ids, dtype=torch.long)
            data["news"].vocab_attention_mask = torch.tensor(vocab_attention, dtype=torch.long)
            if getattr(self, "news_node_emb_cache_root", None):
                data["news"].emb = load_news_node_emb_rows(
                    self.news_node_emb_cache_root,
                    self.companies,
                    news_company_ids,
                    news_dates,
                    news_ids,
                )
        else:
            data["news"].x = torch.zeros((0, 1), dtype=torch.float32)
            data["news"].company_id = torch.zeros((0,), dtype=torch.long)
            data["news"].date = []
            data["news"].day_id = torch.zeros((0,), dtype=torch.long)
            data["news"].type = []
            data["news"].news_id = []
            data["news"].text = []
            data["news"].vocab_input_ids = torch.zeros((0, 100), dtype=torch.long)
            data["news"].vocab_attention_mask = torch.zeros((0, 100), dtype=torch.long)
            if getattr(self, "news_node_emb_cache_root", None):
                data["news"].emb = torch.zeros((0, 768), dtype=torch.float32)
        MAX_NEWS_PER_DAY = int(self.news_padding_k)
        MAX_SEQ_LEN = 100
        n_price = int(data["price"].x.size(0))
        data["news_padding"].x = torch.zeros((n_price, MAX_NEWS_PER_DAY, 1), dtype=torch.float32)
        texts_by_price = [[] for _ in range(n_price)]
        key2price_idx = {}
        for p_idx in range(n_price):
            c = int(data["price"].company_id[p_idx].item())
            d = int(data["price"].day_id[p_idx].item())
            key2price_idx[c, d] = p_idx
        for n_idx, txt in enumerate(news_text_list):
            c = int(news_company_ids[n_idx])
            target_d = int(news_target_day_ids[n_idx])
            p_idx = key2price_idx.get((c, target_d), None)
            if p_idx is None:
                continue
            if len(texts_by_price[p_idx]) < MAX_NEWS_PER_DAY:
                texts_by_price[p_idx].append(txt)
        docs_mat: list[list[str]] = []
        is_real_mat: list[list[float]] = []
        for p_idx in range(n_price):
            day_docs = texts_by_price[p_idx]
            row_docs: list[str] = []
            row_real: list[float] = []
            for slot in range(MAX_NEWS_PER_DAY):
                if slot < len(day_docs):
                    row_docs.append(day_docs[slot])
                    row_real.append(1.0)
                else:
                    row_docs.append("")
                    row_real.append(0.0)
            docs_mat.append(row_docs)
            is_real_mat.append(row_real)
        if n_price > 0:
            data["news_padding"].is_real = torch.tensor(is_real_mat, dtype=torch.float32)
        else:
            data["news_padding"].is_real = torch.zeros((0, MAX_NEWS_PER_DAY), dtype=torch.float32)
        if n_price > 0:
            flat_docs = [d for row in docs_mat for d in row]
            if self.skip_finbert_tokenizer:
                data["news_padding"].input_ids = torch.zeros(
                    (n_price, MAX_NEWS_PER_DAY, MAX_SEQ_LEN), dtype=torch.long
                )
                data["news_padding"].attention_mask = torch.zeros(
                    (n_price, MAX_NEWS_PER_DAY, MAX_SEQ_LEN), dtype=torch.long
                )
            else:
                chunk_size = 4000
                input_ids_list = []
                attention_mask_list = []
                token_type_ids_list = []
                use_token_type_ids = True
                with torch.no_grad():
                    for start in range(0, len(flat_docs), chunk_size):
                        end = min(start + chunk_size, len(flat_docs))
                        enc = self.news_tokenizer(
                            flat_docs[start:end],
                            padding="max_length",
                            truncation=True,
                            max_length=MAX_SEQ_LEN,
                            return_tensors="pt",
                        )
                        input_ids_list.append(enc["input_ids"])
                        attention_mask_list.append(enc["attention_mask"])
                        if "token_type_ids" in enc:
                            token_type_ids_list.append(enc["token_type_ids"])
                        else:
                            use_token_type_ids = False
                _ids = torch.cat(input_ids_list, dim=0)
                _mask = torch.cat(attention_mask_list, dim=0)
                data["news_padding"].input_ids = _ids.view(n_price, MAX_NEWS_PER_DAY, MAX_SEQ_LEN)
                data["news_padding"].attention_mask = _mask.view(
                    n_price, MAX_NEWS_PER_DAY, MAX_SEQ_LEN
                )
                if use_token_type_ids and len(token_type_ids_list) > 0:
                    _tt = torch.cat(token_type_ids_list, dim=0)
                    data["news_padding"].token_type_ids = _tt.view(
                        n_price, MAX_NEWS_PER_DAY, MAX_SEQ_LEN
                    )
            vocab_input_ids = []
            vocab_attention = []
            for doc in flat_docs:
                idx = text_to_vocab_ids(
                    doc,
                    self.vocab,
                    unk_id=self.vocab_unk_id,
                    max_len=MAX_SEQ_LEN,
                    word_level=getattr(self, "vocab_word_level", False),
                    use_jieba=getattr(self, "vocab_use_jieba", False),
                )
                (idx, attn) = pad_vocab_ids(idx, self.vocab_pad_id, MAX_SEQ_LEN)
                vocab_input_ids.append(idx)
                vocab_attention.append(attn)
            _vocab_ids = torch.tensor(vocab_input_ids, dtype=torch.long)
            _vocab_attn = torch.tensor(vocab_attention, dtype=torch.long)
            data["news_padding"].vocab_input_ids = _vocab_ids.view(
                n_price, MAX_NEWS_PER_DAY, MAX_SEQ_LEN
            )
            data["news_padding"].vocab_attention_mask = _vocab_attn.view(
                n_price, MAX_NEWS_PER_DAY, MAX_SEQ_LEN
            )
            _skip_pad_emb = (
                getattr(self, "news_node_emb_cache_root", None)
                and hasattr(data["news"], "emb")
                and isinstance(getattr(data["news"], "emb", None), torch.Tensor)
                and (int(data["news"].emb.numel()) > 0)
            )
            if self.use_cached_news_padding_emb and (not _skip_pad_emb):
                emb_dim = 768
                if (
                    isinstance(self.news_padding_emb_cache_root, str)
                    and self.news_padding_emb_cache_root
                ):
                    emb_out = torch.zeros((n_price, MAX_NEWS_PER_DAY, emb_dim), dtype=torch.float32)
                    is_real_out = data["news_padding"].is_real.clone()
                    for p_idx in range(n_price):
                        comp_idx = int(data["price"].company_id[p_idx].item())
                        day_id = int(data["price"].day_id[p_idx].item())
                        comp_name = self.companies[comp_idx]
                        cur_date = range_dates[day_id]
                        comp_dates = comp_dates_sorted_map.get(comp_name, [])
                        date_to_pos = comp_date_to_pos.get(comp_name, {})
                        prev_date = (
                            comp_dates[date_to_pos[cur_date] - 1]
                            if cur_date in date_to_pos and date_to_pos[cur_date] >= 1
                            else None
                        )
                        date_str = prev_date if prev_date else cur_date
                        fpath = os.path.join(
                            self.news_padding_emb_cache_root, comp_name, f"{date_str}.pt"
                        )
                        if not os.path.exists(fpath):
                            continue
                        obj = torch.load(fpath, map_location="cpu")
                        emb = obj.get("emb", None)
                        ir = obj.get("is_real", None)
                        layout = obj.get("emb_layout", None)
                        n_day = len(texts_by_price[p_idx])
                        if not isinstance(emb, torch.Tensor) or emb.dim() != 2:
                            continue
                        if (
                            layout in ("packed", "by_news_id")
                            or int(emb.size(0)) != MAX_NEWS_PER_DAY
                        ):
                            n_f = int(emb.size(0))
                            n_take = min(n_f, n_day, MAX_NEWS_PER_DAY)
                            if n_take > 0:
                                emb_out[p_idx, :n_take] = emb[:n_take].to(dtype=torch.float32)
                        else:
                            if emb.numel() > 0:
                                emb_out[p_idx] = emb[:MAX_NEWS_PER_DAY].to(dtype=torch.float32)
                            if isinstance(ir, torch.Tensor) and ir.numel() > 0:
                                is_real_out[p_idx] = ir[:MAX_NEWS_PER_DAY].to(dtype=torch.float32)
                    data["news_padding"].emb = emb_out
                    data["news_padding"].is_real = is_real_out
                elif self.news_padding_emb_cache is not None:
                    cache = self.news_padding_emb_cache
                    idx_map = cache.get("index", {})
                    emb_cache = cache.get("emb", None)
                    is_real_cache = cache.get("is_real", None)
                    if isinstance(emb_cache, torch.Tensor) and emb_cache.numel() > 0:
                        Kc = int(cache.get("k", MAX_NEWS_PER_DAY))
                        if Kc == MAX_NEWS_PER_DAY:
                            emb_out = torch.zeros(
                                (n_price, MAX_NEWS_PER_DAY, emb_cache.size(-1)), dtype=torch.float32
                            )
                            is_real_out = data["news_padding"].is_real.clone()
                            for p_idx in range(n_price):
                                comp_idx = int(data["price"].company_id[p_idx].item())
                                day_id = int(data["price"].day_id[p_idx].item())
                                comp_name = self.companies[comp_idx]
                                cur_date = range_dates[day_id]
                                comp_dates = comp_dates_sorted_map.get(comp_name, [])
                                date_to_pos = comp_date_to_pos.get(comp_name, {})
                                prev_date = (
                                    comp_dates[date_to_pos[cur_date] - 1]
                                    if cur_date in date_to_pos and date_to_pos[cur_date] >= 1
                                    else cur_date
                                )
                                date_str = prev_date
                                row = idx_map.get((comp_name, date_str), None)
                                if row is None:
                                    continue
                                emb_out[p_idx] = emb_cache[row]
                                if (
                                    isinstance(is_real_cache, torch.Tensor)
                                    and is_real_cache.numel() > 0
                                ):
                                    is_real_out[p_idx] = is_real_cache[row]
                                else:
                                    is_real_out[p_idx] = (
                                        emb_cache[row].abs().sum(dim=-1) > 0
                                    ).float()
                            data["news_padding"].emb = emb_out
                            data["news_padding"].is_real = is_real_out
                if not hasattr(data["news_padding"], "emb"):
                    data["news_padding"].emb = torch.zeros(
                        (n_price, MAX_NEWS_PER_DAY, emb_dim), dtype=torch.float32
                    )

        def _encode_base_news_type(t: str) -> int:
            if t is None:
                return 2
            s = str(t)
            sl = s.lower().strip()
            if "市场" in s or sl in ("market", "global market", "global_market"):
                return 0
            if "行业" in s or "板块" in s or sl == "industry":
                return 1
            return 2

        news_type_feats = []
        news_to_type_edges = []
        if data["news"].x.size(0) > 0:
            k_str = str(self.news_cluster_k)
            for n_idx in range(data["news"].x.size(0)):
                comp_idx = int(data["news"].company_id[n_idx].item())
                comp_name = self.companies[comp_idx]
                date_str = data["news"].date[n_idx]
                n_type_str = data["news"].type[n_idx]
                n_id = data["news"].news_id[n_idx]
                base_tid = _encode_base_news_type(n_type_str)
                cluster_tid = -1
                if (
                    base_tid == 2
                    and comp_name in self.news_clusters
                    and (date_str in self.news_clusters[comp_name])
                ):
                    cluster_dict_all_k = self.news_clusters[comp_name][date_str]
                    if k_str in cluster_dict_all_k:
                        cluster_for_k = cluster_dict_all_k[k_str]
                        if n_id in cluster_for_k:
                            c_label = int(cluster_for_k[n_id])
                            if 0 <= c_label < self.news_cluster_k:
                                cluster_tid = c_label
                if base_tid == 2 and cluster_tid < 0:
                    sl = str(n_type_str or "").lower().strip()
                    _sent_to_c = {
                        "positive": 1,
                        "bullish": 1,
                        "negative": 2,
                        "bearish": 2,
                        "neutral": 3,
                        "mixed": 4,
                        "hold": 5,
                    }
                    if sl in _sent_to_c:
                        cluster_tid = int(_sent_to_c[sl])
                news_type_feats.append(torch.tensor([base_tid, cluster_tid], dtype=torch.float32))
                news_to_type_edges.append([n_idx, n_idx])
        if news_type_feats:
            data["news_type"].x = torch.stack(news_type_feats, dim=0)
        else:
            data["news_type"].x = torch.zeros((0, 2), dtype=torch.float32)
        if news_to_type_edges:
            data["news", "to", "news_type"].edge_index = torch.tensor(
                news_to_type_edges, dtype=torch.long
            ).t()
        else:
            data["news", "to", "news_type"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        price_type_feats = []
        price_to_type_edges = []
        if data["price"].x.size(0) > 0 and len(range_dates) > 0:
            for p_idx in range(data["price"].x.size(0)):
                comp_idx = int(data["price"].company_id[p_idx].item())
                day_i = int(data["price"].day_id[p_idx].item())
                comp_name = self.companies[comp_idx]
                target_date = range_dates[day_i]
                trend_date = self._resolve_trend_date(comp_name, target_date)
                ct = self.cluster_trend.get(comp_name, {}).get(trend_date, {})
                kmeans_id = (
                    int(ct["kmeans_cluster"])
                    if "kmeans_cluster" in ct and ct["kmeans_cluster"] is not None
                    else -1
                )
                dbscan_id = (
                    int(ct["dbscan_cluster"])
                    if "dbscan_cluster" in ct and ct["dbscan_cluster"] is not None
                    else -1
                )
                rt_cn = self.rule_trend.get(comp_name, {}).get(trend_date, None)
                rule_id = float(rule_trend_cn_to_id(rt_cn))
                feat = torch.tensor([kmeans_id, dbscan_id, rule_id], dtype=torch.float32)
                price_type_feats.append(feat)
                price_to_type_edges.append([p_idx, p_idx])
        if price_type_feats:
            data["price_type"].x = torch.stack(price_type_feats, dim=0)
        else:
            data["price_type"].x = torch.zeros((0, 3), dtype=torch.float32)
        if price_to_type_edges:
            data["price", "to", "price_type"].edge_index = torch.tensor(
                price_to_type_edges, dtype=torch.long
            ).t()
        else:
            data["price", "to", "price_type"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        industry_type_feats = []
        price_to_industry_edges = []
        industry_cn_to_id = getattr(self, "_industry_cn_to_id", None) or {}
        num_industries = int(getattr(self, "_num_industries_graph", 27) or 27)
        default_ind_id = 0
        if data["price"].x.size(0) > 0 and len(range_dates) > 0:
            for p_idx in range(data["price"].x.size(0)):
                comp_idx = int(data["price"].company_id[p_idx].item())
                comp_name = self.companies[comp_idx]
                ind_info = self.industry_info.get(comp_name, {})
                ind_cn = ind_info.get("industry_cn", "")
                ind_id = industry_cn_to_id.get(ind_cn, default_ind_id)
                industry_type_feats.append(torch.tensor([float(ind_id)], dtype=torch.float32))
                price_to_industry_edges.append([p_idx, p_idx])
        if industry_type_feats:
            data["industry_type"].x = torch.stack(industry_type_feats, dim=0)
        else:
            data["industry_type"].x = torch.zeros((0, 1), dtype=torch.float32)
        if price_to_industry_edges:
            data["price", "to", "industry_type"].edge_index = torch.tensor(
                price_to_industry_edges, dtype=torch.long
            ).t()
        else:
            data["price", "to", "industry_type"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        label_bin = []
        label_org = []
        label_strong = []
        vol_bin = []
        vol_org = []
        vol_valid = []
        price_raw_line = []
        for p_idx in range(data["price"].x.size(0)):
            comp_idx = int(data["price"].company_id[p_idx].item())
            day_i = int(data["price"].day_id[p_idx].item())
            comp_name = self.companies[comp_idx]
            cur_date = range_dates[day_i]
            raw_label = self.label.get(comp_name, {}).get(cur_date, None)
            comp_dates = comp_dates_sorted_map.get(comp_name, [])
            date_to_pos = comp_date_to_pos.get(comp_name, {})
            prev_date = None
            if cur_date in date_to_pos and date_to_pos[cur_date] >= 1:
                prev_date = comp_dates[date_to_pos[cur_date] - 1]
                day_dict = self.stock_data.get(comp_name, {})
                if prev_date in day_dict:
                    raw_3 = list(day_dict[prev_date])[:3]
                    wd = datetime.strptime(str(prev_date), "%Y-%m-%d").weekday() if prev_date else 0
                    onehot = [0.0] * 5
                    if wd < 5:
                        onehot[wd] = 1.0
                    raw_line_str = f"3cols:{raw_3} onehot:{onehot}"
                else:
                    last_row = data["price"].x[p_idx, -1, :].tolist()
                    raw_line_str = f"3cols:{last_row[:3]} onehot:{last_row[3:8]}"
            else:
                last_row = data["price"].x[p_idx, -1, :].tolist()
                raw_line_str = f"3cols:{last_row[:3]} onehot:{last_row[3:8]}"
            price_raw_line.append(raw_line_str)
            try:
                y = float(raw_label) if raw_label is not None else 0.0
            except Exception:
                y = 0.0
            y_bin = 1.0 if y > 0.0 else 0.0
            is_strong = 1.0 if y > 0.0055 or y < -0.005 else 0.0
            label_org.append(y)
            label_bin.append(y_bin)
            label_strong.append(is_strong)
            (vb, vo, vv) = _vol_change_from_map(
                getattr(self, "volume", None) or {}, comp_name, cur_date, prev_date
            )
            vol_bin.append(vb)
            vol_org.append(vo)
            vol_valid.append(vv)
        data["price"].raw_line = price_raw_line
        if label_bin:
            data["label"].x = torch.tensor(label_bin, dtype=torch.float32).view(-1, 1)
            data["label"].org = torch.tensor(label_org, dtype=torch.float32).view(-1, 1)
            data["label"].strong_mask = torch.tensor(label_strong, dtype=torch.bool).view(-1, 1)
            data["label"].vol_x = torch.tensor(vol_bin, dtype=torch.float32).view(-1, 1)
            data["label"].vol_org = torch.tensor(vol_org, dtype=torch.float32).view(-1, 1)
            data["label"].vol_valid_mask = torch.tensor(vol_valid, dtype=torch.bool).view(-1, 1)
            if hasattr(data["price"], "valid_mask"):
                data["label"].valid_mask = data["price"].valid_mask.clone()
            else:
                data["label"].valid_mask = torch.ones_like(
                    data["label"].strong_mask, dtype=torch.bool
                )
        else:
            data["label"].x = torch.zeros((0, 1), dtype=torch.float32)
            data["label"].org = torch.zeros((0, 1), dtype=torch.float32)
            data["label"].strong_mask = torch.zeros((0, 1), dtype=torch.bool)
            data["label"].valid_mask = torch.zeros((0, 1), dtype=torch.bool)
            data["label"].vol_x = torch.zeros((0, 1), dtype=torch.float32)
            data["label"].vol_org = torch.zeros((0, 1), dtype=torch.float32)
            data["label"].vol_valid_mask = torch.zeros((0, 1), dtype=torch.bool)
        num_comp = len(self.companies)
        num_seg = len(range_dates)
        n_price = data["price"].x.size(0)
        vw_cap = max(1, int(self.num_virtual_window))
        vw_num = min(vw_cap, num_seg)
        data["virtual_window"].x = torch.zeros((vw_num, 1), dtype=torch.float32)
        vw_edges_price = []
        for p_idx in range(n_price):
            day_id = int(data["price"].day_id[p_idx].item())
            if day_id < vw_num:
                vw_edges_price.append([p_idx, day_id])
        vw_edges_news = []
        for n_idx in range(data["news"].x.size(0)):
            day_id = int(data["news"].day_id[n_idx].item())
            if day_id < vw_num:
                vw_edges_news.append([n_idx, day_id])
        data["price", "to", "virtual_window"].edge_index = (
            torch.tensor(vw_edges_price, dtype=torch.long).t()
            if vw_edges_price
            else torch.zeros((2, 0), dtype=torch.long)
        )
        data["news", "to", "virtual_window"].edge_index = (
            torch.tensor(vw_edges_news, dtype=torch.long).t()
            if vw_edges_news
            else torch.zeros((2, 0), dtype=torch.long)
        )
        vn_num = n_price
        data["virtual_news"].x = torch.zeros((vn_num, 1), dtype=torch.float32)
        p_to_vn = [[p_idx, p_idx] for p_idx in range(n_price)]
        key2pidx = {}
        for p_idx in range(n_price):
            comp_idx = int(data["price"].company_id[p_idx].item())
            day_id = int(data["price"].day_id[p_idx].item())
            key2pidx[comp_idx, day_id] = p_idx
        n_to_vn = []
        for n_idx in range(data["news"].x.size(0)):
            comp_idx = int(data["news"].company_id[n_idx].item())
            day_id = int(data["news"].day_id[n_idx].item())
            p_idx = key2pidx.get((comp_idx, day_id))
            if p_idx is not None:
                n_to_vn.append([n_idx, p_idx])
        data["price", "to", "virtual_news"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["news", "to", "virtual_news"].edge_index = (
            torch.tensor(n_to_vn, dtype=torch.long).t()
            if n_to_vn
            else torch.zeros((2, 0), dtype=torch.long)
        )
        NUM_NEWS_TYPE = 12
        data["virtual_news_type"].x = torch.zeros((NUM_NEWS_TYPE, 1), dtype=torch.float32)
        news_type_to_vnt = []
        if data["news_type"].x.size(0) > 0:
            for t_idx in range(data["news_type"].x.size(0)):
                base_tid = int(data["news_type"].x[t_idx, 0].item())
                cluster_tid = int(data["news_type"].x[t_idx, 1].item())
                if base_tid == 0:
                    vnt_idx = 0
                elif base_tid == 1:
                    vnt_idx = 1
                else:
                    vnt_idx = 2 + (cluster_tid if cluster_tid >= 0 and cluster_tid < 10 else 0)
                vnt_idx = min(vnt_idx, NUM_NEWS_TYPE - 1)
                news_type_to_vnt.append([t_idx, vnt_idx])
        data["news_type", "to", "virtual_news_type"].edge_index = (
            torch.tensor(news_type_to_vnt, dtype=torch.long).t()
            if news_type_to_vnt
            else torch.zeros((2, 0), dtype=torch.long)
        )
        NUM_PRICE_TYPE = 10
        data["virtual_price_type"].x = torch.zeros((NUM_PRICE_TYPE, 1), dtype=torch.float32)
        price_type_to_vpt = []
        if data["price_type"].x.size(0) > 0:
            for t_idx in range(data["price_type"].x.size(0)):
                kmeans_id = (
                    int(data["price_type"].x[t_idx, 0].item())
                    if data["price_type"].x[t_idx, 0].item() >= 0
                    else 0
                )
                dbscan_id = (
                    int(data["price_type"].x[t_idx, 1].item())
                    if data["price_type"].x[t_idx, 1].item() >= 0
                    else 0
                )
                rule_id = (
                    int(data["price_type"].x[t_idx, 2].item())
                    if data["price_type"].x[t_idx, 2].item() >= 0
                    else 0
                )
                src = self.price_type_source
                if src == "kmeans":
                    src_id = kmeans_id
                elif src == "dbscan":
                    src_id = dbscan_id
                elif src == "mix":
                    src_id = kmeans_id + dbscan_id + rule_id
                else:
                    src_id = rule_id
                vpt_idx = int(src_id) % NUM_PRICE_TYPE
                price_type_to_vpt.append([t_idx, vpt_idx])
        data["price_type", "to", "virtual_price_type"].edge_index = (
            torch.tensor(price_type_to_vpt, dtype=torch.long).t()
            if price_type_to_vpt
            else torch.zeros((2, 0), dtype=torch.long)
        )
        data["virtual_price_type", "to", "price_type"].edge_index = (
            data["price_type", "to", "virtual_price_type"].edge_index.flip(0)
            if price_type_to_vpt
            else torch.zeros((2, 0), dtype=torch.long)
        )
        num_days_seg = len(range_dates)
        NUM_VIRTUAL_INDUSTRY = num_industries * num_days_seg
        data["virtual_industry"].x = torch.zeros((NUM_VIRTUAL_INDUSTRY, 1), dtype=torch.float32)
        industry_type_to_vi = []
        if data["industry_type"].x.size(0) > 0:
            for t_idx in range(data["industry_type"].x.size(0)):
                ind_id = int(data["industry_type"].x[t_idx, 0].item())
                ind_id = max(0, min(ind_id, num_industries - 1))
                day_id = int(data["price"].day_id[t_idx].item())
                vi_idx = day_id * num_industries + ind_id
                vi_idx = min(vi_idx, NUM_VIRTUAL_INDUSTRY - 1)
                industry_type_to_vi.append([t_idx, vi_idx])
        data["industry_type", "to", "virtual_industry"].edge_index = (
            torch.tensor(industry_type_to_vi, dtype=torch.long).t()
            if industry_type_to_vi
            else torch.zeros((2, 0), dtype=torch.long)
        )
        data["virtual_industry", "to", "industry_type"].edge_index = (
            data["industry_type", "to", "virtual_industry"].edge_index.flip(0)
            if industry_type_to_vi
            else torch.zeros((2, 0), dtype=torch.long)
        )
        data["range_dates"] = range_dates
        return buffer_to_window_sample(data)


class _CMINUSBuilderDataset(_CMINBuilderDataset):

    def _load_all_data(self, root):
        return load_cmin_us_all_data(
            root, news_type_subdir=getattr(self, "news_type_subdir", "news_type")
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seg_keep_tail = True
        self._rebuild_seg_windows()


class MarketWindowDataset(torch.utils.data.Dataset):

    def __init__(self, profile: DatasetProfile | str, mode: str = "train", **kwargs):
        self.profile = get_profile(profile) if isinstance(profile, str) else profile
        p = self.profile
        common = dict(
            root=kwargs.get("root", p.default_root),
            mode=mode,
            seg_length=kwargs.get("seg_length", p.seg_length),
            price_window=kwargs.get("price_window", p.price_window),
            news_cluster_k=kwargs.get("news_cluster_k", 10),
            price_type_source=kwargs.get("price_type_source", "rule"),
            num_virtual_window=kwargs.get("num_virtual_window", p.num_virtual_window),
            news_padding_k=kwargs.get("news_padding_k", 5),
            use_cached_news_padding_emb=kwargs.get("use_cached_news_padding_emb", False),
            news_padding_emb_cache_path=kwargs.get("news_padding_emb_cache_path"),
            skip_finbert_tokenizer=kwargs.get("skip_finbert_tokenizer", True),
            vocab_path=kwargs.get("vocab_path", p.vocab_path),
            finbert_model_path=kwargs.get("finbert_model_path"),
        )
        _split_mode = str(kwargs.get("split_mode", "contiguous") or "contiguous").strip().lower()
        _purge = bool(kwargs.get("train_lookback_purge", True))
        if p.key in ("cmin_cn", "cmin_us", "massive", "stocknet"):
            common["seg_keep_tail"] = kwargs.get("seg_keep_tail", p.seg_keep_tail)
            common["news_type_subdir"] = kwargs.get("news_type_subdir", "news_type")
            common["split_dates"] = dict(kwargs.get("split_dates") or p.split_dates)
            common["split_mode"] = _split_mode
            common["train_lookback_purge"] = _purge
        if p.key == "cmin_cn":
            common["require_wind_news_source"] = bool(kwargs.get("require_wind_news_source", True))
            common["require_window_news"] = bool(kwargs.get("require_window_news", False))
            common["subset_top_companies"] = int(kwargs.get("subset_top_companies", 0) or 0)
            _wl = kwargs.get("companies_whitelist")
            if _wl:
                common["companies_whitelist"] = list(_wl)
            self._inner = _CMINBuilderDataset(**common)
        elif p.key in ("cmin_us", "massive", "stocknet"):
            self._inner = _CMINUSBuilderDataset(**common)
        elif p.key in ("csmd50", "csmd300", "sse50"):
            common["news_type_subdir"] = kwargs.get("news_type_subdir", "news_type")
            common["csmd_news_source"] = (
                str(kwargs.get("csmd_news_source", "llm_extract") or "llm_extract").strip().lower()
            )
            common["split_dates"] = dict(kwargs.get("split_dates") or p.split_dates)
            common["split_mode"] = _split_mode
            common["train_lookback_purge"] = _purge
            _ip = str(kwargs.get("news_impact_oracle_path", "") or "").strip()
            if _ip:
                common["news_impact_oracle_path"] = _ip
            self._inner = _CSMDBuilderDataset(**common)
        else:
            raise KeyError(p.key)

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        return self._inner[idx]


_PROFILES: dict[str, DatasetProfile] = {
    "cmin_cn": DatasetProfile(
        key="cmin_cn",
        description="CMIN-CN 中文 300 股",
        default_root=os.path.join(_PATTERN_ROOT, "data", "CMIN-CN"),
        vocab_path=os.path.join(DICT_DIR, "dict_cn.pkl"),
        grid_num_stocks=300,
        num_score_news_types=12,
        num_industry=27,
        num_rule_trend=8,
        price_window=5,
        seg_length=10,
        seg_keep_tail=False,
        num_virtual_window=10,
        split_dates={
            "train_start": "2018-01-01",
            "train_end": "2021-04-30",
            "val_start": "2021-05-01",
            "val_end": "2021-08-31",
            "test_start": "2021-09-01",
            "test_end": "2021-12-31",
        },
        pattern_subdir="CMIN-CN",
        include_calendar_frac=True,
        include_industry=True,
        include_rule_trend_node=False,
        news_subtype=False,
        virtual_news_type_count=12,
    ),
    "cmin_us": DatasetProfile(
        key="cmin_us",
        description="CMIN-US 英文 ~110 股",
        default_root=os.path.join(_PATTERN_ROOT, "data", "CMIN-US"),
        vocab_path=os.path.join(DICT_DIR, "dict_us.pkl"),
        grid_num_stocks=110,
        num_score_news_types=12,
        num_industry=27,
        num_rule_trend=8,
        price_window=5,
        seg_length=10,
        seg_keep_tail=True,
        num_virtual_window=10,
        split_dates={
            "train_start": "2018-01-01",
            "train_end": "2021-04-30",
            "val_start": "2021-05-01",
            "val_end": "2021-08-31",
            "test_start": "2021-09-01",
            "test_end": "2021-12-31",
        },
        pattern_subdir="CMIN-US",
        include_calendar_frac=True,
        include_industry=True,
        include_rule_trend_node=False,
        news_subtype=False,
        virtual_news_type_count=12,
    ),
    "massive": DatasetProfile(
        key="massive",
        description="Massive.com StockTable ~88 US stocks (2022-2025, with news sentiment)",
        default_root=os.path.join(_PATTERN_ROOT, "data", "massive_data"),
        vocab_path=os.path.join(DICT_DIR, "dict_massive.pkl"),
        grid_num_stocks=88,
        num_score_news_types=12,
        num_industry=27,
        num_rule_trend=8,
        price_window=5,
        seg_length=10,
        seg_keep_tail=True,
        num_virtual_window=10,
        split_dates={
            "train_start": "2022-01-01",
            "train_end": "2023-12-31",
            "val_start": "2024-01-01",
            "val_end": "2024-12-31",
            "test_start": "2025-01-01",
            "test_end": "2025-12-31",
        },
        pattern_subdir="massive_data",
        include_calendar_frac=True,
        include_industry=True,
        include_rule_trend_node=False,
        news_subtype=False,
        virtual_news_type_count=12,
    ),
    "stocknet": DatasetProfile(
        key="stocknet",
        description="StockNet ACL18 88 stocks (tweets 2014-2015; StockNet 9 sectors)",
        default_root="data/stocknet",
        vocab_path="dict/dict_stocknet.pkl",
        grid_num_stocks=88,
        num_score_news_types=12,
        num_industry=27,
        num_rule_trend=8,
        price_window=5,
        seg_length=10,
        seg_keep_tail=True,
        num_virtual_window=10,
        split_dates={
            "train_start": "2014-01-01",
            "train_end": "2015-06-30",
            "val_start": "2015-07-01",
            "val_end": "2015-09-30",
            "test_start": "2015-10-01",
            "test_end": "2015-12-31",
        },
        pattern_subdir="stocknet",
        include_calendar_frac=True,
        include_industry=True,
        include_rule_trend_node=False,
        news_subtype=False,
        virtual_news_type_count=12,
    ),
    "csmd50": DatasetProfile(
        key="csmd50",
        description="CSMD 50 股",
        default_root=os.path.join(_PATTERN_ROOT, "data", "CSMD50"),
        vocab_path=os.path.join(DICT_DIR, "dict_csmd.pkl"),
        grid_num_stocks=50,
        num_score_news_types=13,
        num_industry=27,
        num_rule_trend=8,
        price_window=5,
        seg_length=10,
        seg_keep_tail=True,
        num_virtual_window=10,
        split_dates={
            "train_start": "2021-01-01",
            "train_end": "2023-01-01",
            "val_start": "2023-01-02",
            "val_end": "2024-01-02",
            "test_start": "2024-01-03",
            "test_end": "2024-12-31",
        },
        pattern_subdir="CSMD50",
        include_calendar_frac=False,
        include_industry=True,
        include_rule_trend_node=True,
        news_subtype=True,
        virtual_news_type_count=13,
    ),
    "csmd300": DatasetProfile(
        key="csmd300",
        description="CSMD 300 股",
        default_root=os.path.join(_PATTERN_ROOT, "data", "CSMD300"),
        vocab_path=os.path.join(DICT_DIR, "dict_csmd.pkl"),
        grid_num_stocks=300,
        num_score_news_types=13,
        num_industry=27,
        num_rule_trend=8,
        price_window=5,
        seg_length=10,
        seg_keep_tail=True,
        num_virtual_window=10,
        split_dates={
            "train_start": "2021-01-01",
            "train_end": "2023-01-01",
            "val_start": "2023-01-02",
            "val_end": "2024-01-02",
            "test_start": "2024-01-03",
            "test_end": "2024-12-31",
        },
        pattern_subdir="CSMD300",
        include_calendar_frac=False,
        include_industry=True,
        include_rule_trend_node=True,
        news_subtype=True,
        virtual_news_type_count=13,
    ),
    "sse50": DatasetProfile(
        key="sse50",
        description="SSE50 from news_weibo (Tushare news + Sina prices, 2022-2025)",
        default_root="data/sse50_news",
        vocab_path="dict/dict_sse50.pkl",
        grid_num_stocks=48,
        num_score_news_types=13,
        num_industry=27,
        num_rule_trend=8,
        price_window=5,
        seg_length=10,
        seg_keep_tail=True,
        num_virtual_window=10,
        split_dates={
            "train_start": "2022-01-01",
            "train_end": "2023-12-31",
            "val_start": "2024-01-02",
            "val_end": "2024-12-31",
            "test_start": "2025-01-02",
            "test_end": "2025-12-31",
        },
        pattern_subdir="sse50_news",
        include_calendar_frac=False,
        include_industry=True,
        include_rule_trend_node=True,
        news_subtype=True,
        virtual_news_type_count=13,
    ),
}
_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_SCRIPT = {
    "cmin_cn": "main_cmin_cn.py",
    "cmin_us": "main_cmin_us.py",
    "csmd50": "main_csmd_50.py",
    "csmd300": "main_csmd_300.py",
}
_LEGACY_DATASET_CLASSES = {
    "CMIN_CN_Dataset": "cmin_cn",
    "CMIN_US_Dataset": "cmin_us",
    "CSMD_50_Dataset": "csmd50",
    "CSMD_300_Dataset": "csmd300",
}
_INJECT_MARKER = "# --- unified legacy bridge ---"


def get_profile(key: str) -> DatasetProfile:
    k = key.strip().lower()
    if k not in _PROFILES:
        raise KeyError(f"unknown dataset {key!r}, choose from {list(_PROFILES)}")
    return _PROFILES[k]


def load_price(directory_path):
    """
    Loads all .txt files from price/preprocessed.
    Returns stock_data (H/L/C), label (Close return), raw_line, volume (abs shares).
    Preprocessed cols: date, label, open_rel, high_rel, low_rel, close_rel, volume.
    """
    stock_data = {}
    label = {}
    raw_line = {}
    volume = {}
    for filename in _list_input_directory(directory_path):
        if filename.endswith(".txt"):
            company_name = filename.replace(".txt", "")
            file_path = os.path.join(directory_path, filename)
            stock_data[company_name] = {}
            label[company_name] = {}
            raw_line[company_name] = {}
            volume[company_name] = {}
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 6:
                        continue
                    time = parts[0]
                    features = [float(x) for x in parts[3:6]]
                    stock_data[company_name][time] = features
                    label[company_name][time] = parts[1]
                    raw_line[company_name][time] = line.strip()
                    try:
                        volume[company_name][time] = float(parts[6]) if len(parts) >= 7 else 0.0
                    except Exception:
                        volume[company_name][time] = 0.0
    return (stock_data, label, raw_line, volume)


def _vol_change_from_map(
    volume_map: dict, comp_name: str, cur_date: str, prev_date: str | None
) -> tuple[float, float, bool]:
    """Vol change from map."""
    if not prev_date:
        return (0.0, 0.0, False)
    by_d = volume_map.get(comp_name, {}) if volume_map else {}
    try:
        v_cur = float(by_d.get(cur_date, 0.0) or 0.0)
        v_prev = float(by_d.get(prev_date, 0.0) or 0.0)
    except Exception:
        return (0.0, 0.0, False)
    if v_cur <= 0.0 or v_prev <= 0.0:
        return (0.0, 0.0, False)
    org = v_cur / v_prev - 1.0
    return (1.0 if org > 0.0 else 0.0, float(org), True)


def _vol_return_seq_for_dates(
    volume_map: dict, comp_name: str, dates: list, all_dates_sorted: list, date_to_pos: dict
) -> list[list[float]]:
    """Vol return seq for dates."""
    out: list[list[float]] = []
    for dd in dates:
        pos = date_to_pos.get(dd)
        prev_d = all_dates_sorted[pos - 1] if isinstance(pos, int) and pos > 0 else None
        (_bin, org, valid) = _vol_change_from_map(volume_map, comp_name, str(dd), prev_d)
        out.append([float(org) if valid else 0.0])
    return out


def _day_of_year_frac(date_str: str) -> float:
    """Day of year frac."""
    try:
        d = datetime.strptime(str(date_str), "%Y-%m-%d")
    except Exception:
        return 0.0
    start = datetime(d.year, 1, 1)
    doy = (d - start).days + 1
    days_in_year = 366 if calendar.isleap(d.year) else 365
    return float(doy) / float(days_in_year)


def load_industry_info(csv_path):
    """Load industry info."""
    industry_map = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            (stock_name, industry_cn, industry_en) = (row[0], row[1], row[2])
            industry_map[stock_name] = {"industry_cn": industry_cn, "industry_en": industry_en}
    return industry_map


def load_industry_dict(pkl_path: str | None = None):
    """Load industry dict."""
    if pkl_path is None:
        pkl_path = os.path.join(DICT_DIR, "industry_dict.pkl")
    elif not os.path.isabs(pkl_path) and (not os.path.isfile(pkl_path)):
        pkl_path = os.path.join(DICT_DIR, os.path.basename(pkl_path))
    if not os.path.isfile(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_code_name_mapping(json_path):
    """Load code name mapping."""
    with open(json_path, "r", encoding="utf-8") as f:
        code_to_name = json.load(f)
    name_to_code = {v: k for (k, v) in code_to_name.items()}
    return (code_to_name, name_to_code)


def _csv_row_source_contains_wind(row: list) -> bool:
    """Csv row source contains wind."""
    if len(row) < 5:
        return False
    return "wind" in str(row[4]).lower()


def load_news_and_types(news_dir, news_type_dir, *, require_wind_source: bool = True):
    """Load news and types."""
    news_texts = defaultdict(lambda: defaultdict(dict))
    news_types = defaultdict(lambda: defaultdict(dict))
    company_list = [
        d for d in _list_input_directory(news_dir) if os.path.isdir(os.path.join(news_dir, d))
    ]
    for company in company_list:
        type_dir = os.path.join(news_type_dir, company)
        company_news_dir = os.path.join(news_dir, company)
        if not os.path.exists(type_dir) or not os.path.exists(company_news_dir):
            continue
        for file_name in _list_input_directory(type_dir):
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
                        if require_wind_source and (not _csv_row_source_contains_wind(row)):
                            continue
                        id_to_text[str(idx)] = row[3]
            except Exception:
                continue
            for news_id, news_type in type_dict.items():
                if news_id in id_to_text:
                    news_texts[company][date_str][news_id] = id_to_text[news_id]
                    news_types[company][date_str][news_id] = news_type
    return (news_texts, news_types)


def load_news_clusters(news_cluster_root):
    """Load news clusters."""
    news_clusters = defaultdict(lambda: defaultdict(dict))
    if not os.path.exists(news_cluster_root):
        return news_clusters
    for company in _list_input_directory(news_cluster_root):
        company_dir = os.path.join(news_cluster_root, company)
        if not os.path.isdir(company_dir):
            continue
        for file_name in _list_input_directory(company_dir):
            if not file_name.endswith(".json"):
                continue
            date_str = file_name.replace(".json", "")
            file_path = os.path.join(company_dir, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                news_clusters[company][date_str] = data
            except Exception:
                continue
    return news_clusters


def load_cluster_trend(cluster_trend_dir, code_to_name):
    """Load cluster trend."""
    cluster_trend = defaultdict(dict)
    if not os.path.exists(cluster_trend_dir):
        return cluster_trend
    for filename in _list_input_directory(cluster_trend_dir):
        if not filename.endswith(".csv"):
            continue
        code = filename.replace(".csv", "")
        stock_name = code_to_name.get(code)
        if stock_name is None:
            continue
        file_path = os.path.join(cluster_trend_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                date = row.get("Date")
                if not date:
                    continue
                kmeans_raw = row.get("kmeans_cluster", "").strip()
                dbscan_raw = row.get("dbscan_cluster", "").strip()

                def _to_int(x):
                    if x == "" or x is None:
                        return None
                    try:
                        return int(float(x))
                    except Exception:
                        return None

                cluster_trend[stock_name][date] = {
                    "kmeans_cluster": _to_int(kmeans_raw),
                    "dbscan_cluster": _to_int(dbscan_raw),
                }
    return cluster_trend


def load_rule_trend(trend_dir, code_to_name):
    """Load rule trend."""
    rule_trend = defaultdict(dict)
    if not os.path.exists(trend_dir):
        return rule_trend
    for filename in _list_input_directory(trend_dir):
        if not filename.endswith(".csv"):
            continue
        code = filename.replace(".csv", "")
        stock_name = code_to_name.get(code)
        if stock_name is None:
            continue
        file_path = os.path.join(trend_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                date = row.get("Date")
                if not date:
                    continue
                main_trend_cn = row.get("main_trend_cn", "").strip()
                rule_trend[stock_name][date] = main_trend_cn
    return rule_trend


def load_cmin_cn_all_data(
    dataset_root, news_type_subdir: str = "news_type", *, require_wind_news_source: bool = True
):
    """Load cmin cn all data."""
    return _load_cmin_common_data(
        dataset_root,
        require_wind_news_source=bool(require_wind_news_source),
        news_type_subdir=news_type_subdir,
    )


def load_cmin_us_all_data(dataset_root, news_type_subdir: str = "news_type"):
    """Load cmin us all data."""
    return _load_cmin_common_data(
        dataset_root, require_wind_news_source=False, news_type_subdir=news_type_subdir
    )


def _cmin_news_type_json_has_embedded_content(news_type_dir: str) -> bool:
    """Cmin news type json has embedded content."""
    if not os.path.isdir(news_type_dir):
        return False
    for company in _list_input_directory(news_type_dir):
        cdir = os.path.join(news_type_dir, company)
        if not os.path.isdir(cdir):
            continue
        for fn in _list_input_directory(cdir):
            if not fn.endswith(".json") or fn.startswith("_"):
                continue
            try:
                with open(os.path.join(cdir, fn), "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            for v in data.values():
                if isinstance(v, dict) and (v.get("内容") or v.get("content")):
                    return True
            return False
    return False


def _load_cmin_common_data(
    dataset_root, *, require_wind_news_source: bool, news_type_subdir: str = "news_type"
):
    price_dir = os.path.join(dataset_root, "price", "preprocessed")
    (stock_data, label, raw_line, volume) = load_price(price_dir)
    news_dir = os.path.join(dataset_root, "news")
    sub = str(news_type_subdir or "news_type").strip() or "news_type"
    news_type_dir = os.path.join(dataset_root, "llm_extract", sub)
    if _cmin_news_type_json_has_embedded_content(news_type_dir):
        (news_texts, news_types) = load_news_and_types_from_llm_extract(news_type_dir)
    else:
        (news_texts, news_types) = load_news_and_types(
            news_dir, news_type_dir, require_wind_source=require_wind_news_source
        )
    pattern_root = "./pattern"
    ds_name = os.path.basename(os.path.normpath(dataset_root))
    pattern_dir = os.path.join(pattern_root, ds_name)
    news_cluster_root = os.path.join(pattern_dir, "news_cluster")
    if not os.path.exists(news_cluster_root):
        news_cluster_root = os.path.join(dataset_root, "pattern", "news_cluster")
    news_clusters = load_news_clusters(news_cluster_root)
    code_name_path = os.path.join(dataset_root, "code_name.json")
    if os.path.exists(code_name_path):
        (code_to_name, name_to_code) = load_code_name_mapping(code_name_path)
    else:
        code_to_name = {str(k): str(k) for k in stock_data.keys()}
        name_to_code = {str(k): str(k) for k in stock_data.keys()}
    cluster_trend_dir = os.path.join(pattern_dir, "cluster_trend")
    if not os.path.exists(cluster_trend_dir):
        cluster_trend_dir = os.path.join(dataset_root, "pattern", "cluster_trend")
    cluster_trend = load_cluster_trend(cluster_trend_dir, code_to_name)
    trend_dir = os.path.join(pattern_dir, "trend")
    if not os.path.exists(trend_dir):
        trend_dir = os.path.join(dataset_root, "pattern", "trend")
    rule_trend = load_rule_trend(trend_dir, code_to_name)
    industry_csv = os.path.join(dataset_root, "comp_industry", "comp_indus.csv")
    industry_info = load_industry_info(industry_csv)
    return {
        "stock_data": stock_data,
        "label": label,
        "raw_line": raw_line,
        "volume": volume,
        "news_texts": news_texts,
        "news_types": news_types,
        "news_clusters": news_clusters,
        "cluster_trend": cluster_trend,
        "rule_trend": rule_trend,
        "industry_info": industry_info,
        "code_to_name": code_to_name,
        "name_to_code": name_to_code,
    }


def default_news_node_emb_dir(
    dataset_root: str,
    news_type_subdir: str = "news_type",
    max_seq_len: int = 128,
    pooling: str = "pooler",
) -> str:
    """Default news node emb dir."""
    sub = str(news_type_subdir or "news_type").strip() or "news_type"
    safe = sub.replace(os.sep, "_").replace("/", "_")
    return os.path.join(
        dataset_root, "precomputed", f"news_node_finbert_len{int(max_seq_len)}_{pooling}", safe
    )


def _encode_texts_finbert(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    max_seq_len: int = 128,
    batch_size: int = 32,
    pooling: str = "pooler",
) -> torch.Tensor:
    """Encode texts finbert."""
    if not texts:
        return torch.zeros((0, 768), dtype=torch.float32)
    outs: list[torch.Tensor] = []
    bs = max(1, int(batch_size))
    for i in range(0, len(texts), bs):
        chunk = texts[i : i + bs]
        inputs = tokenizer(
            chunk, padding=True, truncation=True, max_length=int(max_seq_len), return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        if pooling == "pooler" and getattr(outputs, "pooler_output", None) is not None:
            emb = outputs.pooler_output.detach().cpu()
        else:
            emb = outputs.last_hidden_state[:, 0, :].detach().cpu()
        outs.append(emb.float())
    return torch.cat(outs, dim=0)


def load_news_node_emb_rows(
    cache_root: str,
    companies: list[str],
    company_ids: list[int],
    news_dates: list[str],
    news_ids: list,
    emb_dim: int = 768,
) -> torch.Tensor:
    """Load news node emb rows."""
    n = len(news_ids)
    if n == 0:
        return torch.zeros((0, emb_dim), dtype=torch.float32)
    out = torch.zeros((n, emb_dim), dtype=torch.float32)
    file_cache: dict[tuple[str, str], tuple[dict[str, int], torch.Tensor] | None] = {}
    miss = 0
    for i in range(n):
        cidx = int(company_ids[i])
        comp = companies[cidx] if 0 <= cidx < len(companies) else str(cidx)
        date_str = str(news_dates[i])
        nid = str(news_ids[i])
        key = (comp, date_str)
        if key not in file_cache:
            fpath = os.path.join(cache_root, comp, f"{date_str}.pt")
            if not os.path.isfile(fpath):
                file_cache[key] = None
            else:
                obj = torch.load(fpath, map_location="cpu")
                emb = obj.get("emb", None)
                if not isinstance(emb, torch.Tensor) or emb.dim() != 2:
                    file_cache[key] = None
                else:
                    ids = [str(x) for x in obj.get("news_ids") or []]
                    if ids and len(ids) == int(emb.size(0)):
                        id2i = {x: j for (j, x) in enumerate(ids)}
                    else:
                        id2i = {str(j): j for j in range(int(emb.size(0)))}
                    file_cache[key] = (id2i, emb.float())
        packed = file_cache[key]
        if packed is None:
            miss += 1
            continue
        (id2i, emb) = packed
        j = id2i.get(nid)
        if j is None:
            miss += 1
            continue
        if 0 <= int(j) < int(emb.size(0)):
            out[i] = emb[int(j)]
        else:
            miss += 1
    if miss > 0:
        print(f"[news_node_emb] warn: {miss}/{n} rows missing in cache root={cache_root}")
    return out


def _get_content_and_type_csmd(item):
    """Get content and type csmd."""
    if not isinstance(item, dict):
        return (None, None)
    content = item.get("内容") or item.get("content") or ""
    dtype = item.get("类型") or item.get("type") or ""
    return (content.strip() if content else None, dtype.strip() if dtype else None)


def load_news_from_csmd_raw_csv(news_dir: str):
    """Load news from csmd raw csv."""
    news_texts = defaultdict(lambda: defaultdict(dict))
    news_types = defaultdict(lambda: defaultdict(dict))
    if not os.path.isdir(news_dir):
        return (news_texts, news_types)
    for company in _list_input_directory(news_dir):
        company_dir = os.path.join(news_dir, company)
        if not os.path.isdir(company_dir):
            continue
        for file_name in _list_input_directory(company_dir):
            if not file_name.endswith(".csv"):
                continue
            date_str = file_name.replace(".csv", "")
            file_path = os.path.join(company_dir, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for idx, row in enumerate(reader):
                        text = (row.get("text") or "").strip()
                        if not text:
                            continue
                        news_id = str(idx)
                        news_texts[company][date_str][news_id] = text
                        news_types[company][date_str][news_id] = ""
            except Exception:
                continue
    return (news_texts, news_types)


def load_news_and_types_from_llm_extract(llm_extract_news_type_dir):
    """Load news and types from llm extract."""
    news_texts = defaultdict(lambda: defaultdict(dict))
    news_types = defaultdict(lambda: defaultdict(dict))
    if not os.path.exists(llm_extract_news_type_dir):
        return (news_texts, news_types)
    company_list = [
        d
        for d in _list_input_directory(llm_extract_news_type_dir)
        if os.path.isdir(os.path.join(llm_extract_news_type_dir, d))
    ]
    for company in company_list:
        company_dir = os.path.join(llm_extract_news_type_dir, company)
        for file_name in _list_input_directory(company_dir):
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
                (content, dtype) = _get_content_and_type_csmd(item)
                if not content:
                    continue
                news_texts[company][date_str][news_id] = content
                news_types[company][date_str][news_id] = dtype if dtype else ""
    return (news_texts, news_types)


def load_news_subtypes(news_subtype_root):
    """Load news subtypes."""
    news_subtypes = defaultdict(lambda: defaultdict(dict))
    if not os.path.exists(news_subtype_root):
        return news_subtypes
    for company in _list_input_directory(news_subtype_root):
        company_dir = os.path.join(news_subtype_root, company)
        if not os.path.isdir(company_dir):
            continue
        for file_name in _list_input_directory(company_dir):
            if not file_name.endswith(".json"):
                continue
            date_str = file_name.replace(".json", "")
            file_path = os.path.join(company_dir, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    news_subtypes[company][date_str] = {str(k): int(v) for (k, v) in data.items()}
            except Exception:
                continue
    return news_subtypes


def rule_trend_cn_to_id(trend_cn: str | None) -> int:
    if not trend_cn:
        return 0
    s = str(trend_cn).strip()
    return int(RULE_TREND_CN_TO_ID.get(s, 0))


def load_csmd_all_data(
    dataset_root, news_type_subdir: str = "news_type", *, news_source: str = "raw"
):
    """Load csmd all data."""
    price_dir = os.path.join(dataset_root, "price", "preprocessed")
    (stock_data, label, raw_line, volume) = load_price(price_dir)
    if str(news_source or "raw").strip().lower() != "raw":
        raise ValueError("The released CSMD recipes require raw news CSV inputs.")
    news_dir = os.path.join(dataset_root, "news")
    if not os.path.isdir(news_dir):
        raise FileNotFoundError(f"CSMD raw news dir not found: {news_dir}")
    (news_texts, news_types) = load_news_from_csmd_raw_csv(news_dir)
    # These auxiliary graph fields are not consumed by the released model.
    news_subtypes, cluster_trend, rule_trend = {}, {}, {}
    code_to_name = {c: c for c in stock_data.keys()}
    name_to_code = {c: c for c in stock_data.keys()}
    industry_csv = os.path.join(dataset_root, "comp_industry", "comp_indus.csv")
    industry_info = load_industry_info(industry_csv)
    return {
        "stock_data": stock_data,
        "label": label,
        "raw_line": raw_line,
        "volume": volume,
        "news_texts": news_texts,
        "news_types": news_types,
        "news_subtypes": news_subtypes,
        "cluster_trend": cluster_trend,
        "rule_trend": rule_trend,
        "industry_info": industry_info,
        "code_to_name": code_to_name,
        "name_to_code": name_to_code,
    }


def _pad_ratio_row_to_news_types(row: list[float]) -> list[float]:
    """Pad ratio row to news types."""
    n = NUM_VIRTUAL_NEWS_TYPES
    if len(row) >= n:
        return row[:n]
    row = list(row)
    if len(row) == n - 1 and len(row) >= 3:
        ind_mean = sum(row[2:]) / max(len(row) - 2, 1)
        row.append(ind_mean * 0.5)
    while len(row) < n:
        row.append(row[-1] if row else 0.0)
    return row[:n]


def _builtin_price_type_news_type_ratios() -> dict[int, list[float]]:
    """Builtin price type news type ratios."""
    _raw = {
        0: [
            0.5605,
            0.9864,
            0.4092,
            0.0886,
            0.2622,
            0.1971,
            0.1991,
            0.541,
            0.1618,
            0.1917,
            0.1815,
            0.2969,
        ],
        1: [
            0.5738,
            0.9812,
            0.4506,
            0.0885,
            0.1798,
            0.1279,
            0.159,
            0.4638,
            0.1516,
            0.1263,
            0.1619,
            0.3153,
        ],
        2: [
            0.622,
            0.9622,
            0.4635,
            0.0961,
            0.1263,
            0.2172,
            0.2124,
            0.4444,
            0.1335,
            0.1987,
            0.1458,
            0.2857,
        ],
        3: [
            0.6505,
            0.9544,
            0.4361,
            0.0912,
            0.15,
            0.1458,
            0.1192,
            0.3944,
            0.1234,
            0.1974,
            0.1491,
            0.2836,
        ],
        4: [
            0.5385,
            0.9905,
            0.4135,
            0.1143,
            0.2353,
            0.1667,
            0.1875,
            0.5238,
            0.0645,
            0.05,
            0.1613,
            0.4688,
        ],
        5: [
            0.3333,
            0.9907,
            0.4737,
            0.1935,
            0.2143,
            0.25,
            0.275,
            0.6154,
            0.1026,
            0.3243,
            0.1347,
            0.1935,
        ],
        6: [
            0.6727,
            0.9769,
            0.4673,
            0.102,
            0.1818,
            0.1892,
            0.5,
            0.6341,
            0.1154,
            0.1607,
            0.1362,
            0.1786,
        ],
        7: [
            0.3846,
            0.9829,
            0.4066,
            0.0455,
            0.125,
            0.0417,
            0.5,
            0.5385,
            0.0323,
            0.2609,
            0.1702,
            0.2,
        ],
    }
    return {pt: _pad_ratio_row_to_news_types(row) for (pt, row) in _raw.items()}


def parse_price_type_news_type_ratios_from_report(report_path: str) -> dict[int, list[float]]:
    """Parse price type news type ratios from report."""
    ratios: dict[int, list[float]] = {}
    if not report_path or not os.path.isfile(report_path):
        return ratios
    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("price_type=") or ":" not in s:
                continue
            (head, tail) = s.split(":", 1)
            try:
                pt = int(head.replace("price_type=", "").strip())
            except ValueError:
                continue
            vals = []
            for x in tail.split():
                try:
                    vals.append(float(x))
                except ValueError:
                    pass
            if vals:
                ratios[pt] = _pad_ratio_row_to_news_types(vals)
    return ratios


def _news_type_name(news_type_id: int) -> str:
    if news_type_id == 0:
        return "市场"
    if news_type_id == 1:
        return "行业/板块"
    if 2 <= news_type_id <= 12:
        return _INDIVIDUAL_SUBTYPE_NAMES.get(news_type_id - 1, f"个股细分类{news_type_id - 1}")
    return str(news_type_id)


def _price_type_cn(price_type_id: int) -> str:
    names = {
        0: "冲高回落",
        1: "宽幅震荡",
        2: "弱势上涨",
        3: "弱势下跌",
        4: "强势上涨",
        5: "强势下跌",
        6: "探底回升",
        7: "窄幅横盘",
    }
    return names.get(int(price_type_id), str(price_type_id))


def write_synthesis_by_pt_csv_from_pooled(
    pooled_csv: str, out_csv: str, *, num_price_type: int = 10, tag: str = "synthesis_from_pooled"
) -> int:
    """Write synthesis by pt csv from pooled."""
    pooled_rows: list[tuple[int, int, str, float]] = []
    if not pooled_csv or not os.path.isfile(pooled_csv):
        print(f"[SYNTHESIS_FROM_POOLED] pooled csv not found: {pooled_csv}")
        return 0
    with open(pooled_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                w = int(row.get("w_off", -1))
                nt = int(row.get("news_type_id", row.get("news_type", -1)))
                ratio = float(row.get("keep_ratio", row.get("ratio", "nan")))
            except (TypeError, ValueError):
                continue
            if w < 0 or nt < 0 or ratio <= 0:
                continue
            name = str(row.get("news_type_name", "") or "").strip() or _news_type_name(nt)
            pooled_rows.append((w, nt, name, ratio))
    pooled_rows.sort(key=lambda x: (-x[3], x[0], x[1]))
    n_pt = max(1, int(num_price_type))
    out_rows: list[tuple] = []
    for pt in range(n_pt):
        for rank, (w, nt, name, ratio) in enumerate(pooled_rows, start=1):
            out_rows.append((rank, pt, w, nt, name, ratio))
    _dir = os.path.dirname(out_csv)
    if _dir:
        os.makedirs(__dir, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "price_type_id",
                "price_type_cn",
                "w_off",
                "news_type_id",
                "news_type_name",
                "keep_ratio",
                "source",
            ]
        )
        for rank, pt, w, nt, name, ratio in out_rows:
            writer.writerow([rank, pt, _price_type_cn(pt), w, nt, name, f"{ratio:.4f}", tag])
    print(
        f"[SYNTHESIS_FROM_POOLED] wrote {len(out_rows)} rows ({len(pooled_rows)} pooled edges × {n_pt} price_types) -> {out_csv}"
    )
    return len(out_rows)


def _write_price_type_top_news_type_csvs(
    pt_type_ratio: np.ndarray, top_path: str, full_path: str, *, top_k: int = 5
):
    """Write price type top news type csvs."""
    arr = np.asarray(pt_type_ratio, dtype=np.float64)
    if arr.ndim < 2:
        return
    (n_pt, n_nt) = (arr.shape[0], min(NUM_VIRTUAL_NEWS_TYPES, arr.shape[1]))
    top_k = max(1, int(top_k))

    def _write(path: str, rows: list[list]):
        if not path:
            return
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)

    header = [
        "price_type_id",
        "price_type_cn",
        "rank",
        "news_type_id",
        "news_type_name",
        "keep_ratio",
    ]
    full_rows = [header]
    top_rows = [header]
    for pt in range(n_pt):
        scored = [(nt, float(arr[pt, nt])) for nt in range(n_nt)]
        scored.sort(key=lambda x: (-x[1], x[0]))
        for rank, (nt, ratio) in enumerate(scored, start=1):
            full_rows.append(
                [pt, _price_type_cn(pt), rank, nt, _news_type_name(nt), f"{ratio:.4f}"]
            )
        for rank, (nt, ratio) in enumerate(scored[:top_k], start=1):
            top_rows.append([pt, _price_type_cn(pt), rank, nt, _news_type_name(nt), f"{ratio:.4f}"])
    _write(full_path, full_rows)
    _write(top_path, top_rows)
    print(
        f"[FILTER_REPORT] saved price_type top news CSV: top_k={top_k} -> {top_path} ({len(top_rows) - 1} rows); full -> {full_path} ({len(full_rows) - 1} rows)"
    )


def _pick_news_types_by_ratio_row(
    row: list[float], k: int, pick: str, num_news_type: int = NUM_VIRTUAL_NEWS_TYPES
) -> set[int]:
    """Pick news types by ratio row."""
    n = min(len(row), int(num_news_type))
    kk = max(0, int(k))
    if kk <= 0 or n <= 0:
        return set()
    pick_l = str(pick or "top").lower()
    if pick_l == "bottom":
        order = sorted(range(n), key=lambda i: (float(row[i]), i))
        return set(order[:kk])
    order = sorted(range(n), key=lambda i: (-float(row[i]), i))
    if pick_l == "middle":
        start = max(0, (n - kk) // 2)
        return set(order[start : start + kk])
    return set(order[:kk])


def _pick_rank_slice(n: int, k: int, pick: str) -> tuple[int, int]:
    """Pick rank slice."""
    n = max(0, int(n))
    kk = max(0, min(int(k), n))
    if kk <= 0:
        return (0, 0)
    pick_l = str(pick or "top").lower()
    if pick_l == "bottom":
        return (n - kk, n)
    if pick_l == "middle":
        start = max(0, (n - kk) // 2)
        return (start, start + kk)
    return (0, kk)


def load_synthesis_pooled_edge_keys(
    csv_path: str, k: int, pick: str = "top"
) -> frozenset[tuple[int, int]]:
    """Load synthesis pooled edge keys."""
    rows: list[tuple[int, int]] = []
    if not csv_path or not os.path.isfile(csv_path):
        return frozenset()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                w = int(row.get("w_off", -1))
                nt = int(row.get("news_type_id", row.get("news_type", -1)))
            except (TypeError, ValueError):
                continue
            if w >= 0 and nt >= 0:
                rows.append((w, nt))
    (s, e) = _pick_rank_slice(len(rows), k, pick)
    return frozenset(rows[s:e])


def load_synthesis_pt_edge_keys(
    csv_path: str, k: int, pick: str = "top", num_price_type: int = 10
) -> dict[int, frozenset[tuple[int, int]]]:
    """Load synthesis pt edge keys."""
    by_pt: dict[int, list[tuple[int, int]]] = defaultdict(list)
    if not csv_path or not os.path.isfile(csv_path):
        return {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pt = int(row.get("price_type_id", row.get("price_type", -1)))
                w = int(row.get("w_off", -1))
                nt = int(row.get("news_type_id", row.get("news_type", -1)))
            except (TypeError, ValueError):
                continue
            if pt >= 0 and w >= 0 and (nt >= 0):
                by_pt[pt].append((w, nt))
    mapping: dict[int, frozenset[tuple[int, int]]] = {}
    for pt in range(int(num_price_type)):
        items = by_pt.get(pt, [])
        (s, e) = _pick_rank_slice(len(items), k, pick)
        mapping[pt] = frozenset(items[s:e])
    return mapping


def _w_off_offsets_from_edge_keys(
    edge_keys: frozenset[tuple[int, int]] | set[tuple[int, int]], span: int = 5
) -> list[int]:
    offs = sorted({int(w) for (w, _) in edge_keys})
    return offs if offs else list(range(max(1, int(span))))


def _set_block(buf: BatchBuffer, name: str, **fields) -> None:
    blk = buf[name]
    for k, v in fields.items():
        setattr(blk, k, v)


def _set_link(buf: BatchBuffer, src: str, rel: str, dst: str, index: torch.Tensor) -> None:
    buf[src, rel, dst].edge_index = index


def window_sample_to_buffer(sample: WindowSample) -> BatchBuffer:
    buf = BatchBuffer()
    _price_fields = dict(
        x=sample.price_seq,
        company_id=sample.price_company_id,
        day_id=sample.price_day_id,
        valid_mask=sample.price_valid_mask,
        date=sample.price_target_dates,
        raw_line=sample.price_debug_line,
    )
    if sample.price_vol_seq is not None:
        _price_fields["vol_seq"] = sample.price_vol_seq
    _set_block(buf, "price", **_price_fields)
    if sample.calendar_frac is not None:
        _set_block(buf, "date", x=sample.calendar_frac)
    news_fields = dict(
        x=sample.news_count,
        company_id=sample.news_company_id,
        day_id=sample.news_day_id,
        date=sample.news_dates,
        type=sample.news_types,
        news_id=sample.news_ids,
        text=sample.news_texts,
        vocab_input_ids=sample.news_vocab_ids,
        vocab_attention_mask=sample.news_vocab_mask,
    )
    if sample.news_input_ids is not None:
        news_fields["input_ids"] = sample.news_input_ids
    if sample.news_attention_mask is not None:
        news_fields["attention_mask"] = sample.news_attention_mask
    if sample.news_token_type_ids is not None:
        news_fields["token_type_ids"] = sample.news_token_type_ids
    if sample.news_emb is not None:
        news_fields["emb"] = sample.news_emb
    if sample.news_impact_y is not None:
        news_fields["impact_y"] = sample.news_impact_y
    if sample.news_impact_y_valid is not None:
        news_fields["impact_y_valid"] = sample.news_impact_y_valid
    _set_block(buf, "news", **news_fields)
    npad = dict(
        x=sample.news_pad_x,
        is_real=sample.news_pad_is_real,
        vocab_input_ids=sample.news_pad_vocab_ids,
        vocab_attention_mask=sample.news_pad_vocab_mask,
    )
    if sample.news_pad_input_ids is not None:
        npad["input_ids"] = sample.news_pad_input_ids
    if sample.news_pad_attention_mask is not None:
        npad["attention_mask"] = sample.news_pad_attention_mask
    if sample.news_pad_token_type_ids is not None:
        npad["token_type_ids"] = sample.news_pad_token_type_ids
    if sample.news_pad_emb is not None:
        npad["emb"] = sample.news_pad_emb
    _set_block(buf, "news_padding", **npad)
    _set_block(buf, "news_type", x=sample.news_type_feat)
    _set_block(buf, "price_type", x=sample.price_type_feat)
    if sample.industry_id_feat is not None:
        _set_block(buf, "industry_type", x=sample.industry_id_feat)
    if sample.rule_trend_id_feat is not None:
        _set_block(buf, "rule_trend_type", x=sample.rule_trend_id_feat)
    _set_block(
        buf,
        "label",
        x=sample.label_bin,
        org=sample.label_org,
        strong_mask=sample.label_strong_mask,
        valid_mask=sample.label_valid_mask,
        vol_x=sample.label_vol_bin,
        vol_org=sample.label_vol_org,
        vol_valid_mask=sample.label_vol_valid_mask,
    )
    _set_block(buf, "virtual_window", x=sample.virtual_window_table)
    _set_block(buf, "virtual_news", x=sample.virtual_news_table)
    _set_block(buf, "virtual_news_type", x=sample.virtual_news_type_table)
    _set_block(buf, "virtual_price_type", x=sample.virtual_price_type_table)
    if sample.virtual_industry_table is not None:
        _set_block(buf, "virtual_industry", x=sample.virtual_industry_table)
    if sample.virtual_trend_table is not None:
        _set_block(buf, "virtual_trend", x=sample.virtual_trend_table)
    if sample.virtual_rule_trend_table is not None:
        _set_block(buf, "virtual_rule_trend", x=sample.virtual_rule_trend_table)
    _set_link(buf, "news", "to", "news_type", sample.map_news_to_news_type)
    _set_link(buf, "price", "to", "price_type", sample.map_price_to_price_type)
    _set_link(buf, "price", "to", "virtual_window", sample.map_price_to_virtual_window)
    _set_link(buf, "news", "to", "virtual_window", sample.map_news_to_virtual_window)
    _set_link(buf, "news", "to", "virtual_news", sample.map_news_to_virtual_news)
    _set_link(
        buf, "news_type", "to", "virtual_news_type", sample.map_news_type_to_virtual_news_type
    )
    _set_link(
        buf, "price_type", "to", "virtual_price_type", sample.map_price_type_to_virtual_price_type
    )
    _set_link(
        buf, "virtual_price_type", "to", "price_type", sample.map_virtual_price_type_to_price_type
    )
    if sample.map_price_to_industry is not None:
        _set_link(buf, "price", "to", "industry_type", sample.map_price_to_industry)
    if sample.map_industry_to_virtual_industry is not None:
        _set_link(
            buf, "industry_type", "to", "virtual_industry", sample.map_industry_to_virtual_industry
        )
    if sample.map_virtual_industry_to_industry is not None:
        _set_link(
            buf, "virtual_industry", "to", "industry_type", sample.map_virtual_industry_to_industry
        )
    if sample.map_price_to_virtual_trend is not None:
        _set_link(buf, "price", "to", "virtual_trend", sample.map_price_to_virtual_trend)
    if sample.map_virtual_trend_to_price is not None:
        _set_link(buf, "virtual_trend", "to", "price", sample.map_virtual_trend_to_price)
    if sample.map_price_to_rule_trend is not None:
        _set_link(buf, "price", "to", "rule_trend_type", sample.map_price_to_rule_trend)
    if sample.map_rule_trend_to_virtual_rule_trend is not None:
        _set_link(
            buf,
            "rule_trend_type",
            "to",
            "virtual_rule_trend",
            sample.map_rule_trend_to_virtual_rule_trend,
        )
    if sample.map_virtual_rule_trend_to_rule_trend is not None:
        _set_link(
            buf,
            "virtual_rule_trend",
            "to",
            "rule_trend_type",
            sample.map_virtual_rule_trend_to_rule_trend,
        )
    buf["range_dates"] = sample.range_dates
    return buf


def buffer_to_window_sample(buf: BatchBuffer) -> WindowSample:
    """Buffer to window sample."""
    p = buf["price"]
    n = buf["news"]
    npad = buf["news_padding"]
    sample = WindowSample(
        range_dates=list(
            buf._meta.get("range_dates", buf["range_dates"] if "range_dates" in buf._meta else [])
        ),
        price_seq=p.x,
        price_company_id=p.company_id,
        price_day_id=p.day_id,
        price_valid_mask=p.valid_mask,
        price_target_dates=list(getattr(p, "date", [])),
        price_debug_line=list(getattr(p, "raw_line", [])),
        price_vol_seq=getattr(p, "vol_seq", None),
        news_count=n.x,
        news_company_id=n.company_id,
        news_day_id=n.day_id,
        news_dates=list(getattr(n, "date", [])),
        news_types=list(getattr(n, "type", [])),
        news_ids=list(getattr(n, "news_id", [])),
        news_texts=list(getattr(n, "text", [])),
        news_vocab_ids=getattr(n, "vocab_input_ids", torch.zeros(0, 100, dtype=torch.long)),
        news_vocab_mask=getattr(n, "vocab_attention_mask", torch.zeros(0, 100, dtype=torch.long)),
        news_pad_x=npad.x,
        news_pad_is_real=getattr(npad, "is_real", torch.zeros(0, 0)),
        news_pad_vocab_ids=getattr(
            npad, "vocab_input_ids", torch.zeros(0, 0, 100, dtype=torch.long)
        ),
        news_pad_vocab_mask=getattr(
            npad, "vocab_attention_mask", torch.zeros(0, 0, 100, dtype=torch.long)
        ),
        news_type_feat=buf["news_type"].x,
        price_type_feat=buf["price_type"].x,
        label_bin=buf["label"].x,
        label_org=buf["label"].org,
        label_strong_mask=buf["label"].strong_mask,
        label_valid_mask=buf["label"].valid_mask,
        label_vol_bin=getattr(buf["label"], "vol_x", torch.zeros_like(buf["label"].x)),
        label_vol_org=getattr(buf["label"], "vol_org", torch.zeros_like(buf["label"].org)),
        label_vol_valid_mask=getattr(
            buf["label"], "vol_valid_mask", torch.zeros_like(buf["label"].valid_mask)
        ),
        virtual_window_table=buf["virtual_window"].x,
        virtual_news_table=buf["virtual_news"].x,
        virtual_news_type_table=buf["virtual_news_type"].x,
        virtual_price_type_table=buf["virtual_price_type"].x,
    )
    if "date" in buf.node_types:
        sample.calendar_frac = buf["date"].x
    if hasattr(n, "input_ids"):
        sample.news_input_ids = n.input_ids
    if hasattr(n, "attention_mask"):
        sample.news_attention_mask = n.attention_mask
    if hasattr(n, "token_type_ids"):
        sample.news_token_type_ids = n.token_type_ids
    if hasattr(n, "emb"):
        sample.news_emb = n.emb
    if hasattr(n, "impact_y"):
        sample.news_impact_y = n.impact_y
    if hasattr(n, "impact_y_valid"):
        sample.news_impact_y_valid = n.impact_y_valid
    if hasattr(npad, "input_ids"):
        sample.news_pad_input_ids = npad.input_ids
    if hasattr(npad, "attention_mask"):
        sample.news_pad_attention_mask = npad.attention_mask
    if hasattr(npad, "token_type_ids"):
        sample.news_pad_token_type_ids = npad.token_type_ids
    if hasattr(npad, "emb"):
        sample.news_pad_emb = npad.emb
    if "industry_type" in buf.node_types:
        sample.industry_id_feat = buf["industry_type"].x
    if "rule_trend_type" in buf.node_types:
        sample.rule_trend_id_feat = buf["rule_trend_type"].x
    if "virtual_industry" in buf.node_types:
        sample.virtual_industry_table = buf["virtual_industry"].x
    if "virtual_trend" in buf.node_types:
        sample.virtual_trend_table = buf["virtual_trend"].x
    if "virtual_rule_trend" in buf.node_types:
        sample.virtual_rule_trend_table = buf["virtual_rule_trend"].x

    def _ei(src, rel, dst):
        if (src, rel, dst) in buf.edge_types:
            return buf[src, rel, dst].edge_index
        return torch.zeros(2, 0, dtype=torch.long)

    sample.map_news_to_news_type = _ei("news", "to", "news_type")
    sample.map_price_to_price_type = _ei("price", "to", "price_type")
    sample.map_price_to_virtual_window = _ei("price", "to", "virtual_window")
    sample.map_news_to_virtual_window = _ei("news", "to", "virtual_window")
    sample.map_news_to_virtual_news = _ei("news", "to", "virtual_news")
    sample.map_news_type_to_virtual_news_type = _ei("news_type", "to", "virtual_news_type")
    sample.map_price_type_to_virtual_price_type = _ei("price_type", "to", "virtual_price_type")
    sample.map_virtual_price_type_to_price_type = _ei("virtual_price_type", "to", "price_type")
    if "industry_type" in buf.node_types:
        sample.map_price_to_industry = _ei("price", "to", "industry_type")
        sample.map_industry_to_virtual_industry = _ei("industry_type", "to", "virtual_industry")
        sample.map_virtual_industry_to_industry = _ei("virtual_industry", "to", "industry_type")
    if "virtual_trend" in buf.node_types:
        sample.map_price_to_virtual_trend = _ei("price", "to", "virtual_trend")
        sample.map_virtual_trend_to_price = _ei("virtual_trend", "to", "price")
    if "rule_trend_type" in buf.node_types:
        sample.map_price_to_rule_trend = _ei("price", "to", "rule_trend_type")
        sample.map_rule_trend_to_virtual_rule_trend = _ei(
            "rule_trend_type", "to", "virtual_rule_trend"
        )
        sample.map_virtual_rule_trend_to_rule_trend = _ei(
            "virtual_rule_trend", "to", "rule_trend_type"
        )
    return sample


def load_window_sample_from_cache(path: str, range_dates: list[str] | None = None) -> WindowSample:
    """Load window sample from cache."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, WindowSample):
        return obj
    if isinstance(obj, BatchBuffer):
        if range_dates:
            obj["range_dates"] = range_dates
        return buffer_to_window_sample(obj)
    raise ValueError("Unsupported cached sample format; rebuild it with this data loader.")


def _test_print_sample_alignment(
    ds, batch, i: int, target_date: str | None = None, num_w_off: int = 5
) -> dict:
    """Test print sample alignment."""
    comp_idx = int(batch["price"].company_id[i].item())
    day_i = int(batch["price"].day_id[i].item())
    comp = ds.companies[comp_idx] if comp_idx < len(ds.companies) else "?"
    if target_date is None:
        price_dates_attr = getattr(batch["price"], "date", None)
        if isinstance(price_dates_attr, (list, tuple)) and i < len(price_dates_attr):
            target_date = price_dates_attr[i]
        else:
            target_date = "?"
    if isinstance(target_date, (list, tuple)):
        target_date = target_date[0] if target_date else "?"
    target_date = str(target_date)
    range_dates = getattr(batch, "range_dates", None)
    if range_dates is None and hasattr(batch, "get"):
        range_dates = batch.get("range_dates")
    if isinstance(range_dates, torch.Tensor):
        range_dates = [str(x) for x in range_dates.view(-1).tolist()]
    elif (
        isinstance(range_dates, (list, tuple))
        and len(range_dates) == 1
        and isinstance(range_dates[0], (list, tuple))
    ):
        range_dates = list(range_dates[0])
    comp_dates = sorted(ds.stock_data.get(comp, {}).keys()) if comp else []
    date_to_pos = {d: j for (j, d) in enumerate(comp_dates)}
    pos = date_to_pos.get(target_date)
    pw = int(getattr(ds, "price_window", 5))
    prev_date = ds.prev_date.get(comp, {}).get(target_date, "") if comp else ""
    if not prev_date and pos is not None and (pos >= 1):
        prev_date = comp_dates[pos - 1]
    price_window_dates: list[str] = []
    if pos is not None and pos >= pw:
        price_window_dates = comp_dates[pos - pw : pos]
    label_org = (
        float(batch["label"].org[i].item())
        if hasattr(batch["label"], "org")
        else float(batch["label"].x[i].item())
    )
    news_by_date = ds.news_texts.get(comp, {}) if comp else {}
    news_padding_items = []
    if prev_date and prev_date in news_by_date:
        for _nid, txt in list(news_by_date[prev_date].items())[
            : int(getattr(ds, "news_padding_k", 5))
        ]:
            s = "" if txt is None else str(txt).replace("\n", " ").strip()
            news_padding_items.append(s[:80] + ("..." if len(s) > 80 else ""))
    news_graph_by_woff: list[tuple[int, str, int, str]] = []
    if isinstance(range_dates, (list, tuple)):
        for w in range(num_w_off):
            src_day_i = day_i - w
            if 0 <= src_day_i < len(range_dates):
                tgt_d = str(range_dates[src_day_i])
                pub_d = ds.prev_date.get(comp, {}).get(tgt_d, "") if comp else ""
                if not pub_d and tgt_d in date_to_pos and (date_to_pos[tgt_d] >= 1):
                    pub_d = comp_dates[date_to_pos[tgt_d] - 1]
                n_cnt = len(news_by_date.get(pub_d, {})) if pub_d else 0
                news_graph_by_woff.append((w, pub_d or "?", n_cnt, tgt_d))
    price_rows = batch["price"].x[i].cpu().tolist() if batch["price"].x.numel() else []
    price_hlc = [[round(x, 4) for x in row[:3]] for row in price_rows] if price_rows else []
    return {
        "company": comp,
        "target_date": target_date,
        "prev_date": prev_date or "",
        "price_window_dates": price_window_dates,
        "price_hlc_by_day": (
            list(zip(price_window_dates, price_hlc))
            if len(price_window_dates) == len(price_hlc)
            else []
        ),
        "news_padding_date": prev_date or "",
        "news_padding_items": news_padding_items,
        "news_graph_by_woff": news_graph_by_woff,
        "label_org": label_org,
        "rule_trend": (
            ds.rule_trend.get(comp, {}).get(
                ds._resolve_trend_date(comp, target_date) if comp else "", ""
            )
            if comp
            else ""
        ),
        "trend_date": ds._resolve_trend_date(comp, target_date) if comp else "",
    }


def _print_test_print_alignment_block(align: dict, predict: float, label_bin: float) -> None:
    print("=" * 72)
    print(f"公司: {align['company']}")
    print(
        f"目标日 target_date: {align['target_date']}  |  股价最后一根 prev_date: {align['prev_date']}"
    )
    print(f"rule_trend({align.get('trend_date', align['prev_date'])}): {align['rule_trend']}")
    print("-" * 72)
    print(
        f"【股价输入】共 {len(align['price_window_dates'])} 天 (不含 target_date，最后一根=prev_date):"
    )
    for d, hlc in align.get("price_hlc_by_day", []):
        mark = " ← prev_date(最后一根)" if d == align["prev_date"] else ""
        print(f"  {d}  HLC={hlc}{mark}")
    print("-" * 72)
    pad_d = align["news_padding_date"] or "(无)"
    print(f"【news_padding】新闻发布日 {pad_d} (对应预测 target_date={align['target_date']}):")
    if align["news_padding_items"]:
        for j, t in enumerate(align["news_padding_items"], 1):
            print(f"  [{j}] {t}")
    else:
        print("  (无新闻)")
    print("-" * 72)
    print("【news 图路径 w_off→price】新闻发布日 → 连到 target_date 的 price 节点:")
    for w, pub_d, cnt, tgt_d in align["news_graph_by_woff"]:
        print(f"  w_off={w}: 新闻日 {pub_d}  (#{cnt} 条) → target {tgt_d}")
    print("-" * 72)
    print(f"【输出】label_org(target日收益)={align['label_org']:.6f}")
    print(f"       predict(sigmoid)={predict:.4f}  label_bin={label_bin:.0f}")
    print("=" * 72)
