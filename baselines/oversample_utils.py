#!/usr/bin/env python3
"""Deterministic train oversample extras (same JSONL as dual_tf)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_oversample_extra_list(path: str | Path) -> list[tuple[str, str]]:
    """JSONL rows in file order -> [(company, date), ...]."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"oversample extra not found: {p}")
    rows: list[tuple[str, str]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append((str(row["company"]), str(row["date"])))
    return rows


def extra_counts_from_list(extras: Sequence[tuple[str, str]]) -> dict[tuple[str, str], int]:
    return dict(Counter(extras))


def load_oversample_extra_counts(path: str | Path) -> dict[tuple[str, str], int]:
    return extra_counts_from_list(load_oversample_extra_list(path))


def summarize_oversample_counts(counts: Mapping[tuple[str, str], int]) -> dict:
    n_keys = len(counts)
    n_extra = int(sum(counts.values()))
    return {
        "n_unique_keys": n_keys,
        "n_extra_copies": n_extra,
        "max_copies_one_key": int(max(counts.values())) if counts else 0,
    }


def expand_tuple_samples(
    samples: list[tuple[str, str, float]],
    extras: Sequence[tuple[str, str]],
) -> tuple[list[tuple[str, str, float]], int, int]:
    """Append extra copies of matching (company, date) samples. Val/test unused."""
    by_key = {(c, d): (c, d, mv) for c, d, mv in samples}
    out = list(samples)
    n_hit = n_miss = 0
    for key in extras:
        src = by_key.get(key)
        if src is None:
            n_miss += 1
            continue
        out.append(src)
        n_hit += 1
    return out, n_hit, n_miss


def expand_dict_samples(
    samples: list[dict[str, Any]],
    extras: Sequence[tuple[str, str]],
    *,
    company_key: str = "company",
    date_key: str = "target_date",
) -> tuple[list[dict[str, Any]], int, int]:
    by_key = {(str(s[company_key]), str(s[date_key])): s for s in samples}
    out = list(samples)
    n_hit = n_miss = 0
    for key in extras:
        src = by_key.get(key)
        if src is None:
            n_miss += 1
            continue
        out.append(src)
        n_hit += 1
    return out, n_hit, n_miss


def apply_oversample_to_dataset_samples(dataset, extra_path: str, log_fn=None) -> int:
    """Append extras onto dataset.samples (train only). Returns n_hit."""
    path = str(extra_path or "").strip()
    if not path:
        return 0
    extras = load_oversample_extra_list(path)
    counts = extra_counts_from_list(extras)
    os_sum = summarize_oversample_counts(counts)
    n_before = len(dataset.samples)
    new_samples, n_hit, n_miss = expand_dict_samples(dataset.samples, extras)
    dataset.samples = new_samples
    if log_fn is not None:
        log_fn(
            "train oversample extra=%s unique=%d copies=%d hit=%d miss=%d n %d->%d "
            "(val/test unchanged)",
            path,
            os_sum["n_unique_keys"],
            os_sum["n_extra_copies"],
            n_hit,
            n_miss,
            n_before,
            len(dataset.samples),
        )
    return n_hit
