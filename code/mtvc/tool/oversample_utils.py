#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Massive-train oversample helper (match test positive rate).

Role
----
- Load ``train_oversample_extra.jsonl`` (company, date) → extra sampling weights.
- Used only when ``--train_oversample_extra`` is set (Massive paper runs).
"""
from __future__ import annotations

import json
import os
from typing import Any


def load_oversample_extra_counts(path: str) -> dict[tuple[str, str], int]:
    """Load {(company, date): extra_copies} from json/jsonl. Empty/missing → {}."""
    path = str(path or "").strip()
    if not path or not os.path.isfile(path):
        return {}
    out: dict[tuple[str, str], int] = {}
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                key = (str(obj.get("company", "")), str(obj.get("date", ""))[:10])
                out[key] = int(obj.get("extra", obj.get("count", 1)))
        return out
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(k, str) and "|" in k:
                c, d = k.split("|", 1)
                out[(c, d[:10])] = int(v)
            elif isinstance(v, dict):
                for d, n in v.items():
                    out[(str(k), str(d)[:10])] = int(n)
    elif isinstance(data, list):
        for obj in data:
            if not isinstance(obj, dict):
                continue
            key = (str(obj.get("company", "")), str(obj.get("date", ""))[:10])
            out[key] = int(obj.get("extra", obj.get("count", 1)))
    return out


def summarize_oversample_counts(counts: dict[tuple[str, str], int] | None) -> dict[str, Any]:
    counts = counts or {}
    n_extra = int(sum(max(0, int(v)) for v in counts.values()))
    return {
        "n_unique_keys": len(counts),
        "n_extra_copies": n_extra,
    }
