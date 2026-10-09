#!/usr/bin/env python3
"""Align train/val/test calendar ranges onto a 10-day window lattice.

Problem
-------
``contiguous`` splits flat-chunk trading days by ``seg_length`` (default 10)
starting at each role's *first trading day*. Changing a role's start by even
one trading day remaps every day's ``day_id`` / window membership (phase +1)
and historically collapses CSMD stagger MCC.

This tool takes a *desired* split (year/month intent) and returns dates whose
first trading days sit on a chosen lattice, so window boundaries match a
reference experiment.

Default reference lattice (CSMD50 historical working test)
----------------------------------------------------------
Anchor = first trading day of old test = ``2024-01-03`` (not calendar Jan 1).
Window starts = every ``seg_length`` trading days from that anchor forward
(and optionally backward).

Usage
-----
  cd MTVC_paper_repro/data_preparation

  # Print aligned dates for a requested 3y / 0.5y / 0.5y intent
  python align_split_phase.py \\
      --dataset csmd50 \\
      --train 2021-01-01:2023-12-31 \\
      --val   2024-01-01:2024-06-30 \\
      --test  2024-07-01:2024-12-31

  # Lock to Massive-style: use that profile's own first-trading-day lattice
  python align_split_phase.py --dataset massive --anchor-from-role test \\
      --train 2022-01-01:2023-12-31 --val 2024-01-01:2024-12-31 \\
      --test 2025-01-01:2025-12-31

  # Explicit anchor
  python align_split_phase.py --dataset csmd50 --anchor 2024-01-03 \\
      --train 2021-01-01:2023-12-31 --val 2024-01-01:2024-06-30 \\
      --test 2024-07-01:2024-12-31

Outputs JSON + human summary with:
  - requested ranges
  - aligned CLI dates (safe to pass as --train_start etc.)
  - first trading day / n_days / n_windows / window starts per role
  - verification that val+test window starts lie on the lattice
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))  # MTVC_paper_repro
_MTVC = os.path.join(_REPO, "code", "mtvc")


def _setup_path() -> None:
    os.environ.setdefault("DUAL_TF_MODEL_DIR", _REPO)
    os.chdir(_REPO)
    if _MTVC not in sys.path:
        sys.path.insert(0, _MTVC)


@dataclass
class RoleRange:
    start: str  # YYYY-MM-DD inclusive (calendar intent)
    end: str


@dataclass
class AlignedRole:
    role: str
    requested_start: str
    requested_end: str
    aligned_start: str
    aligned_end: str
    first_trading_day: str
    last_trading_day: str
    n_trading_days: int
    n_windows: int
    window_starts: list[str]
    first_window: list[str]
    last_window: list[str]
    on_lattice: bool


def _parse_range(s: str) -> RoleRange:
    s = s.strip()
    if ":" not in s:
        raise argparse.ArgumentTypeError(f"range must be START:END, got {s!r}")
    a, b = s.split(":", 1)
    a, b = a.strip(), b.strip()
    for x in (a, b):
        datetime.strptime(x, "%Y-%m-%d")
    if a > b:
        raise argparse.ArgumentTypeError(f"start>end: {s}")
    return RoleRange(a, b)


def _default_root(dataset: str) -> str:
    roots = {
        "csmd50": "/home/zhaokx/Pattern/Pattern_Mining/dataset/CSMD50",
        "csmd300": "/home/zhaokx/Pattern/Pattern_Mining/dataset/CSMD300",
        "sse50": "/home/zhaokx/Pattern/Pattern_Mining/dataset/sse50_news",
        "massive": "/home/zhaokx/Pattern/Pattern_Mining/dataset/massive_data",
        "stocknet": "/home/zhaokx/Pattern/Pattern_Mining/dataset/stocknet",
    }
    if dataset not in roots:
        raise ValueError(f"unknown dataset {dataset}; known={sorted(roots)}")
    return roots[dataset]


def _load_trading_days(
    dataset: str,
    root: str,
    cal_start: str,
    cal_end: str,
    *,
    csmd_news_source: str = "raw",
) -> list[str]:
    """All trading days in [cal_start, cal_end] via MarketWindowDataset test mode."""
    from data import MarketWindowDataset, get_profile

    profile = get_profile(dataset)
    # Dummy train/val so builder accepts; put the span on test.
    # Use a tiny prior train/val that does not need to be in-range for listing.
    split = {
        "train_start": "2000-01-01",
        "train_end": "2000-06-30",
        "val_start": "2000-07-01",
        "val_end": "2000-12-31",
        "test_start": cal_start,
        "test_end": cal_end,
    }
    kwargs: dict = {
        "root": root,
        "split_dates": split,
        "split_mode": "contiguous",
        "train_lookback_purge": True,
        "news_padding_k": 5,
    }
    if dataset.startswith("csmd") or dataset == "sse50":
        kwargs["csmd_news_source"] = csmd_news_source
    ds = MarketWindowDataset(profile, mode="test", **kwargs)
    return [str(d) for d in ds._inner.valid_dates]


def _first_on_or_after(days: list[str], ymd: str) -> str | None:
    for d in days:
        if d >= ymd:
            return d
    return None


def _last_on_or_before(days: list[str], ymd: str) -> str | None:
    hit = None
    for d in days:
        if d <= ymd:
            hit = d
        else:
            break
    return hit


def _lattice_starts(days: list[str], anchor: str, seg: int) -> list[str]:
    if anchor not in days:
        raise ValueError(f"anchor {anchor} not in trading calendar")
    i0 = days.index(anchor)
    out = []
    # forward
    i = i0
    while i < len(days):
        out.append(days[i])
        i += seg
    # backward (exclusive of anchor already listed)
    i = i0 - seg
    pref = []
    while i >= 0:
        pref.append(days[i])
        i -= seg
    return list(reversed(pref)) + out


def _snap_start_to_lattice(days: list[str], requested_start: str, lattice: set[str] | Iterable[str]) -> str:
    """First trading day >= requested_start that is a lattice window start."""
    lat = set(lattice)
    for d in days:
        if d >= requested_start and d in lat:
            return d
    raise ValueError(
        f"no lattice window start on/after {requested_start} "
        f"(last day {days[-1] if days else None})"
    )


def _days_in_closed(days: list[str], start: str, end: str) -> list[str]:
    return [d for d in days if start <= d <= end]


def _chunk_windows(role_days: list[str], seg: int) -> list[list[str]]:
    if not role_days:
        return []
    out = []
    for i in range(0, len(role_days), seg):
        out.append(role_days[i : i + seg])
    return out


def align_split(
    *,
    dataset: str,
    root: str,
    train: RoleRange,
    val: RoleRange,
    test: RoleRange,
    anchor: str | None,
    anchor_from_role: str | None,
    seg_length: int = 10,
    csmd_news_source: str = "raw",
    snap_roles: Iterable[str] = ("train", "val", "test"),
    end_mode: str = "before_next",
) -> dict:
    """Return aligned dates + diagnostics.

    Policy
    ------
    1. Build full trading calendar covering min(requested)..max(requested).
    2. Resolve anchor:
         - ``--anchor YYYY-MM-DD``, or
         - ``--anchor-from-role {train,val,test}`` = first trading day of that
           role's *requested* range (Massive-style self-lattice), or
         - default for csmd*: ``2024-01-03`` (historical working test head).
    3. For roles in ``snap_roles``, ``aligned_start`` = first lattice window
       start >= requested_start. Other roles use the first trading day
       >= requested_start (calendar intent; may be off-lattice — same as
       legacy CSMD train which starts at 2021-01-06 while test lattice
       anchors at 2024-01-03).
    4. End policy (``end_mode``):
         - ``before_next``: last trading day strictly before next role start
           (and <= requested_end).
         - ``requested``: last trading day <= requested_end only.
         - ``complete_windows``: like before_next, then trim to a multiple of
           seg_length (drop short tail).
    """
    cal_lo = min(train.start, val.start, test.start)
    cal_hi = max(train.end, val.end, test.end)
    days = _load_trading_days(dataset, root, cal_lo, cal_hi, csmd_news_source=csmd_news_source)
    if not days:
        raise RuntimeError(f"empty trading calendar for {dataset} in [{cal_lo},{cal_hi}]")

    roles_req = {"train": train, "val": val, "test": test}
    snap_set = {str(x).strip() for x in snap_roles if str(x).strip()}
    bad = snap_set - {"train", "val", "test"}
    if bad:
        raise ValueError(f"bad snap_roles {bad}")

    if anchor:
        anc = anchor
        if anc not in days:
            # allow calendar date → first trading on/after
            ft = _first_on_or_after(days, anc)
            if ft is None:
                raise ValueError(f"anchor {anchor} beyond calendar")
            anc = ft
    elif anchor_from_role:
        rr = roles_req[anchor_from_role]
        anc = _first_on_or_after(days, rr.start)
        if anc is None:
            raise ValueError(f"no trading day on/after {rr.start} for anchor-from-role")
    elif dataset.startswith("csmd"):
        anc = "2024-01-03"
        if anc not in days:
            anc = _first_on_or_after(days, "2024-01-03") or days[0]
    else:
        # self-lattice from train first trading day
        anc = _first_on_or_after(days, train.start) or days[0]

    lat_list = _lattice_starts(days, anc, seg_length)
    lat_set = set(lat_list)

    # Resolve each role start
    order = ["train", "val", "test"]
    snapped_starts: dict[str, str] = {}
    for role in order:
        rr = roles_req[role]
        if role in snap_set:
            snapped_starts[role] = _snap_start_to_lattice(days, rr.start, lat_set)
        else:
            ft = _first_on_or_after(days, rr.start)
            if ft is None:
                raise ValueError(f"{role}: no trading day on/after {rr.start}")
            snapped_starts[role] = ft

    aligned: dict[str, AlignedRole] = {}
    for i, role in enumerate(order):
        rr = roles_req[role]
        a_start = snapped_starts[role]
        # tentative end
        a_end_td = _last_on_or_before(days, rr.end)
        if a_end_td is None or a_end_td < a_start:
            raise ValueError(f"{role}: empty after snap (start={a_start}, req_end={rr.end})")
        if end_mode in ("before_next", "complete_windows") and i + 1 < len(order):
            nxt = snapped_starts[order[i + 1]]
            prev = None
            for d in days:
                if d >= nxt:
                    break
                prev = d
            if prev is not None and prev >= a_start:
                a_end_td = min(a_end_td, prev)
        role_days = _days_in_closed(days, a_start, a_end_td)
        if end_mode == "complete_windows" and len(role_days) >= seg_length:
            n_keep = (len(role_days) // seg_length) * seg_length
            role_days = role_days[:n_keep]
            a_end_td = role_days[-1]
        wins = _chunk_windows(role_days, seg_length)
        # on_lattice: role head sits on the global anchor lattice
        on_lat = a_start in lat_set and all(w[0] in lat_set for w in wins if len(w) == seg_length)
        aligned[role] = AlignedRole(
            role=role,
            requested_start=rr.start,
            requested_end=rr.end,
            aligned_start=a_start,  # safe CLI: use trading day itself as start
            aligned_end=a_end_td,
            first_trading_day=role_days[0],
            last_trading_day=role_days[-1],
            n_trading_days=len(role_days),
            n_windows=len(wins),
            window_starts=[w[0] for w in wins],
            first_window=list(wins[0]) if wins else [],
            last_window=list(wins[-1]) if wins else [],
            on_lattice=bool(on_lat),
        )

    # Cross-check: no overlapping trading days
    sets = {r: set(_days_in_closed(days, aligned[r].aligned_start, aligned[r].aligned_end)) for r in order}
    overlap = {
        "train∩val": sorted(sets["train"] & sets["val"]),
        "val∩test": sorted(sets["val"] & sets["test"]),
        "train∩test": sorted(sets["train"] & sets["test"]),
    }

    cli = {
        "train_start": aligned["train"].aligned_start,
        "train_end": aligned["train"].aligned_end,
        "val_start": aligned["val"].aligned_start,
        "val_end": aligned["val"].aligned_end,
        "test_start": aligned["test"].aligned_start,
        "test_end": aligned["test"].aligned_end,
    }

    return {
        "dataset": dataset,
        "root": root,
        "seg_length": seg_length,
        "anchor": anc,
        "snap_roles": sorted(snap_set),
        "end_mode": end_mode,
        "lattice_starts_head": lat_list[:8],
        "lattice_starts_tail": lat_list[-8:],
        "n_lattice_starts": len(lat_list),
        "requested": {k: asdict(v) for k, v in roles_req.items()},
        "aligned": {k: asdict(v) for k, v in aligned.items()},
        "cli_dates": cli,
        "overlap_trading_days": {k: v for k, v in overlap.items() if v},
        "notes": [
            "Pass cli_dates as --train_start/--train_end/--val_start/--val_end/--test_start/--test_end.",
            "Roles in snap_roles start on the anchor lattice (flat 10-day phase matches).",
            "CSMD historical working test lattice anchor=2024-01-03; Jan-01→first TD 2024-01-02 is phase +1.",
            "Phase lock ≠ good MCC: old eqEnc3x6 s42 is MCC~0.105 on 2024H1 but ~0.005 on 2024H2 "
            "(same lattice). 3y/6m/6m with val=H1 test=H2 can look 'aligned but bad' for that reason.",
        ],
    }


def _print_human(result: dict) -> None:
    print(f"dataset={result['dataset']}  anchor={result['anchor']}  seg={result['seg_length']}")
    print("--- requested → aligned ---")
    for role in ("train", "val", "test"):
        a = result["aligned"][role]
        print(
            f"{role:5s}  req [{a['requested_start']} .. {a['requested_end']}]  →  "
            f"[{a['aligned_start']} .. {a['aligned_end']}]  "
            f"tdays={a['n_trading_days']} wins={a['n_windows']}  "
            f"first_win={a['first_window'][:1]}.. on_lattice={a['on_lattice']}"
        )
    print("--- CLI ---")
    for k, v in result["cli_dates"].items():
        print(f"  --{k} {v}")
    if result["overlap_trading_days"]:
        print("WARNING overlap:", result["overlap_trading_days"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="csmd50", choices=["csmd50", "csmd300", "sse50", "massive", "stocknet"])
    p.add_argument("--root", default="", help="dataset root (default by --dataset)")
    p.add_argument("--train", type=_parse_range, required=True, help="START:END")
    p.add_argument("--val", type=_parse_range, required=True, help="START:END")
    p.add_argument("--test", type=_parse_range, required=True, help="START:END")
    p.add_argument("--anchor", default="", help="lattice anchor YYYY-MM-DD (trading or calendar)")
    p.add_argument(
        "--anchor-from-role",
        default="",
        choices=["", "train", "val", "test"],
        help="set anchor = first trading day of that requested role",
    )
    p.add_argument("--seg-length", type=int, default=10)
    p.add_argument(
        "--snap-roles",
        default="val,test",
        help="comma list of roles to snap onto the lattice (default: val,test; "
        "use train,val,test for a single global phase including train)",
    )
    p.add_argument(
        "--end-mode",
        default="before_next",
        choices=["before_next", "requested", "complete_windows"],
        help="how to choose aligned_end (default: before_next)",
    )
    p.add_argument("--csmd-news-source", default="raw")
    p.add_argument("--json-out", default="", help="write full result JSON to path")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    _setup_path()
    root = args.root or _default_root(args.dataset)
    snap_roles = [x.strip() for x in str(args.snap_roles).split(",") if x.strip()]
    result = align_split(
        dataset=args.dataset,
        root=root,
        train=args.train,
        val=args.val,
        test=args.test,
        anchor=args.anchor or None,
        anchor_from_role=args.anchor_from_role or None,
        seg_length=args.seg_length,
        csmd_news_source=args.csmd_news_source,
        snap_roles=snap_roles,
        end_mode=args.end_mode,
    )
    if not args.quiet:
        _print_human(result)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"wrote {args.json_out}")
    elif args.quiet:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
