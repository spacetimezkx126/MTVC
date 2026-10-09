#!/usr/bin/env python3
"""Summarize Acc/MCC for all new sep13 casel6 runs under MTVC_paper_repro/checkpoints."""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

REPRO = Path(__file__).resolve().parents[2]
CKPT = REPRO / "checkpoints"
OUT = REPRO / "results" / "casel6_acc_mcc"
SEEDS = [42, 43, 44, 45, 46]
DATASETS = ["csmd50", "csmd300", "massive"]

# role -> method dir pattern relative to checkpoints/{ds}/
# Massive tags sometimes include _noglobal_
# role -> {dataset: method_dir}
ROLES = [
    ("full", {ds: "full_casel6" for ds in DATASETS}),
    ("casel1", {ds: "full_casel1" for ds in DATASETS}),
    ("casel2", {ds: "full_casel2" for ds in DATASETS}),
    ("casel3", {ds: "full_casel3" for ds in DATASETS}),
    ("casel4", {ds: "full_casel4" for ds in DATASETS}),
    ("casel5", {ds: "full_casel5" for ds in DATASETS}),
    ("casel6", {ds: "full_casel6" for ds in DATASETS}),
    ("abl_noglobal", {"csmd50": "abl_noglobal_casel6", "csmd300": "abl_noglobal_casel6"}),
    ("abl_addglobal", {"massive": "abl_addglobal_casel6"}),
    ("abl_noindustry", {ds: "abl_noindustry_casel6" for ds in DATASETS}),
    ("abl_nogate", {ds: "abl_nogate_casel6" for ds in DATASETS}),
    ("abl_nocontrast", {ds: "abl_nocontrast_casel6" for ds in DATASETS}),
    ("abl_noprice", {ds: "abl_noprice_casel6" for ds in DATASETS}),
    ("multinews", {ds: "multinews_casel6" for ds in DATASETS}),
    ("crossattn", {ds: "crossattn_casel6" for ds in DATASETS}),
    ("fullunified", {ds: "fullunified_casel6" for ds in DATASETS}),
    ("vin_ind_as_mkt", {ds: "vin_ind_as_mkt_casel6" for ds in DATASETS}),
    ("vin_mkt_as_ind", {ds: "vin_mkt_as_ind_casel6" for ds in DATASETS}),
]


def method_for(ds: str, mapping: dict) -> str | None:
    return mapping.get(ds)


def load_seed(ds: str, method: str, seed: int, allow_partial: bool = True):
    ck = CKPT / ds / method / f"s{seed}"
    meta = ck / "best.pt.meta.json"
    log = ck / "train.log"
    if not meta.is_file():
        return None
    txt = log.read_text(errors="ignore") if log.is_file() else ""
    finished = "Training finished" in txt
    bad = (not finished) and any(k in txt for k in ("CUDA_FATAL", "OutOfMemoryError", "Traceback"))
    if bad and not finished:
        return None
    if (not finished) and (not allow_partial):
        return None
    m = json.loads(meta.read_text())["metrics"]
    return {
        "seed": seed,
        "finished": finished,
        "acc": 100.0 * float(m["best_test_acc"]),
        "mcc": float(m["best_test_mcc"]),
        "best_epoch": int(m.get("best_epoch", -1)),
        "path": str(ck),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    lines = []
    lines.append("role\tds\tn\tAcc\tMCC\tΔAcc vs full\tΔMCC vs full\tbest_seed(MCC)")
    full_ref = {}

    # first pass: full
    for ds in DATASETS:
        meth = method_for(ds, ROLES[0][1])
        seeds = [load_seed(ds, meth, s) for s in SEEDS]
        seeds = [x for x in seeds if x]
        if seeds:
            full_ref[ds] = (st.mean([x["acc"] for x in seeds]), st.mean([x["mcc"] for x in seeds]))

    for role, mapping in ROLES:
        for ds in DATASETS:
            meth = method_for(ds, mapping)
            if not meth:
                continue
            seeds = [load_seed(ds, meth, s) for s in SEEDS]
            seeds = [x for x in seeds if x]
            if not seeds:
                continue
            accs = [x["acc"] for x in seeds]
            mccs = [x["mcc"] for x in seeds]
            best = max(seeds, key=lambda x: x["mcc"])
            fa, fm = full_ref.get(ds, (None, None))
            row = {
                "role": role,
                "dataset": ds,
                "method": meth,
                "n_seeds": len(seeds),
                "n_finished": sum(1 for x in seeds if x["finished"]),
                "acc_mean": round(st.mean(accs), 4),
                "acc_std": round(st.pstdev(accs), 4) if len(accs) > 1 else 0.0,
                "mcc_mean": round(st.mean(mccs), 6),
                "mcc_std": round(st.pstdev(mccs), 6) if len(mccs) > 1 else 0.0,
                "delta_acc": round(st.mean(accs) - fa, 4) if fa is not None else None,
                "delta_mcc": round(st.mean(mccs) - fm, 6) if fm is not None else None,
                "best_seed": best["seed"],
                "best_acc": round(best["acc"], 4),
                "best_mcc": round(best["mcc"], 6),
                "seeds": seeds,
            }
            rows.append(row)
            da = f"{row['delta_acc']:+.2f}" if row["delta_acc"] is not None else ""
            dm = f"{row['delta_mcc']:+.4f}" if row["delta_mcc"] is not None else ""
            lines.append(
                f"{role}\t{ds}\t{row['n_seeds']}\t"
                f"{row['acc_mean']:.2f}±{row['acc_std']:.2f}\t"
                f"{row['mcc_mean']:.4f}±{row['mcc_std']:.4f}\t"
                f"{da}\t{dm}\ts{best['seed']}"
            )

    (OUT / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    with (OUT / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "role",
                "dataset",
                "method",
                "n_seeds",
                "n_finished",
                "acc_mean",
                "acc_std",
                "mcc_mean",
                "mcc_std",
                "delta_acc",
                "delta_mcc",
                "best_seed",
                "best_acc",
                "best_mcc",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"wrote {OUT} ({len(rows)} rows)")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
