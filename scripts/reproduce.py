"""Evaluate archived checkpoints or train their recorded recipes from scratch."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import statistics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/mtvc"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["evaluate", "train", "evaluate-all"])
    p.add_argument("--dataset", choices=["csmd50", "csmd300", "massive"], default="csmd50")
    p.add_argument("--case", default="full_casel6")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--checkpoint-root", type=Path, default=ROOT.parent / "MTVC_checkpoints")
    p.add_argument("--recipe-file", type=Path, help="Override the checkpoint collection recipes.")
    p.add_argument("--data-root", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--tag", choices=["best", "last", "both"], default="best")
    p.add_argument("--epochs", type=int)
    p.add_argument("--all-datasets", action="store_true")
    p.add_argument("--cases", nargs="+", help="Optional case filter for batch evaluation.")
    p.add_argument(
        "--seeds", nargs="+", type=int, help="Optional fixed seed filter for batch evaluation."
    )
    p.add_argument("--tolerance", type=float, default=1e-12)
    p.add_argument("--acc-tolerance", type=float, help="Absolute ACC tolerance in probability units; 0.0001 equals 0.01 percentage points.")
    p.add_argument("--mcc-tolerance", type=float, help="Absolute MCC tolerance, reported independently from ACC.")
    a = p.parse_args()
    if any(v is not None and v < 0 for v in [a.tolerance, a.acc_tolerance, a.mcc_tolerance]):
        p.error("Metric tolerances must be nonnegative.")
    a.checkpoint_root = a.checkpoint_root.resolve()
    recipe_file = a.recipe_file or (
        ROOT / "configs/paper_recipes.json"
        if a.action == "train"
        else a.checkpoint_root / "paper_recipes.json"
    )
    if not recipe_file.is_file():
        if a.recipe_file:
            p.error("Recipe file does not exist.")
        release_recipes = ROOT / "configs/release_checkpoint_recipes.json"
        recipe_file = (
            release_recipes
            if a.action != "train" and release_recipes.is_file()
            else ROOT / "configs/paper_recipes.json"
        )
    a.recipe_file = recipe_file.resolve()
    collection = json.loads(a.recipe_file.read_text())
    recipes = [
        {**r, "args": {**collection.get("defaults", {}), **r.get("args", {})}}
        for r in collection["runs"]
    ]
    selected = [
        r
        for r in recipes
        if (a.all_datasets or r["dataset"] == a.dataset)
        and (not a.cases or r["case"] in a.cases)
        and (not a.seeds or r["seed"] in a.seeds)
    ]
    if a.action == "evaluate-all":
        output = (a.output or ROOT / "results/evaluation").resolve()
        output.mkdir(parents=True, exist_ok=True)
        failed = []
        skipped = []
        reports = []
        tags = ["best", "last"] if a.tag == "both" else [a.tag]
        planned = 0
        for r in selected:
            for tag in tags:
                ckpt = a.checkpoint_root / r["id"] / (tag + ".pt")
                if tag == "last" and (not ckpt.exists()):
                    skipped.append(
                        {"run": r["id"], "tag": tag, "reason": "Not saved in the original archive."}
                    )
                    continue
                planned += 1
                report = output / r["dataset"] / r["case"] / ("s%d.%s.json" % (r["seed"], tag))
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "evaluate",
                    "--dataset",
                    r["dataset"],
                    "--case",
                    r["case"],
                    "--seed",
                    str(r["seed"]),
                    "--device",
                    a.device,
                    "--checkpoint-root",
                    str(a.checkpoint_root),
                    "--recipe-file",
                    str(a.recipe_file),
                    "--output",
                    str(report),
                    "--tolerance",
                    str(a.tolerance),
                    "--tag",
                    tag,
                ]
                for name in ["acc_tolerance", "mcc_tolerance"]:
                    value = getattr(a, name)
                    if value is not None:
                        cmd += ["--" + name.replace("_", "-"), str(value)]
                if a.data_root:
                    cmd += ["--data-root", str(a.data_root)]
                log = report.with_suffix(".log")
                log.parent.mkdir(parents=True, exist_ok=True)
                report.unlink(missing_ok=True)
                with log.open("w") as f:
                    rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
                if rc:
                    failed.append({"run": r["id"], "tag": tag, "returncode": rc})
                if rc in (0, 1) and report.exists():
                    reports.append(json.loads(report.read_text()))
                print(r["id"], tag, "OK" if rc == 0 else "CHECK REPORT", flush=True)
        groups = []
        for ds, case, tag in sorted(
            {(r["dataset"], r["case"], tag) for r in selected for tag in tags}
        ):
            rows = [
                r
                for r in reports
                if r["dataset"] == ds
                and r["case"] == case
                and (r["tag"] == tag)
                and (r["actual"]["acc"] is not None)
                and (r["actual"]["mcc"] is not None)
            ]
            g = {"dataset": ds, "case": case, "tag": tag, "n": len(rows)}
            for k in ["acc", "mcc"]:
                values = [r["actual"][k] for r in rows]
                g[k + "_mean"] = statistics.mean(values) if values else None
                g[k + "_population_std"] = statistics.pstdev(values) if len(values) > 1 else None
            groups.append(g)
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "groups": groups,
                    "failures_or_mismatches": failed,
                    "skipped_unsaved_checkpoints": skipped,
                    "reported_checkpoints": len(reports),
                    "planned_available_checkpoints": planned,
                },
                indent=2,
            )
            + "\n"
        )
        return int(bool(failed))
    if a.action == "evaluate" and a.tag == "both":
        p.error("Use evaluate-all for --tag both.")
    r = next((r for r in selected if r["case"] == a.case and r["seed"] == a.seed), None)
    if r is None:
        p.error("No recorded recipe for this dataset/case/seed in the selected checkpoint collection.")
    from reproduction import configure_environment, recipe_args, build_model, load_archive_weights

    configure_environment()
    import torch

    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    (args, profile) = recipe_args(r, a.device, a.data_root)
    from train import (
        set_all_seeds,
        device_from_args,
        _builder_dataset,
        make_model_dataloader,
        move_model_batch_to_device,
        _eval_acc_mcc_on_loader,
        dataset_kwargs_from_args,
        run,
    )

    if a.action == "train":
        output = (a.output or ROOT / "runs" / r["id"]).resolve()
        if output.exists():
            p.error("Output exists; use a fresh --output directory.")
        if output == a.checkpoint_root or a.checkpoint_root in output.parents:
            p.error("Training must not overwrite the paper checkpoint archive.")
        args.ckpt_dir = str(output)
        args.no_save_ckpt = False
        if a.epochs is not None:
            args.epochs = a.epochs
        run(profile, args)
        return 0
    from data import MarketWindowDataset

    set_all_seeds(args.seed)
    device = device_from_args(args)
    ds = MarketWindowDataset(profile, mode="test", **dataset_kwargs_from_args(args, profile))
    companies = list(getattr(_builder_dataset(ds), "companies", []) or [])
    model = build_model(args, len(companies) or profile.grid_num_stocks)
    ckpt = a.checkpoint_root / r["id"] / (a.tag + ".pt")
    mapping = load_archive_weights(model, ckpt)
    model.to(device)
    loader = make_model_dataloader(ds, batch_size=1, shuffle=False)
    (acc, mcc, _) = _eval_acc_mcc_on_loader(
        loader,
        model,
        device,
        move_batch=lambda b: move_model_batch_to_device(b, device),
        split_name="PAPER TEST",
    )
    expected = r["expected"] if a.tag == "best" else None
    delta = {
        k: None if expected is None or v is None or expected.get(k) is None else v - expected[k]
        for (k, v) in {"acc": acc, "mcc": mcc}.items()
    }
    tolerances = {
        "acc": a.tolerance if a.acc_tolerance is None else a.acc_tolerance,
        "mcc": a.tolerance if a.mcc_tolerance is None else a.mcc_tolerance,
    }
    match = expected is not None and all(
        v is not None and abs(v) <= tolerances[k] for k, v in delta.items()
    )
    exact = expected is not None and all(
        v is not None and abs(v) <= 1e-12 for v in delta.values()
    )
    report = {
        "dataset": r["dataset"],
        "case": r["case"],
        "seed": r["seed"],
        "checkpoint": str(ckpt.relative_to(a.checkpoint_root)),
        "tag": a.tag,
        "actual": {"acc": acc, "mcc": mcc},
        "expected": expected,
        "delta": delta,
        "status": "PASS" if exact else "PASS_WITHIN_TOLERANCE" if match else "MEASURED" if expected is None else "MISMATCH",
        "tolerances": tolerances,
        "exact_acc_mcc_match": exact,
        "mapping": mapping,
        "scope": "Checkpoint inference; not training from scratch.",
    }
    output = (
        a.output
        or ROOT / "results/evaluation" / r["dataset"] / r["case"] / ("s%d.json" % r["seed"])
    ).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for (k, v) in report.items() if k != "mapping"}, indent=2))
    return 0 if match or expected is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
