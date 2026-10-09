"""Backtest entrypoints for vol_screen_trend_attn checkpoints."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.vol_screen_engine import (  # noqa: E402
    VolScreenPredictor,
    build_test_loader,
    load_vol_screen_checkpoint,
    run_classification_eval,
    run_portfolio_strategy,
    run_single_stock_backtest,
)

DEFAULT_CKPT = (
    ROOT
    / "checkpoints/csmd50_train_edge_volatility_ablation-vol_screen_trend_attn-window-top-r0p4-seed43/best.pt"
)

PORTFOLIO_MODES = ("label_ew", "model_ew", "model_topk", "all")


def _save_portfolio_csv(path: Path, results: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["date"]
    for r in results:
        cols.extend([f"budget_{r.name}", f"daily_ret_{r.name}", f"picks_{r.name}"])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        n = len(results[0].dates)
        for i in range(n):
            row = [results[0].dates[i]]
            for r in results:
                row.extend([f"{r.budget[i]:.4f}", f"{r.daily_ret[i]:.6f}", "|".join(r.picks[i])])
            w.writerow(row)


def _save_plot(path: Path, results: list, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"[backtest] skip plot: {e}")
        return
    days = np.arange(len(results[0].budget))
    plt.figure(figsize=(10, 6))
    for r in results:
        plt.plot(days, r.budget, label=r.name)
    plt.title(title)
    plt.ylabel("Budget")
    plt.xticks([])
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[backtest] plot -> {path}")


def backtest_portfolio(args) -> dict:
    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt).resolve()
    _, ckpt_args, profile, model, eop, eopt = load_vol_screen_checkpoint(ckpt_path, device)
    ds, loader = build_test_loader(ckpt_args, profile, split=args.split, max_windows=args.max_windows)
    store = VolScreenPredictor(
        model, eop, eopt, ckpt_args, loader, ds._inner.companies, device=device, strong_only=args.strong_only
    ).collect()

    modes = PORTFOLIO_MODES[:-1] if args.mode == "all" else (args.mode,)
    results = []
    for mode in modes:
        name = {
            "label_ew": "Label EW",
            "model_ew": f"Model EW (>={args.model_ew_threshold})",
            "model_topk": f"Model top-{args.topk}",
        }[mode]
        results.append(
            run_portfolio_strategy(
                store,
                name=name,
                mode=mode,
                budget=args.budget,
                fees=args.fees,
                topk=args.topk,
                prob_threshold=args.prob_threshold,
                model_ew_threshold=args.model_ew_threshold,
            )
        )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "ckpt": str(ckpt_path),
        "dataset": profile.key,
        "split": args.split,
        "mode": args.mode,
        "strategies": {r.name: r.metrics for r in results},
    }
    with (out_dir / "portfolio_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    _save_portfolio_csv(out_dir / "portfolio_daily.csv", results)
    if args.plot:
        _save_plot(out_dir / "portfolio_budget.png", results, f"{profile.key} {args.split} portfolio backtest")

    cls = run_classification_eval(store, prob_threshold=args.model_ew_threshold, strong_only=args.strong_only)
    summary["classification"] = cls
    print(f"[portfolio] ckpt={ckpt_path} dataset={profile.key} split={args.split}")
    for r in results:
        m = r.metrics
        print(
            f"  {r.name}: final={m['final_budget']:.2f} ARR={m['arr']:.4f} SR={m['sr']:.4f} MDD={m['mdd']:.4f}"
        )
    print(f"  classification ACC={cls['acc']:.4f} MCC={cls['mcc']:.4f} n={cls['n_samples']}")
    return summary


def backtest_single_stock(args) -> dict:
    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt).resolve()
    _, ckpt_args, profile, model, eop, eopt = load_vol_screen_checkpoint(ckpt_path, device)
    ds, loader = build_test_loader(ckpt_args, profile, split=args.split, max_windows=args.max_windows)
    store = VolScreenPredictor(
        model, eop, eopt, ckpt_args, loader, ds._inner.companies, device=device, strong_only=args.strong_only
    ).collect()

    per_stock, summary = run_single_stock_backtest(
        store,
        budget=args.budget,
        fees=args.fees,
        allow_short=not args.long_only,
        prob_threshold=args.model_ew_threshold,
    )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "single_stock.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["company", "acc", "arr", "sr", "mdd", "final_budget", "n_days"])
        w.writeheader()
        w.writerows(per_stock)

    payload = {
        "ckpt": str(ckpt_path),
        "dataset": profile.key,
        "split": args.split,
        "long_only": args.long_only,
        "summary": summary,
    }
    with (out_dir / "single_stock_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[single_stock] ckpt={ckpt_path} dataset={profile.key} split={args.split} n_stocks={summary['n_stocks']}")
    print(
        f"  mean ACC={summary['mean_acc']:.4f} mean ARR={summary['mean_arr']:.4f} "
        f"mean SR={summary['mean_sr']:.4f} mean MDD={summary['mean_mdd']:.4f}"
    )
    return payload


def backtest_classification(args) -> dict:
    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt).resolve()
    _, ckpt_args, profile, model, eop, eopt = load_vol_screen_checkpoint(ckpt_path, device)
    ds, loader = build_test_loader(ckpt_args, profile, split=args.split, max_windows=args.max_windows)
    store = VolScreenPredictor(
        model, eop, eopt, ckpt_args, loader, ds._inner.companies, device=device, strong_only=False
    ).collect()

    all_eval = run_classification_eval(store, prob_threshold=args.model_ew_threshold, strong_only=False)
    strong_eval = run_classification_eval(store, prob_threshold=args.model_ew_threshold, strong_only=True)

    payload = {
        "ckpt": str(ckpt_path),
        "dataset": profile.key,
        "split": args.split,
        "all_samples": all_eval,
        "strong_only": strong_eval,
    }
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "classification_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[classification] ckpt={ckpt_path} dataset={profile.key} split={args.split}")
    print(
        f"  all: ACC={all_eval['acc']:.4f} MCC={all_eval['mcc']:.4f} n={all_eval['n_samples']}\n"
        f"  strong: ACC={strong_eval['acc']:.4f} MCC={strong_eval['mcc']:.4f} n={strong_eval['n_samples']}"
    )
    return payload


def backtest_topk_sweep(args) -> dict:
    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt).resolve()
    _, ckpt_args, profile, model, eop, eopt = load_vol_screen_checkpoint(ckpt_path, device)
    ds, loader = build_test_loader(ckpt_args, profile, split=args.split, max_windows=args.max_windows)
    store = VolScreenPredictor(
        model, eop, eopt, ckpt_args, loader, ds._inner.companies, device=device, strong_only=args.strong_only
    ).collect()

    sweep = []
    for k in args.topk_list:
        r = run_portfolio_strategy(
            store,
            name=f"Model top-{k}",
            mode="model_topk",
            budget=args.budget,
            fees=args.fees,
            topk=int(k),
            prob_threshold=args.prob_threshold,
            model_ew_threshold=args.model_ew_threshold,
        )
        sweep.append({"topk": int(k), **r.metrics})

    label = run_portfolio_strategy(
        store, name="Label EW", mode="label_ew", budget=args.budget, fees=args.fees
    )
    payload = {
        "ckpt": str(ckpt_path),
        "dataset": profile.key,
        "split": args.split,
        "label_ew": label.metrics,
        "model_topk_sweep": sweep,
    }
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "topk_sweep.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[topk_sweep] ckpt={ckpt_path} dataset={profile.key}")
    print(f"  Label EW: ARR={label.metrics['arr']:.4f} final={label.metrics['final_budget']:.2f}")
    for row in sweep:
        print(f"  top-{row['topk']}: ARR={row['arr']:.4f} final={row['final_budget']:.2f} SR={row['sr']:.4f}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Backtest vol_screen_trend_attn checkpoint")
    ap.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--budget", type=float, default=10_000.0)
    ap.add_argument("--fees", type=float, default=0.0)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--topk_list", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--prob_threshold", type=float, default=0.0)
    ap.add_argument("--model_ew_threshold", type=float, default=0.5)
    ap.add_argument("--strong_only", action="store_true")
    ap.add_argument("--long_only", action="store_true", help="single_stock: disable short selling")
    ap.add_argument("--max_windows", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT / "backtest/result/vol_screen_csmd50"),
    )
    ap.add_argument(
        "--task",
        required=True,
        choices=["portfolio", "single_stock", "classification", "topk_sweep", "all"],
        help="backtest mode",
    )
    ap.add_argument(
        "--mode",
        default="all",
        choices=list(PORTFOLIO_MODES),
        help="portfolio sub-mode (portfolio task only)",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()
    base_out = Path(args.out_dir)
    if args.task == "portfolio":
        args.out_dir = str(base_out / "portfolio")
        backtest_portfolio(args)
    elif args.task == "single_stock":
        args.out_dir = str(base_out / "single_stock")
        backtest_single_stock(args)
    elif args.task == "classification":
        args.out_dir = str(base_out / "classification")
        backtest_classification(args)
    elif args.task == "topk_sweep":
        args.out_dir = str(base_out / "topk_sweep")
        backtest_topk_sweep(args)
    elif args.task == "all":
        for sub, fn in [
            ("portfolio", backtest_portfolio),
            ("single_stock", backtest_single_stock),
            ("classification", backtest_classification),
            ("topk_sweep", backtest_topk_sweep),
        ]:
            sub_args = argparse.Namespace(**vars(args))
            sub_args.out_dir = str(base_out / sub)
            if sub == "portfolio":
                sub_args.mode = "all"
            print(f"\n===== {sub} =====")
            fn(sub_args)


if __name__ == "__main__":
    main()
