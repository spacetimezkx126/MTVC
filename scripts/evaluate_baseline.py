"""Evaluate one historical no-volume baseline checkpoint without retraining."""
from __future__ import annotations
import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["pen", "stocknet", "lstm", "bi_lstm", "alstm", "adv_alstm", "dtml"], required=True)
    parser.add_argument("--dataset", choices=["csmd50", "csmd300", "massive"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True, help="Directory containing method_dataset_seedNN run folders.")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=ROOT / "baselines", help="Baseline implementation directory.")
    parser.add_argument("--predictions", type=Path, help="Also export all valid dated PEN/StockNet test rows for portfolio analysis.")
    parser.add_argument("--prediction-batch-size", type=int, default=256)
    a = parser.parse_args()
    if a.predictions and a.method not in ("pen", "stocknet"):
        parser.error("Dated export currently supports PEN and StockNet; price-only checkpoints can still be evaluated.")
    a.output = a.output.resolve()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    data = a.data_root or ROOT / "data" / {"csmd50": "CSMD50", "csmd300": "CSMD300", "massive": "massive_data"}[a.dataset]
    folder = a.checkpoint_root.resolve() / f"{a.method}_{a.dataset}_seed{a.seed}"
    matches = sorted(folder.rglob(f"best_{a.dataset}.pt"))
    if len(matches) != 1:
        raise ValueError(f"Expected one best checkpoint, found {len(matches)} in {folder}")
    checkpoint = matches[0]
    sys.path.insert(0, str(a.code_root.resolve()))
    import torch
    torch.set_num_threads(1)
    kind = a.method if a.method in ("pen", "stocknet") else "single"
    module = importlib.import_module("main_" + kind)
    catalog_file = ROOT / "configs/baseline_recipes.json"
    catalog = json.loads(catalog_file.read_text())["runs"] if catalog_file.is_file() else []
    recipe = next((r for r in catalog if r["method"] == a.method and r["dataset"] == a.dataset and r["seed"] == a.seed), {})
    if kind == "stocknet" and recipe.get("training_device"):
        original_load = module.load_ckpt
        def load_with_device(*args, **kwargs):
            state, meta = original_load(*args, **kwargs)
            meta.setdefault("training_device", recipe["training_device"])
            return state, meta
        module.load_ckpt = load_with_device
    function = "evaluate" if kind == "single" else "evaluate_flat"
    original = getattr(module, function)
    results = []
    def record(*args, **kwargs):
        value = original(*args, **kwargs)
        split = kwargs.get("split_name", args[5] if kind == "single" and len(args)>5 else args[4] if kind != "single" else "")
        results.append({"split": split, "loss": float(value[0]), "acc": float(value[1]), "mcc": float(value[2]), "n": int(value[3]) if len(value)>3 else None})
        if a.predictions and "TEST" in split:
            from common import collate_pen_batch, set_seed
            import numpy as np
            set_seed(a.seed)
            model, source_loader, device, days = args[:4]
            loader = torch.utils.data.DataLoader(source_loader.dataset, batch_size=a.prediction_batch_size,
                                                  shuffle=False, collate_fn=lambda batch: batch)
            a.predictions = a.predictions.resolve()
            a.predictions.parent.mkdir(parents=True, exist_ok=True)
            with a.predictions.open("w", newline="") as handle, torch.no_grad():
                writer = csv.writer(handle)
                writer.writerow(["date", "company", "probability", "return", "label", "strong"])
                for raw in loader:
                    batch = collate_pen_batch(raw, days)
                    out = model(*(batch[key].to(device) for key in
                                  ("word_ids", "n_words", "ss_index", "n_msgs", "price", "y", "t_idx")),
                                compute_loss=False)
                    probabilities = torch.softmax(out["y_T"], dim=-1)[:, 1].cpu().tolist()
                    for i, probability in enumerate(probabilities):
                        realized = float(batch["main_mv"][i])
                        if np.isfinite(realized):
                            writer.writerow([batch["dates"][i], batch["companies"][i], probability,
                                             realized, int(realized > 0), int(batch["strong"][i])])
        return value
    setattr(module, function, record)
    sys.argv = [f"main_{kind}.py", "--mode", "eval", "--dataset_root", str(data.resolve()),
                "--seed", str(a.seed), "--device", a.device, "--ckpt", str(checkpoint),
                "--best_metric", "val_acc_mcc", "--split_mode", "contiguous",
                "--log_file", str(a.output.with_suffix('.log'))]
    sys.argv += ["--model", a.method, "--input_size", "3"] if kind == "single" else ["--price_input_size", "3"]
    module.main()
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = {key: raw.get("test_"+key) for key in ("acc", "mcc", "n")}
    source = "checkpoint_metadata"
    if expected["acc"] is None and recipe.get("expected"):
        expected = recipe["expected"]
        source = recipe.get("reference_source", "recorded_recipe")
    actual = next(row for row in reversed(results) if "TEST" in row["split"])
    delta = {key: actual[key] - expected[key] if expected[key] is not None else None for key in ("acc", "mcc")}
    report = {"method": a.method, "dataset": a.dataset, "seed": a.seed,
              "checkpoint": str(checkpoint.relative_to(a.checkpoint_root.resolve())),
              "actual": actual, "checkpoint_reference": expected, "delta": delta,
              "exact_match": all(delta[key] is not None and abs(delta[key]) < 1e-12 for key in ("acc", "mcc")),
              "reference_source": source,
              "matches_logged_precision": all(delta[key] is not None and abs(delta[key]) <= 0.00005 for key in ("acc", "mcc")),
              "test_sampling_state_available": bool(raw.get("test_sampling_state") or raw.get("test_rng_state")),
              "prediction_export": {"file": a.predictions.name, "seed": a.seed, "batch_size": a.prediction_batch_size,
                                    "scope": "All valid samples; independent fixed-seed export; historical double-softmax ranking convention."} if a.predictions else None,
              "evaluations": results, "scope": "Independent checkpoint inference, not retraining."}
    report["status"] = "EXACT_MATCH" if report["exact_match"] else "MATCHES_LOG_PRECISION" if source == "rounded_training_log" and report["matches_logged_precision"] else "MISMATCH" if expected["acc"] is not None else "NO_REFERENCE"
    a.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
