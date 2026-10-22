"""Export dated checkpoint predictions for classification audits and backtesting."""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/mtvc"))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--dataset", choices=["csmd50", "csmd300", "massive"], default="csmd50")
p.add_argument("--case", default="full_casel6")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--checkpoint-root", type=Path, default=ROOT.parent / "MTVC_checkpoints")
p.add_argument("--data-root", type=Path)
p.add_argument("--device", default="cuda:0")
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output = a.output.resolve()
a.checkpoint_root = a.checkpoint_root.resolve()
if a.data_root:
    a.data_root = a.data_root.resolve()
from reproduction import configure_environment, recipe_args, build_model, load_archive_weights

configure_environment()
import torch
from data import MarketWindowDataset
from train import (
    set_all_seeds,
    device_from_args,
    dataset_kwargs_from_args,
    _builder_dataset,
    make_model_dataloader,
    move_model_batch_to_device,
    calculate_mcc,
)
from tool.cuda_guard import cuda_device_lock, forward_with_cuda_retry

torch.set_num_threads(1)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
recipe_file = a.checkpoint_root / "paper_recipes.json"
if not recipe_file.is_file():
    recipe_file = ROOT / "configs/release_checkpoint_recipes.json"
if not recipe_file.is_file():
    recipe_file = ROOT / "configs/paper_recipes.json"
recipes = json.loads(recipe_file.read_text())["runs"]
r = next(
    (
        r
        for r in recipes
        if r["dataset"] == a.dataset and r["case"] == a.case and r["seed"] == a.seed
    ),
    None,
)
if r is None:
    p.error("No matching recorded recipe.")
args, profile = recipe_args(r, a.device, a.data_root)
set_all_seeds(a.seed)
device = device_from_args(args)
ds = MarketWindowDataset(profile, mode="test", **dataset_kwargs_from_args(args, profile))
companies = list(getattr(_builder_dataset(ds), "companies", []) or [])
model = build_model(args, len(companies) or profile.grid_num_stocks)
load_archive_weights(model, a.checkpoint_root / r["id"] / "best.pt")
model.to(device)
model.eval()
a.output.parent.mkdir(parents=True, exist_ok=True)
count = 0
truth = []
predicted = []
with a.output.open("w", newline="") as f, torch.no_grad():
    writer = csv.writer(f)
    writer.writerow(["date", "company", "probability", "return", "label", "strong"])
    for batch in make_model_dataloader(ds, batch_size=1, shuffle=False):
        if batch["price"].x.numel() == 0 or batch["label"].x.numel() == 0:
            continue
        move_model_batch_to_device(batch, device)
        with cuda_device_lock(device):
            logits = forward_with_cuda_retry(
                lambda: model(batch), device=device, tag="prediction-export"
            )
        probs = torch.sigmoid(logits.reshape(-1)).cpu().tolist()
        dates = batch["price"].date
        if len(dates) == 1 and isinstance(dates[0], (list, tuple)):
            dates = dates[0]
        ids = batch["price"].company_id.reshape(-1).cpu().tolist()
        labels = batch["label"].x.reshape(-1).cpu().tolist()
        returns = batch["label"].org.reshape(-1).cpu().tolist()
        valid = batch["label"].valid_mask.reshape(-1).cpu().tolist()
        strong = batch["label"].strong_mask.reshape(-1).cpu().tolist()
        assert len(probs) == len(ids) == len(dates) == len(labels) == len(returns)
        for i, prob in enumerate(probs):
            if not valid[i]:
                continue
            cid = int(ids[i])
            company = companies[cid] if 0 <= cid < len(companies) else str(cid)
            writer.writerow([dates[i], company, prob, returns[i], int(labels[i]), int(strong[i])])
            count += 1
            if strong[i]:
                truth.append(int(labels[i]))
                predicted.append(int(prob > 0.5))
acc = sum(t == p for t, p in zip(truth, predicted)) / len(truth) if truth else None
mcc = calculate_mcc(truth, predicted) if truth else None
report = {
    "dataset": a.dataset,
    "case": a.case,
    "seed": a.seed,
    "valid_rows": count,
    "strong_rows": len(truth),
    "acc": acc,
    "mcc": mcc,
    "scope": "Dated test predictions; valid rows are retained for the original portfolio convention.",
}
a.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
