"""PEN train/eval helpers (extracted from Pattern_Mining/main_pen.py)."""
from __future__ import annotations

import logging
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from pen1_data import calculate_mcc

LOGGER = logging.getLogger("pen1_train")
def slice_batch(batch: dict, idx: torch.Tensor) -> dict:
    out = {}
    base_n = batch["t_idx"].size(0)
    for k, v in batch.items():
        if k in ("companies", "dates"):
            out[k] = [v[i] for i in idx.tolist()]
        elif isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) == base_n:
            out[k] = v[idx]
        else:
            out[k] = v
    return out


def _filter_strong_batch(batch: dict, only_strong: bool):
    if not only_strong or "strong" not in batch:
        return batch
    sm = batch["strong"]
    if not sm.any():
        return None
    idx = sm.nonzero(as_tuple=True)[0]
    return slice_batch(batch, idx)


def _forward_batch(model, batch, device):
    return model(
        batch["word_ids"].to(device),
        batch["n_words"].to(device),
        batch["ss_index"].to(device),
        batch["n_msgs"].to(device),
        batch["price"].to(device),
        batch["y"].to(device),
        batch["t_idx"].to(device),
        compute_loss=True,
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    max_n_days: int,
    split_name: str = "",
    logger=None,
    strong_only: bool = True,
):
    model.eval()
    all_true, all_pred = [], []
    total_loss, n_batches = 0.0, 0
    for raw in loader:
        batch = collate_pen_batch(raw, max_n_days)
        batch = _filter_strong_batch(batch, strong_only)
        if batch is None:
            continue
        out = _forward_batch(model, batch, device)
        if out["loss"] is not None:
            total_loss += float(out["loss"].item())
            n_batches += 1
        all_true.extend(out["y_true_class"].cpu().tolist())
        all_pred.extend(out["y_pred_class"].cpu().tolist())
    acc = accuracy_score(all_true, all_pred) if all_true else 0.0
    mcc = calculate_mcc(all_true, all_pred) if all_true else 0.0
    avg_loss = total_loss / max(n_batches, 1)
    if logger and split_name:
        logger.info(
            "[%s] loss=%.6f acc=%.4f mcc=%.4f n=%d strong_only=%s",
            split_name,
            avg_loss,
            acc,
            mcc,
            len(all_true),
            strong_only,
        )
    return avg_loss, acc, mcc, len(all_true)


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    max_n_days: int,
    grad_clip: float,
    epoch: int,
    log_every: int,
    logger=None,
    strong_only: bool = True,
):
    model.train()
    total_loss, n_batches = 0.0, 0
    n_steps = len(loader)
    if logger:
        logger.info("[Epoch %03d] train start, %d batches strong_only=%s", epoch, n_steps, strong_only)
    for batch_idx, raw in enumerate(loader, start=1):
        batch = collate_pen_batch(raw, max_n_days)
        batch = _filter_strong_batch(batch, strong_only)
        if batch is None:
            continue
        optimizer.zero_grad(set_to_none=True)
        out = _forward_batch(model, batch, device)
        out["loss"].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(out["loss"].item())
        n_batches += 1
        if logger and log_every > 0 and (batch_idx % log_every == 0 or batch_idx == n_steps):
            logger.info(
                "[Epoch %03d] batch %d/%d loss=%.6f",
                epoch,
                batch_idx,
                n_steps,
                float(out["loss"].item()),
            )
    avg_loss = total_loss / max(n_batches, 1)

    model.eval()
    eval_correct, eval_samples = 0, 0
    with torch.no_grad():
        for raw in loader:
            batch = collate_pen_batch(raw, max_n_days)
            batch = _filter_strong_batch(batch, strong_only)
            if batch is None:
                continue
            out = _forward_batch(model, batch, device)
            eval_correct += int((out["y_pred_class"] == out["y_true_class"]).sum().item())
            eval_samples += int(out["y_true_class"].numel())
    train_acc = float(eval_correct) / float(eval_samples) if eval_samples else 0.0
    if logger:
        logger.info(
            "[Epoch %03d] train done loss=%.6f train_acc=%.4f (eval mode, strong_only=%s)",
            epoch,
            avg_loss,
            train_acc,
            strong_only,
        )
    model.train()
    return avg_loss, train_acc


def save_ckpt(path: str, model, **meta) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"state_dict": model.state_dict(), **meta}
    torch.save(payload, path)


def load_ckpt(path: str, device):
    try:
        raw = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location=device)
    if isinstance(raw, dict) and "state_dict" in raw:
        meta = {k: v for k, v in raw.items() if k != "state_dict"}
        return raw["state_dict"], meta
    return raw, {}


def kl_lambda_at_step(step: int, anneal_rate: float, start_step: int, constant: float | None) -> float:
    if step < start_step:
        return 0.0
    if constant is not None:
        return constant
    return min(anneal_rate * step, 1.0)


def collate_pen_batch(batch, max_n_days: int):
    """Flat PEN batch with ``main_mv`` / ``strong`` (CminCnPenDataset samples)."""
    b = len(batch)
    m = batch[0]["word_ids"].size(1) if torch.is_tensor(batch[0]["word_ids"]) else batch[0]["word_ids"].shape[1]
    w = batch[0]["word_ids"].size(2) if torch.is_tensor(batch[0]["word_ids"]) else batch[0]["word_ids"].shape[2]
    y_size = batch[0]["y"].size(-1) if torch.is_tensor(batch[0]["y"]) else batch[0]["y"].shape[-1]

    word_ids = torch.zeros(b, max_n_days, m, w, dtype=torch.long)
    ss_index = torch.zeros(b, max_n_days, m, dtype=torch.long)
    n_words = torch.zeros(b, max_n_days, m, dtype=torch.long)
    n_msgs = torch.zeros(b, max_n_days, dtype=torch.long)
    p0 = batch[0]["price"]
    price_dim = int(p0.size(-1) if torch.is_tensor(p0) else np.asarray(p0).shape[-1])
    price = torch.zeros(b, max_n_days, price_dim, dtype=torch.float32)
    y = torch.zeros(b, max_n_days, y_size, dtype=torch.float32)
    t_idx = torch.zeros(b, dtype=torch.long)
    stock_id = torch.zeros(b, dtype=torch.long)
    main_mv = torch.full((b,), float("nan"), dtype=torch.float32)
    strong = torch.zeros((b,), dtype=torch.bool)
    companies, dates = [], []

    for i, item in enumerate(batch):
        t = int(item["T"])
        word_ids[i, :t] = item["word_ids"] if torch.is_tensor(item["word_ids"]) else torch.as_tensor(item["word_ids"], dtype=torch.long)
        ss_index[i, :t] = item["ss_index"] if torch.is_tensor(item["ss_index"]) else torch.as_tensor(item["ss_index"], dtype=torch.long)
        n_words[i, :t] = item["n_words"] if torch.is_tensor(item["n_words"]) else torch.as_tensor(item["n_words"], dtype=torch.long)
        n_msgs[i, :t] = item["n_msgs"] if torch.is_tensor(item["n_msgs"]) else torch.as_tensor(item["n_msgs"], dtype=torch.long)
        price[i, :t] = item["price"] if torch.is_tensor(item["price"]) else torch.as_tensor(item["price"], dtype=torch.float32)
        y[i, :t] = item["y"] if torch.is_tensor(item["y"]) else torch.as_tensor(item["y"], dtype=torch.float32)
        t_idx[i] = t
        stock_id[i] = item["stock_id"]
        if "main_mv" in item:
            main_mv[i] = float(item["main_mv"])
        if "strong" in item:
            strong[i] = bool(item["strong"])
        companies.append(item["company"])
        dates.append(item["target_date"])

    return {
        "word_ids": word_ids,
        "ss_index": ss_index,
        "n_words": n_words,
        "n_msgs": n_msgs,
        "price": price,
        "y": y,
        "t_idx": t_idx,
        "stock_id": stock_id,
        "main_mv": main_mv,
        "strong": strong,
        "companies": companies,
        "dates": dates,
    }


def pen_strong_eval_mask(main_mv: torch.Tensor) -> torch.Tensor:
    mv = main_mv.reshape(-1).float()
    return (mv > MV_STRONG_HIGH) | (mv < MV_NEUTRAL_LOW)


def majority_acc_from_classes(all_true: list[int]) -> float:
    if not all_true:
        return 0.0
    pos_rate = float(sum(all_true)) / float(len(all_true))
    return max(pos_rate, 1.0 - pos_rate)


def filter_pen_predictions_by_strong_mask(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    main_mv: torch.Tensor,
    *,
    strong_eval: bool = True,
    strong_mask: torch.Tensor | None = None,
) -> tuple[list[int], list[int]]:
    yt = y_true.detach().cpu().reshape(-1)
    yp = y_pred.detach().cpu().reshape(-1)
    if not strong_eval:
        return yt.tolist(), yp.tolist()
    if strong_mask is not None:
        sm = strong_mask.detach().cpu().reshape(-1).bool()
        return yt[sm].tolist(), yp[sm].tolist()
    mask = pen_strong_eval_mask(main_mv).cpu()
    return yt[mask].tolist(), yp[mask].tolist()


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_logger(log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("pen")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info("log file: %s", log_file)
    return logger


# --- Split-lookback flat PEN dataset (CMIN-CN / CSMD / CMIN-US; same sample rules) ---

SEG_LENGTH = 10


