"""Checkpoints: model, optimizer, RNG, and the CLI config."""

from __future__ import annotations
import json
import math
import os
import random
from datetime import datetime
from typing import Any
import numpy as np
import torch

FORMAT_VERSION = 1
TRAIN_MODES = frozenset({"train", "train_simple"})


def namespace_to_dict(args) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in vars(args).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            out[k] = str(v)
    root = os.environ.get("DUAL_TF_MODEL_DIR", os.getcwd())
    for key, value in out.items():
        if isinstance(value, str) and os.path.isabs(value):
            out[key] = os.path.relpath(value, root)
    return out


def is_ckpt_enabled(args) -> bool:
    if getattr(args, "no_save_ckpt", False):
        return False
    mode = str(getattr(args, "mode", ""))
    return mode in TRAIN_MODES


def resolve_ckpt_dir(args, profile) -> str:
    """Checkpoint directory. Default is ./checkpoints/{dataset}_{mode}."""
    custom = str(getattr(args, "ckpt_dir", "") or "").strip()
    if custom:
        return os.path.abspath(custom)
    return os.path.abspath(os.path.join("./checkpoints", f"{profile.key}_{args.mode}"))


def module_state_dict(module) -> dict[str, torch.Tensor] | None:
    if module is None:
        return None
    mod = module.module if hasattr(module, "module") else module
    return {k: v.detach().cpu().clone() for (k, v) in mod.state_dict().items()}


def capture_rng_states() -> dict[str, Any]:
    states: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng_states(states: dict[str, Any] | None) -> None:
    if not states:
        return
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch" in states:
        torch.set_rng_state(states["torch"])
    if "cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


def build_checkpoint_payload(
    *,
    profile,
    args,
    epoch: int,
    tag: str,
    model,
    metrics: dict[str, Any] | None = None,
    edge_operator=None,
    edge_operator_type=None,
    edge_operator_aux=None,
    edge_operator_aux_type=None,
    model_optimizer=None,
    denoise_optimizer=None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "tag": tag,
        "saved_at": datetime.now().isoformat(),
        "profile_key": profile.key,
        "mode": args.mode,
        "seed": int(getattr(args, "seed", 42)),
        "epoch": int(epoch),
        "args": namespace_to_dict(args),
        "metrics": dict(metrics or {}),
        "model_state_dict": module_state_dict(model),
        "rng_states": capture_rng_states(),
    }
    for name, mod in (
        ("edge_operator_state_dict", edge_operator),
        ("edge_operator_type_state_dict", edge_operator_type),
        ("edge_operator_aux_state_dict", edge_operator_aux),
        ("edge_operator_aux_type_state_dict", edge_operator_aux_type),
    ):
        sd = module_state_dict(mod)
        if sd is not None:
            payload[name] = sd
    if model_optimizer is not None:
        payload["model_optimizer_state_dict"] = model_optimizer.state_dict()
    if denoise_optimizer is not None:
        payload["denoise_optimizer_state_dict"] = denoise_optimizer.state_dict()
    if extra:
        payload.update(extra)
    payload["state_dict"] = payload["model_state_dict"]
    return payload


def save_checkpoint_file(path: str, payload: dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(payload, path)
    meta_path = path + ".meta.json"
    meta = {
        k: payload.get(k)
        for k in (
            "format_version",
            "tag",
            "saved_at",
            "profile_key",
            "mode",
            "seed",
            "epoch",
            "metrics",
        )
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return path


class CheckpointManager:
    """Writes best.pt, last.pt, and an optional per-epoch file."""

    def __init__(self, args, profile):
        self.args = args
        self.profile = profile
        self.enabled = is_ckpt_enabled(args)
        self.ckpt_dir = resolve_ckpt_dir(args, profile) if self.enabled else ""
        self.save_every_epoch = bool(getattr(args, "save_ckpt_every_epoch", False))
        self.best_epoch = 0
        self.best_val_acc_plus_mcc = float("-inf")
        self.best_val_loss = float("inf")
        self.es_metric = str(getattr(args, "es_metric", "acc_mcc") or "acc_mcc")
        self.best_metrics: dict[str, Any] = {}
        if self.enabled:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            print(f"[CKPT] save enabled -> {self.ckpt_dir} (es_metric={self.es_metric})")

    def _common_modules(
        self,
        model,
        edge_operator=None,
        edge_operator_type=None,
        edge_operator_aux=None,
        edge_operator_aux_type=None,
        model_optimizer=None,
        denoise_optimizer=None,
    ):
        return dict(
            model=model,
            edge_operator=edge_operator,
            edge_operator_type=edge_operator_type,
            edge_operator_aux=edge_operator_aux,
            edge_operator_aux_type=edge_operator_aux_type,
            model_optimizer=model_optimizer,
            denoise_optimizer=denoise_optimizer,
        )

    def save(
        self,
        filename: str,
        *,
        epoch: int,
        tag: str,
        model,
        metrics: dict[str, Any] | None = None,
        edge_operator=None,
        edge_operator_type=None,
        edge_operator_aux=None,
        edge_operator_aux_type=None,
        model_optimizer=None,
        denoise_optimizer=None,
        extra: dict[str, Any] | None = None,
    ) -> str | None:
        if not self.enabled:
            return None
        path = os.path.join(self.ckpt_dir, filename)
        payload = build_checkpoint_payload(
            profile=self.profile,
            args=self.args,
            epoch=epoch,
            tag=tag,
            metrics=metrics,
            extra=extra,
            **self._common_modules(
                model,
                edge_operator,
                edge_operator_type,
                edge_operator_aux,
                edge_operator_aux_type,
                model_optimizer,
                denoise_optimizer,
            ),
        )
        save_checkpoint_file(path, payload)
        print(f"[CKPT] saved {tag} epoch={epoch} -> {path}")
        return path

    def maybe_save_best(
        self,
        *,
        epoch: int,
        val_acc,
        val_mcc,
        model,
        edge_operator=None,
        edge_operator_type=None,
        edge_operator_aux=None,
        edge_operator_aux_type=None,
        model_optimizer=None,
        denoise_optimizer=None,
        test_acc=None,
        test_mcc=None,
        val_loss=None,
        es_metric: str | None = None,
    ) -> bool:
        metric = str(es_metric or self.es_metric or "acc_mcc")
        if metric == "val_loss":
            if val_loss is None:
                return False
            vloss = float(val_loss)
            if not math.isfinite(vloss):
                return False
            if vloss >= self.best_val_loss - 1e-08:
                return False
            self.best_val_loss = vloss
            self.best_epoch = int(epoch)
            val_sum = (
                float(val_acc) + float(val_mcc)
                if val_acc is not None and val_mcc is not None
                else None
            )
            if val_sum is not None:
                self.best_val_acc_plus_mcc = val_sum
            self.best_metrics = {
                "best_epoch": self.best_epoch,
                "best_val_loss": vloss,
                "best_es_metric": "val_loss",
                "best_val_acc": None if val_acc is None else float(val_acc),
                "best_val_mcc": None if val_mcc is None else float(val_mcc),
                "best_val_acc_plus_mcc": val_sum,
                "best_test_acc": None if test_acc is None else float(test_acc),
                "best_test_mcc": None if test_mcc is None else float(test_mcc),
            }
        else:
            if val_acc is None or val_mcc is None:
                return False
            val_sum = float(val_acc) + float(val_mcc)
            if val_sum <= self.best_val_acc_plus_mcc + 1e-08:
                return False
            self.best_val_acc_plus_mcc = val_sum
            self.best_epoch = int(epoch)
            self.best_metrics = {
                "best_epoch": self.best_epoch,
                "best_es_metric": "acc_mcc",
                "best_val_acc": float(val_acc),
                "best_val_mcc": float(val_mcc),
                "best_val_acc_plus_mcc": val_sum,
                "best_val_loss": None if val_loss is None else float(val_loss),
                "best_test_acc": None if test_acc is None else float(test_acc),
                "best_test_mcc": None if test_mcc is None else float(test_mcc),
            }
        self.save(
            "best.pt",
            epoch=epoch,
            tag="best",
            model=model,
            metrics=self.best_metrics,
            edge_operator=edge_operator,
            edge_operator_type=edge_operator_type,
            edge_operator_aux=edge_operator_aux,
            edge_operator_aux_type=edge_operator_aux_type,
            model_optimizer=model_optimizer,
            denoise_optimizer=denoise_optimizer,
        )
        return True

    def save_last(
        self,
        *,
        epoch: int,
        model,
        edge_operator=None,
        edge_operator_type=None,
        edge_operator_aux=None,
        edge_operator_aux_type=None,
        model_optimizer=None,
        denoise_optimizer=None,
        finished: bool = True,
    ) -> str | None:
        metrics = dict(self.best_metrics)
        metrics["last_epoch"] = int(epoch)
        metrics["finished"] = finished
        return self.save(
            "last.pt",
            epoch=epoch,
            tag="last",
            model=model,
            metrics=metrics,
            edge_operator=edge_operator,
            edge_operator_type=edge_operator_type,
            edge_operator_aux=edge_operator_aux,
            edge_operator_aux_type=edge_operator_aux_type,
            model_optimizer=model_optimizer,
            denoise_optimizer=denoise_optimizer,
        )

    def maybe_save_epoch(
        self,
        *,
        epoch: int,
        model,
        edge_operator=None,
        edge_operator_type=None,
        edge_operator_aux=None,
        edge_operator_aux_type=None,
        model_optimizer=None,
        denoise_optimizer=None,
    ) -> None:
        if not self.enabled or not self.save_every_epoch:
            return
        self.save(
            f"epoch-{epoch:03d}.pt",
            epoch=epoch,
            tag=f"epoch-{epoch:03d}",
            model=model,
            metrics={
                "best_epoch": self.best_epoch,
                "best_val_acc_plus_mcc": self.best_val_acc_plus_mcc,
            },
            edge_operator=edge_operator,
            edge_operator_type=edge_operator_type,
            edge_operator_aux=edge_operator_aux,
            edge_operator_aux_type=edge_operator_aux_type,
            model_optimizer=model_optimizer,
            denoise_optimizer=denoise_optimizer,
        )

    def write_run_manifest(self) -> None:
        if not self.enabled:
            return
        manifest = {
            "profile_key": self.profile.key,
            "mode": self.args.mode,
            "seed": int(getattr(self.args, "seed", 42)),
            "ckpt_dir": os.path.relpath(
                self.ckpt_dir, os.environ.get("DUAL_TF_MODEL_DIR", os.getcwd())
            ),
            "args": namespace_to_dict(self.args),
            "best_metrics": self.best_metrics,
            "updated_at": datetime.now().isoformat(),
        }
        path = os.path.join(self.ckpt_dir, "run_manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)


def load_checkpoint(path: str, map_location="cpu") -> dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    if not isinstance(ckpt, dict):
        raise ValueError(f"invalid checkpoint format: {path}")
    return ckpt


def _filter_compatible_state(
    module, state_dict: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], list[str]]:
    mod = module.module if hasattr(module, "module") else module
    current = mod.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in state_dict.items():
        if key not in current:
            continue
        if current[key].shape != value.shape:
            skipped.append(key)
            continue
        filtered[key] = value
    return (filtered, skipped)


def _load_module_state(module, state_dict: dict[str, torch.Tensor] | None, name: str) -> None:
    if module is None or not state_dict:
        return
    mod = module.module if hasattr(module, "module") else module
    try:
        mod.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        (filtered, skipped) = _filter_compatible_state(module, state_dict)
        incompatible = mod.load_state_dict(filtered, strict=False)
        print(
            f"[CKPT] loaded {name} (shape-filtered={len(skipped)}, missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)})"
        )
        if skipped:
            print(
                f"[CKPT] skipped shape-mismatch keys: {', '.join(skipped[:8])}"
                + (" ..." if len(skipped) > 8 else "")
            )
        return
    print(f"[CKPT] loaded {name}")


def apply_checkpoint_to_modules(
    ckpt: dict[str, Any],
    *,
    model,
    edge_operator=None,
    edge_operator_type=None,
    edge_operator_aux=None,
    edge_operator_aux_type=None,
    model_optimizer=None,
    denoise_optimizer=None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    model_sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model")
    _load_module_state(model, model_sd, "model")
    _load_module_state(edge_operator, ckpt.get("edge_operator_state_dict"), "edge_operator")
    _load_module_state(
        edge_operator_type, ckpt.get("edge_operator_type_state_dict"), "edge_operator_type"
    )
    _load_module_state(
        edge_operator_aux, ckpt.get("edge_operator_aux_state_dict"), "edge_operator_aux"
    )
    _load_module_state(
        edge_operator_aux_type,
        ckpt.get("edge_operator_aux_type_state_dict"),
        "edge_operator_aux_type",
    )
    if model_optimizer is not None and ckpt.get("model_optimizer_state_dict"):
        model_optimizer.load_state_dict(ckpt["model_optimizer_state_dict"])
        print("[CKPT] loaded model optimizer")
    if denoise_optimizer is not None and ckpt.get("denoise_optimizer_state_dict"):
        denoise_optimizer.load_state_dict(ckpt["denoise_optimizer_state_dict"])
        print("[CKPT] loaded denoise optimizer")
    if restore_rng:
        restore_rng_states(ckpt.get("rng_states"))
    metrics = ckpt.get("metrics") or {}
    info = {
        "epoch": int(ckpt.get("epoch", 0)),
        "best_epoch": int(metrics.get("best_epoch", ckpt.get("epoch", 0))),
        "best_val_acc_plus_mcc": float(metrics.get("best_val_acc_plus_mcc", float("-inf"))),
        "metrics": metrics,
        "profile_key": ckpt.get("profile_key"),
        "mode": ckpt.get("mode"),
        "seed": ckpt.get("seed"),
        "tag": ckpt.get("tag"),
    }
    return info


def maybe_resume_from_args(
    args,
    profile,
    *,
    model,
    edge_operator=None,
    edge_operator_type=None,
    edge_operator_aux=None,
    edge_operator_aux_type=None,
    model_optimizer=None,
    denoise_optimizer=None,
    ckpt_manager: CheckpointManager | None = None,
) -> tuple[int, dict[str, Any]]:
    """Load --resume_ckpt into the model and optimizer. Returns (start_epoch, info)."""
    resume_path = str(getattr(args, "resume_ckpt", "") or "").strip()
    if not resume_path:
        return (1, {})
    resume_path = os.path.abspath(resume_path)
    ckpt = load_checkpoint(resume_path, map_location="cpu")
    if ckpt.get("profile_key") and ckpt["profile_key"] != profile.key:
        print(
            f"[CKPT][WARN] checkpoint profile={ckpt.get('profile_key')} != current {profile.key}; continue anyway."
        )
    weights_only = bool(getattr(args, "resume_weights_only", False))
    info = apply_checkpoint_to_modules(
        ckpt,
        model=model,
        edge_operator=edge_operator,
        edge_operator_type=edge_operator_type,
        edge_operator_aux=edge_operator_aux,
        edge_operator_aux_type=edge_operator_aux_type,
        model_optimizer=None if weights_only else model_optimizer,
        denoise_optimizer=None if weights_only else denoise_optimizer,
        restore_rng=not weights_only,
    )
    if weights_only:
        start_epoch = 1
        if ckpt_manager is not None:
            ckpt_manager.best_epoch = 0
            ckpt_manager.best_val_acc_plus_mcc = float("-inf")
            ckpt_manager.best_val_loss = float("inf")
            ckpt_manager.best_metrics = {}
        print(
            f"[CKPT] resume_weights_only from {resume_path} (model only, start_epoch=1, fresh optim/ES)",
            flush=True,
        )
        return (start_epoch, info)
    start_epoch = int(info["epoch"]) + 1
    if ckpt_manager is not None:
        ckpt_manager.best_epoch = int(info.get("best_epoch") or info["epoch"])
        ckpt_manager.best_val_acc_plus_mcc = float(info.get("best_val_acc_plus_mcc", float("-inf")))
        ckpt_manager.best_metrics = dict(info.get("metrics") or {})
    print(f"[CKPT] resume from {resume_path} -> start_epoch={start_epoch}")
    return (start_epoch, info)


def load_model_state_for_eval(model, ckpt_path: str, device) -> None:
    """Load model weights from best.pt, last.pt, or an older checkpoint for evaluation."""
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model_sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model")
    if model_sd is None:
        model.load_state_dict(ckpt, strict=True)
    else:
        _load_module_state(model, model_sd, "model")
