#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Numeric helpers for attention / logits (``tool.numeric``).

Role
----
- ``sanitize_*`` / ``stable_attn_softmax`` / ``sanitize_logits_for_bce``:
  used on every forward / BCE step.
- Soft-token-prune APIs are **stubs** (always off). Paper runs set
  ``MTVC_SOFT_PRUNE=0``; enabling prune is not supported in this package.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def sanitize_pad_mask(pad_mask: torch.Tensor | None) -> torch.Tensor | None:
    """Ensure no row is all-True (avoids softmax(-inf) → nan)."""
    if pad_mask is None or pad_mask.numel() == 0:
        return pad_mask
    all_pad = pad_mask.all(dim=-1)
    if not bool(all_pad.any()):
        return pad_mask
    out = pad_mask.clone()
    out[all_pad, 0] = False
    return out


def sanitize_hidden(h: torch.Tensor) -> torch.Tensor:
    """Replace non-finite hidden states."""
    if torch.isfinite(h).all():
        return h
    return torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)


def sanitize_keep(keep: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(keep, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def stable_attn_softmax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax that stays finite when a row is all -inf / nan."""
    finite = torch.isfinite(scores)
    scores_safe = torch.where(finite, scores, torch.full_like(scores, float("-inf")))
    has_valid = finite.any(dim=dim, keepdim=True)
    if not bool(has_valid.all()):
        filler = torch.full_like(scores_safe, float("-inf"))
        idx = [slice(None)] * scores_safe.dim()
        idx[dim] = 0
        filler[tuple(idx)] = 0.0
        scores_safe = torch.where(has_valid, scores_safe, filler)
    attn = torch.softmax(scores_safe, dim=dim)
    return torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)


def sanitize_logits_for_bce(logits: torch.Tensor, lim: float = 40.0) -> torch.Tensor:
    """Keep BCEWithLogits finite."""
    lim = float(lim)
    return torch.nan_to_num(logits, nan=0.0, posinf=lim, neginf=-lim).clamp(-lim, lim)


# ---------------------------------------------------------------------------
# Soft-prune stubs (always off) — keep call sites in encoder/train importable
# ---------------------------------------------------------------------------

_PRUNE_EPOCH = 0


def set_prune_epoch(epoch: int) -> None:
    global _PRUNE_EPOCH
    _PRUNE_EPOCH = int(epoch)


def get_prune_epoch() -> int:
    return int(_PRUNE_EPOCH)


def prune_warmup_epochs() -> int:
    return 0


def prune_in_warmup() -> bool:
    return False


def prune_site_enabled(site: str) -> bool:
    return False


def prune_identity_enabled() -> bool:
    return True


def prune_detach_keep_enabled() -> bool:
    return False


def prune_kw_once_enabled() -> bool:
    return False


def hard_finetune_enabled() -> bool:
    return False


def freeze_prune_thresholds(model: nn.Module) -> int:
    return 0


def is_delta_prune_mode(mode: str | None = None) -> bool:
    return False


def is_topk_prune_mode(mode: str | None = None) -> bool:
    return False


def is_attn_decay_thr_mode(mode: str | None = None) -> bool:
    return False


def is_attn_decay_rule_mode(mode: str | None = None) -> bool:
    return False


def is_attn_decay_prune_mode(mode: str | None = None) -> bool:
    return False


def decay_residual_enabled() -> bool:
    return False


def decay_halt_pairs(num_layers: int) -> list[tuple[int, int]]:
    return []


def prune_front_layers(module: str | None, num_layers: int) -> int:
    return int(num_layers)


def prune_aux_loss(*args, **kwargs):
    return None


def soft_keep_rate(*args, **kwargs):
    return None


def delta_attn_importance(*args, **kwargs):
    raise RuntimeError("token prune is disabled in the MTVC paper package")


def residual_halt_mix(*args, **kwargs):
    raise RuntimeError("token prune is disabled in the MTVC paper package")


class SoftGateMLP(nn.Module):
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "SoftGateMLP unavailable: MTVC_SOFT_PRUNE is not supported in MTVC_paper_repro"
        )


class SoftThresholdGate(nn.Module):
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "SoftThresholdGate unavailable: MTVC_SOFT_PRUNE is not supported in MTVC_paper_repro"
        )
