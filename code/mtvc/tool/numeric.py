"""Numeric helpers for attention / logits (``tool.numeric``).

Role
----
- ``sanitize_*`` / ``stable_attn_softmax`` / ``sanitize_logits_for_bce``:
  used on every forward / BCE step. Finite inputs are returned unchanged.
"""

from __future__ import annotations
import torch


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
    """Clamp a keep-score to [0, 1] and replace non-finite values with 1."""
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
