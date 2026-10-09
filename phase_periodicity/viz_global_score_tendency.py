#!/usr/bin/env python3
"""Visualize 10-day periodic global VIN scalars after reduce_dim (Linear D→1).

Style matches the reference "tendency" plot: blue scatter + red cubic-spline line.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import CubicSpline


def _load_state(ckpt: Path) -> dict:
    try:
        obj = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(str(ckpt), map_location="cpu")
    if isinstance(obj, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        # plain state_dict
        if any(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise RuntimeError(f"unrecognized checkpoint format: {ckpt}")


def _find_tensor(sd: dict, *suffixes: str) -> torch.Tensor:
    for key, val in sd.items():
        if not isinstance(val, torch.Tensor):
            continue
        for s in suffixes:
            if key == s or key.endswith("." + s) or key.endswith(s):
                return val
    raise KeyError(f"missing keys matching {suffixes} in ckpt ({len(sd)} keys)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="best.pt / last.pt")
    ap.add_argument("--out", type=str, default="", help="output png path")
    ap.add_argument("--title", type=str, default="tendency")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    sd = _load_state(ckpt)
    gp = _find_tensor(sd, "global_parameter")  # [10, 1, D]
    w = _find_tensor(sd, "reduce_dim.weight")  # [1, D]
    b = _find_tensor(sd, "reduce_dim.bias")  # [1]

    # Pure VIN phase vectors → 1-d scores (no day_mean modulation)
    # gp[d, 0, :] @ W^T + b
    vecs = gp[:, 0, :].float()  # [10, D]
    scores = (vecs @ w.float().T).squeeze(-1) + b.float().view(-1)
    y = scores.detach().cpu().numpy().astype(np.float64)
    x = np.arange(len(y), dtype=np.float64)

    # smooth red trendency line (cubic spline on denser grid)
    xs = np.linspace(float(x.min()), float(x.max()), 400)
    if len(y) >= 4:
        cs = CubicSpline(x, y)
        ys = cs(xs)
    else:
        ys = np.interp(xs, x, y)

    out = Path(args.out) if args.out else (ckpt.parent / "global_score_tendency.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=140)
    ax.scatter(x, y, s=55, c="#4C78A8", alpha=0.75, label="data", zorder=3)
    ax.plot(xs, ys, color="#E45756", linewidth=2.4, label="trendency line", zorder=2)
    ax.set_title(args.title)
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.set_xticks(list(range(0, 10, 2)))
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.35)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    csv_path = out.with_suffix(".csv")
    np.savetxt(csv_path, np.column_stack([x, y]), delimiter=",", header="time,value", comments="")
    print(f"[ok] wrote {out}")
    print(f"[ok] wrote {csv_path}")
    print("scores:", ", ".join(f"{v:.6f}" for v in y.tolist()))


if __name__ == "__main__":
    main()
