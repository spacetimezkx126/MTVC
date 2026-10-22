"""Cross-process GPU lock and CUDA error recovery for several jobs on one card."""

from __future__ import annotations
import contextlib
import fcntl
import os
import time
from typing import Iterator
import torch


def _lock_enabled() -> bool:
    """On unless DUAL_TF_CUDA_LOCK is 0/false/off/no."""
    return str(os.environ.get("DUAL_TF_CUDA_LOCK", "1")).strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def device_index(device: torch.device | str | int | None) -> int:
    """CUDA index from a device, or 0."""
    if device is None:
        return 0
    if isinstance(device, int):
        return int(device)
    if isinstance(device, str):
        if device.startswith("cuda:"):
            return int(device.split(":")[-1])
        return 0
    if getattr(device, "type", None) == "cuda":
        idx = getattr(device, "index", None)
        return int(idx) if idx is not None else int(torch.cuda.current_device())
    return 0


def _physical_cuda_index(device: torch.device | str | int | None = None) -> int:
    """Physical GPU id, honoring CUDA_VISIBLE_DEVICES."""
    "Map logical cuda:N → physical GPU id.\n\n    Training scripts use CUDA_VISIBLE_DEVICES=<phys> with --device cuda:0, so every\n    process would otherwise lock the same /tmp/dual_tf_cuda0.lock and serialize\n    across all GPUs. Prefer the first id in CUDA_VISIBLE_DEVICES when set.\n    "
    vis = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if vis and vis.lower() not in ("", "none"):
        first = vis.split(",")[0].strip()
        if first.isdigit():
            return int(first)
    return device_index(device)


@contextlib.contextmanager
def cuda_device_lock(device: torch.device | str | int | None = None) -> Iterator[None]:
    """Serialize forward and backward across processes that share one physical GPU."""
    if not _lock_enabled():
        yield
        return
    if isinstance(device, torch.device) and device.type != "cuda":
        yield
        return
    if isinstance(device, str) and (not device.startswith("cuda")):
        yield
        return
    idx = _physical_cuda_index(device)
    path = f"/tmp/dual_tf_cuda{idx}.lock"
    fd = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        finally:
            fd.close()


def cuda_sync_and_clear(device: torch.device | None = None) -> None:
    """Synchronize the device and drop the caching allocator."""
    if not torch.cuda.is_available():
        return
    try:
        if device is not None and getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)
        else:
            torch.cuda.synchronize()
    except RuntimeError:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def is_cuda_fatal(err: BaseException) -> bool:
    """True for illegal access, device asserts, and similar driver errors."""
    msg = str(err).lower()
    keys = (
        "invalid program counter",
        "illegal memory access",
        "cuda error",
        "cublas",
        "cudnn",
        "device-side assert",
    )
    return any((k in msg for k in keys))


def forward_with_cuda_retry(fn, *, device: torch.device, max_retries: int = 2, tag: str = ""):
    """Run ``fn``. On an async CUDA error or OOM, clear the cache and retry."""
    try:
        return fn()
    except RuntimeError as e:
        msg = str(e).lower()
        is_oom = "out of memory" in msg or "cuda out of memory" in msg
        if not (is_cuda_fatal(e) or is_oom) or max_retries <= 0:
            if is_cuda_fatal(e) and max_retries <= 0:
                abort_on_unrecoverable_cuda(e, tag=tag)
            raise
        prefix = f"[CUDA_RETRY]{(' ' + tag if tag else '')}"
        print(f"{prefix} caught: {e}", flush=True)
        cuda_sync_and_clear(device)
        time.sleep(0.5)
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass
        return forward_with_cuda_retry(fn, device=device, max_retries=max_retries - 1, tag=tag)


def abort_on_unrecoverable_cuda(err: BaseException, *, tag: str = "") -> None:
    """Fatal CUDA state is process-local but can confuse the driver; exit fast."""
    if is_cuda_fatal(err):
        prefix = f"[CUDA_FATAL]{(' ' + tag if tag else '')}"
        print(f"{prefix} unrecoverable: {err}", flush=True)
        os._exit(1)


def harden_cuda_for_multiproc() -> None:
    """Best-effort: reduce flaky multi-process CUDA on the same physical GPU."""
    if not torch.cuda.is_available():
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
