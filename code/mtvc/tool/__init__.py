"""Training / numeric helpers used across MTVC.

Submodules
----------
- ``tool.numeric``: sanitize / stable softmax (soft-prune stubs always off).
- ``tool.cuda_guard``: GPU file lock + forward retry.
- ``tool.oversample_utils``: Massive train oversample weights.

Import submodules directly, e.g. ``from tool.numeric import sanitize_logits_for_bce``.
"""
from __future__ import annotations
