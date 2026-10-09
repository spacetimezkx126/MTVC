#!/usr/bin/env python3
"""Package entrypoint: set PYTHONPATH and launch ``mtvc/main.py``.

Usage (from anywhere)::

    python code/run_mtvc.py --dataset csmd50 --mode train ...

Does not contain model logic; only path setup + ``runpy``.
"""
from __future__ import annotations

import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../MTVC/code
_MTVC_ROOT = os.path.abspath(os.path.join(_HERE, ".."))     # .../MTVC
_MTVC_PKG = os.path.join(_HERE, "mtvc")

# Package root for data/dict resolution; override with DUAL_TF_MODEL_DIR if needed.
os.environ.setdefault("DUAL_TF_MODEL_DIR", _MTVC_ROOT)
os.chdir(_MTVC_ROOT)

if _MTVC_ROOT not in sys.path:
    sys.path.append(_MTVC_ROOT)
if _MTVC_PKG not in sys.path:
    sys.path.insert(0, _MTVC_PKG)

runpy.run_path(os.path.join(_MTVC_PKG, "main.py"), run_name="__main__")