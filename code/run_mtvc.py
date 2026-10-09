#!/usr/bin/env python3
"""Package entrypoint: set PYTHONPATH and launch ``mtvc/main.py``.

Usage (from anywhere)::

    python MTVC_paper_repro/code/run_mtvc.py --dataset csmd50 --mode train ...

Does not contain model logic; only path setup + ``runpy``.
"""
from __future__ import annotations

import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DUAL_TF = os.path.abspath(os.path.join(_HERE, "..", ".."))
_MTVC = os.path.join(_HERE, "mtvc")

os.environ["DUAL_TF_MODEL_DIR"] = _DUAL_TF
os.chdir(_DUAL_TF)

if _DUAL_TF not in sys.path:
    sys.path.append(_DUAL_TF)
if _MTVC not in sys.path:
    sys.path.insert(0, _MTVC)

runpy.run_path(os.path.join(_MTVC, "main.py"), run_name="__main__")
