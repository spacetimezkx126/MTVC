"""Train MTVC using the single public implementation."""

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/mtvc"))
from reproduction import configure_environment

configure_environment()
from main import main

if __name__ == "__main__":
    main()
