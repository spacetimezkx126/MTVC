"""Verify the content of each exported dataset against its recorded checksums."""

from pathlib import Path
import argparse
import hashlib
import json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("root", type=Path, help="Dataset folder containing .mtvc_file_order.json.")
a = p.parse_args()
m = json.loads((a.root / ".mtvc_file_order.json").read_text())
for name, v in m["files"].items():
    path = a.root / name
    if not path.is_file():
        raise SystemExit("Missing file: " + name)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    if path.stat().st_size != v["bytes"] or h.hexdigest() != v["sha256"]:
        raise SystemExit("Mismatch: " + name)
print("Verified", len(m["files"]), "dataset files")
