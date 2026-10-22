"""Verify the portable checkpoint archive against its SHA256 index."""

import argparse, hashlib, json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("root", type=Path)
a = p.parse_args()
index = json.loads((a.root / "index.json").read_text())
for entry in index["files"] + index.get("metadata_files", []):
    f = a.root / entry["file"]
    h = hashlib.sha256()
    with f.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    if f.stat().st_size != entry["bytes"] or h.hexdigest() != entry["sha256"]:
        raise SystemExit("Mismatch: " + entry["file"])
print("Verified", len(index["files"]), "checkpoints")
