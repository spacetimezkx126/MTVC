#!/usr/bin/env python3
"""
Convert news CSV folders to StockNet/PEN JSONL (one line per row: text list, created_at, user_id_str).

Tokenization:
  en — CJK one char per token; Latin runs as words (A-Za-z0-9_@$'+); punctuation separate. (CMIN-US)
  cn — CJK one char per token; Latin letters and digits each as own token (夹杂英文/数字拆开).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

# en: Chinese one char; English/number runs; punctuation.
TOKEN_RE_EN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z0-9_@$']+|[^\w\s]",
    re.UNICODE,
)

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_ASCII_WORDCHAR_RE = re.compile(r"[A-Za-z0-9_@$']")


def tokenize_en(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE_EN.findall(text or "")]


def tokenize_cn(text: str) -> list[str]:
    """CJK 单字；夹杂的 ASCII 字母、数字、_@$' 各单独成 token（字母转小写）。"""
    text = unicodedata.normalize("NFKC", text or "")
    out: list[str] = []
    for ch in text:
        if ch.isspace():
            continue
        if _CJK_RE.fullmatch(ch):
            out.append(ch)
        elif _ASCII_WORDCHAR_RE.fullmatch(ch):
            out.append(ch.lower() if ch.isalpha() else ch)
        else:
            out.append(ch)
    return out


def infer_tokenize_mode(src_root: Path) -> str:
    """CMIN-CN / CSMD* 用 cn（字母数字拆开）；CMIN-US 等用 en。"""
    parts = src_root.parts
    if "CMIN-CN" in parts:
        return "cn"
    if any(p.startswith("CSMD") for p in parts):
        return "cn"
    return "en"


def convert_one_csv(src_csv: Path, dst_txt: Path, *, tokenize_mode: str) -> int:
    tok = tokenize_en if tokenize_mode == "en" else tokenize_cn
    n = 0
    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    with src_csv.open("r", encoding="utf-8", errors="replace", newline="") as f_in, dst_txt.open(
        "w", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        for idx, row in enumerate(reader):
            created_at = (row.get("created_at") or "").strip()
            text = row.get("text") or ""
            item = {
                "text": tok(text),
                "created_at": created_at,
                "user_id_str": str(idx),
            }
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert news csv files to stocknet preprocessed text format (JSONL)."
    )
    parser.add_argument("--src", required=True, help="source folder, e.g. dataset/CMIN-US/news")
    parser.add_argument("--dst", required=True, help="target folder, e.g. dataset/CMIN-US/news_txt")
    parser.add_argument(
        "--tokenize",
        choices=("en", "cn", "auto"),
        default="auto",
        help="en: Latin/number runs as words (CMIN-US). cn: each ASCII letter/digit separate. "
        "auto: cn if --src path contains folder CMIN-CN or CSMD*, else en.",
    )
    args = parser.parse_args()

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()
    if not src_root.is_dir():
        raise SystemExit(f"source dir not found: {src_root}")

    mode = args.tokenize
    if mode == "auto":
        mode = infer_tokenize_mode(src_root)

    n_files = 0
    n_rows = 0
    for src_csv in sorted(src_root.rglob("*.csv")):
        rel = src_csv.relative_to(src_root)
        dst_txt = (dst_root / rel).with_suffix("")
        n_rows += convert_one_csv(src_csv, dst_txt, tokenize_mode=mode)
        n_files += 1

    print(f"done. tokenize={mode} (flag={args.tokenize}), files={n_files}, rows={n_rows}, dst={dst_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
