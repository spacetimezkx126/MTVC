#!/usr/bin/env python3
"""
Convert /home/zhaokx/Pattern/massive_stocktable/data
  -> /home/zhaokx/Pattern/Pattern_Mining/dataset/massive_data

Layout mirrors CMIN-US so dual_tf can load via profile key ``massive``:
  price/raw/{SYM}.csv
  price/preprocessed/{SYM}.txt   # date,label,open_rel,high_rel,low_rel,close_rel,volume
  news/{SYM}/{YYYY-MM-DD}.csv    # code_name,ticker,created_at,text,sentiment,...
  llm_extract/news_type/{SYM}/{YYYY-MM-DD}.json  # id -> sentiment|individual
  llm_extract/news_meta/{SYM}/{YYYY-MM-DD}.json  # full article meta (tickers/url/insights)
  comp_industry/comp_indus.csv
  company_name.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path

SRC = Path("/home/zhaokx/Pattern/massive_stocktable/data")
TICKERS_CSV = Path("/home/zhaokx/Pattern/massive_stocktable/tickers.csv")
DST = Path("/home/zhaokx/Pattern/Pattern_Mining/dataset/massive_data")

# StockNet appendix 9 sectors (tickers.csv uses typo "Basic Matierials")
SECTOR_TO_INDUSTRY = {
    "Basic Materials": ("基础材料", "Basic Materials"),
    "Basic Matierials": ("基础材料", "Basic Materials"),
    "Consumer Goods": ("消费品", "Consumer Goods"),
    "Healthcare": ("医疗健康", "Healthcare"),
    "Services": ("服务业", "Services"),
    "Utilities": ("公用事业", "Utilities"),
    "Conglomerates": ("综合企业", "Conglomerates"),
    "Financial": ("金融", "Financial"),
    "Industrial Goods": ("工业品", "Industrial Goods"),
    "Technology": ("科技", "Technology"),
}

# Keep StockNet 9 sectors as-is (no finer overrides).
TICKER_INDUSTRY_OVERRIDE: dict[str, tuple[str, str]] = {}


def _safe_float(x):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def normalize_sentiment(s: str | None) -> str:
    if not s:
        return ""
    t = str(s).strip().lower()
    aliases = {
        "bullish": "positive",
        "bearish": "negative",
        "pos": "positive",
        "neg": "negative",
        "neutral/positive": "positive",
        "neutral/negative": "negative",
        "na": "",
        "n/a": "",
        "cautious": "neutral",
    }
    return aliases.get(t, t)


def load_ticker_meta() -> dict[str, dict]:
    meta = {}
    with TICKERS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sym = row["symbol_raw"].strip()
            meta[sym] = row
    return meta


def industry_for(sym: str, sector: str) -> tuple[str, str]:
    if sym in TICKER_INDUSTRY_OVERRIDE:
        return TICKER_INDUSTRY_OVERRIDE[sym]
    return SECTOR_TO_INDUSTRY.get(sector, ("可选消费", "Consumer Discretionary"))


def write_raw_and_preprocessed(sym: str, prices_csv: Path, raw_dir: Path, prep_dir: Path) -> int:
    rows = []
    with prices_csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = (r.get("date") or "").strip()
            o = _safe_float(r.get("open"))
            h = _safe_float(r.get("high"))
            l = _safe_float(r.get("low"))
            c = _safe_float(r.get("close"))
            v = _safe_float(r.get("volume")) or 0.0
            if not d or None in (o, h, l, c):
                continue
            rows.append((d, o, h, l, c, v))
    rows.sort(key=lambda x: x[0])
    if len(rows) < 2:
        return 0

    raw_path = raw_dir / f"{sym}.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])
        for d, o, h, l, c, v in rows:
            # Massive daily bars: use Close as Adj Close (already adjusted in crawl)
            w.writerow([d, o, h, l, c, c, int(v) if float(v).is_integer() else v])

    # CMIN-US formula: field-wise vs previous day; label = close_t/close_{t-1}-1
    prep_lines = []
    for i in range(1, len(rows)):
        d, o, h, l, c, v = rows[i]
        po, ph, pl, pc = rows[i - 1][1], rows[i - 1][2], rows[i - 1][3], rows[i - 1][4]
        if min(po, ph, pl, pc) <= 0:
            continue
        label = c / pc - 1.0
        open_rel = o / po - 1.0
        high_rel = h / ph - 1.0
        low_rel = l / pl - 1.0
        close_rel = c / pc - 1.0
        prep_lines.append(
            f"{d}\t{label}\t{open_rel}\t{high_rel}\t{low_rel}\t{close_rel}\t{v}"
        )
    (prep_dir / f"{sym}.txt").write_text("\n".join(prep_lines) + ("\n" if prep_lines else ""), encoding="utf-8")
    # reversed (optional convenience mirror of CMIN-US)
    rev = list(reversed(prep_lines))
    rev_dir = prep_dir.parent / "reversed"
    rev_dir.mkdir(parents=True, exist_ok=True)
    (rev_dir / f"{sym}.txt").write_text("\n".join(rev) + ("\n" if rev else ""), encoding="utf-8")
    return len(prep_lines)


def convert_news(sym: str, news_dir: Path, out_news: Path, out_type: Path, out_meta: Path) -> tuple[int, int, Counter]:
    out_news.mkdir(parents=True, exist_ok=True)
    out_type.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)
    n_days = 0
    n_arts = 0
    sent_c = Counter()
    if not news_dir.is_dir():
        return 0, 0, sent_c

    for jp in sorted(news_dir.glob("*.json")):
        day = jp.stem
        if day == "unknown" or len(day) < 10:
            continue
        try:
            payload = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        articles = payload.get("articles") or []
        if not isinstance(articles, list) or not articles:
            continue

        csv_rows = []
        type_map = {}
        meta_map = {}
        for i, a in enumerate(articles):
            if not isinstance(a, dict):
                continue
            title = (a.get("title") or "").strip()
            desc = (a.get("description") or "").strip()
            # 与 CMIN-US 打标一致：新闻文本只用 title
            text = title
            if not text:
                continue
            pub = a.get("published_utc") or f"{day}T00:00:00Z"
            sent = normalize_sentiment(a.get("sentiment"))
            if sent:
                sent_c[sent] += 1
            else:
                sent_c["(none)"] += 1
            # news_type for model: sentiment label if present, else individual
            ntype = sent if sent else "individual"
            nid = str(i)
            type_map[nid] = ntype
            meta_map[nid] = {
                "id": a.get("id"),
                "published_utc": pub,
                "title": title,
                "description": desc,
                "author": a.get("author"),
                "source": a.get("source"),
                "article_url": a.get("article_url"),
                "tickers": a.get("tickers") or [],
                "sentiment": sent or None,
                "sentiment_reasoning": a.get("sentiment_reasoning"),
                "insights": a.get("insights") or [],
                "keywords": a.get("keywords") or [],
            }
            csv_rows.append(
                {
                    "code_name": sym,
                    "ticker": sym,
                    "created_at": pub.replace("T", " ").replace("Z", ""),
                    "text": text,
                    "sentiment": sent,
                    "sentiment_reasoning": (a.get("sentiment_reasoning") or "").replace("\n", " "),
                    "tickers": "|".join(a.get("tickers") or []),
                    "article_url": a.get("article_url") or "",
                    "source": a.get("source") or "",
                }
            )
            n_arts += 1

        if not csv_rows:
            continue
        n_days += 1
        with (out_news / f"{day}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "code_name",
                    "ticker",
                    "created_at",
                    "text",
                    "sentiment",
                    "sentiment_reasoning",
                    "tickers",
                    "article_url",
                    "source",
                ],
            )
            w.writeheader()
            w.writerows(csv_rows)
        with (out_type / f"{day}.json").open("w", encoding="utf-8") as f:
            json.dump(type_map, f, ensure_ascii=False, indent=2)
        with (out_meta / f"{day}.json").open("w", encoding="utf-8") as f:
            json.dump(meta_map, f, ensure_ascii=False, indent=2)
    return n_days, n_arts, sent_c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--dst", type=Path, default=DST)
    ap.add_argument("--fresh", action="store_true", help="Delete dst before write")
    args = ap.parse_args()

    src: Path = args.src
    dst: Path = args.dst
    if args.fresh and dst.exists():
        shutil.rmtree(dst)

    raw_dir = dst / "price" / "raw"
    prep_dir = dst / "price" / "preprocessed"
    news_root = dst / "news"
    type_root = dst / "llm_extract" / "news_type"
    meta_root = dst / "llm_extract" / "news_meta"
    ind_dir = dst / "comp_industry"
    for p in (raw_dir, prep_dir, news_root, type_root, meta_root, ind_dir):
        p.mkdir(parents=True, exist_ok=True)

    tickers = load_ticker_meta()
    company_rows = []
    ind_rows = [("stock_name", "行业中文", "行业英文")]
    stats = []
    sent_all = Counter()

    syms = sorted([d.name for d in src.iterdir() if d.is_dir()])
    print(f"[convert] src={src} dst={dst} n_sym={len(syms)}")

    for sym in syms:
        sdir = src / sym
        prices = sdir / "prices.csv"
        if not prices.exists():
            print(f"  SKIP {sym}: no prices.csv")
            continue
        meta = tickers.get(sym, {})
        sector = meta.get("sector", "")
        company = meta.get("company", sym)
        cn, en = industry_for(sym, sector)

        n_bars = write_raw_and_preprocessed(sym, prices, raw_dir, prep_dir)
        n_days, n_arts, sc = convert_news(
            sym, sdir / "news", news_root / sym, type_root / sym, meta_root / sym
        )
        sent_all.update(sc)
        company_rows.append({"file_name": sym, "company_name": company})
        ind_rows.append((sym, cn, en))
        stats.append((sym, n_bars, n_days, n_arts))
        print(f"  {sym}: price_bars={n_bars} news_days={n_days} news_arts={n_arts}")

    with (dst / "company_name.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "company_name"], delimiter="\t")
        w.writeheader()
        w.writerows(company_rows)

    with (ind_dir / "comp_indus.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(ind_rows)

    # code_name.json: ticker -> ticker (CMIN-US style optional)
    code_name = {r["file_name"]: r["file_name"] for r in company_rows}
    (dst / "code_name.json").write_text(json.dumps(code_name, indent=2), encoding="utf-8")

    summary = {
        "n_companies": len(stats),
        "n_price_bars": sum(s[1] for s in stats),
        "n_news_days": sum(s[2] for s in stats),
        "n_news_articles": sum(s[3] for s in stats),
        "sentiment_counts": dict(sent_all),
        "src": str(src),
        "dst": str(dst),
    }
    (dst / "convert_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[convert] DONE", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
