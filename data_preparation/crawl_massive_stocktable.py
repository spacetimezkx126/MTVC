#!/usr/bin/env python3
"""
Crawl Massive.com via RESTClient for StockTable companies.

  prices: daily OHLCV (incl. volume) -> prices.csv
  news:   one JSON per calendar day  -> news/YYYY-MM-DD.json

Layout under /home/zhaokx/Pattern/massive_stocktable/data/{SYMBOL}/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from massive import RESTClient

ROOT = Path("/home/zhaokx/Pattern/massive_stocktable")
DATA = ROOT / "data"
LOGS = ROOT / "logs"
TICKERS_CSV = ROOT / "tickers.csv"

DEFAULT_KEY = os.environ.get(
    "POLYGON_API_KEY",
    os.environ.get("MASSIVE_API_KEY", "ZstU7A0MG2m3wKDdM7oHemcbDJ3eSOpc"),
)
DEFAULT_SLEEP = 0.35


def log(msg: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with (LOGS / "crawl.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_tickers(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def published_to_date(published_utc: str | None) -> str | None:
    if not published_utc:
        return None
    s = str(published_utc).strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return None


def article_from_api(item: Any, query_ticker: str) -> dict[str, Any]:
    insights = item.insights or []
    insight_dicts = []
    for ins in insights:
        insight_dicts.append(
            {
                "ticker": getattr(ins, "ticker", None),
                "sentiment": getattr(ins, "sentiment", None),
                "sentiment_reasoning": getattr(ins, "sentiment_reasoning", None),
            }
        )
    sent = None
    sent_reason = None
    for d in insight_dicts:
        if str(d.get("ticker") or "").upper() == query_ticker.upper():
            sent = d.get("sentiment")
            sent_reason = d.get("sentiment_reasoning")
            break
    if sent is None and insight_dicts:
        sent = insight_dicts[0].get("sentiment")
        sent_reason = insight_dicts[0].get("sentiment_reasoning")

    pub = item.publisher
    source = getattr(pub, "name", None) if pub is not None else None
    tickers = list(item.tickers or [])
    return {
        "id": item.id,
        "published_utc": item.published_utc,
        "title": item.title,
        "description": item.description,
        "author": item.author,
        "source": source,
        "article_url": item.article_url,
        "tickers": tickers,
        "sentiment": sent,
        "sentiment_reasoning": sent_reason,
        "insights": insight_dicts,
        "keywords": list(item.keywords or []),
    }


def write_news_daily(news_dir: Path, query_ticker: str, articles: list[dict]) -> dict[str, Any]:
    """Group articles by UTC date and write news/YYYY-MM-DD.json."""
    if news_dir.exists():
        shutil.rmtree(news_dir)
    news_dir.mkdir(parents=True, exist_ok=True)

    by_day: dict[str, list[dict]] = defaultdict(list)
    for a in articles:
        day = published_to_date(a.get("published_utc"))
        if not day:
            day = "unknown"
        by_day[day].append(a)

    for day in sorted(by_day):
        day_arts = sorted(by_day[day], key=lambda x: x.get("published_utc") or "")
        # dedupe by id within day
        seen: dict[str, dict] = {}
        for a in day_arts:
            aid = a.get("id") or json.dumps(a, sort_keys=True, ensure_ascii=False)
            seen[aid] = a
        day_arts = list(seen.values())
        day_arts.sort(key=lambda x: x.get("published_utc") or "")
        payload = {
            "date": day,
            "query_ticker": query_ticker,
            "n": len(day_arts),
            "articles": day_arts,
        }
        with (news_dir / f"{day}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return {"n_news": len(articles), "n_days": len(by_day), "news_dir": str(news_dir)}


def news_done(news_dir: Path) -> bool:
    return news_dir.is_dir() and any(news_dir.glob("*.json"))


def fetch_prices(client: RESTClient, ticker: str, start: str, end: str, sleep_s: float) -> tuple[list[dict], dict]:
    meta: dict[str, Any] = {"ticker": ticker, "errors": []}
    rows: list[dict] = []
    try:
        for i, a in enumerate(
            client.list_aggs(
                ticker,
                1,
                "day",
                from_=start,
                to=end,
                adjusted=True,
                sort="asc",
                limit=50000,
            )
        ):
            ts = int(a.timestamp) / 1000.0
            date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            rows.append(
                {
                    "date": date,
                    "open": a.open,
                    "high": a.high,
                    "low": a.low,
                    "close": a.close,
                    "volume": a.volume,
                    "vwap": a.vwap,
                    "transactions": a.transactions,
                    "ticker_api": ticker,
                }
            )
            if sleep_s and i > 0 and i % 1000 == 0:
                time.sleep(sleep_s)
    except Exception as exc:
        meta["errors"].append(str(exc))
        log(f"  prices {ticker} ERROR {exc}")

    by_date = {r["date"]: r for r in rows}
    rows = [by_date[k] for k in sorted(by_date)]
    meta["n_bars"] = len(rows)
    if rows:
        meta["date_min"] = rows[0]["date"]
        meta["date_max"] = rows[-1]["date"]
    log(f"  prices {ticker} n={len(rows)}")
    return rows, meta


def fetch_news(client: RESTClient, ticker: str, start: str, end: str, sleep_s: float) -> tuple[list[dict], dict]:
    meta: dict[str, Any] = {"ticker": ticker, "errors": [], "n_iter": 0}
    rows: list[dict] = []
    try:
        for item in client.list_ticker_news(
            ticker=ticker,
            published_utc_gte=f"{start}T00:00:00Z",
            published_utc_lte=f"{end}T23:59:59Z",
            limit=1000,
            sort="published_utc",
            order="asc",
        ):
            meta["n_iter"] += 1
            rows.append(article_from_api(item, ticker))
            if sleep_s and meta["n_iter"] % 200 == 0:
                time.sleep(sleep_s)
                log(f"  news {ticker} progress={meta['n_iter']}")
    except Exception as exc:
        meta["errors"].append(str(exc))
        log(f"  news {ticker} ERROR {exc}")

    by_id = {r["id"]: r for r in rows if r.get("id")}
    rows = sorted(by_id.values(), key=lambda r: r.get("published_utc") or "")
    meta["n_news"] = len(rows)
    log(f"  news {ticker} n={len(rows)}")
    return rows, meta


def convert_news_csv_to_daily(sym_dir: Path) -> bool:
    """Convert legacy news.csv -> news/YYYY-MM-DD.json; return True if converted."""
    csv_path = sym_dir / "news.csv"
    if not csv_path.exists():
        return False
    articles: list[dict] = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tickers = []
            raw_t = row.get("tickers") or ""
            try:
                tickers = json.loads(raw_t) if raw_t.startswith("[") else [
                    t for t in (row.get("tickers_join") or "").split("|") if t
                ]
            except json.JSONDecodeError:
                tickers = [t for t in (row.get("tickers_join") or "").split("|") if t]
            insights = []
            raw_i = row.get("insights_json") or ""
            try:
                insights = json.loads(raw_i) if raw_i else []
            except json.JSONDecodeError:
                insights = []
            keywords = []
            raw_k = row.get("keywords") or ""
            try:
                keywords = json.loads(raw_k) if raw_k else []
            except json.JSONDecodeError:
                keywords = []
            articles.append(
                {
                    "id": row.get("id"),
                    "published_utc": row.get("published_utc"),
                    "title": row.get("title"),
                    "description": row.get("description"),
                    "author": row.get("author"),
                    "source": row.get("source"),
                    "article_url": row.get("article_url"),
                    "tickers": tickers,
                    "sentiment": row.get("sentiment") or None,
                    "sentiment_reasoning": row.get("sentiment_reasoning") or None,
                    "insights": insights,
                    "keywords": keywords,
                }
            )
    query = articles[0].get("query_ticker") if articles else sym_dir.name
    # query_ticker was in CSV; fall back
    if articles and "query_ticker" not in articles[0]:
        # already not in article dict; use folder / first CSV col
        pass
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    query_ticker = (rows[0].get("query_ticker") if rows else None) or sym_dir.name
    wmeta = write_news_daily(sym_dir / "news", query_ticker, articles)
    csv_path.unlink()
    log(f"  converted {sym_dir.name} news.csv -> news/ days={wmeta['n_days']} n={wmeta['n_news']}")
    return True


def crawl_one(
    client: RESTClient,
    row: dict[str, str],
    start: str,
    end: str,
    sleep_s: float,
    do_prices: bool,
    do_news: bool,
    force: bool,
) -> None:
    sym_raw = row["symbol_raw"]
    sym_api = row["symbol_api"]
    out_dir = DATA / sym_raw
    out_dir.mkdir(parents=True, exist_ok=True)
    prices_path = out_dir / "prices.csv"
    news_dir = out_dir / "news"
    meta_path = out_dir / "meta.json"

    # migrate legacy news.csv if present
    if (out_dir / "news.csv").exists() and (force or not news_done(news_dir)):
        convert_news_csv_to_daily(out_dir)

    meta: dict[str, Any] = {
        "sector": row.get("sector"),
        "company": row.get("company"),
        "symbol_raw": sym_raw,
        "symbol_api": sym_api,
        "mapped_from": row.get("mapped_from"),
        "range": [start, end],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "client": "massive.RESTClient",
    }
    log(f"== {sym_raw} (api={sym_api}) ==")

    used_price_sym = sym_api
    if do_prices and (force or not prices_path.exists() or prices_path.stat().st_size < 50):
        price_rows, pmeta = fetch_prices(client, sym_api, start, end, sleep_s)
        if not price_rows and sym_api != sym_raw:
            log(f"  retry prices raw {sym_raw}")
            price_rows, pmeta2 = fetch_prices(client, sym_raw, start, end, sleep_s)
            pmeta["fallback_raw"] = pmeta2
            if price_rows:
                used_price_sym = sym_raw
        write_csv(
            prices_path,
            price_rows,
            ["date", "open", "high", "low", "close", "volume", "vwap", "transactions", "ticker_api"],
        )
        meta["prices"] = pmeta
        meta["prices_used_symbol"] = used_price_sym
        time.sleep(sleep_s)
    else:
        meta["prices"] = {"skipped": True, "exists": prices_path.exists()}

    if do_news and (force or not news_done(news_dir)):
        news_rows, nmeta = fetch_news(client, sym_api, start, end, sleep_s)
        if not news_rows and sym_api != sym_raw:
            log(f"  retry news raw {sym_raw}")
            news_rows, nmeta2 = fetch_news(client, sym_raw, start, end, sleep_s)
            nmeta["fallback_raw"] = nmeta2
        wmeta = write_news_daily(news_dir, sym_api, news_rows)
        nmeta.update(wmeta)
        meta["news"] = nmeta
        # remove legacy csv if any
        legacy = out_dir / "news.csv"
        if legacy.exists():
            legacy.unlink()
        time.sleep(sleep_s)
    else:
        meta["news"] = {"skipped": True, "exists": news_done(news_dir)}

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", default=DEFAULT_KEY)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    ap.add_argument("--symbols", default="")
    ap.add_argument("--skip_prices", action="store_true")
    ap.add_argument("--skip_news", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--convert_only",
        action="store_true",
        help="Only convert existing news.csv -> news/*.json, no API calls",
    )
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    if args.convert_only:
        for p in sorted(DATA.iterdir()):
            if p.is_dir() and (p / "news.csv").exists():
                convert_news_csv_to_daily(p)
        log("convert_only DONE")
        return

    client = RESTClient(args.api_key)
    tickers = load_tickers(TICKERS_CSV)
    if args.symbols:
        want = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        tickers = [t for t in tickers if t["symbol_raw"].upper() in want]

    log(f"start crawl n={len(tickers)} range={args.start}..{args.end} sleep={args.sleep} key=...{args.api_key[-4:]}")
    for t in tickers:
        try:
            crawl_one(
                client,
                t,
                start=args.start,
                end=args.end,
                sleep_s=args.sleep,
                do_prices=not args.skip_prices,
                do_news=not args.skip_news,
                force=args.force,
            )
        except Exception as exc:
            log(f"FATAL {t.get('symbol_raw')}: {exc}")
            time.sleep(5)
    log("ALL DONE")


if __name__ == "__main__":
    main()
