# Massive StockTable crawl

数据根目录：`/home/zhaokx/Pattern/massive_stocktable`

## 来源
- 公司列表：`/home/zhaokx/Pattern/StockTable`（88 家）
- API：`massive.RESTClient`（Polygon / Massive 兼容）
- Key：环境变量 `POLYGON_API_KEY` / `MASSIVE_API_KEY`，或脚本默认 key

## 目录结构
```
tickers.csv                 # sector / symbol_raw / symbol_api / company
data/{SYMBOL}/
  prices.csv                # 日线 OHLCV（含 volume）
  news/YYYY-MM-DD.json      # 每天一个 JSON（含当日全部新闻）
  meta.json                 # 抓取状态
logs/crawl.log
scripts/crawl_massive_stocktable.py
```

## news/YYYY-MM-DD.json
```json
{
  "date": "2022-01-01",
  "query_ticker": "AAPL",
  "n": 3,
  "articles": [
    {
      "id": "...",
      "published_utc": "...",
      "title": "...",
      "description": "...",
      "tickers": ["AAPL", "..."],
      "sentiment": "positive",
      "sentiment_reasoning": "...",
      "insights": [...],
      "keywords": [...]
    }
  ]
}
```
- `tickers`：该新闻适用的公司列表
- `sentiment` / `insights`：情感（来自 Massive insights）

## 已知代码映射
| StockTable | API |
|------------|-----|
| RDS-B | SHEL |
| TOT | TTE |
| BBL | BHP |
| PCLN | BKNG |
| FB | META |
| UTX | RTX |
| BRK-A | BRK.A |

## 验证（当前 key）
AAPL 冒烟：`prices` **1003** 根（2022-01-03 … 2025-12-31），`news` **16560** 条；约 16.7% 带 `insights` sentiment。

## 运行
```bash
cd /home/zhaokx/Pattern/massive_stocktable
export POLYGON_API_KEY='你的key'
nohup /home/zhaokx/miniconda3/envs/CAMEF/bin/python -u scripts/crawl_massive_stocktable.py \
  --start 2022-01-01 --end 2025-12-31 --sleep 0.25 \
  > logs/nohup.out 2>&1 &
```
