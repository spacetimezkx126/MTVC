# data_preparation

CSMD / Massive 数据准备与申万行业拉取脚本副本。

## 文件

| 文件 | 用途 |
|------|------|
| `align_split_phase.py` | 给定 train/val/test 意图区间 → 10 日窗 phase 对齐日期（`--train_start` 等） |
| `data_processing_csmd.py` | CSMD/CMIN 原始行情 → `price/preprocessed` |
| `news_csv_to_news_txt.py` | 新闻 CSV → 模型可用 txt |
| `xlsx_news_raw_to_cmin_cn_csv.py` | XLSX 新闻 → CMIN-CN CSV |
| `fetch_csmd_industry_sw.py` | **申万**一/二级行业（akshare）下载缓存 |
| `write_csmd_comp_industry.py` | 写 `comp_industry/comp_indus.csv` |
| `industry_remap/` | 实验冻结的申万 remap CSV |
| `crawl_massive_stocktable.py` | Massive/Polygon 拉价格+新闻 |
| `README_massive_crawl.md` | Massive 爬虫说明 |
| `build_massive_data.py` | 爬虫结果 → `dataset/massive_data` |

## 数据根目录（本机约定）

- CSMD50: `/home/zhaokx/Pattern/Pattern_Mining/dataset/CSMD50`
- CSMD300: `/home/zhaokx/Pattern/Pattern_Mining/dataset/CSMD300`
- Massive: `/home/zhaokx/Pattern/Pattern_Mining/dataset/massive_data`
- 词典: `/home/zhaokx/Pattern/Pattern_Mining/dict/`

运行前请按脚本内路径/依赖（akshare、API key 等）配置环境。

## 相位对齐 split

```bash
cd data_preparation
python align_split_phase.py \
  --dataset csmd50 \
  --train 2021-01-01:2023-12-31 \
  --val   2024-01-01:2024-06-30 \
  --test  2024-07-01:2024-12-31
```

输出可直接用作训练 CLI 的 `--train_start/--train_end/...`。
