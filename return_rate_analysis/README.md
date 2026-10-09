# return_rate_analysis（论文正式栈）

收益率回测与组合曲线。面向 **MTVC L6 / pre-2024-09-24（CSMD）**。

## 目录

| 路径 | 用途 |
|------|------|
| `backtest/` | 组合回测引擎 |
| `scripts/` | 论文图 / Acc·MCC 汇总 |
| `plots/paper_casel6_pre924/` | 论文收益率图（正式） |
| `cache/prediction_stores_casel6/` | L6 PredictionStore 缓存 |
| `results/` | 本包内零散输出（Acc/MCC 见仓库 `../results/casel6_acc_mcc/`） |

## 正式脚本

| 脚本 | 作用 |
|------|------|
| `plot_paper_casel6_pre924.py` | **论文主图**：`return_*.png` + `paper_portfolio_summary.json` |
| `summarize_acc_mcc_casel6.py` | L6 Acc/MCC 表 → `../results/casel6_acc_mcc/` |
| `plot_compare_mtvc_casel6.py` | 可选：从 ckpt 重算 MTVC/基线曲线 |
| `lib_pen_stocknet.py` | PEN / StockNet 推理辅助 |
| `lib_portfolio_store.py` | PredictionStore 读写辅助 |

## 常用命令

```bash
PY=/home/zhaokx/miniconda3/envs/CAMEF/bin/python
RRA=/home/zhaokx/Pattern/Pattern_Mining/dual_tf/MTVC_paper_repro/return_rate_analysis

# 论文组合曲线（读已有 cache，无需 GPU）
$PY $RRA/scripts/plot_paper_casel6_pre924.py

# Acc/MCC
$PY $RRA/scripts/summarize_acc_mcc_casel6.py
```
