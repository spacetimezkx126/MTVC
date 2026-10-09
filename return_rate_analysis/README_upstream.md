# 收益率分析代码汇总

从 `Pattern_Mining/normal` 汇总的 vol_screen 组合回测与收益曲线绘图代码。

**源项目路径：** `/home/zkx/Pattern_Mining/normal`

## 目录结构

```
return_rate_analysis/
├── backtest/
│   ├── vol_screen_engine.py    # 核心：checkpoint 推理 + top-K 组合策略 + budget 曲线
│   ├── backtest_vol_screen.py  # CLI 回测入口（portfolio / single_stock / classification）
│   ├── run_backtest.py         # 统一回测 runner（--model vol_screen）
│   └── metrics.py              # ARR / SR / MDD / Calmar / IR 等指标
├── scripts/
│   ├── simulator_vol_screen.py # 单 checkpoint 模拟器 + simulator_budget.png
│   └── plot_simulator_*.py     # 多配置对比收益曲线（CSMD50/300）
└── reference/
    └── dtml_simulator.py       # DTML 原版 TradingSimulator（参考）
```

## 常用命令

在 `Pattern_Mining/normal` 下运行（需完整项目依赖：model、data、train、checkpoint 等）：

```bash
cd /home/zkx/Pattern_Mining/normal

# 单模型收益曲线
python scripts/simulator_vol_screen.py \
  --ckpt checkpoints/.../best.pt \
  --plot \
  --out_dir ablation_logs/.../simulator_run

# 统一回测 CLI
python backtest/run_backtest.py --model vol_screen --ckpt checkpoints/.../best.pt --task portfolio --plot

# 多模型对比图（示例）
python scripts/plot_simulator_top04_sources_vs_mean_topk1_csmd50.py --out_dir ablation_logs/...
```

## 输出说明

- `simulator_budget.png`：Label EW / Model EW / Model top-K 累计资金曲线
- 指标：ARR（年化收益）、SR（夏普）、MDD（最大回撤）

## 依赖说明

本文件夹仅包含收益率分析核心逻辑；运行时仍依赖 `Pattern_Mining/normal` 下的：
`checkpoint.py`, `data.py`, `model.py`, `train.py`, `count_kept_news_csmd300_0p5.py` 等。

备份日期：2026-08-23
