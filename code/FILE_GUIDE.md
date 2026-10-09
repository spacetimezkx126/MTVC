# MTVC 代码文件说明

路径：`MTVC_paper_repro/code/`

入口：`run_mtvc.py` → `mtvc/main.py` → `mtvc/train.py` → `Model` + `contrast_aux`。

```
run_mtvc.py
    └─ mtvc/main.py / train.py
           ├─ data.py / checkpoint.py
           ├─ tool/                 # numeric / cuda_guard / oversample
           ├─ model.py              # 单文件论文模型（内部分区，见下）
           └─ contrast_aux.py       # 对比分支：挖 pack + L_pair（复用 Model 编码）
```

## `model.py` 内部分区

| 分区 | 内容 |
|------|------|
| **§1 Modality encoders** | `SeqTransformerPool` / `NewsTransformerEncoder` (alias BertTokTransformer) / `TokenMLP` / `PredFusionTransformer` / `CrossAttentionEncoder` |
| **§2 Partial-unified Transformer** | `PartialUnifiedModel`（门控联合层 + case-mine 缓存） |
| **§3 Top-level Model** | 虚节点 Parameter、拼装 §1/§2、`forward` → logits |

数据流::

    batch → Model
              ├─ §1 encoders → day / news tokens
              ├─ 虚节点（industry / window / type / global …）
              ├─ §2 PartialUnifiedModel（或 DualTF）
              └─ PredFusion → logits → BCE
    train 另调 contrast_aux(Model) → L_pair

---

## 入口与训练

| 文件 | 作用 |
|------|------|
| `run_mtvc.py` | 包入口 |
| `mtvc/main.py` | CLI |
| `mtvc/train.py` | 训练循环 |
| `mtvc/checkpoint.py` | 检查点 |
| `mtvc/data.py` | 数据（`contiguous`） |
| `mtvc/contrast_aux.py` | 对比辅助损失 |
| `mtvc/tool/` | 数值稳定 / CUDA / 过采样 |
| `smoke_strict_load.py` | 严格加载冒烟 |

---

## 论文消融（CLI ↔ checkpoint）

| role | 关键开关 | ckpt 目录 |
|------|----------|-----------|
| `full_mtvc` | `--contrast_mode mtvc --mtvc_case_layer 6` | `full_casel6` |
| `casel{1..6}` | `--mtvc_case_layer L` | `full_casel{L}` |
| `naive_fusion` | `--contrast_aux_weight 0` | `abl_nocontrast_casel6` |
| `noprice` | `--no_price_token`（+ 无对比） | `abl_noprice_casel6` |
| `contrast_only` / nogate | `--contrast_mode mtvc_nogate` | `abl_nogate_casel6` |
| `no_global` | `--main_mlp_ablation_mode no_global` | `abl_noglobal_casel6` |
| `add_global`（Massive） | 打开 global | `abl_addglobal_casel6` |
| `no_industry` | `--main_mlp_ablation_mode no_industry` | `abl_noindustry_casel6` |
| `crossattn` | `--price_encoder cross_attn` | `crossattn_casel6` |
| `fullunified` | `--price_encoder full_unified` | `fullunified_casel6` |
| `multinews` | `--mtvc_contrast_news_mode multinews` | `multinews_casel6` |
| `vin_*` | VIN connect 消融 | `vin_*_casel6` |

复评对齐 meta（容许 ≈1–2/8653 FP 误差）::

```bash
/home/zhaokx/miniconda3/envs/CAMEF/bin/python code/eval_acc_mcc_vs_meta.py --roles all
# 或
/home/zhaokx/miniconda3/envs/CAMEF/bin/python code/eval_acc_mcc_vs_meta.py --from-catalog
```

重训脚本（与 `checkpoints/` 同名）：

- `scripts/launch_full_casel_layers_3ds.sh` → `full_casel{1..6}`
- `scripts/launch_abl_casel6_3ds.sh` → 组件消融
- `scripts/launch_arch_casel6_3ds.sh` → multinews / crossattn / fullunified
- `scripts/launch_vin_casel6_3ds.sh` → VIN 连接消融

正式栈 = PredFusion tokens=3（无 global_score）；full case-mine = L6。

