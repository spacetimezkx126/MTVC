# scripts（与 `checkpoints/` 正式栈对齐）

| 脚本 | 写入目录 |
|------|----------|
| `launch_full_casel_layers_3ds.sh` | `full_casel{1..6}` |
| `launch_abl_casel6_3ds.sh` | `abl_{nocontrast,nogate,noglobal\|addglobal,noindustry,noprice}_casel6` |
| `launch_arch_casel6_3ds.sh` | `multinews_casel6` / `crossattn_casel6` / `fullunified_casel6` |
| `launch_vin_casel6_3ds.sh` | `vin_{mkt_as_ind,ind_as_mkt}_casel6` |
| `wait_abl_then_arch_casel6.sh` | 等 abl 完再启 arch |
| `run_mtvc_example.sh` | 单次训练入口包装 |
| `verify_checkpoints.sh` | 检查 best.pt / 基线 / 代码布局 |

Acc/MCC 汇总见 `../return_rate_analysis/scripts/summarize_acc_mcc_casel6.py`。
