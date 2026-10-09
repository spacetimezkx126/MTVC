#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI argument definitions and process entry for MTVC.

Role
----
- Parse all training / data / model flags (``argparse``).
- Call ``train.run(profile_key, args)``.

Does not implement the model or the train loop.
"""
from __future__ import annotations

import argparse
import os
import sys

from train import run

try:
    from tool.cuda_guard import harden_cuda_for_multiproc
except ImportError:
    from .tool.cuda_guard import harden_cuda_for_multiproc

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    # ---- 基本 ----
    parser.add_argument(
        "--contrast_proj_dim",
        type=int,
        default=64,
        help="MTVC 对比投影维（contrast_proj_p / contrast_proj_n）",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="数据集根目录（默认见 profile：CSMD50 / CSMD300 / massive_data）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--bce_pos_weight",
        type=float,
        default=1.0,
        help="BCEWithLogitsLoss 的 pos_weight（>1 加重正类；默认 1.0）",
    )
    parser.add_argument(
        "--bce_class_weight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="按 train strong 标签 pos-rate 设 pos_weight=(1-p)/p（反频率 BCE；覆盖 --bce_pos_weight）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device: gpu / cuda:0 / cpu / …",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=110,
        help="number of training epochs",
    )
    parser.add_argument(
        "--es_start_epoch",
        type=int,
        default=1,
        help="从第几个 epoch 开始累计早停 patience（之前只训不计数）",
    )
    parser.add_argument(
        "--es_patience",
        type=int,
        default=10,
        help="早停 patience：连续多少个 epoch 监控指标无提升则停",
    )
    parser.add_argument(
        "--es_metric",
        type=str,
        default="acc_mcc",
        choices=["acc_mcc", "val_loss"],
        help="早停监控：acc_mcc=val_acc+val_MCC 越大越好；val_loss=val_task_loss 越小越好",
    )

    # ---- 数据 / 新闻编码 ----
    parser.add_argument("--train_start", type=str, default="", help="覆盖 train 起始日 YYYY-MM-DD")
    parser.add_argument("--train_end", type=str, default="", help="覆盖 train 结束日 YYYY-MM-DD")
    parser.add_argument("--val_start", type=str, default="", help="覆盖 val 起始日 YYYY-MM-DD")
    parser.add_argument("--val_end", type=str, default="", help="覆盖 val 结束日 YYYY-MM-DD")
    parser.add_argument("--test_start", type=str, default="", help="覆盖 test 起始日 YYYY-MM-DD")
    parser.add_argument("--test_end", type=str, default="", help="覆盖 test 结束日 YYYY-MM-DD")
    parser.add_argument(
        "--news_ablation_mode",
        type=str,
        default="full",
        choices=["full", "no_news"],
        help=(
            "新闻消融（nonews）：full=保留新闻支路（默认）；"
            "no_news=去掉新闻塔/新闻 PredFusion token（价格 Enc 仍在）"
        ),
    )
    parser.add_argument(
        "--train_oversample_extra",
        type=str,
        default="",
        help=(
            "可选：train_oversample_extra.jsonl（company,date 有放回副本）。"
            "仅 train loss：对该 (company,date) 的 strong 样本权重=1+额外次数；val/test 不变。"
        ),
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=128,
        help="股价 / 日状态 embedding 宽度（默认 128）",
    )
    parser.add_argument(
        "--news_emb_dim",
        type=int,
        default=32,
        help=(
            "新闻 TextTF 输出维（默认 32）；进 joint Encoder 前经 unified_news_proj "
            "升到 emb_dim"
        ),
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="CUDA autocast（默认关；部分 index_add 路径与 bf16 不兼容，确认无误再用 --amp）",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="",
        help="可选多卡 DataParallel，如 '0,1' 或 'cuda:0,cuda:1'（batch_size=1 时收益有限）",
    )
    parser.add_argument(
        "--price_encoder",
        type=str,
        default="partial_unified",
        choices=[
            "cross_attn",
            "partial_unified",
            "full_unified",
        ],
        help=(
            "Paper price/news stack: "
            "partial_unified=JMTVC partial (HLC-SeqTF+news TextTF+gated joint Encoder); "
            "cross_attn=Cross-Attention ablation "
            "(CrossAttentionEncoder: Enc + causal price←news Cross TF + CrossAttnPostFusion); "
            "full_unified=price TokenMLP + joint Encoder."
        ),
    )
    parser.add_argument(
        "--pred_fusion_ablation_mode",
        type=str,
        default="full",
        choices=["full", "no_global", "no_industry", "no_global_industry"],
        help=(
            "PredFusion 虚拟结点消融："
            "full=保留 market global + industry；"
            "no_global / no_industry / no_global_industry=去掉对应结点。"
            "CSMD 论文主结果用 full；Massive 主结果用 no_global，"
            "add_global 消融即改回 full。"
        ),
    )
    parser.add_argument(
        "--industry_fuse_site",
        type=str,
        default="pred_fusion",
        choices=["pred_fusion", "cross", "fusion"],
        help=(
            "行业虚拟结点位置："
            "pred_fusion=最终 PredFusion 与 global 一起（默认）；"
            "cross=只进 Cross Transformer 最后一层；"
            "fusion=进 CrossAttnPostFusion（仅 cross_attn）。"
        ),
    )
    parser.add_argument(
        "--vin_connect_mode",
        type=str,
        default="default",
        choices=["default", "mkt_as_ind", "ind_as_mkt"],
        help=(
            "Virtual-node connect ablation. default: market=Proj(mean)⊙g_day, "
            "industry=Proj([mean‖v_k]). mkt_as_ind: market=Proj([mean‖g_day]) "
            "(keeps day-phase g). ind_as_mkt: industry=mean⊙v_k."
        ),
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="",
        help="覆盖 profile 词表路径（默认 csmd→dict_csmd.pkl，massive→dict_massive.pkl）",
    )
    parser.add_argument(
        "--contrast_aux_weight",
        type=float,
        default=0.0,
        help=(
            "联合训练：在 BCE 预测损失上叠加 λ * MTVC 对比损失（同一次 backward）。"
            "λ=0 或 contrast_mode=temporal 为无对比。"
        ),
    )
    parser.add_argument(
        "--contrast_mode",
        type=str,
        default="mtvc",
        choices=[
            "mtvc",
            "mtvc_news_peer",
            "mtvc_price_peer",
            "mtvc_gate_only",
            "mtvc_nogate",
            "temporal",
        ],
        help=(
            "MTVC contrast: mtvc (gate+L_pair) / mtvc_gate_only / mtvc_nogate / "
            "mtvc_news_peer / mtvc_price_peer; temporal or --contrast_aux_weight 0 = no-contrast."
        ),
    )
    parser.add_argument(
        "--use_pred_fusion_global_score",
        action="store_true",
        help=(
            "Add PredFusion token from reduce_dim(global)→Linear(1,D). "
            "Default off (tokens=3 with global+industry). Needed to load older "
            "cross_attn / full_unified checkpoints that ship this head."
        ),
    )
    parser.add_argument(
        "--sim_quantile",
        type=float,
        default=0.75,
        help="MTVC case-mine peer thr = batch pairwise sim quantile (0.75≈top 25%%)",
    )
    parser.add_argument("--mtvc_lambda_pair", type=float, default=0.2)
    parser.add_argument("--mtvc_beta_hard", type=float, default=0.5)
    parser.add_argument("--mtvc_tau_y", type=float, default=0.5)
    parser.add_argument("--mtvc_temp_pair", type=float, default=0.07)
    parser.add_argument(
        "--mtvc_case_layer",
        type=int,
        default=6,
        help=(
            "MTVC: 1-indexed joint layer whose price/news pools mine hard/easy cases "
            "(official full = 6; L1–L5 kept for layer ablations)"
        ),
    )
    parser.add_argument(
        "--mtvc_contrast_news_mode",
        type=str,
        default="nbag",
        choices=["nbag", "multinews"],
        help=(
            "L_pair joint news input: nbag=mean-pool 1 token (default full); "
            "multinews=per-news tokens (token-level contrast ablation)"
        ),
    )
    parser.add_argument("--contrast_max_anchors", type=int, default=48)
    parser.add_argument("--contrast_max_news_per_bag", type=int, default=6)
    parser.add_argument(
        "--contrast_day_mode",
        type=str,
        default="all",
        choices=["all", "last"],
        help=(
            "MTVC L_pair 锚点取哪些交易日："
            "all=窗内所有有新闻+有效标签的日（默认）；"
            "last=仅窗末一日"
        ),
    )
    parser.add_argument(
        "--news_text_tf_layers",
        type=int,
        default=3,
        help="新闻文本 Transformer 层数（默认 1；更深预训练须同构加载，如 sent_nextday L3）",
    )
    parser.add_argument(
        "--news_padding_k",
        type=int,
        default=5,
        help=(
            "仅 --price_encoder cross_attn：每个 (day,stock) 固定 M 个新闻槽 "
            "（不足用「当日无文本」填充）。MTVC partial_unified 走变长 news 节点，不用此值"
        ),
    )
    parser.add_argument(
        "--no_price_token",
        action="store_true",
        help=(
            "价格消融（noprice）：股价日 token 置 0，不编码真实 HLC；"
            "价格 Enc + 跨注意力结构仍保留（与 --news_ablation_mode no_news 正交）"
        ),
    )
    parser.add_argument(
        "--seg_keep_tail",
        action="store_true",
        help="CMIN-CN：保留不足 seg_length 的尾段",
    )

    # ---- 可复现检查点 ----
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="",
        help="训练检查点目录（默认 ./checkpoints/{dataset}_{mode}_...）",
    )
    parser.add_argument(
        "--no_save_ckpt",
        action="store_true",
        help="不保存训练检查点",
    )
    parser.add_argument(
        "--save_ckpt_every_epoch",
        action="store_true",
        help="每个 epoch 额外保存 epoch-NNN.pt（仍始终更新 best.pt / last.pt）",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default="",
        help="从检查点续训（best.pt / last.pt / epoch-NNN.pt）",
    )
    parser.add_argument(
        "--no_restore_rng_on_resume",
        action="store_true",
        help="续训时不恢复 python/numpy/torch 随机数状态",
    )
    parser.add_argument(
        "--resume_weights_only",
        action="store_true",
        help="只加载 model 权重，不恢复 optimizer/RNG，epoch 从 1 开始（SSL→finetune）",
    )

    
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MTVC 统一入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", "-d",
        choices=["csmd50", "csmd300", "massive"],
        help="数据集",
    )
    add_cli_arguments(parser)
    return parser


def main() -> None:
    harden_cuda_for_multiproc()
    parser = build_parser()
    args, _unknown = parser.parse_known_args()
    if args.dataset is None:
        parser.error("必须指定 --dataset")
    from model import normalize_price_encoder

    args.price_encoder = normalize_price_encoder(getattr(args, "price_encoder", None))
    run(args.dataset, args)


if __name__ == "__main__":
    main()
