from __future__ import annotations
import argparse


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "train_simple"], help="Mode."
    )
    parser.add_argument("--ssl_objective", type=str, default="n2p_mlm", help="Ssl objective.")
    parser.add_argument("--ssl_mask_ratio", type=float, default=0.3, help="Ssl mask ratio.")
    parser.add_argument("--ssl_temperature", type=float, default=0.07, help="InfoNCE temperature")
    parser.add_argument(
        "--ssl_proj_dim",
        type=int,
        default=64,
        help="MTVC contrast projection dim (legacy name; not SSL pretrain)",
    )
    parser.add_argument("--ssl_log_every", type=int, default=20, help="Ssl log every.")
    parser.add_argument("--dataset_root", type=str, default=None, help="Dataset root.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--bce_pos_weight", type=float, default=1.0, help="Bce pos weight.")
    parser.add_argument(
        "--bce_class_weight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Bce class weight.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device.")
    parser.add_argument("--epochs", type=int, default=110, help="Epochs.")
    parser.add_argument("--es_start_epoch", type=int, default=1, help="Es start epoch.")
    parser.add_argument("--es_patience", type=int, default=10, help="Es patience.")
    parser.add_argument(
        "--es_metric",
        type=str,
        default="acc_mcc",
        choices=["acc_mcc", "val_loss"],
        help="Es metric.",
    )
    parser.add_argument(
        "--price_type_source",
        type=str,
        default="rule",
        choices=["rule", "kmeans", "dbscan", "mix"],
        help="Price type source.",
    )
    parser.add_argument(
        "--news_type_subdir", type=str, default="news_type", help="News type subdir."
    )
    parser.add_argument(
        "--csmd_news_source",
        type=str,
        default="llm_extract",
        choices=["llm_extract", "raw"],
        help="Csmd news source.",
    )
    parser.add_argument("--cmin_cn_all_news", action="store_true", help="Cmin cn all news.")
    parser.add_argument("--require_window_news", action="store_true", help="Require window news.")
    parser.add_argument("--subset_top_companies", type=int, default=0, help="Subset top companies.")
    parser.add_argument("--train_start", type=str, default="", help="Train start.")
    parser.add_argument("--train_end", type=str, default="", help="Train end.")
    parser.add_argument("--val_start", type=str, default="", help="Val start.")
    parser.add_argument("--val_end", type=str, default="", help="Val end.")
    parser.add_argument("--test_start", type=str, default="", help="Test start.")
    parser.add_argument("--test_end", type=str, default="", help="Test end.")
    parser.add_argument(
        "--split_mode",
        type=str,
        default="contiguous",
        choices=[
            "contiguous",
            "seasonal_h1q3q4",
            "stagger_4y",
            "week_roundrobin",
            "month_roundrobin",
            "timefrac_622",
            "day_rr_1022",
            "day_rr_411",
            "quarter_rr_211",
            "quarter_rr_311",
            "year_gap",
        ],
        help="Split mode.",
    )
    parser.add_argument(
        "--no_train_lookback_purge", action="store_true", help="No train lookback purge."
    )
    parser.add_argument("--dual_own_prop_news", action="store_true", help="Dual own prop news.")
    parser.add_argument("--own_news_only", action="store_true", help="Own news only.")
    parser.add_argument("--prop_same_trend_only", action="store_true", help="Prop same trend only.")
    parser.add_argument(
        "--news_encode_mode",
        type=str,
        default="vocab",
        choices=["finbert", "vocab", "finbert_embedding", "finbert_tok"],
        help="News encode mode.",
    )
    parser.add_argument("--finbert_model_path", type=str, default="", help="Finbert model path.")
    parser.add_argument(
        "--finbert_freeze",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Finbert freeze.",
    )
    parser.add_argument("--finbert_chunk_size", type=int, default=32, help="Finbert chunk size.")
    parser.add_argument(
        "--news_window_agg_mode",
        type=str,
        default="false",
        choices=["sum", "false"],
        help="News window agg mode.",
    )
    parser.add_argument(
        "--news_freq_gate_mode",
        type=str,
        default="hard",
        choices=["soft", "hard"],
        help="News freq gate mode.",
    )
    parser.add_argument(
        "--news_ablation_mode",
        type=str,
        default="full",
        choices=["full", "no_news"],
        help="News branch: full, or no_news (abl_nonews). Price-off is --no_price_token (abl_noprice).",
    )
    parser.add_argument(
        "--train_oversample_extra", type=str, default="", help="Train oversample extra."
    )
    parser.add_argument("--emb_dim", type=int, default=128, help="Emb dim.")
    parser.add_argument("--news_emb_dim", type=int, default=32, help="News emb dim.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False, help="Amp.")
    parser.add_argument("--devices", type=str, default="", help="Devices.")
    parser.add_argument(
        "--price_encoder",
        type=str,
        default="partial_unified",
        choices=["cross_attn", "partial_unified", "full_unified"],
        help="Joint encoder stack (paper): partial_unified = HLC SeqTransformer + news TextTF + gated joint Encoder (default); cross_attn = Cross-Attention ablation (price←news Cross Transformer); full_unified = price TokenMLP + joint Encoder. ",
    )
    parser.add_argument(
        "--main_mlp_ablation_mode",
        type=str,
        default="full",
        choices=["full", "no_global", "no_industry", "no_global_industry"],
        help="Main mlp ablation mode.",
    )
    parser.add_argument(
        "--industry_fuse_site",
        type=str,
        default="pred_fusion",
        choices=["pred_fusion", "cross", "fusion"],
        help="Industry virtual-node site: pred_fusion=final PredFusion with global (default); cross=last joint Transformer layer only; fusion=PriceNewsFusion.",
    )
    parser.add_argument(
        "--vin_ablation_mode",
        type=str,
        default="full",
        choices=["full", "with_news_type", "no_window", "with_news_type_no_window"],
        help="VIN slots: full=window on, no news-type slots (default); with_news_type=add news-type slots; no_window=drop window; with_news_type_no_window=type slots on, window off",
    )
    parser.add_argument("--vocab_path", type=str, default="", help="Vocab path.")
    parser.add_argument(
        "--news_word_emb",
        type=str,
        default="random",
        choices=["random", "pretrained"],
        help="News word embeddings: random (default) or pretrained",
    )
    parser.add_argument("--news_pretrain_ckpt", type=str, default="", help="News pretrain ckpt.")
    parser.add_argument(
        "--contrast_aux_weight", type=float, default=0.0, help="Contrast aux weight."
    )
    parser.add_argument(
        "--contrast_mode",
        type=str,
        default="mtvc",
        choices=["mtvc", "mtvc_nogate", "nocontrast"],
        help="Contrast loss. mtvc = news gate + L_pair. mtvc_nogate = abl_nogate (L_pair, news gate off). nocontrast = abl_nocontrast, and abl_fusion / abl_nonews / abl_noprice (weight 0).",
    )
    parser.add_argument(
        "--mtvc_contrast_news_mode",
        type=str,
        default="nbag",
        choices=["nbag", "multinews"],
        help="L_pair news input: nbag=one pooled token; multinews=per-news tokens (multinews_casel6).",
    )
    parser.add_argument(
        "--vin_connect_mode",
        type=str,
        default="default",
        choices=["default", "mkt_as_ind", "ind_as_mkt"],
        help="VIN connect: default, mkt_as_ind (vin_mkt_as_ind_casel6), ind_as_mkt (vin_ind_as_mkt_casel6).",
    )
    parser.add_argument(
        "--sim_quantile",
        type=float,
        default=0.75,
        help="MTVC contrast: peer threshold = batch pairwise similarity quantile (0.75 ≈ top 25%%)",
    )
    parser.add_argument("--mtvc_lambda_pair", type=float, default=0.2, help="MTVC L_pair weight")
    parser.add_argument("--mtvc_beta_hard", type=float, default=0.5)
    parser.add_argument("--mtvc_tau_y", type=float, default=0.5)
    parser.add_argument("--mtvc_temp_pair", type=float, default=0.07)
    parser.add_argument("--mtvc_lambda_clip_start", type=float, default=0.3)
    parser.add_argument("--mtvc_lambda_clip_end", type=float, default=0.05)
    parser.add_argument("--mtvc_alpha_bad", type=float, default=0.3)
    parser.add_argument("--mtvc_tau_pos", type=float, default=0.3)
    parser.add_argument("--mtvc_window_k", type=int, default=3)
    parser.add_argument(
        "--mtvc_case_layer",
        type=int,
        default=6,
        help="1-indexed joint layer for hard/easy case mining (paper full = 6)",
    )
    parser.add_argument(
        "--mtvc_case_feat",
        type=str,
        default="joint",
        choices=["joint", "encoder"],
        help="Case-mining features: joint mid-layer pools (default) or pre-joint encoder pools",
    )
    parser.add_argument("--contrast_temperature", type=float, default=0.07)
    parser.add_argument("--contrast_num_neg", type=int, default=4)
    parser.add_argument("--contrast_max_anchors", type=int, default=48)
    parser.add_argument("--contrast_max_news_per_bag", type=int, default=6)
    parser.add_argument(
        "--contrast_day_mode",
        type=str,
        default="all",
        choices=["all", "last"],
        help="Contrast day mode.",
    )
    parser.add_argument("--news_text_tf_layers", type=int, default=1, help="News text tf layers.")
    parser.add_argument("--freeze_news_text_tf", action="store_true", help="Freeze news text tf.")
    parser.add_argument(
        "--freeze_price_news_encoders", action="store_true", help="Freeze price news encoders."
    )
    parser.add_argument(
        "--news_process_mode",
        type=str,
        default="default",
        choices=["default"],
        help="News process mode (paper: default only)",
    )
    parser.add_argument("--news_padding_k", type=int, default=5, help="News padding k.")
    parser.add_argument("--use_vol_aux", action="store_true", help="Use vol aux.")
    parser.add_argument(
        "--vol_aux_from",
        type=str,
        default="unitrans_h",
        choices=["unitrans_h", "pred_fusion"],
        help="Vol aux from.",
    )
    parser.add_argument("--use_vol_token", action="store_true", help="Use vol token.")
    parser.add_argument("--use_vol_in_price", action="store_true", help="Use vol in price.")
    parser.add_argument(
        "--no_price_token",
        action="store_true",
        help="Price-off ablation (abl_noprice): day token is fixed at 0. The price encoder stays.",
    )
    parser.add_argument("--no_price_token_pad", action="store_true", help="No price token pad.")
    parser.add_argument("--no_news_token", action="store_true", help="No news token.")
    parser.add_argument("--no_news_token_pad", action="store_true", help="No news token pad.")
    parser.add_argument(
        "--modality_ablation",
        type=str,
        default="full",
        choices=["full", "price_only", "news_only"],
        help="Modality ablation.",
    )
    parser.add_argument("--industry_no_mean", action="store_true", help="Industry no mean.")
    parser.add_argument(
        "--aux_task_loss_weight", type=float, default=0.1, help="Aux task loss weight."
    )
    parser.add_argument(
        "--num_score_news_types", type=int, default=None, help="Num score news types."
    )
    parser.add_argument("--num_virtual_window", type=int, default=10, help="Num virtual window.")
    parser.add_argument("--use_cached_news_emb", action="store_true", help="Use cached news emb.")
    parser.add_argument("--news_emb_cache_path", type=str, default="", help="News emb cache path.")
    parser.add_argument("--seg_keep_tail", action="store_true", help="Seg keep tail.")
    parser.add_argument("--ckpt", type=str, default="", help="Ckpt.")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="",
        help="Checkpoint directory (default ./checkpoints/{dataset}_{mode})",
    )
    parser.add_argument("--no_save_ckpt", action="store_true", help="No save ckpt.")
    parser.add_argument(
        "--save_ckpt_every_epoch", action="store_true", help="Save ckpt every epoch."
    )
    parser.add_argument("--resume_ckpt", type=str, default="", help="Resume ckpt.")
    parser.add_argument("--resume_weights_only", action="store_true", help="Resume weights only.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mtvc.", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        "-d",
        choices=["cmin_cn", "cmin_us", "csmd50", "csmd300", "sse50", "massive", "stocknet"],
        help="Dataset.",
    )
    add_cli_arguments(parser)
    parser.add_argument(
        "--init_news_vocab_size",
        type=int,
        default=None,
        help="Recorded vocabulary size for frozen initialization companions.",
    )
    parser.add_argument(
        "--init_news_embedding_dim",
        type=int,
        default=None,
        help="Recorded word width for frozen initialization companions.",
    )
    return parser


from train import run
from tool.cuda_guard import harden_cuda_for_multiproc


def main() -> None:
    harden_cuda_for_multiproc()
    parser = build_parser()
    (args, _unknown) = parser.parse_known_args()
    if args.dataset is None:
        parser.error("Specify --dataset.")
    run(args.dataset, args)


if __name__ == "__main__":
    main()
