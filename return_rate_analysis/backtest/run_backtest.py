import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Unified backtest runner")

    parser.add_argument(
        '--model',
        default="vol_screen",
        type=str,
        choices=["vol_screen", "alstm", "dtml", "scinet", "StockNet"],
        help="vol_screen uses our checkpoint backtest; others need legacy model/utils deps",
    )

    # vol_screen args (forwarded)
    parser.add_argument('--ckpt', type=str, default="")
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--split', default="test", choices=["train", "val", "test"])
    parser.add_argument('--task', default="all", choices=["portfolio", "single_stock", "classification", "topk_sweep", "all"])
    parser.add_argument('--mode', default="all", choices=["label_ew", "model_ew", "model_topk", "all"])
    parser.add_argument('--budget', type=float, default=10_000.0)
    parser.add_argument('--fees', type=float, default=0.0)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--topk_list', type=int, nargs='+', default=[3, 5, 10])
    parser.add_argument('--prob_threshold', type=float, default=0.0)
    parser.add_argument('--model_ew_threshold', type=float, default=0.5)
    parser.add_argument('--strong_only', action='store_true')
    parser.add_argument('--long_only', action='store_true')
    parser.add_argument('--max_windows', type=int, default=0)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument(
        '--out_dir',
        type=str,
        default=os.path.join(project_root, 'backtest', 'result', 'vol_screen_csmd50'),
    )

    # legacy placeholders (kept for compatibility)
    parser.add_argument('--dataset', default="CSMD50", type=str)
    parser.add_argument('--useGPU', default=True, type=bool)
    parser.add_argument('--GPU_ID', default=1, type=int)

    args = parser.parse_args()

    if args.model == "vol_screen":
        from backtest.backtest_vol_screen import main as vol_screen_main
        # Rebuild argv for backtest_vol_screen.main()
        import sys as _sys
        _argv = [
            "backtest_vol_screen.py",
            "--task", args.task,
            "--split", args.split,
            "--device", args.device,
            "--budget", str(args.budget),
            "--fees", str(args.fees),
            "--topk", str(args.topk),
            "--prob_threshold", str(args.prob_threshold),
            "--model_ew_threshold", str(args.model_ew_threshold),
            "--max_windows", str(args.max_windows),
            "--out_dir", args.out_dir,
            "--mode", args.mode,
        ]
        if args.ckpt:
            _argv.extend(["--ckpt", args.ckpt])
        if args.strong_only:
            _argv.append("--strong_only")
        if args.long_only:
            _argv.append("--long_only")
        if args.plot:
            _argv.append("--plot")
        _argv.extend(["--topk_list", *[str(k) for k in args.topk_list]])
        _sys.argv = _argv
        vol_screen_main()
    else:
        from backtest_single import backtest_single
        from backtest_multi import backtest_multi
        if not args.useGPU:
            pass
        args.test_price_folder = os.path.join(project_root, 'dataset', args.dataset, 'test', 'price')
        args.test_news_folder = os.path.join(project_root, 'dataset', args.dataset, 'test', 'news_embedding')
        args.use_news = False
        args.model_save_folder = os.path.join(project_root, 'result', args.dataset, 'model_saved')
        args.backtest_result_save_folder = os.path.join(project_root, 'backtest', 'result', args.dataset)
        backtest_single(args)
