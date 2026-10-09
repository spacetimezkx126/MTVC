"""CSMD / CMIN 原始行情 → price/preprocessed（Tab 分隔 txt）。

与 dataset/*/price/preprocessed/*.txt 对齐：
1) move_per 用 Adj Close 的前日收益
2) Open/High/Low/Close 均以前一日 Adj Close 为分母
3) 丢弃前两行（首行无前值、并与现有 preprocessed 起始日期一致）
"""
import argparse
import os

import pandas as pd

DATASET_ROOT = "/home/zhaokx/Pattern/Pattern_Mining/dataset"
SKIP_INITIAL_ROWS = 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="CSMD50",
        choices=["CSMD50", "CSMD300", "CMIN-US", "CMIN-CN"],
    )
    args = parser.parse_args()
    path_from = os.path.join(DATASET_ROOT, args.dataset, "price", "raw")
    path_to = os.path.join(DATASET_ROOT, args.dataset, "price", "preprocessed")

    os.makedirs(path_to, exist_ok=True)
    files = [f for f in os.listdir(path_from) if f.endswith(".csv")]
    output_columns = ["Date", "move_per", "Open", "High", "Low", "Close", "Volume"]

    for file in files:
        file_path = os.path.join(path_from, file)
        dataframe = pd.read_csv(file_path)
        prev_adj = dataframe["Adj Close"].shift(1)
        denom = prev_adj.replace(0.0, pd.NA)
        dataframe["move_per"] = (dataframe["Adj Close"] - prev_adj) / denom
        dataframe["Open"] = (dataframe["Open"] - prev_adj) / denom
        dataframe["High"] = (dataframe["High"] - prev_adj) / denom
        dataframe["Low"] = (dataframe["Low"] - prev_adj) / denom
        dataframe["Close"] = (dataframe["Close"] - prev_adj) / denom
        dataframe = dataframe.iloc[SKIP_INITIAL_ROWS:].copy()

        out_path = os.path.join(path_to, file.replace(".csv", ".txt"))
        dataframe[output_columns].to_csv(out_path, sep="\t", index=False, header=False)
        print(f"  {file} -> {out_path}")

    print("Done.")
