#!/usr/bin/env python3
"""
把 fetch_csmd_industry_sw.py 拉到的申万行业，写成数据集目录下的:

  dataset/<CSMD50|CSMD300>/comp_industry/comp_indus.csv
  dataset/<CSMD50|CSMD300>/comp_industry/huizong.csv

列格式与 dual_tf.data.load_industry_info 一致:
  stock_name,行业中文,行业英文

申万一级/二级会映射到项目 industry_dict 中的中英文行业标签。

用法:
  python write_csmd_comp_industry.py --input industry_cache/csmd_sw_industry_XXXX.csv
  python write_csmd_comp_industry.py --input ... --datasets CSMD50 --dry-run
"""
from __future__ import annotations

import argparse
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_PATTERN_ROOT = _HERE.parent
_DEFAULT_DATASET_ROOT = _PATTERN_ROOT / "dataset"
_DEFAULT_INDUSTRY_DICT = _PATTERN_ROOT / "dict" / "industry_dict.pkl"
if not _DEFAULT_INDUSTRY_DICT.is_file():
    _DEFAULT_INDUSTRY_DICT = _PATTERN_ROOT / "dual_tf" / "dict" / "industry_dict.pkl"


def map_sw_to_project(l1_name: str, l2_name: str = "") -> str:
    """申万一级(+二级) -> 项目行业中文。"""
    l1_name = (l1_name or "").strip()
    l2_name = (l2_name or "").strip()
    if not l1_name:
        return ""

    if l1_name == "银行":
        return "银行"
    if l1_name == "非银金融":
        if "保险" in l2_name:
            return "保险"
        return "资本市场"  # 证券 / 多元金融
    if l1_name == "医药生物":
        return "医疗健康"
    if l1_name == "基础化工":
        return "化工"
    if l1_name == "电子":
        if "半导体" in l2_name:
            return "半导体与新能源设备"
        return "硬件与电子设备"
    if l1_name == "电力设备":
        if any(k in l2_name for k in ("光伏", "风电", "电池")):
            return "半导体与新能源设备"
        return "工业制造"
    if l1_name == "公用事业":
        return "能源"
    if l1_name == "煤炭":
        return "能源"
    if l1_name == "石油石化":
        return "石油与天然气"
    if l1_name in ("有色金属", "钢铁"):
        return "矿业与金属"
    if l1_name == "房地产":
        return "房地产"
    if l1_name in ("食品饮料", "农林牧渔", "美容护理"):
        return "必需消费"
    if l1_name in ("家用电器", "纺织服饰", "轻工制造"):
        return "可选消费"
    if l1_name == "汽车":
        return "汽车与零部件"
    if l1_name == "交通运输":
        return "物流与运输"
    if l1_name == "商贸零售":
        if "互联网电商" in l2_name:
            return "互联网服务"
        return "零售"
    if l1_name == "通信":
        if "通信服务" in l2_name:
            return "电信服务"
        return "硬件与电子设备"
    if l1_name == "计算机":
        if "计算机设备" in l2_name:
            return "硬件与电子设备"
        return "软件与信息技术"
    if l1_name == "传媒":
        if any(k in l2_name for k in ("游戏", "数字媒体")):
            return "互联网服务"
        return "媒体与娱乐"
    if l1_name == "国防军工":
        return "航空航天与国防"
    if l1_name in ("建筑材料", "建筑装饰", "机械设备", "环保"):
        return "工业制造"
    if l1_name == "综合":
        return "多元化控股"
    if l1_name == "社会服务":
        if "专业服务" in l2_name:
            return "商业服务"
        return "可选消费"
    return "工业制造"


def load_cn_to_en(pkl_path: Path) -> dict[str, str]:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    return {
        d["id_to_industry_cn"][i]: d["id_to_industry_en"][i]
        for i in d["id_to_industry_cn"]
    }


def write_one_dataset(
    df_ds: pd.DataFrame,
    out_dir: Path,
    cn_to_en: dict[str, str],
    *,
    backup: bool,
    dry_run: bool,
) -> pd.DataFrame:
    rows = []
    for _, r in df_ds.iterrows():
        ind_cn = map_sw_to_project(str(r.get("sw_l1", "")), str(r.get("sw_l2", "")))
        ind_en = cn_to_en.get(ind_cn, "")
        if not ind_cn:
            raise ValueError(
                f"no industry for {r.get('dataset')} {r.get('stock_name')} "
                f"code={r.get('code')} sw_l1={r.get('sw_l1')}"
            )
        rows.append(
            {
                "stock_name": r["stock_name"],
                "行业中文": ind_cn,
                "行业英文": ind_en,
            }
        )
    out_df = pd.DataFrame(rows).sort_values("stock_name").reset_index(drop=True)
    hz = (
        out_df.groupby(["行业中文", "行业英文"], as_index=False)
        .size()
        .rename(columns={"size": "公司数量"})
        .sort_values("行业中文")
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    ind_path = out_dir / "comp_indus.csv"
    hz_path = out_dir / "huizong.csv"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if dry_run:
        print(f"[dry-run] would write {ind_path} ({len(out_df)} rows)")
        print(hz.to_string(index=False))
        return out_df

    if backup:
        for p in (ind_path, hz_path):
            if p.is_file():
                bak = p.with_name(f"{p.stem}.bak_{ts}{p.suffix}")
                shutil.copy2(p, bak)
                print(f"backup -> {bak}")

    out_df.to_csv(ind_path, index=False)
    hz.to_csv(hz_path, index=False)
    print(f"wrote {ind_path}")
    print(f"wrote {hz_path}")
    return out_df


def main() -> int:
    ap = argparse.ArgumentParser(description="Write CSMD comp_industry CSVs from SW fetch result")
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV from fetch_csmd_industry_sw.py",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["CSMD50", "CSMD300"],
        choices=["CSMD50", "CSMD300"],
    )
    ap.add_argument("--dataset-root", type=Path, default=_DEFAULT_DATASET_ROOT)
    ap.add_argument("--industry-dict", type=Path, default=_DEFAULT_INDUSTRY_DICT)
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    raw = pd.read_csv(args.input, dtype={"code": str})
    # 兼容 code 被读成 float 的缓存文件
    if "code" in raw.columns:
        raw["code"] = (
            raw["code"]
            .fillna("")
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(6)
            .replace("0000nan", "")
            .replace("000000", "")
        )
        raw.loc[raw["code"].isin(["", "000nan", "00nan"]), "code"] = ""

    cn_to_en = load_cn_to_en(args.industry_dict)

    for ds in args.datasets:
        df_ds = raw[raw["dataset"] == ds].copy()
        if df_ds.empty:
            print(f"[skip] {ds}: no rows in input")
            continue
        out_dir = args.dataset_root / ds / "comp_industry"
        write_one_dataset(
            df_ds,
            out_dir,
            cn_to_en,
            backup=not args.no_backup,
            dry_run=bool(args.dry_run),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
