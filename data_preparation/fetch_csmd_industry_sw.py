#!/usr/bin/env python3
"""
从申万（SWS）行业分类拉取 CSMD 公司所属行业。

数据来源（网站）:
  - 申万宏源研究 / 申万行业分类: https://www.swsresearch.com/
  - 本脚本通过 akshare 调用申万指数成份接口:
      ak.sw_index_first_info() / ak.sw_index_second_info()
      ak.index_component_sw(symbol=<行业代码>)
      ak.stock_info_a_code_name()  # A 股代码-名称（东方财富）

输出:
  data_process/industry_cache/csmd_sw_industry_<timestamp>.csv
  列: dataset, stock_name, code, name_on_exchange, sw_l1, sw_l2, note

用法:
  python fetch_csmd_industry_sw.py --datasets CSMD50 CSMD300
  python fetch_csmd_industry_sw.py --datasets CSMD50 --out /tmp/sw.csv
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_PATTERN_ROOT = _HERE.parent
_DEFAULT_DATASET_ROOT = _PATTERN_ROOT / "dataset"
_DEFAULT_CACHE = _HERE / "industry_cache"

# 数据集内公司名 -> 现用/并入后的证券代码（改名、合并时用）
NAME_CODE_ALIASES: dict[str, str] = {
    "韦尔股份": "603501",  # 豪威集团
    "中伟股份": "300919",  # 中伟新材
    "北京君正": "300223",  # 君正股份
    "国泰君安": "601211",  # 国泰海通
    "海通证券": "601211",  # 已并入国泰海通
    "国泰海通": "601211",  # 国泰君安现用名
    "纳思达": "002180",  # 奔图科技
    "闻泰科技": "600745",  # *ST闻泰
    "中国重工": "601989",  # 已退出/并入船舶体系；若无成份则仅记代码
    "中国国旅": "601888",  # 现中国中免
    "隆基股份": "601012",  # 现隆基绿能
    "赛力斯": "601127",
    "华电新能": "600930",
}

# 需要拆二级行业的一级行业（银行等一级已足够）
SPLIT_L1_PARENTS = {
    "非银金融",
    "电子",
    "电力设备",
    "公用事业",
    "石油石化",
    "计算机",
    "通信",
    "商贸零售",
    "社会服务",
    "煤炭",
    "有色金属",
    "传媒",
}


def _norm_name(s: str) -> str:
    return (
        str(s)
        .replace(" ", "")
        .replace("\u3000", "")
        .replace("Ａ", "A")
        .replace("Ｂ", "B")
        .strip()
    )


def list_companies(dataset_root: Path, dataset: str) -> list[str]:
    price_dir = dataset_root / dataset / "price" / "preprocessed"
    if not price_dir.is_dir():
        raise FileNotFoundError(f"missing price dir: {price_dir}")
    return sorted(p.stem for p in price_dir.glob("*.txt"))


def load_dataset_name_codes(dataset_root: Path, dataset: str) -> dict[str, str]:
    """Optional name->code from dataset/company_name.csv."""
    path = dataset_root / dataset / "company_name.csv"
    if not path.is_file():
        return {}
    df = pd.read_csv(path, dtype=str)
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("name") or cols.get("stock_name")
    code_col = cols.get("code") or cols.get("ts_code")
    if not name_col or not code_col:
        return {}
    out: dict[str, str] = {}
    for _, r in df.iterrows():
        name = _norm_name(r[name_col])
        raw = str(r[code_col]).strip()
        code = raw.split(".")[0].zfill(6) if raw else ""
        if name and code.isdigit():
            out[name] = code
            # also keep original folder name if different
            out[str(r[name_col]).strip()] = code
    return out


def resolve_code(
    name: str,
    name_to_codes: dict[str, list[str]],
    prefer_code: str | None = None,
) -> tuple[str | None, str]:
    if prefer_code:
        c = str(prefer_code).split(".")[0].zfill(6)
        if c.isdigit():
            return c, "dataset_csv"
    if name in NAME_CODE_ALIASES:
        return NAME_CODE_ALIASES[name], "alias"
    n = _norm_name(name)
    if n in NAME_CODE_ALIASES:
        return NAME_CODE_ALIASES[n], "alias"
    codes = name_to_codes.get(n)
    if codes:
        return codes[0], "exact"
    # 唯一模糊匹配
    cands = sorted(
        {
            c
            for nn, cs in name_to_codes.items()
            if n in nn or nn in n
            for c in cs
        }
    )
    if len(cands) == 1:
        return cands[0], "fuzzy"
    return None, "unresolved"


def build_name_to_codes() -> tuple[dict[str, list[str]], dict[str, str]]:
    import akshare as ak

    df = ak.stock_info_a_code_name()
    name_to_codes: dict[str, list[str]] = {}
    code_to_name: dict[str, str] = {}
    for _, r in df.iterrows():
        code = str(r["code"]).zfill(6)
        name = _norm_name(r["name"])
        name_to_codes.setdefault(name, []).append(code)
        code_to_name[code] = str(r["name"]).strip()
    return name_to_codes, code_to_name


def _fetch_cons(symbol: str, retries: int = 3) -> pd.DataFrame | None:
    import akshare as ak

    last_err: Exception | None = None
    for i in range(retries):
        try:
            return ak.index_component_sw(symbol=symbol)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.2 * (i + 1))
    print(f"  [warn] fail {symbol}: {last_err}")
    return None


def fetch_sw_maps(sleep_s: float = 0.25) -> tuple[dict[str, str], dict[str, str]]:
    """返回 code -> 申万一级名称, code -> 申万二级名称。"""
    import akshare as ak

    print("fetch Shenwan L1 list ...")
    l1 = ak.sw_index_first_info()
    code_to_l1: dict[str, str] = {}
    for _, r in l1.iterrows():
        sym = str(r["行业代码"]).replace(".SI", "")
        l1_name = str(r["行业名称"])
        print(f"  L1 {sym} {l1_name}")
        cons = _fetch_cons(sym)
        if cons is None:
            continue
        for c in cons["证券代码"].astype(str).str.zfill(6):
            code_to_l1[c] = l1_name
        time.sleep(sleep_s)

    print("fetch Shenwan L2 (selected parents) ...")
    l2 = ak.sw_index_second_info()
    code_to_l2: dict[str, str] = {}
    for _, r in l2.iterrows():
        parent = str(r["上级行业"])
        if parent not in SPLIT_L1_PARENTS:
            continue
        sym = str(r["行业代码"]).replace(".SI", "")
        l2_name = str(r["行业名称"])
        print(f"  L2 {sym} {l2_name} ({parent})")
        cons = _fetch_cons(sym)
        if cons is None:
            continue
        for c in cons["证券代码"].astype(str).str.zfill(6):
            code_to_l2[c] = l2_name
        time.sleep(sleep_s)

    return code_to_l1, code_to_l2


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch CSMD company industries from Shenwan via akshare")
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["CSMD50", "CSMD300"],
        choices=["CSMD50", "CSMD300"],
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=_DEFAULT_DATASET_ROOT,
        help="Pattern_Mining/dataset",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: industry_cache/csmd_sw_industry_<ts>.csv)",
    )
    ap.add_argument("--sleep", type=float, default=0.25, help="Sleep between SW API calls")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or (_DEFAULT_CACHE / f"csmd_sw_industry_{ts}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    print("source: 申万行业分类 https://www.swsresearch.com/ (via akshare index_component_sw)")
    print("fetch A-share code-name ...")
    name_to_codes, code_to_exch_name = build_name_to_codes()
    code_to_l1, code_to_l2 = fetch_sw_maps(sleep_s=float(args.sleep))

    rows: list[dict] = []
    for ds in args.datasets:
        prefer_codes = load_dataset_name_codes(args.dataset_root, ds)
        for stock_name in list_companies(args.dataset_root, ds):
            prefer = prefer_codes.get(stock_name) or prefer_codes.get(_norm_name(stock_name))
            code, how = resolve_code(stock_name, name_to_codes, prefer_code=prefer)
            l1 = code_to_l1.get(code or "") if code else ""
            l2 = code_to_l2.get(code or "") if code else ""
            note = how
            if code and not l1:
                # 退市/并入后可能不在当前成份；人工兜底
                if stock_name == "中国重工":
                    l1, note = "国防军工", "manual_delisted_like"
                else:
                    note = f"{how};missing_sw_member"
            rows.append(
                {
                    "dataset": ds,
                    "stock_name": stock_name,
                    "code": code or "",
                    "name_on_exchange": code_to_exch_name.get(code or "", ""),
                    "sw_l1": l1 or "",
                    "sw_l2": l2 or "",
                    "note": note,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    miss = df[(df["code"] == "") | (df["sw_l1"] == "")]
    print(f"\nwrote {out}  rows={len(df)}  unresolved_or_no_l1={len(miss)}")
    if len(miss):
        print(miss[["dataset", "stock_name", "code", "note"]].to_string(index=False))
    print(
        "\nnext: python write_csmd_comp_industry.py "
        f"--input {out} --datasets {' '.join(args.datasets)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
