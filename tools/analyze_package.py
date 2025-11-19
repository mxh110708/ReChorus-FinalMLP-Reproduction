# -*- coding: utf-8 -*-
"""
Analyze package (no hardcoded paths):
- 输入：Dev/Test 的 mean-std 汇总（topk_meanstd.csv / ctr_meanstd.csv）
- 可选：merged_*（不用也行，仅保留做交叉核对）与校准汇总 calibration_summary.csv
- 输出：quick_report 目录下的：
    topk_summary.csv, ctr_summary.csv
    topk_main.csv,   ctr_auc.csv, ctr_logloss.csv
    calibration_pick.csv（若提供校准文件）
"""
import argparse
from pathlib import Path
import os
import pandas as pd

def safe_read(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p and p.exists() else pd.DataFrame()

def find_file(dir_path: Path, candidates):
    """在目录下按候选文件名（或包含关键字）寻找第一份存在的文件。"""
    if not dir_path:
        return None
    if isinstance(candidates, (str, Path)):
        candidates = [candidates]
    # 先精确匹配
    for name in candidates:
        p = dir_path / name
        if p.exists():
            return p
    # 再模糊（包含关键子串）
    for f in dir_path.glob("*.csv"):
        for key in candidates:
            key = str(key).lower()
            if key in f.name.lower():
                return f
    return None

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """规范部分列名，兼容大小写/别名；保留未知列。"""
    if df.empty:
        return df
    colmap = {
        'model':'model_key',
        'mode':'task',
        'dataset_name':'dataset',
        'auc_mean':'AUC_mean',
        'auc_std':'AUC_std',
        'logloss_mean':'LOG_LOSS_mean',
        'logloss_std':'LOG_LOSS_std',
    }
    df = df.rename(columns={c: colmap.get(c, c) for c in df.columns})

    # 常见变体统一（如 'AUC@All_mean' -> 'AUC_mean'）
    def _canon_metric_col(c: str) -> str:
        if c.endswith("_mean") or c.endswith("_std"):
            base, suf = c.rsplit("_", 1)
            base = base.replace("@All", "").replace("@all", "")
            return f"{base}_{suf}"
        return c

    df.columns = [_canon_metric_col(c) for c in df.columns]
    # 关键列确保存在
    for c in ["model_key", "dataset"]:
        if c not in df.columns:
            df[c] = ""
    return df

def add_split_suffix(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """给所有数值指标列的 mean/std 后缀加 _{split}，避免 dev/test 冲突。"""
    if df.empty:
        return df
    out = df.copy()
    for c in list(out.columns):
        if c.endswith("_mean") or c.endswith("_std"):
            out.rename(columns={c: f"{c}_{split}"}, inplace=True)
    return out

def summarize(dev: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """合并 Dev/Test：按 (model_key, dataset) 外连接"""
    dev = add_split_suffix(dev, "dev")
    test = add_split_suffix(test, "test")
    key_cols = ["model_key", "dataset"]
    return pd.merge(dev, test, on=key_cols, how="outer")

def pick_metric(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """从 summary 中抽取某一主指标（含 dev/test 两套 mean/std 列）。"""
    if df.empty:
        return df
    # 规范目标指标名（去掉@all）
    norm = metric_name.replace("@All", "").replace("@all", "")
    cols = [c for c in df.columns if c.startswith(f"{norm}_") and ("_dev" in c or "_test" in c)]
    keep = ["model_key", "dataset"] + sorted(cols)
    return df[keep]

def pick_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """从校准汇总中抽取关键列（ECE/Brier等）。"""
    if df.empty:
        return df
    cols = []
    for c in df.columns:
        lc = c.lower()
        if lc in {"model_key","dataset","split","bins"} or ("ece" in lc) or ("brier" in lc):
            cols.append(c)
    if not cols:
        return pd.DataFrame()
    return df[cols]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_dir",    required=True, help="Dev meanstd 所在目录（含 topk_meanstd.csv / ctr_meanstd.csv）")
    ap.add_argument("--test_dir",   required=True, help="Test meanstd 所在目录（含 topk_meanstd.csv / ctr_meanstd.csv）")
    ap.add_argument("--merged_dir", default="",    help="可选，merged 目录（仅作交叉核对，不强依赖）")
    ap.add_argument("--calib_dir",  default="",    help="可选，校准汇总目录（含 calibration_summary.csv）")
    ap.add_argument("--out_dir",    required=True, help="输出目录")
    ap.add_argument("--topk_metric", default="NDCG@10", help="TopK 主指标（默认 NDCG@10）")
    ap.add_argument("--ctr_metrics", default="AUC,LOG_LOSS", help="CTR 主指标，逗号分隔（默认 AUC,LOG_LOSS）")
    args = ap.parse_args()

    dev_dir  = Path(args.dev_dir)
    test_dir = Path(args.test_dir)
    merged_dir = Path(args.merged_dir) if args.merged_dir else None
    calib_dir  = Path(args.calib_dir)  if args.calib_dir  else None
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 寻找输入文件（大小写/变体兼容）----
    dev_topk_p  = find_file(dev_dir,  ["topk_meanstd.csv",  "TOPK_meanstd.csv"])
    dev_ctr_p   = find_file(dev_dir,  ["ctr_meanstd.csv",   "CTR_meanstd.csv"])
    test_topk_p = find_file(test_dir, ["topk_meanstd.csv",  "TOPK_meanstd.csv"])
    test_ctr_p  = find_file(test_dir, ["ctr_meanstd.csv",   "CTR_meanstd.csv"])

    m_topk_p = find_file(merged_dir, "merged_topk.csv") if merged_dir else None
    m_ctr_p  = find_file(merged_dir, "merged_ctr.csv")  if merged_dir else None
    calib_p  = find_file(calib_dir,  "calibration_summary.csv") if calib_dir else None

    # ---- 读取并规范列名 ----
    dev_topk  = norm_cols(safe_read(dev_topk_p))
    dev_ctr   = norm_cols(safe_read(dev_ctr_p))
    test_topk = norm_cols(safe_read(test_topk_p))
    test_ctr  = norm_cols(safe_read(test_ctr_p))
    m_topk    = norm_cols(safe_read(m_topk_p))
    m_ctr     = norm_cols(safe_read(m_ctr_p))
    calib     = safe_read(calib_p)

    # ---- 合并 Dev/Test 为总表 ----
    topk_summary = summarize(dev_topk, test_topk)
    ctr_summary  = summarize(dev_ctr,  test_ctr)

    topk_summary.to_csv(out_dir / "topk_summary.csv", index=False, encoding="utf-8-sig")
    ctr_summary.to_csv(out_dir  / "ctr_summary.csv",  index=False, encoding="utf-8-sig")

    # ---- 主指标表 ----
    topk_main = pick_metric(topk_summary, args.topk_metric)
    topk_main.to_csv(out_dir / "topk_main.csv", index=False, encoding="utf-8-sig")

    ctr_list = [m.strip() for m in args.ctr_metrics.split(",") if m.strip()]
    for cm in ctr_list:
        picked = pick_metric(ctr_summary, cm)
        out_name = "ctr_auc.csv" if cm.upper()=="AUC" else ("ctr_logloss.csv" if cm.upper() in {"LOG_LOSS","LOGLOSS"} else f"ctr_{cm}.csv")
        picked.to_csv(out_dir / out_name, index=False, encoding="utf-8-sig")

    # ---- 校准（可选）----
    cal_pick = pick_calibration(calib)
    if not cal_pick.empty:
        cal_pick.to_csv(out_dir / "calibration_pick.csv", index=False, encoding="utf-8-sig")

    # ---- 备查（可选打印）----
    for tag, p in [("dev_topk",dev_topk_p),("dev_ctr",dev_ctr_p),("test_topk",test_topk_p),("test_ctr",test_ctr_p),
                   ("merged_topk",m_topk_p),("merged_ctr",m_ctr_p),("calibration",calib_p)]:
        print(f"[{tag}] -> {p if p else 'N/A'}")
    print(f"[OK] 导出到 {out_dir}")

if __name__ == "__main__":
    main()
