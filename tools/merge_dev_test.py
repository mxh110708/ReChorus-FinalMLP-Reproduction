# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd
import re

def _read_wide(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    # 规范列
    for need in ["model_key","dataset","seed"]:
        if need not in df.columns:
            df[need] = ""
    # 确保是字符串，避免 merge 出错
    df["model_key"] = df["model_key"].astype(str)
    df["dataset"]   = df["dataset"].astype(str)
    df["seed"]      = df["seed"].astype(str)
    return df

def _norm(s: str) -> str:
    # 粗暴归一化用于匹配：小写、去下划线和空格
    return re.sub(r'[\s_]+','', s.strip().lower())

def _find_metric_column(df: pd.DataFrame, desired: str) -> str:
    """
    在 df 的列里寻找与 desired（如 NDCG@10 / AUC）最匹配的一列。
    规则：
      - 忽略大小写匹配
      - AUC 可匹配 AUC 或 AUC@All
      - LOG_LOSS / LogLoss / logloss 互相兼容
      - TopK 指标可用前缀匹配（如 ndcg@10）
    """
    cols = list(df.columns)
    if not cols:
        return ""

    d = desired.strip()
    d_norm = _norm(d)

    # 优先：完全忽略大小写的精确匹配
    for c in cols:
        if c.lower() == d.lower():
            return c

    # LOG_LOSS 同义
    if d_norm in {"logloss","logloss@all"}:
        for c in cols:
            if _norm(c) in {"logloss","logloss@all","log_loss","log_loss@all"}:
                return c

    # AUC：允许匹配 AUC 或 AUC@All
    if d_norm in {"auc","auc@all"}:
        for c in cols:
            cn = _norm(c)
            if cn in {"auc","auc@all"} or cn.startswith("auc"):
                return c

    # TopK：如 NDCG@10 / HR@10 等，允许前缀匹配（忽略大小写）
    for c in cols:
        if c.lower().startswith(d.lower()):
            return c

    # 最后：模糊包含
    for c in cols:
        if d.lower() in c.lower():
            return c

    return ""  # 没找到就返回空

def _is_minimize(metric_col: str) -> bool:
    # 只对 logloss 族使用最小化；其余默认最大化
    return "logloss" in _norm(metric_col) or "log_loss" in _norm(metric_col)

def _pick_best_by_metric(dev_wide: pd.DataFrame, metric_hint: str) -> (pd.DataFrame, str, str):
    """
    从 dev_wide 中对每个 (model_key, dataset) 选择最佳 seed
    返回：best_dev（带全部列）、真实匹配到的列名 metric_col、方向 dir_str（'max'或'min'）
    """
    metric_col = _find_metric_column(dev_wide, metric_hint)
    if not metric_col:
        raise ValueError(f"在 Dev 表中找不到指标列：{metric_hint}；可用列有：{[c for c in dev_wide.columns if '@' in c or c.upper() in ['AUC','LOG_LOSS']]}")

    minimize = _is_minimize(metric_col)
    dir_str = "min" if minimize else "max"

    # 排序取每组最优
    sort_cols = ["model_key","dataset", metric_col, "seed"]
    tmp = dev_wide.copy()
    # 数值列强转 float，出错的置 NaN
    tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
    # 排序方向
    tmp = tmp.sort_values(
        by=["model_key","dataset", metric_col, "seed"],
        ascending=[True, True, minimize, True]
    )
    # 每组取最后/最前一行（按上面排序已经处理）
    best = tmp.groupby(["model_key","dataset"], as_index=False).tail(1)
    return best, metric_col, dir_str

def _prefix_metrics(df: pd.DataFrame, prefix: str, keep_keys):
    out = df.copy()
    metric_cols = [c for c in out.columns if c not in keep_keys and pd.api.types.is_numeric_dtype(out[c])]
    rename_map = {c: f"{prefix}{c}" for c in metric_cols}
    out = out.rename(columns=rename_map)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_dir",  required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--topk_metric", default="NDCG@10",
                    help="用于 TopK 选 seed 的指标列名（模糊匹配，默认 NDCG@10）")
    ap.add_argument("--ctr_metric",  default="AUC",
                    help="用于 CTR 选 seed 的指标列名（模糊匹配，AUC 或 LOG_LOSS）")
    args = ap.parse_args()

    dev_dir  = Path(args.dev_dir)
    test_dir = Path(args.test_dir)
    out_dir  = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 读取 Dev/Test 宽表
    dev_topk  = _read_wide(dev_dir  / "topk_wide.csv")
    dev_ctr   = _read_wide(dev_dir  / "ctr_wide.csv")
    test_topk = _read_wide(test_dir / "topk_wide.csv")
    test_ctr  = _read_wide(test_dir / "ctr_wide.csv")

    keep_keys = ["model_key","dataset","seed","exp_dir","checkpoint"]

    # ---------- TopK ----------
    if not dev_topk.empty and not test_topk.empty:
        best_dev_topk, metric_col_topk, dir_topk = _pick_best_by_metric(dev_topk, args.topk_metric)

        dev_pref = _prefix_metrics(best_dev_topk, "DEV_", keep_keys)
        test_pick = test_topk.merge(
            best_dev_topk[["model_key","dataset","seed"]],
            on=["model_key","dataset","seed"], how="inner"
        )
        test_pref = _prefix_metrics(test_pick, "TEST_", keep_keys)

        merged_topk = dev_pref.merge(
            test_pref[["model_key","dataset","seed"] + [c for c in test_pref.columns if c.startswith("TEST_")]],
            on=["model_key","dataset","seed"], how="left"
        )
        # 补充用于排序/阅读的两列
        merged_topk["selected_metric"] = metric_col_topk
        merged_topk["direction"] = dir_topk

        # 导出
        out_topk = out_dir / "merged_topk.csv"
        merged_topk.to_csv(out_topk, index=False, encoding="utf-8-sig")
        print(f"[OK] 写出 {out_topk}")
    else:
        print("[INFO] TopK：Dev 或 Test 缺失，跳过。")

    # ---------- CTR ----------
    if not dev_ctr.empty and not test_ctr.empty:
        best_dev_ctr, metric_col_ctr, dir_ctr = _pick_best_by_metric(dev_ctr, args.ctr_metric)

        dev_pref = _prefix_metrics(best_dev_ctr, "DEV_", keep_keys)
        test_pick = test_ctr.merge(
            best_dev_ctr[["model_key","dataset","seed"]],
            on=["model_key","dataset","seed"], how="inner"
        )
        test_pref = _prefix_metrics(test_pick, "TEST_", keep_keys)

        merged_ctr = dev_pref.merge(
            test_pref[["model_key","dataset","seed"] + [c for c in test_pref.columns if c.startswith("TEST_")]],
            on=["model_key","dataset","seed"], how="left"
        )
        merged_ctr["selected_metric"] = metric_col_ctr
        merged_ctr["direction"] = dir_ctr

        out_ctr = out_dir / "merged_ctr.csv"
        merged_ctr.to_csv(out_ctr, index=False, encoding="utf-8-sig")
        print(f"[OK] 写出 {out_ctr}")
    else:
        print("[INFO] CTR：Dev 或 Test 缺失，跳过。")

if __name__ == "__main__":
    main()