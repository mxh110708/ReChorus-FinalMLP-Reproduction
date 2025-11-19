# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def _read_csv(p: Path) -> pd.DataFrame:
    if p is None: return pd.DataFrame()
    if not p.exists(): return pd.DataFrame()
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame()
    return df

def _safe_get(df, cols):
    """取存在的列（按顺序第一优先），都不存在则返回None"""
    for c in cols:
        if c in df.columns: return c
    return None

def _best_by_metric(df, metric_col, larger_is_better=True):
    if df.empty or metric_col not in df.columns:
        return pd.DataFrame()
    key_cols = [c for c in ["dataset","model_key"] if c in df.columns]
    rows = []
    for ds, g in df.groupby("dataset"):
        g = g.copy()
        if larger_is_better:
            ridx = g[metric_col].astype(float).idxmax()
        else:
            ridx = g[metric_col].astype(float).idxmin()
        rows.append(g.loc[ridx])
    out = pd.DataFrame(rows)
    return out

def _merge_dev_test(dev_df, test_df, metric):
    """把 dev/test 的 mean/std 贴到一张表（要求列名形如 <metric>_mean/_std 或 <metric>@All_mean/_std）"""
    if dev_df.empty and test_df.empty:
        return pd.DataFrame()
    df_d = dev_df.copy()
    df_t = test_df.copy()
    # 尝试两种命名
    cand_mean = [f"{metric}_mean", f"{metric}@All_mean", f"{metric}@all_mean"]
    cand_std  = [f"{metric}_std",  f"{metric}@All_std",  f"{metric}@all_std"]
    m_d = _safe_get(df_d, cand_mean); s_d = _safe_get(df_d, cand_std)
    m_t = _safe_get(df_t, cand_mean); s_t = _safe_get(df_t, cand_std)
    if m_d is None and m_t is None:
        return pd.DataFrame()

    cols = ["dataset","model_key"]
    base = None
    if not df_d.empty:
        base = df_d[ [c for c in cols if c in df_d.columns] ].drop_duplicates()
    if base is None or base.empty:
        base = df_t[ [c for c in cols if c in df_t.columns] ].drop_duplicates()

    def pick(df, mean_col, std_col, suffix):
        if df.empty or mean_col is None: 
            return pd.DataFrame(columns=[*cols, f"{suffix}_mean", f"{suffix}_std"])
        use = [c for c in cols if c in df.columns] + [mean_col]
        if std_col and std_col in df.columns: use += [std_col]
        tmp = df[use].copy()
        tmp = tmp.rename(columns={mean_col: f"{suffix}_mean"})
        if std_col and std_col in df.columns:
            tmp = tmp.rename(columns={std_col: f"{suffix}_std"})
        else:
            tmp[f"{suffix}_std"] = np.nan
        return tmp

    left = base
    dev_part  = pick(df_d, m_d, s_d, "dev")
    test_part = pick(df_t, m_t, s_t, "test")
    for part in [dev_part, test_part]:
        if not part.empty:
            left = pd.merge(left, part, on=[c for c in cols if c in part.columns], how="left")
    return left

def _attach_exp_tracking(df, merged_df):
    """从 merged_* 里把 exp_dir / checkpoint 贴回（如果用户提供了 merged 对齐表）"""
    if df.empty or merged_df.empty:
        return df
    cols = [c for c in ["dataset","model_key","exp_dir","checkpoint"] if c in merged_df.columns]
    if not {"dataset","model_key"}.issubset(set(cols)):
        return df
    meta = merged_df[cols].drop_duplicates()
    out = pd.merge(df, meta, on=["dataset","model_key"], how="left")
    return out

def _nice_pm(mean, std, digits=4):
    if pd.isna(mean): return ""
    if pd.isna(std) or float(std) == 0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_meanstd_topk",  default="", help="dev_topk_meanstd.csv")
    ap.add_argument("--test_meanstd_topk", default="", help="test_topk_meanstd.csv")
    ap.add_argument("--dev_meanstd_ctr",   default="", help="dev_ctr_meanstd.csv")
    ap.add_argument("--test_meanstd_ctr",  default="", help="test_ctr_meanstd.csv")
    ap.add_argument("--merged_topk",       default="", help="merged_topk.csv（含exp_dir/checkpoint）")
    ap.add_argument("--merged_ctr",        default="", help="merged_ctr.csv（含exp_dir/checkpoint）")
    ap.add_argument("--ctr_auc",           default="", help="ctr_auc.csv（来自 analyze_package.py）")
    ap.add_argument("--ctr_logloss",       default="", help="ctr_logloss.csv（来自 analyze_package.py）")
    ap.add_argument("--topk_main",         default="", help="topk_main.csv（来自 analyze_package.py）")
    ap.add_argument("--calibration_pick",  default="", help="calibration_pick.csv（可选）")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--topk_metric", default="NDCG@10")
    ap.add_argument("--ctr_metrics", default="AUC,LOG_LOSS")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    dev_topk  = _read_csv(Path(args.dev_meanstd_topk))
    test_topk = _read_csv(Path(args.test_meanstd_topk))
    dev_ctr   = _read_csv(Path(args.dev_meanstd_ctr))
    test_ctr  = _read_csv(Path(args.test_meanstd_ctr))

    merged_topk = _read_csv(Path(args.merged_topk))
    merged_ctr  = _read_csv(Path(args.merged_ctr))

    # 1) TopK：按 NDCG@10 选最优（Test为准），并附上 Dev/Test ±、泛化差和追溯
    topk_join = _merge_dev_test(dev_topk, test_topk, args.topk_metric)
    best_topk = _best_by_metric(topk_join, "test_mean", larger_is_better=True)
    if not best_topk.empty:
        best_topk["generalization_gap"] = best_topk["test_mean"] - best_topk["dev_mean"]
        best_topk = _attach_exp_tracking(best_topk, merged_topk)
        best_topk["pretty_dev"]  = best_topk.apply(lambda r: _nice_pm(r.get("dev_mean"),  r.get("dev_std")),  axis=1)
        best_topk["pretty_test"] = best_topk.apply(lambda r: _nice_pm(r.get("test_mean"), r.get("test_std")), axis=1)
        cols = ["dataset","model_key","pretty_dev","pretty_test","generalization_gap","exp_dir","checkpoint"]
        best_topk[cols].to_csv(out_dir/"topk_winners.csv", index=False, encoding="utf-8-sig")

    # 2) CTR：同时给 AUC 与 LOG_LOSS 两套结论
    out_ctr = []
    for m in [x.strip() for x in args.ctr_metrics.split(",") if x.strip()]:
        ctr_join = _merge_dev_test(dev_ctr, test_ctr, m)
        if ctr_join.empty: 
            continue
        # AUC 越大越好；LOG_LOSS 越小越好
        larger = (m.upper() != "LOG_LOSS")
        best = _best_by_metric(ctr_join, "test_mean", larger_is_better=larger)
        if best.empty: 
            continue
        best["generalization_gap"] = best["test_mean"] - best["dev_mean"]
        best = _attach_exp_tracking(best, merged_ctr)
        best["metric"] = m
        best["pretty_dev"]  = best.apply(lambda r: _nice_pm(r.get("dev_mean"),  r.get("dev_std")),  axis=1)
        best["pretty_test"] = best.apply(lambda r: _nice_pm(r.get("test_mean"), r.get("test_std")), axis=1)
        out_ctr.append(best)
    if out_ctr:
        ctr_winners = pd.concat(out_ctr, ignore_index=True)
        cols = ["metric","dataset","model_key","pretty_dev","pretty_test","generalization_gap","exp_dir","checkpoint"]
        ctr_winners[cols].to_csv(out_dir/"ctr_winners.csv", index=False, encoding="utf-8-sig")

    # 3) 校准（若提供）
    calib = _read_csv(Path(args.calibration_pick))
    if not calib.empty:
        # 尝试通用列名
        keep = [c for c in ["dataset","model_key","split","ece","brier","nll","bins","samples"] if c in calib.columns]
        if keep:
            calib[keep].to_csv(out_dir/"calibration_overview.csv", index=False, encoding="utf-8-sig")

    # 4) 汇总 markdown
    parts = []
    if (out_dir/"topk_winners.csv").exists():
        parts.append("## Top-K Winners (by Test NDCG@10)\n\n" + (out_dir/"topk_winners.csv").read_text(encoding="utf-8-sig"))
    if (out_dir/"ctr_winners.csv").exists():
        parts.append("\n\n## CTR Winners (AUC & LOG_LOSS)\n\n" + (out_dir/"ctr_winners.csv").read_text(encoding="utf-8-sig"))
    if (out_dir/"calibration_overview.csv").exists():
        parts.append("\n\n## Calibration Overview\n\n" + (out_dir/"calibration_overview.csv").read_text(encoding="utf-8-sig"))
    (out_dir/"README.md").write_text("\n\n".join(parts), encoding="utf-8-sig")

    print(f"[OK] 结果已输出到: {out_dir}")
    if (out_dir/"topk_winners.csv").exists():
        print(" - topk_winners.csv")
    if (out_dir/"ctr_winners.csv").exists():
        print(" - ctr_winners.csv")
    if (out_dir/"calibration_overview.csv").exists():
        print(" - calibration_overview.csv")
    print(" - README.md")

if __name__ == "__main__":
    main()
