# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
from pathlib import Path

def _pretty_pm(mean, std, digits=4):
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        std = 0.0
    return f"{mean:.{digits}f} ± {std:.{digits}f}"

def _norm_metric_name(s: str) -> str:
    """
    归一化指标名/列名前缀，用于宽松匹配：
    - 大小写不敏感
    - 去空格
    - 去掉 '@all'（有的表是 AUC@All_mean）
    - LOG_LOSS / LOGLOSS / logloss / log_loss 都统一到 'logloss'
    """
    x = (s or "").strip().lower().replace(" ", "")
    x = x.replace("@all", "")
    if x in {"logloss", "log_loss"}:
        x = "logloss"
    return x

def _strip_suffix(col: str, suffix: str) -> str:
    return col[:-len(suffix)] if col.lower().endswith(suffix.lower()) else col

def _pick_metric_cols(df: pd.DataFrame, metric_name: str):
    """
    在 df 中找到给定 metric 的 mean/std 列。
    兼容：
      - AUC_mean / AUC_std
      - AUC@All_mean / AUC@All_std
      - LOG_LOSS_mean / LOGLOSS_std / log_loss_mean
    """
    # 先尝试严格列名
    exact_mean = f"{metric_name}_mean"
    exact_std  = f"{metric_name}_std"
    if exact_mean in df.columns and exact_std in df.columns:
        return exact_mean, exact_std

    # 构造一个“归一化列名 -> 原列名”的索引（去掉 @all / 空格 / 大小写）
    columns = list(df.columns)
    norm_map_mean = {}  # 归一化前缀 -> 原列名（mean）
    norm_map_std  = {}  # 归一化前缀 -> 原列名（std）

    for c in columns:
        lc = c.lower().strip()
        if lc.endswith("_mean"):
            prefix = _strip_suffix(lc, "_mean")
            prefix = _norm_metric_name(prefix)
            norm_map_mean[prefix] = c
        elif lc.endswith("_std"):
            prefix = _strip_suffix(lc, "_std")
            prefix = _norm_metric_name(prefix)
            norm_map_std[prefix] = c

    target = _norm_metric_name(metric_name)

    # 直接匹配
    if target in norm_map_mean and target in norm_map_std:
        return norm_map_mean[target], norm_map_std[target]

    # 再宽松一点：允许目标是前缀（极少数情况）
    cand_mean = [norm_map_mean[k] for k in norm_map_mean if k.startswith(target)]
    cand_std  = [norm_map_std[k]  for k in norm_map_std  if k.startswith(target)]
    if cand_mean and cand_std:
        return cand_mean[0], cand_std[0]

    raise KeyError(
        f"在表中未找到 '{metric_name}' 的 mean/std 列。"
        f"\n可用列：{columns}"
    )

def _load_pair(dev_path: Path, test_path: Path, metric_name: str, key_cols=("model_key","dataset")):
    """
    读入 Dev/Test 两份 meanstd 表（topk 或 ctr），抽取指定 metric 的 mean/std，并完成合并。
    返回：summary DataFrame，包含
      [model_key, dataset, dev_mean, dev_std, test_mean, test_std, n_runs_dev, n_runs_test, exp_dir_dev, exp_dir_test, checkpoint_dev, checkpoint_test]
    若没有这些附加列也不会报错。
    """
    dev  = pd.read_csv(dev_path)
    test = pd.read_csv(test_path)

    m_dev, s_dev = _pick_metric_cols(dev,  metric_name)
    m_tst, s_tst = _pick_metric_cols(test, metric_name)

    keep_dev = list(key_cols) + [m_dev, s_dev]
    for extra in ("n_runs","exp_dir","checkpoint"):
        if extra in dev.columns: keep_dev.append(extra)
    dev_small = dev[keep_dev].copy()
    dev_small = dev_small.rename(columns={
        m_dev: "dev_mean",
        s_dev: "dev_std",
        "n_runs": "n_runs_dev",
        "exp_dir": "exp_dir_dev",
        "checkpoint": "checkpoint_dev",
    })

    keep_tst = list(key_cols) + [m_tst, s_tst]
    for extra in ("n_runs","exp_dir","checkpoint"):
        if extra in test.columns: keep_tst.append(extra)
    test_small = test[keep_tst].copy()
    test_small = test_small.rename(columns={
        m_tst: "test_mean",
        s_tst: "test_std",
        "n_runs": "n_runs_test",
        "exp_dir": "exp_dir_test",
        "checkpoint": "checkpoint_test",
    })

    merged = pd.merge(dev_small, test_small, on=list(key_cols), how="outer")
    return merged

def _emit_tables(summary: pd.DataFrame, metric_name: str, task_title: str, out_dir: Path):
    """
    将汇总表导出为 CSV / Markdown / LaTeX 三种格式。
    summary 需要具有：model_key, dataset, dev_mean, dev_std, test_mean, test_std 列。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    basic_cols = ["model_key","dataset","dev_mean","dev_std","test_mean","test_std",
                  "n_runs_dev","n_runs_test","exp_dir_dev","exp_dir_test","checkpoint_dev","checkpoint_test"]
    exist_cols = [c for c in basic_cols if c in summary.columns]
    t = summary[exist_cols].copy()

    # 美化列
    if "dev_mean" in t.columns and "dev_std" in t.columns:
        t["Dev (mean±std)"]  = t.apply(lambda r: _pretty_pm(r.get("dev_mean"),  r.get("dev_std")), axis=1)
    if "test_mean" in t.columns and "test_std" in t.columns:
        t["Test (mean±std)"] = t.apply(lambda r: _pretty_pm(r.get("test_mean"), r.get("test_std")), axis=1)

    # 导出“漂亮表”和“原始表”
    pretty_cols = [c for c in ["model_key","dataset","Dev (mean±std)","Test (mean±std)"] if c in t.columns]
    pretty = t[pretty_cols].copy() if pretty_cols else t.copy()

    # 保存 CSV
    summary.to_csv(out_dir / f"{task_title}_summary_raw.csv", index=False, encoding="utf-8-sig")
    pretty.to_csv(out_dir / f"{task_title}_summary_pretty.csv", index=False, encoding="utf-8-sig")

    # 保存 Markdown
    with open(out_dir / f"{task_title}_summary_pretty.md", "w", encoding="utf-8") as f:
        f.write(pretty.to_markdown(index=False))

    # 保存 LaTeX（可直接进论文，未来 pandas 可能更换实现，warning 可忽略）
    with open(out_dir / f"{task_title}_summary_pretty.tex", "w", encoding="utf-8") as f:
        f.write(pretty.to_latex(index=False, escape=False))

def _emit_checkpoints(merged_file: Path, out_dir: Path, title: str, topk_metric: str = None, ctr_metric: str = None):
    """
    从 merged_xxx.csv 里导出“最佳 checkpoint 对应表”。
    对 TopK 会优先抓取 DEV_{topk_metric} / TEST_{topk_metric}，CTR 则抓取 DEV_{ctr_metric} / TEST_{ctr_metric}（若存在）。
    """
    if not merged_file.exists():
        return

    df = pd.read_csv(merged_file)

    cols = ["model_key","dataset","seed","selected_metric","direction","best_exp_dir","checkpoint"]
    cols = [c for c in cols if c in df.columns]

    # 可选地加入对应 metric 的 Dev/Test 列（如果存在）
    if title.lower().startswith("topk") and topk_metric:
        for side in ("DEV", "TEST"):
            col = f"{side}_{topk_metric}"
            if col in df.columns:
                cols.append(col)
    if title.lower().startswith("ctr") and ctr_metric:
        # ctr 的列里经常带 @All
        for side in ("DEV", "TEST"):
            variants = [f"{side}_{ctr_metric}", f"{side}_{ctr_metric}@All", f"{side}_{ctr_metric}@all"]
            hit = next((v for v in variants if v in df.columns), None)
            if hit:
                cols.append(hit)

    out = df[cols].copy() if cols else df.copy()
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"checkpoints_{title}.csv", index=False, encoding="utf-8-sig")

    # 也给一份 Markdown
    with open(out_dir / f"checkpoints_{title}.md", "w", encoding="utf-8") as f:
        f.write(out.to_markdown(index=False))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_dir",   required=True, help="目录里应包含 topk_meanstd.csv / ctr_meanstd.csv（Dev）")
    ap.add_argument("--test_dir",  required=True, help="目录里应包含 topk_meanstd.csv / ctr_meanstd.csv（Test）")
    ap.add_argument("--merged_dir",required=True, help="目录里应包含 merged_topk.csv / merged_ctr.csv")
    ap.add_argument("--out_dir",   required=True, help="输出目录")
    ap.add_argument("--topk_metric", default="NDCG@10", help="TopK 主指标（列名前缀），如 NDCG@10 / HR@10")
    ap.add_argument("--ctr_metric",  default="AUC",      help="CTR 主指标（列名前缀），如 AUC / LOG_LOSS / LOGLOSS")
    args = ap.parse_args()

    dev_dir   = Path(args.dev_dir)
    test_dir  = Path(args.test_dir)
    merged_dir= Path(args.merged_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== 1) TopK：合并 Dev/Test 的 mean±std =====
    topk_dev  = dev_dir  / "topk_meanstd.csv"
    topk_test = test_dir / "topk_meanstd.csv"
    if topk_dev.exists() and topk_test.exists():
        topk_summary = _load_pair(topk_dev, topk_test, args.topk_metric)
        _emit_tables(topk_summary, args.topk_metric, "TopK", out_dir / "tables_topk")
    else:
        print(f"[WARN] TopK meanstd 未找到：{topk_dev} 或 {topk_test}")

    # ===== 2) CTR：合并 Dev/Test 的 mean±std =====
    ctr_dev  = dev_dir  / "ctr_meanstd.csv"
    ctr_test = test_dir / "ctr_meanstd.csv"
    if ctr_dev.exists() and ctr_test.exists():
        ctr_summary = _load_pair(ctr_dev, ctr_test, args.ctr_metric)
        _emit_tables(ctr_summary, args.ctr_metric, "CTR", out_dir / "tables_ctr")
    else:
        print(f"[WARN] CTR meanstd 未找到：{ctr_dev} 或 {ctr_test}")

    # ===== 3) 最佳 checkpoint 对照表 =====
    merged_topk = merged_dir / "merged_topk.csv"
    _emit_checkpoints(merged_topk, out_dir / "checkpoints", "topk", topk_metric=args.topk_metric)

    merged_ctr  = merged_dir / "merged_ctr.csv"
    _emit_checkpoints(merged_ctr,  out_dir / "checkpoints", "ctr",  ctr_metric=args.ctr_metric)

    print(f"[OK] 产物已生成到：{out_dir.resolve()}")

if __name__ == "__main__":
    main()

