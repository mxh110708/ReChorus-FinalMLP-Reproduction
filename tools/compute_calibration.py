# -*- coding: utf-8 -*-
"""
compute_calibration.py
对 CTR 预测结果（rec-*-dev.csv / rec-*-test.csv）计算：
- Brier score
- ECE (Expected Calibration Error, 加权 |acc - conf|)
- MCE (Maximum Calibration Error)
并生成可靠性图（校准曲线）与汇总表。

用法（PowerShell 单行）：
python tools\compute_calibration.py --log_dir "E:\log" --out_dir "E:\log\calib" --bins 15

注意：
- 仅处理包含 label/prediction (大小写/下划线不敏感) 两列的文件；
- 会跳过 TopK 预测（通常没有概率与label）。
"""

import os
import re
import glob
import argparse
import warnings
import numpy as np
import pandas as pd

# 尝试导入 matplotlib；若失败，仅跳过作图，不影响数值指标
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception as e:
    warnings.warn(f"matplotlib 不可用，可靠性图将跳过。原因：{e}")
    HAS_PLT = False


def _read_any_csv(path: str) -> pd.DataFrame:
    """自动判断分隔符读取 CSV/TSV。"""
    # pandas 的 sep=None + engine='python' 会自动猜测分隔符（逗号/制表）
    return pd.read_csv(path, sep=None, engine="python")


def _find_cols(df: pd.DataFrame):
    """从 DataFrame 中找出 label 与 prediction 列，返回标准名 ('label','pred')。"""
    cols = {c.lower().replace("-", "").replace("_", ""): c for c in df.columns}

    # 常见命名兼容
    label_cands = ["label", "y", "target", "click", "clicked"]
    pred_cands  = ["prediction", "pred", "score", "prob", "pctr", "yhat"]

    label_col = None
    pred_col  = None

    for k, c in cols.items():
        if label_col is None and any(k == s for s in label_cands):
            label_col = c
        if pred_col is None  and any(k == s for s in pred_cands):
            pred_col = c

    # 再次宽松匹配（包含子串）
    if label_col is None:
        for k, c in cols.items():
            if "label" in k or "click" in k or "target" in k:
                label_col = c
                break
    if pred_col is None:
        for k, c in cols.items():
            if "pred" in k or "score" in k or "prob" in k or "pctr" in k or "yhat" in k:
                pred_col = c
                break

    return label_col, pred_col


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier 分数（概率均方误差），越小越好。"""
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    return float(np.mean((y_prob - y_true) ** 2))


def ece_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15):
    """计算 ECE/MCE 与每个 bin 的统计用于画校准曲线。"""
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1  # 0..n_bins-1
    idx = np.clip(idx, 0, n_bins - 1)

    bin_stats = []
    n = len(y_true)
    ece = 0.0
    mce = 0.0

    for b in range(n_bins):
        mask = (idx == b)
        cnt = int(mask.sum())
        if cnt == 0:
            bin_stats.append({
                "bin_left": float(bins[b]),
                "bin_right": float(bins[b + 1]),
                "avg_conf": np.nan,
                "avg_acc":  np.nan,
                "count":    0,
            })
            continue

        conf = float(np.mean(y_prob[mask]))
        acc  = float(np.mean(y_true[mask]))
        gap  = abs(acc - conf)

        w = cnt / n
        ece += w * gap
        mce = max(mce, gap)

        bin_stats.append({
            "bin_left": float(bins[b]),
            "bin_right": float(bins[b + 1]),
            "avg_conf": conf,
            "avg_acc":  acc,
            "count":    cnt,
        })

    bin_df = pd.DataFrame(bin_stats)
    return float(ece), float(mce), bin_df


def _infer_ctx_from_path(path: str):
    """
    从路径推断 (model_tag, dataset, split, exp_dir, filename)。
    兼容类似：
    E:\log\FinalMLPReImplCTR\FinalMLPReImplCTR__MovieLens_1M\ML_1MCTR_context000__0__lr=...\rec-FinalMLPReImplCTR-dev.csv
    """
    norm_path = path.replace("/", "\\")
    parts = norm_path.split("\\")
    model_tag, dataset, split = "UnknownModel", "UnknownData", "UnknownSplit"
    exp_dir = os.path.dirname(path)

    # 从文件名推断 split
    fname = os.path.basename(path)
    if re.search(r"-dev\.csv$", fname, flags=re.I):
        split = "dev"
    elif re.search(r"-test\.csv$", fname, flags=re.I):
        split = "test"

    # 从 “…\{ModelTag}\{ModelTag__Dataset}\...” 推断
    # 找到第一个形如 AAA__BBB 的段
    for p in parts:
        m = re.match(r"^(.+?)__([^\\]+)$", p)
        if m:
            # 再看它的上一级是否就是 model_tag 目录
            up = parts[parts.index(p) - 1] if parts.index(p) > 0 else ""
            if up and m.group(1) == up:
                model_tag = m.group(1)
                dataset = m.group(2)
                break

    return model_tag, dataset, split, exp_dir, fname


def _plot_reliability(bin_df: pd.DataFrame, title: str, save_path: str):
    if not HAS_PLT:
        return
    plt.figure(figsize=(4.2, 4.2), dpi=160)
    # 只画有效 bin
    v = bin_df.dropna(subset=["avg_conf", "avg_acc"])
    if len(v) > 0:
        plt.plot(v["avg_conf"], v["avg_acc"], marker="o", linewidth=1)
    # 理想对角
    xs = np.linspace(0, 1, 101)
    plt.plot(xs, xs, linestyle="--")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical accuracy")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir",  required=True, help="递归扫描的日志根目录（含 rec-*-dev/test.csv）")
    ap.add_argument("--out_dir",  required=True, help="输出目录（summary 与 图）")
    ap.add_argument("--bins",     type=int, default=15, help="校准分箱个数 (默认15)")
    ap.add_argument("--glob_pat", default="**/rec-*.csv", help="匹配预测文件的通配符")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    curves_dir = os.path.join(args.out_dir, "curves")
    plots_dir  = os.path.join(args.out_dir, "plots")
    os.makedirs(curves_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    files = glob.glob(os.path.join(args.log_dir, args.glob_pat), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        print(f"[WARN] 未在 {args.log_dir} 下找到任何 '{args.glob_pat}' 预测文件。")
        return

    rows = []
    for f in files:
        try:
            df = _read_any_csv(f)
        except Exception as e:
            print(f"[SKIP] 读取失败：{f}  -> {e}")
            continue

        label_col, pred_col = _find_cols(df)
        if label_col is None or pred_col is None:
            # 多半是 TopK 的 top-100 列表，没有概率与label，跳过
            print(f"[SKIP] 非 CTR 预测（缺少 label/prediction）：{f}")
            continue

        y_true = df[label_col].values.astype(float)
        y_prob = df[pred_col].values.astype(float)
        # clip 至 [0,1]，若你的模型输出已是 sigmoid 概率，此处只是保险
        y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)

        brier = brier_score(y_true, y_prob)
        ece, mce, bin_df = ece_mce(y_true, y_prob, n_bins=args.bins)

        model_tag, dataset, split, exp_dir, fname = _infer_ctx_from_path(f)
        # 保存曲线数据
        curve_name = f"calib_{model_tag}__{dataset}__{split}__{os.path.basename(exp_dir)}.csv"
        curve_path = os.path.join(curves_dir, curve_name)
        bin_df.to_csv(curve_path, index=False)

        # 保存可靠性图
        plot_name = f"reliability_{model_tag}__{dataset}__{split}__{os.path.basename(exp_dir)}.png"
        plot_path = os.path.join(plots_dir, plot_name)
        _plot_reliability(bin_df, title=f"{model_tag} | {dataset} | {split}", save_path=plot_path)

        rows.append({
            "model": model_tag,
            "dataset": dataset,
            "split": split,
            "exp_dir": exp_dir,
            "file": f,
            "brier": brier,
            "ece": ece,
            "mce": mce,
            "n": int(len(y_true)),
            "bins": args.bins,
            "curve_csv": curve_path,
            "plot_png": plot_path if HAS_PLT else ""
        })

    if rows:
        summary = pd.DataFrame(rows).sort_values(["dataset", "model", "split"])
        out_csv = os.path.join(args.out_dir, "calibration_summary.csv")
        summary.to_csv(out_csv, index=False)
        print(f"[OK] 已生成：\n- 汇总：{out_csv}\n- 曲线CSV：{curves_dir}\n- 可靠性图：{plots_dir}")
    else:
        print("[WARN] 没有任何有效 CTR 预测文件被处理。")


if __name__ == "__main__":
    main()
