# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd

def _read(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    for c in ["model_key","dataset","seed"]:
        if c not in df.columns:
            df[c] = ""
    df["model_key"] = df["model_key"].astype(str)
    df["dataset"]   = df["dataset"].astype(str)
    df["seed"]      = df["seed"].astype(str)
    return df

def _num_metric_cols(df: pd.DataFrame) -> list:
    # 去掉非数值/非指标列
    drop = {"model_key","dataset","seed","exp_dir","checkpoint"}
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]

def _agg_meanstd(df: pd.DataFrame, tag: str, out_dir: Path):
    if df.empty:
        print(f"[INFO] {tag}: 输入为空，跳过。")
        return
    metrics = _num_metric_cols(df)
    if not metrics:
        print(f"[INFO] {tag}: 未找到数值指标列，跳过。")
        return

    # 计算 n_runs（按 seed 去重）
    runs = df.groupby(["model_key","dataset"], as_index=False)["seed"].nunique().rename(columns={"seed":"n_runs"})

    # 对所有数值列做 mean/std
    agg_map = {m:["mean","std"] for m in metrics}
    stat = df.groupby(["model_key","dataset"]).agg(agg_map)
    # 扁平化 MultiIndex 列名： metric_mean / metric_std
    stat.columns = [f"{m}_{k}" for (m,k) in stat.columns.to_flat_index()]
    stat = stat.reset_index()

    # exp_dir / checkpoint 汇总（唯一集合，分号拼接，方便追溯）
    def uniq_join(series):
        vals = sorted(set(str(x) for x in series if pd.notna(x)))
        return ";".join(vals)

    meta = df.groupby(["model_key","dataset"], as_index=False).agg({
        "exp_dir": uniq_join if "exp_dir" in df.columns else "first",
        "checkpoint": uniq_join if "checkpoint" in df.columns else "first"
    }) if "exp_dir" in df.columns or "checkpoint" in df.columns else pd.DataFrame()

    out = stat.merge(runs, on=["model_key","dataset"], how="left")
    if not meta.empty:
        out = out.merge(meta, on=["model_key","dataset"], how="left")

    out_path = out_dir / f"{tag.lower()}_meanstd.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 写出 {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",  required=True, help="包含 topk_wide.csv / ctr_wide.csv 的目录（Dev 或 Test 任一）")
    ap.add_argument("--out_dir", required=True, help="输出目录")
    args = ap.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    topk = _read(in_dir / "topk_wide.csv")
    ctr  = _read(in_dir / "ctr_wide.csv")

    _agg_meanstd(topk, "TOPK", out_dir)
    _agg_meanstd(ctr,  "CTR",  out_dir)

if __name__ == "__main__":
    main()
