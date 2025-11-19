# -*- coding: utf-8 -*-
import argparse, re
from pathlib import Path
import pandas as pd

CANON_MAP = {
    # 兜底关键词 -> (TopK 规范名, CTR 规范名)
    "grocery": ("Grocery_and_Gourmet_Food/GGFTOPK", "Grocery_and_Gourmet_Food/GGFCTR"),
    "movielens": ("MovieLens_1M/ML_1MTOPK", "MovieLens_1M/ML_1MCTR"),
}

DS_REGEXES = [
    # 从 exp_dir / checkpoint 等路径中提取  数据集/子目录
    re.compile(r"__([A-Za-z0-9_]+/[A-Za-z0-9_]+)[\\/](?:ContextReader|BaseReader)?", re.I),
    re.compile(r"data[\\/]+([A-Za-z0-9_]+/[A-Za-z0-9_]+)[\\/]", re.I),
]

def guess_task_from_model(model_key: str) -> str:
    mk = (model_key or "").lower()
    return "ctr" if mk.endswith("ctr") else "topk"

def canonical_from_paths(row: pd.Series) -> str | None:
    # 从 exp_dir / checkpoint / exp_dirs / checkpoints 中尽量还原数据集规范名
    candidates = []
    for col in ["exp_dir", "checkpoint", "exp_dirs", "checkpoints"]:
        if col in row and pd.notna(row[col]):
            candidates.extend(str(row[col]).split(";"))
    for s in candidates:
        s = s.strip()
        if not s:
            continue
        for rgx in DS_REGEXES:
            m = rgx.search(s)
            if m:
                ds = m.group(1).replace("\\", "/")
                # 只接受带斜杠的二级目录，如 MovieLens_1M/ML_1MTOPK
                if "/" in ds:
                    return ds
    return None

def canonical_fallback(ds_raw: str, task: str) -> str | None:
    s = (ds_raw or "").lower()
    for kw, (topk_name, ctr_name) in CANON_MAP.items():
        if kw in s:
            return topk_name if task == "topk" else ctr_name
    # 明显有 MovieLens 字样但被截断
    if "movielens" in s:
        return "MovieLens_1M/ML_1MTOPK" if task == "topk" else "MovieLens_1M/ML_1MCTR"
    # 明显有 Grocery 字样但被截断
    if "grocery" in s:
        return "Grocery_and_Gourmet_Food/GGFTOPK" if task == "topk" else "Grocery_and_Gourmet_Food/GGFCTR"
    return None

def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: 
        return df
    need_cols = ["model_key", "dataset"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = ""
    out = df.copy()
    norm = []
    for _, row in out.iterrows():
        task = guess_task_from_model(str(row["model_key"]))
        ds0  = str(row["dataset"])
        ds   = None
        # 1) 有 / 并且没有省略号，直接用
        if "/" in ds0 and "..." not in ds0:
            ds = ds0.replace("\\", "/")
        # 2) 从路径推断
        if ds is None:
            ds = canonical_from_paths(row)
        # 3) 兜底按任务猜
        if ds is None:
            ds = canonical_fallback(ds0, task)
        # 4) 再不行就原样
        if ds is None:
            ds = ds0
        norm.append(ds)
    out["dataset_norm"] = norm
    # 统一写回 dataset 列
    out["dataset"] = out["dataset_norm"]
    out = out.drop(columns=["dataset_norm"])
    return out

def repair_csv(path: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = normalize_dataset(df)

    # 统一 seed 类型
    if "seed" not in df.columns:
        df["seed"] = ""
    df["seed"] = df["seed"].astype(str).str.extract(r"(\d+)", expand=False).fillna("")

    # 去重（如果同 seed 有多条，保留最后一条）
    key_cols = [c for c in ["model_key","dataset","seed"] if c in df.columns]
    if key_cols:
        df = df.sort_index().drop_duplicates(subset=key_cols, keep="last")

    # 统计每组 seeds 数
    if {"model_key","dataset","seed"}.issubset(df.columns):
        g = df.groupby(["model_key","dataset"])["seed"].nunique().rename("n_seeds")
        bad = g[g != 5]
        if not bad.empty:
            print("\n[WARN] 仍有分组的种子数 != 5（以下是 ‘规范化后’ 的统计）：")
            print(bad)
        else:
            print(f"\n[OK] {path} 每组 5 seeds。")
    else:
        print("\n[INFO] 缺少 model_key/dataset/seed 某些列，跳过种子统计。")

    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] 已修复并覆盖：{path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_dir",  required=True)
    ap.add_argument("--test_dir", required=True)
    args = ap.parse_args()

    dev = Path(args.dev_dir);  test = Path(args.test_dir)
    for p in [dev/"topk_wide.csv", dev/"ctr_wide.csv", test/"topk_wide.csv", test/"ctr_wide.csv"]:
        if p.exists():
            print(f"\n== 处理 {p} ==")
            repair_csv(p)
        else:
            print(f"[SKIP] 不存在：{p}")

if __name__ == "__main__":
    main()
