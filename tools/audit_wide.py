import pandas as pd, sys, re
from pathlib import Path

def audit(p):
    df = pd.read_csv(p)
    df["dataset_norm"] = (
        df["dataset"].astype(str)
          .str.replace("\\","/", regex=False)
          .str.strip().str.replace(r"\s+","", regex=True)
    )
    print(f"\n== {p} ==")
    print("分组种子数（规范化后）≠5 的：")
    bad = df.groupby(["model_key","dataset_norm"])["seed"].nunique()
    bad = bad[bad != 5]
    if len(bad): print(bad)
    else: print("OK，全是 5。")

for f in sys.argv[1:]:
    audit(Path(f))
