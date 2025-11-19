# -*- coding: utf-8 -*-
import re
import os
import sys
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd

"""
更健壮的 summarize_all.py

用法（DEV 汇总，用于选型/选超参/选 seed）：
python tools/summarize_all.py --log_dir "E:\\log" --out_dir "E:\\log" --project_root "E:\\" --line_pattern "Dev After Training"

用法（TEST 汇总，用于报告最终结果）：
python tools/summarize_all.py --log_dir "E:\\log" --out_dir "E:\\log" --project_root "E:\\" --line_pattern "Test After Training"

输出：
- topk_long.csv / topk_wide.csv
- ctr_long.csv  / ctr_wide.csv
均包含 exp_dir 与 checkpoint 两列（实验目录与最佳权重路径）
"""

def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return p.read_text(encoding="latin1", errors="replace")

def norm_spaces(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip())

def find_dataset(text: str) -> str:
    # 优先从“Reading data from ... dataset = "xxx"”解析
    m = re.search(r'Reading data from .*?dataset\s*=\s*["\']([^"\']+)["\']', text, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    # 退化从 Arguments 表解析（dataset | xxx），日志里可能被截断，不一定可靠
    m = re.search(r'\bdataset\s*\|\s*([^\r\n]+)', text, flags=re.IGNORECASE)
    if m:
        return norm_spaces(m.group(1))
    return ""

def find_seed(text: str, path: Path) -> str:
    # 从 Arguments 表解析：random_seed | 42
    m = re.search(r'\brandom_seed\s*\|\s*(\d+)', text, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    # 从文件名模式解析：__<seed>__lr=
    m = re.search(r'__([0-9]+)__lr=', path.name, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return ""

def find_checkpoint(text: str, project_root: Path) -> str:
    # 形如：Load model from ../model/FinalMLP.../xxx.pt
    m = re.search(r'Load model from\s+([^\s]+?\.pt)', text, flags=re.IGNORECASE)
    if not m:
        return ""
    raw = m.group(1)
    # 统一路径分隔
    raw = raw.replace('\\', os.sep).replace('/', os.sep)
    # 相对路径转绝对路径
    p = Path(raw)
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return str(p)

def parse_metric_line(line: str) -> dict:
    """
    输入：一整行类似
    Dev  After Training: (HR@5:0.3411,NDCG@5:0.2253,HR@10:...,NDCG@10:...)
    或 CTR: (AUC@All:0.78,LOG_LOSS@All:0.58)
    输出：{ 'HR@5': 0.3411, 'NDCG@5': 0.2253, ... }
    """
    # 括号内容
    m = re.search(r'\((.*)\)', line)
    if not m:
        return {}
    inside = m.group(1)
    # 分割逗号（日志不会包含数值中的逗号）
    parts = [x.strip() for x in inside.split(',') if x.strip()]
    out = {}
    for tok in parts:
        # 形如  HR@5:0.3411  或  LOG_LOSS@All:0.58
        kv = tok.split(':', 1)
        if len(kv) != 2:
            continue
        k = kv[0].strip()
        v = kv[1].strip()
        # 去掉可能的尾随符号
        k = k.replace(' ', '')
        # 取数值
        try:
            out[k] = float(v)
        except ValueError:
            # 截断或异常时忽略该项
            pass
    return out

def detect_mode(model_key: str, metric_keys: set) -> str:
    mk = model_key.lower()
    if mk.endswith('topk'):
        return 'TopK'
    if mk.endswith('ctr'):
        return 'CTR'
    # 通过指标推断
    if any(m.lower().startswith(('hr@', 'ndcg@')) for m in metric_keys):
        return 'TopK'
    if any(m.lower().startswith(('auc@', 'log_loss@', 'logloss@')) or m.lower() in ('auc', 'log_loss', 'logloss') for m in metric_keys):
        return 'CTR'
    return ''  # 未知

def find_line(text: str, pattern_words: str) -> str:
    """
    在文本中寻找一行“* After Training”。
    pattern_words 通过空格分词，逐词匹配（忽略大小写与多空格）。
    例： "Dev After Training" / "Test After Training"
    """
    words = [w.lower() for w in pattern_words.split() if w.strip()]
    for raw_line in text.splitlines():
        line = raw_line.strip()
        low = line.lower()
        if all(w in low for w in words) and 'after training' in low:
            return raw_line
    return ""

def walk_logs(log_dir: Path):
    for p in log_dir.rglob('*.txt'):
        # 仅拿有内容的 .txt
        try:
            if p.stat().st_size > 0:
                yield p
        except Exception:
            # 某些路径权限问题时跳过
            continue

def pivot_wide(df_long: pd.DataFrame, value_col: str = 'value') -> pd.DataFrame:
    """
    将 long 变成宽：索引（model_key, dataset[, seed]），列为 metric
    额外保留 exp_dir 与 checkpoint：若同一索引有多个值，取第一个（或聚合后保留唯一）
    """
    if df_long.empty:
        return df_long

    keep_cols = ['model_key', 'dataset', 'seed']
    meta_cols = [c for c in ['exp_dir', 'checkpoint'] if c in df_long.columns]

    base = df_long.copy()
    # 去重：同一 (model, dataset, seed, metric) 取最后一次
    base = base.sort_values(['model_key', 'dataset', 'seed', 'metric']).drop_duplicates(
        subset=['model_key', 'dataset', 'seed', 'metric'], keep='last'
    )

    wide = base.pivot_table(index=keep_cols, columns='metric', values=value_col, aggfunc='mean').reset_index()
    # 合并 meta 信息：同一 (model, dataset, seed) 保留第一条
    metas = base.groupby(keep_cols)[meta_cols].agg(lambda x: x.iloc[0] if len(x) else "").reset_index() if meta_cols else None
    if metas is not None:
        wide = wide.merge(metas, on=keep_cols, how='left')

    # 列顺序：索引列 + 指标列 + meta
    metric_cols = [c for c in wide.columns if c not in keep_cols + meta_cols]
    wide = wide[keep_cols + metric_cols + meta_cols]
    return wide

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log_dir', required=True, help='日志根目录（包含各模型子目录的 .txt）')
    ap.add_argument('--out_dir', required=True, help='CSV 输出目录')
    ap.add_argument('--project_root', default='.', help='工程根目录（用于拼接相对 checkpoint 路径）')
    ap.add_argument('--line_pattern', default='Dev After Training', help='匹配哪一行（Dev After Training / Test After Training）')
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(args.project_root)

    rows_topk = []
    rows_ctr  = []

    txt_files = list(walk_logs(log_dir))
    if not txt_files:
        print(f'[WARN] 在 {log_dir} 下未发现 .txt 日志。')
        sys.exit(0)

    for p in txt_files:
        text = read_text(p)
        # model_key: 取 log 根下一层目录名（如 FinalMLPReImplTopK / FMCTR / BPRMFTopK）
        try:
            rel = p.relative_to(log_dir)
            parts = rel.parts
            model_key = parts[0] if len(parts) > 1 else p.stem
        except Exception:
            model_key = p.stem

        dataset = find_dataset(text)
        seed = find_seed(text, p)
        picked_line = find_line(text, args.line_pattern)
        if not picked_line:
            # 没找到指定行，跳过该日志
            # print(f'[SKIP] {p} 未找到行：{args.line_pattern}')
            continue

        metrics = parse_metric_line(picked_line)
        if not metrics:
            # print(f'[SKIP] {p} 指标解析为空：{picked_line}')
            continue

        mode = detect_mode(model_key, set(metrics.keys()))
        exp_dir = str(p.parent.resolve())
        ckpt = find_checkpoint(text, project_root)

        base_info = dict(
            model_key=model_key,
            dataset=dataset,
            seed=seed,
            exp_dir=exp_dir,
            checkpoint=ckpt
        )

        # 写入长表
        if mode == 'TopK':
            for k, v in metrics.items():
                rows_topk.append({**base_info, 'metric': k, 'value': v})
        elif mode == 'CTR':
            for k, v in metrics.items():
                rows_ctr.append({**base_info, 'metric': k, 'value': v})
        else:
            # 无法判断任务类型，则尝试通过 key 名再分流
            if any(k.lower().startswith(('hr@', 'ndcg@')) for k in metrics):
                for k, v in metrics.items():
                    rows_topk.append({**base_info, 'metric': k, 'value': v})
            else:
                for k, v in metrics.items():
                    rows_ctr.append({**base_info, 'metric': k, 'value': v})

    # 生成 DataFrame 并落盘
    def save_pair(rows, tag):
        if not rows:
            print(f'[INFO] {tag}: 无数据，跳过保存。')
            return
        df_long = pd.DataFrame(rows)
        df_wide = pivot_wide(df_long, value_col='value')
        long_csv = out_dir / f'{tag.lower()}_long.csv'
        wide_csv = out_dir / f'{tag.lower()}_wide.csv'
        df_long.to_csv(long_csv, index=False, encoding='utf-8-sig')
        df_wide.to_csv(wide_csv, index=False, encoding='utf-8-sig')
        print(f'[OK] 保存 {tag}：\n  - {long_csv}\n  - {wide_csv}')

    save_pair(rows_topk, 'TOPK')
    save_pair(rows_ctr,  'CTR')

if __name__ == '__main__':
    main()
