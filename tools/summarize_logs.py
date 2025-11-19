# -*- coding: utf-8 -*-
"""
Robust log summarizer for ReChorus experiments.

Usage examples:
  python tools/summarize_logs.py --log_dir ./log --model_key FinalMLPReImplTopK
  python tools/summarize_logs.py --log_dir ./log --model_key FMCTR --line_pattern "Test After Training" --save_csv

It will:
  - Recursively scan log_dir for .txt/.log/.out files whose path contains model_key
  - Find the LAST line that matches "<line_pattern>: (<metrics...>)"
  - Parse metrics like "HR@5:0.3312", "AUC@All:0.78" (comma or semicolon separated)
  - Aggregate mean/std across logs, and optionally save to CSV
"""

import os
import re
import json
import argparse
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple

# ---------- Helpers ----------

METRIC_ITEM_RE = re.compile(r'\s*([A-Za-z_]+)@([^:]+):\s*([0-9]*\.?[0-9]+)\s*')

# Split by comma or semicolon, allowing whitespace
SPLIT_RE = re.compile(r'[;,]\s*')

# A line like: "Test After Training: (HR@5:0.33,NDCG@5:0.22, ...)"
def extract_metrics_from_line(line: str, pattern: str) -> Dict[str, float]:
    """
    Return a dict of metric_token -> value for a single matched line.
    metric_token is kept as "METRIC@K" exactly as in logs, e.g., "HR@5", "NDCG@10", "AUC@All", "LOG_LOSS@All".
    """
    # Find "(...)" after "<pattern>:"
    m = re.search(rf'{re.escape(pattern)}\s*:\s*\((.*?)\)\s*', line)
    if not m:
        return {}
    metrics_str = m.group(1).strip()
    parts = [p for p in SPLIT_RE.split(metrics_str) if p.strip()]
    out: Dict[str, float] = {}

    for part in parts:
        m2 = METRIC_ITEM_RE.match(part)
        if not m2:
            # silently skip unknown fragments
            continue
        metric_name, metric_at, metric_val = m2.group(1), m2.group(2), m2.group(3)
        token = f'{metric_name}@{metric_at}'.upper()  # unify to upper
        try:
            val = float(metric_val)
        except ValueError:
            continue
        out[token] = val
    return out


def find_last_metrics_in_file(path: Path, pattern: str) -> Tuple[Dict[str, float], str]:
    """
    Scan a file and return metrics from the LAST occurrence of the given pattern line.
    Returns (metrics_dict, raw_line_text)
    """
    last_line = None
    try:
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if pattern in line:
                    last_line = line.strip()
    except Exception:
        # Try gbk fallback for Windows if needed
        try:
            with path.open('r', encoding='gbk', errors='ignore') as f:
                for line in f:
                    if pattern in line:
                        last_line = line.strip()
        except Exception:
            last_line = None

    if not last_line:
        return {}, ''

    metrics = extract_metrics_from_line(last_line, pattern)
    return metrics, last_line


def collect_files(log_dir: Path, model_key: str) -> List[Path]:
    """
    Recursively collect log files whose path contains model_key and extension is txt/log/out.
    """
    exts = {'.txt', '.log', '.out'}
    files: List[Path] = []
    for p in log_dir.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts and model_key in str(p):
            files.append(p)
    return files


def aggregate(all_records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Given a list of metric dicts (one per file), aggregate to mean/std/n for each metric token.
    """
    bucket: Dict[str, List[float]] = {}
    for md in all_records:
        for k, v in md.items():
            bucket.setdefault(k, []).append(v)

    stats: Dict[str, Dict[str, float]] = {}
    for k, arr in bucket.items():
        if not arr:
            continue
        if len(arr) == 1:
            stats[k] = {'mean': arr[0], 'std': 0.0, 'n': 1}
        else:
            stats[k] = {'mean': mean(arr), 'std': pstdev(arr), 'n': float(len(arr))}
    return stats


def save_csv(out_path: Path, stats: Dict[str, Dict[str, float]], details: List[Tuple[str, Dict[str, float]]]) -> None:
    """
    Save both summary and per-file details to CSV files.
    """
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Summary CSV
        with out_path.open('w', encoding='utf-8', newline='') as f:
            f.write('metric,mean,std,n\n')
            for k, v in sorted(stats.items()):
                f.write(f'{k},{v["mean"]:.6f},{v["std"]:.6f},{int(v["n"])}\n')
        # Details CSV
        det_path = out_path.with_name(out_path.stem + '_details.csv')
        # Collect all metric keys
        keys = sorted({mk for _, md in details for mk in md.keys()})
        with det_path.open('w', encoding='utf-8', newline='') as f:
            f.write('file,' + ','.join(keys) + '\n')
            for fp, md in details:
                row = [fp] + [f'{md.get(k, "")}' for k in keys]
                f.write(','.join(row) + '\n')
        print(f'Saved summary to: {out_path}')
        print(f'Saved details to: {det_path}')
    except Exception as e:
        print(f'[WARN] CSV save failed: {e}')


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description='Summarize ReChorus logs (Test After Training).')
    parser.add_argument('--log_dir', type=str, required=True, help='Root log directory to scan.')
    parser.add_argument('--model_key', type=str, required=True, help='Substring to filter log files by model.')
    parser.add_argument('--line_pattern', type=str, default='Test After Training',
                        help='Line prefix/pattern to extract metrics from. Default: "Test After Training"')
    parser.add_argument('--save_csv', action='store_true', help='Also save summary and details CSV to log_dir/')
    args = parser.parse_args()

    log_dir = Path(args.log_dir).resolve()
    if not log_dir.exists():
        print(f'[ERROR] log_dir does not exist: {log_dir}')
        return

    files = collect_files(log_dir, args.model_key)
    print(f'Found {len(files)} log file(s) for key="{args.model_key}" under {log_dir}')

    records: List[Dict[str, float]] = []
    details: List[Tuple[str, Dict[str, float]]] = []

    for fp in sorted(files):
        metrics, raw = find_last_metrics_in_file(fp, args.line_pattern)
        if metrics:
            records.append(metrics)
            details.append((str(fp), metrics))

    if not records:
        print('[WARN] No metrics found. Check --model_key or --line_pattern or log content.')
        return

    stats = aggregate(records)

    # Print summary JSON
    print('\n=== Summary (mean/std/n) ===')
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # Print details
    print('\n=== Detailed rows ===')
    for fp, md in details:
        print(fp)
        # Consistent order: sort keys
        for k in sorted(md.keys()):
            print(f'  {k}: {md[k]:.6f}')

    # Optional CSV
    if args.save_csv:
        out_csv = log_dir / f'summary_{args.model_key}.csv'
        save_csv(out_csv, stats, details)


if __name__ == '__main__':
    main()
