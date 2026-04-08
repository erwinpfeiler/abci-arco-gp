#!/usr/bin/env python3
"""
Debug helper for ABCI benchmark result folders.

What it does:
1. Scans a results directory for CSV files.
2. Reproduces the current plotting.py selection logic.
3. Separately parses filenames with a safer regex-based parser.
4. Reads every CSV and reports row counts, columns, duplicates, and selected files.
5. Writes a manifest CSV for manual inspection.

Usage:
    python debug_benchmark_results.py /path/to/results_dir --target-exp 50
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd


EXP_RE = re.compile(r"^(?P<prefix>.+)-exp-(?P<exp>\d+)\.csv$")
COS_RE = re.compile(r"^(?P<prefix>.+)-(?P<exp>\d+)-cos\.csv$")


@dataclass
class FileRecord:
    basename: str
    abspath: str
    n_rows: Optional[int]
    n_cols: Optional[int]
    columns: str
    plotting_env_id: Optional[str]
    plotting_run_id: Optional[str]
    plotting_exp_num: Optional[int]
    plotting_result_type: Optional[str]
    regex_prefix: Optional[str]
    regex_exp_num: Optional[int]
    regex_result_type: Optional[str]
    selected_by_plotting: bool
    row_count_matches_target: Optional[bool]


def plotting_parse_file_name(filename: str):
    """Exact copy of current plotting.py parser logic."""
    tokens = filename.split('-')
    result_type = 'default'
    if len(tokens) >= 2 and tokens[-2] == 'exp':
        run_id = tokens[-3]
        env_id = tokens[-4]
        exp_num = int(tokens[-1][:-4])
    elif len(tokens) >= 1 and tokens[-1] == 'cos.csv':
        run_id = tokens[-3]
        env_id = tokens[-4]
        exp_num = int(tokens[-2])
        result_type = 'cos_variance'
    else:
        run_id = tokens[-1][:-4]
        env_id = tokens[-2]
        exp_num = 1
    return env_id, run_id, exp_num, result_type


def regex_parse_file_name(filename: str):
    m = EXP_RE.match(filename)
    if m:
        return m.group('prefix'), int(m.group('exp')), 'default'
    m = COS_RE.match(filename)
    if m:
        return m.group('prefix'), int(m.group('exp')), 'cos_variance'
    return None, None, None


def safe_read_csv(path: str):
    try:
        df = pd.read_csv(path)
        return len(df), len(df.columns), ','.join(df.columns)
    except Exception as e:
        return None, None, f"<READ ERROR: {type(e).__name__}: {e}>"


def build_manifest(results_dir: str, target_exp: int):
    records: list[FileRecord] = []
    results_dir = os.path.abspath(results_dir)

    for entry in sorted(os.scandir(results_dir), key=lambda x: x.name):
        if not entry.is_file() or not entry.name.endswith('.csv'):
            continue

        n_rows, n_cols, columns = safe_read_csv(entry.path)

        try:
            penv, prun, pexp, pres = plotting_parse_file_name(entry.name)
            selected_by_plotting = (pexp == target_exp and pres == 'default')
        except Exception:
            penv = prun = pres = None
            pexp = None
            selected_by_plotting = False

        rprefix, rexp, rres = regex_parse_file_name(entry.name)

        records.append(
            FileRecord(
                basename=entry.name,
                abspath=os.path.abspath(entry.path),
                n_rows=n_rows,
                n_cols=n_cols,
                columns=columns,
                plotting_env_id=penv,
                plotting_run_id=prun,
                plotting_exp_num=pexp,
                plotting_result_type=pres,
                regex_prefix=rprefix,
                regex_exp_num=rexp,
                regex_result_type=rres,
                selected_by_plotting=selected_by_plotting,
                row_count_matches_target=(None if n_rows is None else n_rows == target_exp),
            )
        )

    return records


def print_hist(counter: collections.Counter, title: str, limit: int = 40):
    print(f"\n{title}")
    if not counter:
        print("  <empty>")
        return
    for idx, (key, val) in enumerate(sorted(counter.items(), key=lambda kv: (str(kv[0])))):
        if idx >= limit:
            print(f"  ... and {len(counter) - limit} more")
            break
        print(f"  {key}: {val}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results_dir', help='Directory containing benchmark csv files')
    ap.add_argument('--target-exp', type=int, default=50, help='Experiment index the notebook is trying to load')
    ap.add_argument('--manifest-out', default=None, help='Optional path for manifest csv')
    args = ap.parse_args()

    records = build_manifest(args.results_dir, args.target_exp)
    if not records:
        print(f"No CSV files found in {args.results_dir}")
        sys.exit(1)

    print(f"Scanned {len(records)} CSV files in {os.path.abspath(args.results_dir)}")

    plotting_exp_hist = collections.Counter(r.plotting_exp_num for r in records if r.plotting_exp_num is not None)
    regex_exp_hist = collections.Counter(r.regex_exp_num for r in records if r.regex_exp_num is not None)
    print_hist(plotting_exp_hist, 'Histogram of experiment numbers seen by plotting.py parser:')
    print_hist(regex_exp_hist, 'Histogram of experiment numbers seen by regex parser:')

    selected = [r for r in records if r.selected_by_plotting]
    print(f"\nFiles selected by current plotting.py for target exp={args.target_exp}: {len(selected)}")
    for r in selected:
        print(
            f"  env={r.plotting_env_id!r:>12} run={r.plotting_run_id!r:>20} "
            f"rows={str(r.n_rows):>4} cols={str(r.n_cols):>3} file={r.basename}"
        )

    duplicate_keys = collections.defaultdict(list)
    for r in selected:
        duplicate_keys[(r.plotting_env_id, r.plotting_exp_num)].append(r)

    dup_count = 0
    print("\nDuplicate selections for the same (env_id, exp_num):")
    for key, vals in duplicate_keys.items():
        if len(vals) > 1:
            dup_count += 1
            print(f"  {key}: {len(vals)} files")
            for v in vals:
                print(f"    - {v.basename}")
    if dup_count == 0:
        print("  none")

    bad_rows = [r for r in selected if r.n_rows is not None and r.n_rows != args.target_exp]
    print(f"\nSelected files whose CSV row count != target exp ({args.target_exp}): {len(bad_rows)}")
    for r in bad_rows:
        print(f"  rows={r.n_rows:>4} file={r.basename}")

    weird = [r for r in records if r.plotting_exp_num is not None and r.regex_exp_num is not None and r.plotting_exp_num != r.regex_exp_num]
    print(f"\nFiles where plotting parser and regex parser disagree on exp number: {len(weird)}")
    for r in weird[:50]:
        print(f"  plotting={r.plotting_exp_num:>4} regex={r.regex_exp_num:>4} file={r.basename}")
    if len(weird) > 50:
        print(f"  ... and {len(weird) - 50} more")

    if args.manifest_out is None:
        args.manifest_out = str(Path(args.results_dir) / f'debug_manifest_exp_{args.target_exp}.csv')

    with open(args.manifest_out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    print(f"\nWrote manifest to: {args.manifest_out}")
    print("\nWhat to look for:")
    print("  1. If the histogram shows many exp numbers from 1..N, those are checkpoint files; the warning is only telling you they are being skipped.")
    print("  2. If selected files have row count < target-exp, then data export really is incomplete.")
    print("  3. If there are duplicate selected files per env, plotting.py is averaging duplicates silently.")
    print("  4. If plotting parser and regex parser disagree, filename parsing is still broken.")


"""
python /ceph/home/TUG/epfeiler-tug/abci-arco-gp/src/scripts/debug_benchmark_results.py \
  /ceph/home/TUG/epfeiler-tug/abci-arco-gp/results/BarabasiAlbert/10_nodes_debug/20260303_163047_test_arcogp_random \
  --target-exp 50
"""

if __name__ == '__main__':
    main()