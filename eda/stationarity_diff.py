import argparse
from pathlib import Path

import numpy as np
from statsmodels.tsa.stattools import adfuller


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/home/wmh/lob/TLOB/data/FI_2010")
    parser.add_argument("--feature-rows", type=int, default=144)
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--maxlag", type=int, default=1)
    return parser.parse_args()


def load_features(path, feature_rows, max_points, stride):
    arr = np.loadtxt(path)
    feats = arr[:feature_rows, :]
    if stride > 1:
        feats = feats[:, ::stride]
    if max_points and feats.shape[1] > max_points:
        feats = feats[:, :max_points]
    return feats


def stationarity_stats(mat, alpha, maxlag):
    passed = 0
    total = 0
    failed_idx = []
    for idx, row in enumerate(mat):
        if np.ptp(row) == 0:
            continue
        total += 1
        result = adfuller(row, maxlag=maxlag, autolag=None)
        if result[1] < alpha:
            passed += 1
        else:
            failed_idx.append(idx)
    ratio = passed / total if total else 0.0
    return ratio, failed_idx


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    files = sorted([p for p in data_dir.glob("*.txt") if p.is_file()])
    if not files:
        return
    print(f"data_dir={data_dir}")
    print(f"files={len(files)} feature_rows={args.feature_rows} max_points={args.max_points} stride={args.stride}")
    for path in files:
        feats = load_features(path, args.feature_rows, args.max_points, args.stride)
        diff = np.diff(feats, axis=1)
        pre, pre_failed = stationarity_stats(feats, args.alpha, args.maxlag)
        post, _ = stationarity_stats(diff, args.alpha, args.maxlag)
        print(f"{path.name}: shape={feats.shape} adf_pass={pre:.3f} diff_pass={post:.3f}")
        if pre_failed:
            print(f"  non_stationary_rows_before_diff={pre_failed}")


if __name__ == "__main__":
    main()

