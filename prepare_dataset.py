"""
Script to prepare the Zenodo pointing gesture dataset.
======================================================
Dataset format (no header row, 22 columns):
  col 0       : label (0 = no pointing, 1 = pointing)
  cols 1-21   : Euclidean distances of MediaPipe landmarks 0-20 from the wrist
                (col 1 = wrist = always 0.0, will be dropped since there is no signal)

Download hand-gestures.csv from:
  https://zenodo.org/records/16420298
Save it to: data/hand-gestures.csv
Then run this script to generate data/gesture_data.csv for training.
"""

import pandas as pd
import numpy as np
import os

SRC  = "data/hand-gestures.csv"
DEST = "data/gesture_data.csv"

assert os.path.exists(SRC), (
    f"'{SRC}' not found!\n"
    f"Download from: https://zenodo.org/records/16420298"
)

df = pd.read_csv(SRC, header=None, on_bad_lines="skip")
print(f"Loaded {len(df):,} samples, {df.shape[1]} columns")

# Col 0 = label, cols 1–21 = landmark distances
col_names = ["label"] + [f"dist_{i}" for i in range(21)]
df.columns = col_names

# ── Check class balance ────────────────────────────────────────────────────────
counts = df["label"].value_counts().sort_index()
print(f"\nClass 0 (no pointing) : {counts.get(0, 0):>6} samples")
print(f"Class 1 (pointing)    : {counts.get(1, 0):>6} samples")
print(f"Total                 : {len(df):>6} samples")

if len(counts) < 2:
    raise ValueError("Dataset only has 1 class — check the source file.")

imbalance_ratio = counts.min() / counts.max()
print(f"\nClass balance ratio   : {imbalance_ratio:.2f}  (1.0 = perfectly balanced)")
if imbalance_ratio < 0.7:
    print("  WARNING: Extreme imbalance detected.")
    print("  Consider: oversampling the minority class, or using class_weight='balanced' in train_model.py")
elif imbalance_ratio < 0.9:
    print("  INFO: Slight imbalance, but still safe for standard MLP/RF.")
else:
    print("  OK: Class distribution is balanced.")

# ── Remove dist_0 (wrist → wrist, always 0.0, zero signal) ────────────────────

if "dist_0" in df.columns:
    zero_variance = df["dist_0"].std()
    print(f"\ndist_0 std dev        : {zero_variance:.6f}  (should be 0.0 or very small)")
    df = df.drop(columns=["dist_0"])
    print("dist_0 removed → 20 active features remaining")

# ── Feature statistics summary (early anomaly detection) ──────────────────────
feat_names = [c for c in df.columns if c != "label"]
feat_data  = df[feat_names]

print(f"\nSummary of {len(feat_names)} active features:")
print(f"  Global min value : {feat_data.values.min():.4f}")
print(f"  Global max value : {feat_data.values.max():.4f}")
print(f"  Global mean      : {feat_data.values.mean():.4f}")

# Warning if there are values outside a reasonable range (0 - ~1.5 for normalized distances)
n_outliers = (feat_data.values > 2.0).sum()
if n_outliers > 0:
    print(f"  WARNING: {n_outliers} values > 2.0 found — check source dataset normalization")

# ── Save ───────────────────────────────────────────────────────────────────────
df_out = df[feat_names + ["label"]].dropna()

os.makedirs("data", exist_ok=True)
df_out.to_csv(DEST, index=False)

print(f"\nSaved → {DEST}  ({len(df_out):,} rows, {len(feat_names)} features + label)")
print(f"Dropped rows (NaN) : {len(df) - len(df_out)}")
print("Next: python train_model.py")