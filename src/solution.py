"""Team solution for monthly retail revenue forecasting (March 2025).

Pipeline:
  Step 0: rescue_jan + rescue_feb cascade — fix anomalous monthly values
  Step 1: bag6_median — ensemble of 6 feb-heavy formulas at peak mult 1.131
  Step 2: HuberRegressor on cleaned per-store features (static + time-varying)
  Step 3: Additive blend (w=0.15): final = bag6 × (w × correction + (1-w))

Key insights baked in:
  - rescue cascade (jan first, then feb) cleans broken inputs
  - bag6 averaging across 6 weighting schemes for noise reduction
  - HuberRegressor is robust to outliers (vs Ridge L2)
  - OHE for categoricals + StandardScaler for numerics
  - Train Huber on combined 2023+2024 deviations
  - Target clipping ±0.9 prevents extreme training samples
"""
from __future__ import annotations
import argparse
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================
parser = argparse.ArgumentParser(description="Forecast March store revenue from the supplied monthly training CSV.")
parser.add_argument("--input", type=Path, default=Path("data/train.csv"), help="Path to the permitted training CSV")
parser.add_argument("--output", type=Path, default=Path("outputs/submission.csv"), help="Where to write predictions")
args = parser.parse_args()
SRC = args.input
OUT = args.output
if SRC.resolve() == OUT.resolve():
    parser.error("input and output paths must be different")
RTO = "РТО"
ID = "new_id"

MULT_2024 = 1.111      # bag6 multiplier for Mar 2024 (validation)
MULT_2025 = 1.131      # bag6 multiplier for Mar 2025 (production peak from empirical)
TARGET_CLIP_VAL = 0.9  # clip log_ratio target to ±0.9
HUBER_ALPHA = 100.0    # Huber L2-style regularization
HUBER_EPSILON = 1.35   # standard
BLEND_WEIGHT = 0.15    # weight on Huber correction (small = trust baseline)


# ============================================================
# DATA LOADING
# ============================================================
print("Loading data...")
raw = pd.read_csv(SRC)
raw["yyyymm"] = raw["Год"] * 100 + raw["Месяц"]

def piv(month: int) -> pd.Series:
    return raw[raw["yyyymm"] == month].set_index(ID)[RTO]

mar23 = piv(202303); feb23 = piv(202302); jan23 = piv(202301)
mar24 = piv(202403); feb24 = piv(202402); jan24 = piv(202401); dec23 = piv(202312)
feb25 = piv(202502); jan25 = piv(202501); dec24 = piv(202412)


# ============================================================
# STEP 0: RESCUE CASCADE (jan first, then feb)
# ============================================================
print("\nStep 0: Rescue cascade (jan → feb)...")

def rescue_month(target: pd.Series, anchor: pd.Series,
                 lo_thr=0.6, hi_thr=1.5,
                 normal_lo=0.7, normal_hi=1.3,
                 default_ratio=0.9) -> pd.Series:
    """Detect broken target values vs anchor, replace with anchor × pop median ratio."""
    if anchor is None: return target
    anchor_aligned = anchor.reindex(target.index)
    ratio = target / anchor_aligned
    normal_mask = (ratio > normal_lo) & (ratio < normal_hi)
    median_transition = ratio[normal_mask].median()
    if pd.isna(median_transition): median_transition = default_ratio
    broken_mask = (target < (lo_thr * anchor_aligned)) | (target > (hi_thr * anchor_aligned))
    broken_mask = broken_mask.fillna(False)
    fixed = target.copy()
    fixed.loc[broken_mask] = anchor_aligned.loc[broken_mask] * median_transition
    return fixed

# Step 0a: rescue Jan (using Dec as anchor).
jan24_fixed = rescue_month(jan24, dec23)
jan25_fixed = rescue_month(jan25, dec24)
# jan23 has no dec22 → no rescue.

# Step 0b: rescue Feb (using FIXED Jan as anchor).
feb23_fixed = rescue_month(feb23, jan23)
feb24_fixed = rescue_month(feb24, jan24_fixed)
feb25_fixed = rescue_month(feb25, jan25_fixed)
print("  Rescue cascade complete.")


# ============================================================
# STEP 1: BAG6 — 6 feb-heavy formulas at target effective mult
# ============================================================
def get_bag6(feb_data: pd.Series, jan_data: pd.Series, target_mult: float) -> pd.Series:
    """Median ensemble of 6 weighted feb+jan formulas, calibrated to target effective mult on feb."""
    formulas = {
        "feb":      feb_data,
        "wfj_2":    (feb_data + jan_data) / 2,
        "w2f1j_3":  (2 * feb_data + jan_data) / 3,
        "w3f1j_4":  (3 * feb_data + jan_data) / 4,
        "w4f1j_5":  (4 * feb_data + jan_data) / 5,
        "w5f1j_6":  (5 * feb_data + jan_data) / 6,
    }
    ratios = {n: float((f / feb_data).median()) for n, f in formulas.items()}
    preds = []
    for name, formula in formulas.items():
        mult = target_mult / ratios[name]
        preds.append((formula * mult).values)
    bag_array = np.column_stack(preds)
    return pd.Series(np.median(bag_array, axis=1), index=feb_data.index)


# ============================================================
# STEP 2: BUILD FEATURES with proper OHE
# ============================================================
print("\nStep 2: Building features (5 numeric + 2 categorical OHE + 3 time-varying lags)...")

static_num = ["Численность населения", "Количество домохозяйств",
              "Трафик пеший, в час", "Трафик авто, в час", "Количество касс"]
static_cat = ["Регион", "Торговая площадь, категориальный"]
tv_cols = ["Среднее количество промо товаров в чеке",
           "Среднее количество товаров в чеке",
           "Среднее количество отмен"]

def get_feature_row(month: int, year_marker: int) -> pd.DataFrame:
    snap = raw[raw["yyyymm"] == month].set_index(ID)
    df = snap[static_num + static_cat].copy()
    df['year_marker'] = year_marker
    year = month // 100
    months_back = [month,
                   month - 1 if month % 100 > 1 else (year - 1) * 100 + 12,
                   month - 2 if month % 100 > 2 else (year - 1) * 100 + (10 + month % 100)]
    for col in tv_cols:
        for i, m in enumerate(months_back):
            try:
                vals = raw[raw["yyyymm"] == m].set_index(ID)[col]
                df[f"{col}_t{i}"] = vals
            except Exception: pass
        col_t0 = f"{col}_t0"; col_t1 = f"{col}_t1"
        if col_t0 in df.columns and col_t1 in df.columns:
            df[f"{col}_change"] = df[col_t0] - df[col_t1]
    for c in static_cat:
        if c in df.columns: df[c] = df[c].astype(str).fillna("Unknown")
    return df

feat23_raw = get_feature_row(202302, 23)
feat24_raw = get_feature_row(202402, 24)
feat25_raw = get_feature_row(202502, 25)

# Combine for OHE (so columns match across years).
all_feats = pd.concat([feat23_raw, feat24_raw, feat25_raw], axis=0)
all_feats = pd.get_dummies(all_feats, columns=static_cat, drop_first=True)

feat23 = all_feats[all_feats['year_marker'] == 23].drop('year_marker', axis=1).fillna(0)
feat24 = all_feats[all_feats['year_marker'] == 24].drop('year_marker', axis=1).fillna(0)
feat25 = all_feats[all_feats['year_marker'] == 25].drop('year_marker', axis=1).fillna(0)
print(f"  Feature shape: {feat25.shape}")


# ============================================================
# STEP 3: TRAIN HUBER on combined 2023+2024 deviations
# ============================================================
print(f"\nStep 3: Train Huber (alpha={HUBER_ALPHA}, epsilon={HUBER_EPSILON})...")

log_ratio_23 = np.log(mar23 / feb23_fixed).dropna().clip(-TARGET_CLIP_VAL, TARGET_CLIP_VAL)
log_ratio_24 = np.log(mar24 / feb24_fixed).dropna().clip(-TARGET_CLIP_VAL, TARGET_CLIP_VAL)

idx23 = log_ratio_23.index.intersection(feat23.index)
idx24 = log_ratio_24.index.intersection(feat24.index)
idx25 = feat25.index.intersection(feb25_fixed.index)

y23 = log_ratio_23.loc[idx23].values - log_ratio_23.mean()
y24 = log_ratio_24.loc[idx24].values - log_ratio_24.mean()

X23 = feat23.loc[idx23]; X24 = feat24.loc[idx24]; X25 = feat25.loc[idx25]

# LOCAL VALIDATION: train on 23, predict 24
scaler_val = StandardScaler()
X23_sc = scaler_val.fit_transform(X23.values)
X24_sc = scaler_val.transform(X24.values)
model_val = HuberRegressor(alpha=HUBER_ALPHA, epsilon=HUBER_EPSILON, max_iter=500)
model_val.fit(X23_sc, y23)
log_dev_24 = model_val.predict(X24_sc)
correction_24 = np.exp(log_dev_24)
bag6_24 = get_bag6(feb24_fixed, jan24_fixed, MULT_2024).loc[idx24].values
blended_24 = BLEND_WEIGHT * correction_24 + (1 - BLEND_WEIGHT) * 1.0
final_24 = bag6_24 * blended_24
real_24 = mar24.loc[idx24].values
mape_24 = np.mean(np.abs((real_24 - final_24) / real_24)) * 100
print(f"  LOCAL VAL MAPE (Mar 2024): {mape_24:.4f}%")
print(f"  Local score: {100 * ((100 - mape_24) / 100)**2:.2f}")

# PRODUCTION: train on 23+24 combined, predict 25
print("\nStep 4: Train final on combined 2023+2024...")
X_train = pd.concat([X23, X24], axis=0).values
y_train = np.concatenate([y23, y24])
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X25_sc = scaler.transform(X25.values)
model_final = HuberRegressor(alpha=HUBER_ALPHA, epsilon=HUBER_EPSILON, max_iter=500)
model_final.fit(X_train_sc, y_train)
log_dev_25 = model_final.predict(X25_sc)
correction_25 = np.exp(log_dev_25)


# ============================================================
# STEP 4: BLEND bag6 with Huber correction
# ============================================================
print(f"\nStep 5: Blend bag6 with Huber correction (w={BLEND_WEIGHT})...")
bag6_25 = get_bag6(feb25_fixed, jan25_fixed, MULT_2025).loc[idx25].values
blended_25 = BLEND_WEIGHT * correction_25 + (1 - BLEND_WEIGHT) * 1.0
final_pred = bag6_25 * blended_25


# ============================================================
# STEP 5: SAVE submission
# ============================================================
OUT.parent.mkdir(parents=True, exist_ok=True)
print(f"\nStep 6: Saving to {OUT}...")
out = pd.DataFrame({ID: idx25.values, "rto": final_pred}).sort_values(ID).reset_index(drop=True)
out.to_csv(OUT, index=False)
print(f"  Wrote {OUT}  rows={len(out)}  size={OUT.stat().st_size} bytes")


