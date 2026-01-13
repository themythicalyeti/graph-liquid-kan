#!/usr/bin/env python3
"""
Transfer environmental features from v4 to v5.

V4 has actual environmental data (temperature, salinity, currents)
while v5's extraction failed. This script maps v4's features to v5's
tensor format for the common farms.

V4 format: (N_farms, T_days, F_features) = (1122, 1096, 7)
V5 format: (T_days, N_farms, F_features) = (366, 1777, 8)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime

def main():
    print("=" * 60)
    print("TRANSFER V4 ENVIRONMENTAL FEATURES TO V5")
    print("=" * 60)

    # Paths
    V4_TENSOR_PATH = Path("../v4/data/processed/tensors.npz")
    V5_TENSOR_PATH = Path("data/processed/tensors.npz")
    OUTPUT_PATH = Path("data/processed/tensors.npz")
    BACKUP_PATH = Path("data/processed/tensors_empty_backup.npz")

    # Load both datasets
    print("\n1. Loading datasets...")
    v4 = np.load(V4_TENSOR_PATH, allow_pickle=True)
    v5 = np.load(V5_TENSOR_PATH, allow_pickle=True)

    print(f"   V4: X={v4['X'].shape}, site_ids={len(v4['site_ids'])}")
    print(f"   V5: X={v5['X'].shape}, site_ids={len(v5['site_ids'])}")

    # Get arrays
    v4_X = v4['X']  # (N, T, F) = (1122, 1096, 7)
    v4_Y = v4['Y']  # (N, T, 3)
    v4_mask = v4['mask']  # (N, T)
    v4_site_ids = v4['site_ids']
    v4_dates = pd.to_datetime(v4['dates'])

    v5_X = v5['X']  # (T, N, F) = (366, 1777, 8)
    v5_Y = v5['Y']  # (T, N, 3)
    v5_mask = v5['mask']  # (T, N)
    v5_site_ids = v5['site_ids']
    v5_times = pd.to_datetime(v5['times'])

    print(f"\n   V4 date range: {v4_dates[0]} to {v4_dates[-1]}")
    print(f"   V5 date range: {v5_times[0]} to {v5_times[-1]}")

    # Create ID mapping
    print("\n2. Creating farm ID mapping...")
    v4_id_to_idx = {int(sid): i for i, sid in enumerate(v4_site_ids)}
    v5_id_to_idx = {int(sid): i for i, sid in enumerate(v5_site_ids)}

    common_ids = set(v4_id_to_idx.keys()) & set(v5_id_to_idx.keys())
    print(f"   Common farms: {len(common_ids)}")
    print(f"   V4-only farms: {len(v4_id_to_idx) - len(common_ids)}")
    print(f"   V5-only farms: {len(v5_id_to_idx) - len(common_ids)}")

    # Create date mapping (V5 dates -> V4 time indices)
    print("\n3. Creating date mapping...")
    v4_date_to_idx = {d.date(): i for i, d in enumerate(v4_dates)}

    # Find 2024 indices in V4
    v4_2024_start = None
    v4_2024_end = None
    for i, d in enumerate(v4_dates):
        if d.year == 2024:
            if v4_2024_start is None:
                v4_2024_start = i
            v4_2024_end = i

    if v4_2024_start is not None:
        print(f"   V4 2024 indices: [{v4_2024_start}, {v4_2024_end}]")
        print(f"   V4 2024 days: {v4_2024_end - v4_2024_start + 1}")

    # Create new X tensor
    print("\n4. Transferring environmental features...")

    # V4 has 7 features, V5 expects 8
    # V4 features: [temp, salinity, u, v, current_speed, ?, ?] - check actual
    # V5 features: [temp, salinity, current_speed, sin, cos, treatment1, treatment2, treatment3]

    # First, check v4 feature statistics
    print("\n   V4 feature statistics (for common farms, 2024 data):")
    for f in range(v4_X.shape[2]):
        feat = v4_X[:, v4_2024_start:v4_2024_end+1, f]
        valid = ~np.isnan(feat) & (feat != 0)
        if valid.sum() > 0:
            print(f"     Feature {f}: mean={np.nanmean(feat[valid]):.3f}, std={np.nanstd(feat[valid]):.3f}, coverage={100*valid.sum()/feat.size:.1f}%")
        else:
            print(f"     Feature {f}: NO VALID DATA")

    # Initialize new X tensor with zeros
    new_X = np.zeros_like(v5_X)
    transfer_count = 0
    feature_count = 0

    # Map features from V4 to V5
    # V4: (N, T, F=7) -> V5: (T, N, F=8)
    # We'll use the first 5 environmental features from V4

    for farm_id in common_ids:
        v4_idx = v4_id_to_idx[farm_id]
        v5_idx = v5_id_to_idx[farm_id]

        for v5_t_idx, v5_date in enumerate(v5_times):
            v4_t_idx = v4_date_to_idx.get(v5_date.date())

            if v4_t_idx is not None:
                # Copy environmental features (first 5)
                v4_features = v4_X[v4_idx, v4_t_idx, :5]

                # Only copy non-NaN values
                valid_mask = ~np.isnan(v4_features)
                if valid_mask.any():
                    new_X[v5_t_idx, v5_idx, :5] = np.nan_to_num(v4_features, nan=0.0)
                    feature_count += valid_mask.sum()
                    transfer_count += 1

    print(f"\n   Transferred {transfer_count} (time, farm) pairs")
    print(f"   Total features transferred: {feature_count}")

    # Keep V5's treatment features (last 3 features)
    # They should be in positions 5, 6, 7
    print("\n5. Preserving V5 treatment features...")
    # v5_X already has treatment features in positions 5-7 (if any)
    # But since v5_X is all zeros, we need to check the original builder logic

    # Check if v5 had any non-zero treatment features
    original_treatments = v5_X[:, :, 5:8].sum()
    print(f"   Original V5 treatment feature sum: {original_treatments}")

    # If treatments were stored, keep them
    if original_treatments > 0:
        new_X[:, :, 5:8] = v5_X[:, :, 5:8]

    # Verify new tensor
    print("\n6. Verifying new tensor...")
    print(f"   New X shape: {new_X.shape}")
    print(f"   New X range: [{new_X.min():.4f}, {new_X.max():.4f}]")
    print(f"   New X mean: {new_X.mean():.4f}")
    print(f"   Non-zero elements: {(new_X != 0).sum()} ({100*(new_X != 0).sum()/new_X.size:.1f}%)")

    # Feature-by-feature stats
    print("\n   Per-feature statistics:")
    feature_names = ['temp', 'salinity', 'current_speed', 'current_sin', 'current_cos',
                     'treat_mech', 'treat_med', 'treat_fish']
    for f in range(new_X.shape[2]):
        feat = new_X[:, :, f]
        valid = feat != 0
        if valid.sum() > 0:
            name = feature_names[f] if f < len(feature_names) else f'feat_{f}'
            print(f"     {name}: mean={feat[valid].mean():.3f}, std={feat[valid].std():.3f}, coverage={100*valid.sum()/feat.size:.1f}%")
        else:
            name = feature_names[f] if f < len(feature_names) else f'feat_{f}'
            print(f"     {name}: ALL ZEROS")

    # Normalize features
    print("\n7. Normalizing features...")
    feature_mean = np.zeros(new_X.shape[2])
    feature_std = np.ones(new_X.shape[2])

    for f in range(new_X.shape[2]):
        feat = new_X[:, :, f]
        valid = feat != 0
        if valid.sum() > 100:  # Need enough samples
            feature_mean[f] = feat[valid].mean()
            feature_std[f] = feat[valid].std() + 1e-8

    # Apply normalization
    new_X_norm = (new_X - feature_mean) / feature_std
    new_X_norm = np.nan_to_num(new_X_norm, nan=0.0, posinf=0.0, neginf=0.0)
    new_X_norm = np.clip(new_X_norm, -10, 10)

    print(f"   Feature means: {feature_mean}")
    print(f"   Feature stds: {feature_std}")

    # Backup original and save new
    print("\n8. Saving tensors...")

    # Backup original (empty) tensors
    np.savez(
        BACKUP_PATH,
        X=v5_X,
        Y=v5_Y,
        mask=v5_mask,
        times=v5_times.to_numpy(),
        site_ids=v5_site_ids,
    )
    print(f"   Backed up original to: {BACKUP_PATH}")

    # Save new tensors
    np.savez(
        OUTPUT_PATH,
        X=new_X_norm.astype(np.float32),
        Y=v5_Y,
        mask=v5_mask,
        times=v5_times.to_numpy(),
        site_ids=v5_site_ids,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    print(f"   Saved new tensors to: {OUTPUT_PATH}")

    # Final verification
    print("\n" + "=" * 60)
    print("TRANSFER COMPLETE")
    print("=" * 60)

    # Reload and verify
    verify = np.load(OUTPUT_PATH, allow_pickle=True)
    print(f"\nVerification:")
    print(f"  X shape: {verify['X'].shape}")
    print(f"  X range: [{verify['X'].min():.3f}, {verify['X'].max():.3f}]")
    print(f"  X mean: {verify['X'].mean():.3f}")
    print(f"  X std: {verify['X'].std():.3f}")
    print(f"  Non-zero: {(verify['X'] != 0).sum() / verify['X'].size * 100:.1f}%")

    print("\nNow you can run training with actual environmental features!")
    print("  python scripts/train_glkan.py --max-epochs 50")


if __name__ == "__main__":
    main()
