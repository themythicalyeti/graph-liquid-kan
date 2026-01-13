#!/usr/bin/env python3
"""
Package data for Google Colab deployment.

Creates a zip file containing:
- tensors.npz (features, targets, masks)
- spatial_graph.pt (edge_index, degree)

Upload the resulting glkan_data.zip to Google Drive root folder.
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

def main():
    print("=" * 60)
    print("PACKAGE DATA FOR GOOGLE COLAB")
    print("=" * 60)

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
    OUTPUT_DIR = PROJECT_ROOT / "colab_package"

    TENSOR_PATH = DATA_DIR / "tensors.npz"
    GRAPH_PATH = DATA_DIR / "spatial_graph.pt"

    # Check if data exists
    print("\n1. Checking data files...")

    if not TENSOR_PATH.exists():
        print(f"   [ERROR] {TENSOR_PATH} not found")
        print("   Run Phase 2 data pipeline first to generate tensors.npz")
        return False

    if not GRAPH_PATH.exists():
        print(f"   [ERROR] {GRAPH_PATH} not found")
        print("   Run Phase 2 data pipeline first to generate spatial_graph.pt")
        return False

    print(f"   [OK] Found: {TENSOR_PATH}")
    print(f"   [OK] Found: {GRAPH_PATH}")

    # Load and verify data
    print("\n2. Verifying data integrity...")

    try:
        data = np.load(TENSOR_PATH, allow_pickle=True)
        X = data['X']
        Y = data['Y']
        mask = data['mask']

        print(f"   Tensors:")
        print(f"     X (features): {X.shape} - {X.dtype}")
        print(f"     Y (targets):  {Y.shape} - {Y.dtype}")
        print(f"     mask:         {mask.shape} - {mask.dtype}")
        print(f"     Time steps:   {X.shape[0]}")
        print(f"     Nodes:        {X.shape[1]}")
        print(f"     Features:     {X.shape[2]}")
        print(f"     Observations: {mask.sum()} ({100*mask.sum()/(mask.shape[0]*mask.shape[1]):.1f}%)")

    except Exception as e:
        print(f"   [ERROR] ERROR loading tensors: {e}")
        return False

    try:
        graph = torch.load(GRAPH_PATH, weights_only=False)
        edge_index = graph['edge_index']

        print(f"   Graph:")
        print(f"     edge_index: {edge_index.shape}")
        print(f"     Edges:      {edge_index.shape[1]}")

    except Exception as e:
        print(f"   [ERROR] ERROR loading graph: {e}")
        return False

    # Check for data issues
    print("\n3. Data quality checks...")

    # Check for NaN/Inf
    has_nan_x = np.isnan(X).any()
    has_nan_y = np.isnan(Y).any()
    has_inf_x = np.isinf(X).any()
    has_inf_y = np.isinf(Y).any()

    if has_nan_x or has_nan_y:
        print(f"   [WARN] WARNING: NaN values found in data")
    if has_inf_x or has_inf_y:
        print(f"   [WARN] WARNING: Inf values found in data")

    if not (has_nan_x or has_nan_y or has_inf_x or has_inf_y):
        print(f"   [OK] No NaN/Inf values")

    # Check value ranges
    print(f"   X range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"   Y range: [{Y.min():.4f}, {Y.max():.4f}]")

    # Check outbreak statistics
    adult_female = Y[:, :, 0]
    valid_af = adult_female[mask]
    n_outbreaks = (valid_af > 0.5).sum()
    print(f"   Outbreaks (>0.5): {n_outbreaks} ({100*n_outbreaks/len(valid_af):.2f}%)")

    # Create output directory
    print("\n4. Creating package...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy files to staging
    staging_dir = OUTPUT_DIR / "staging"
    staging_dir.mkdir(exist_ok=True)

    shutil.copy(TENSOR_PATH, staging_dir / "tensors.npz")
    shutil.copy(GRAPH_PATH, staging_dir / "spatial_graph.pt")

    # Create zip file
    timestamp = datetime.now().strftime("%Y%m%d")
    zip_name = f"glkan_data.zip"
    zip_path = OUTPUT_DIR / zip_name

    # Remove old zip if exists
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(staging_dir / "tensors.npz", "tensors.npz")
        zf.write(staging_dir / "spatial_graph.pt", "spatial_graph.pt")

    # Get file sizes
    zip_size = zip_path.stat().st_size / (1024 * 1024)  # MB
    tensor_size = TENSOR_PATH.stat().st_size / (1024 * 1024)
    graph_size = GRAPH_PATH.stat().st_size / (1024 * 1024)

    # Cleanup staging
    shutil.rmtree(staging_dir)

    print(f"   [OK] Created: {zip_path}")
    print(f"   Size: {zip_size:.2f} MB (compressed)")
    print(f"   Contents:")
    print(f"     - tensors.npz ({tensor_size:.2f} MB)")
    print(f"     - spatial_graph.pt ({graph_size:.2f} MB)")

    # Create info file
    info_path = OUTPUT_DIR / "README.txt"
    with open(info_path, 'w') as f:
        f.write("GLKAN Data Package for Google Colab\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Contents:\n")
        f.write(f"  - tensors.npz: Features, targets, masks\n")
        f.write(f"  - spatial_graph.pt: Graph topology\n\n")
        f.write("Data Shape:\n")
        f.write(f"  - X (features): {X.shape}\n")
        f.write(f"  - Y (targets): {Y.shape}\n")
        f.write(f"  - mask: {mask.shape}\n")
        f.write(f"  - edges: {edge_index.shape[1]}\n\n")
        f.write("Instructions:\n")
        f.write("  1. Upload glkan_data.zip to Google Drive root\n")
        f.write("  2. Open deployment.ipynb in Google Colab\n")
        f.write("  3. Run all cells\n")

    print(f"   [OK] Created: {info_path}")

    # Summary
    print("\n" + "=" * 60)
    print("PACKAGE READY")
    print("=" * 60)
    print(f"\nOutput: {zip_path}")
    print(f"Size:   {zip_size:.2f} MB")
    print(f"\nNext steps:")
    print(f"  1. Upload {zip_name} to Google Drive root folder")
    print(f"  2. Open deployment.ipynb in Google Colab")
    print(f"  3. Select GPU runtime (T4 or A100)")
    print(f"  4. Run all cells")
    print(f"\nData summary:")
    print(f"  - {X.shape[0]} time steps")
    print(f"  - {X.shape[1]} nodes (farms)")
    print(f"  - {X.shape[2]} features")
    print(f"  - {edge_index.shape[1]} edges")
    print(f"  - {mask.sum()} observations")
    print(f"  - {n_outbreaks} outbreaks ({100*n_outbreaks/len(valid_af):.2f}%)")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
