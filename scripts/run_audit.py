#!/usr/bin/env python3
"""
Run Scientific Audit Tests on trained GLKAN model.

Tests:
1. Counterfactual: Temperature +5°C should increase lice growth
2. Long-Horizon: 90-day rollout should remain stable
3. Graphon: N vs 2N nodes should have <10% deviation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")

def main():
    logger.info("=" * 60)
    logger.info("SCIENTIFIC VALIDATION AUDIT")
    logger.info("=" * 60)

    # Load data (use reduced subset for speed)
    logger.info("\nLoading data...")
    data = np.load("data/processed/tensors.npz", allow_pickle=True)
    graph_data = torch.load("data/processed/spatial_graph.pt", weights_only=False)

    X = torch.from_numpy(data["X"]).float()
    Y = torch.from_numpy(data["Y"]).float()
    mask = torch.from_numpy(data["mask"]).bool()
    edge_index = graph_data["edge_index"]

    # Load feature indices for SeaLiceGLKAN
    if "feature_indices" in data:
        feature_indices = data["feature_indices"].item()
        logger.info(f"Loaded feature_indices: {list(feature_indices.keys())}")
    else:
        feature_indices = None
        logger.warning("No feature_indices found - using defaults")

    # Subset to 200 nodes for speed
    N_subset = 200
    X = X[:, :N_subset, :]
    Y = Y[:, :N_subset, :]
    mask = mask[:, :N_subset]

    valid_edges = (edge_index[0] < N_subset) & (edge_index[1] < N_subset)
    edge_index = edge_index[:, valid_edges]

    logger.info(f"Data: X={X.shape}, edges={edge_index.shape[1]}")

    # Create sample batch
    window = 30
    x_batch = X[:window].unsqueeze(0)  # (1, 30, N, F)
    y_batch = Y[:window].unsqueeze(0)
    mask_batch = mask[:window].unsqueeze(0)

    batch = {
        'x': x_batch,
        'y': y_batch,
        'mask': mask_batch,
        'edge_index': edge_index,
        'time_points': torch.linspace(0, 1, window),
        'feature_indices': feature_indices,
    }

    # Load or create model
    logger.info("\nLoading SeaLicePredictor model...")
    from src.models.sea_lice_network import SeaLicePredictor

    model = SeaLicePredictor(
        input_dim=X.shape[-1],
        hidden_dim=64,
        output_dim=3,
        n_bases=8,
        k_hops=3,
    )

    # Try to load checkpoint
    checkpoint_path = Path("checkpoints/best_model.pt")
    if checkpoint_path.exists():
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Checkpoint loaded successfully")
    else:
        logger.warning("No checkpoint found - using untrained model")

    model.eval()

    # Run audit
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING SCIENTIFIC VALIDATION TESTS")
    logger.info("=" * 60)

    from src.training.audit import ScientificAuditor, AuditConfig

    config = AuditConfig(
        temperature_delta=5.0,
        temperature_feature_idx=0,  # Temperature is first feature
        rollout_days=90,
        graphon_tolerance=0.10,
    )

    auditor = ScientificAuditor(model, config)
    results = auditor.run_all_tests(batch)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL AUDIT RESULTS")
    logger.info("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "[x]" if passed else "[ ]"
        logger.info(f"  {symbol} {test_name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("\n[SUCCESS] All scientific validation tests PASSED!")
        logger.info("The model has learned physically meaningful dynamics.")
    else:
        logger.warning("\n[WARNING] Some tests FAILED")
        logger.warning("Review model physics and training parameters.")

    return results

if __name__ == "__main__":
    results = main()
