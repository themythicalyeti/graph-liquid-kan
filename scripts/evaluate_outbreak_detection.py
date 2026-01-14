#!/usr/bin/env python3
"""
Evaluate model for outbreak detection metrics with Conformal Prediction.

Uses ConformalSeaLicePredictor to provide:
- Point predictions
- Uncertainty intervals (90% coverage)
- Risk-aware outbreak detection using upper bounds

Goal: 90% Recall, 80% Precision (F1 ≈ 0.85)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score, confusion_matrix
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")

# Norwegian regulatory threshold
OUTBREAK_THRESHOLD = 0.5  # adult female lice per fish
CONFORMAL_COVERAGE = 0.90  # 90% prediction intervals

def main():
    logger.info("=" * 60)
    logger.info("OUTBREAK DETECTION WITH CONFORMAL PREDICTION")
    logger.info("=" * 60)
    logger.info(f"Outbreak threshold: {OUTBREAK_THRESHOLD} lice/fish")
    logger.info(f"Conformal coverage: {CONFORMAL_COVERAGE*100:.0f}%")

    # Load data
    logger.info("\nLoading data...")
    data = np.load("data/processed/tensors.npz", allow_pickle=True)
    graph_data = torch.load("data/processed/spatial_graph.pt", weights_only=False)

    X = torch.from_numpy(data["X"]).float()
    Y = torch.from_numpy(data["Y"]).float()  # (T, N, 3) - adult_female is index 0
    mask = torch.from_numpy(data["mask"]).bool()
    edge_index = graph_data["edge_index"]

    # Load feature indices for SeaLiceGLKAN
    if "feature_indices" in data:
        feature_indices = data["feature_indices"].item()
        logger.info(f"Loaded feature_indices: {list(feature_indices.keys())}")
    else:
        feature_indices = None
        logger.warning("No feature_indices found - using defaults")

    # Subset to 200 nodes (matching training)
    N_subset = 200
    X = X[:, :N_subset, :]
    Y = Y[:, :N_subset, :]
    mask = mask[:, :N_subset]

    valid_edges = (edge_index[0] < N_subset) & (edge_index[1] < N_subset)
    edge_index = edge_index[:, valid_edges]

    logger.info(f"Data shape: X={X.shape}, Y={Y.shape}")

    # Load base model
    logger.info("\nLoading SeaLicePredictor model...")
    from src.models.sea_lice_network import SeaLicePredictor
    from src.models.conformal import ConformalSeaLicePredictor

    base_model = SeaLicePredictor(
        input_dim=X.shape[-1],
        hidden_dim=64,
        output_dim=3,
        n_bases=8,
        k_hops=3,
    )

    checkpoint_path = Path("checkpoints/best_model.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded trained model")
    else:
        logger.warning("No checkpoint found - using untrained model")

    base_model.eval()

    # Wrap with conformal prediction
    logger.info(f"\nWrapping with ConformalSeaLicePredictor (coverage={CONFORMAL_COVERAGE})...")
    conformal_model = ConformalSeaLicePredictor(
        base_model=base_model,
        coverage=CONFORMAL_COVERAGE,
        calibration_window=100,
        use_adaptive=True,
    )

    # Find time indices with observations
    T_total = X.shape[0]
    obs_per_time = mask.sum(dim=1).numpy()
    valid_times = np.where(obs_per_time > 0)[0]
    logger.info(f"Time steps with observations: {len(valid_times)}")

    if len(valid_times) == 0:
        logger.error("No observations in data!")
        return

    # Split: first half for calibration, second half for evaluation
    calib_times = valid_times[:len(valid_times)//2]
    eval_times = valid_times[len(valid_times)//2:]
    logger.info(f"Calibration time steps: {len(calib_times)}")
    logger.info(f"Evaluation time steps: {len(eval_times)}")

    # =========================================================================
    # STEP 1: Calibrate conformal predictor
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("CALIBRATING CONFORMAL PREDICTOR")
    logger.info("=" * 60)

    window = 14
    calib_preds = []
    calib_targets = []
    calib_masks = []

    with torch.no_grad():
        for t_center in calib_times:
            t_start = max(0, t_center - window // 2)
            t_end = min(T_total, t_start + window)

            if t_end - t_start < window:
                continue

            batch = {
                'x': X[t_start:t_end].unsqueeze(0),
                'y': Y[t_start:t_end].unsqueeze(0),
                'mask': mask[t_start:t_end].unsqueeze(0),
                'edge_index': edge_index,
                'feature_indices': feature_indices,
            }

            output = base_model(batch)
            pred = output['predictions'].squeeze(0)  # (T, N, 3)

            calib_preds.append(pred)
            calib_targets.append(Y[t_start:t_end])
            calib_masks.append(mask[t_start:t_end])

    if len(calib_preds) == 0:
        logger.error("No calibration sequences!")
        return

    calib_preds = torch.cat(calib_preds, dim=0)
    calib_targets = torch.cat(calib_targets, dim=0)
    calib_masks = torch.cat(calib_masks, dim=0)

    # Calibrate the conformal predictor
    conformal_model.conformal.calibrate(
        calib_preds, calib_targets, calib_masks
    )
    calib_diagnostics = conformal_model.conformal.get_diagnostics()
    logger.info(f"Calibration complete:")
    logger.info(f"  Residuals collected: {calib_diagnostics.get('n_residuals', 'N/A')}")
    quantile = calib_diagnostics.get('current_quantile', None)
    if quantile is not None:
        logger.info(f"  Current quantile: {quantile:.4f}")
    else:
        logger.info(f"  Current quantile: N/A")

    # =========================================================================
    # STEP 2: Make predictions with uncertainty on evaluation set
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("MAKING PREDICTIONS WITH UNCERTAINTY")
    logger.info("=" * 60)

    all_point_preds = []
    all_lower_bounds = []
    all_upper_bounds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for t_center in eval_times:
            t_start = max(0, t_center - window // 2)
            t_end = min(T_total, t_start + window)

            if t_end - t_start < window:
                continue

            x_seq = X[t_start:t_end]  # (T, N, F)

            # Get prediction with uncertainty interval
            interval = conformal_model.predict_with_uncertainty(
                x_seq, edge_index, feature_indices=feature_indices
            )

            all_point_preds.append(interval.point_prediction)
            all_lower_bounds.append(interval.lower_bound)
            all_upper_bounds.append(interval.upper_bound)
            all_targets.append(Y[t_start:t_end])
            all_masks.append(mask[t_start:t_end])

    if len(all_point_preds) == 0:
        logger.error("No test sequences available!")
        return

    # Concatenate all predictions
    point_preds = torch.cat(all_point_preds, dim=0)  # (total_T, N, 3)
    lower_bounds = torch.cat(all_lower_bounds, dim=0)
    upper_bounds = torch.cat(all_upper_bounds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    masks = torch.cat(all_masks, dim=0)

    logger.info(f"Test predictions shape: {point_preds.shape}")

    # Extract adult female lice (index 0)
    pred_af = point_preds[:, :, 0].numpy()
    lower_af = lower_bounds[:, :, 0].numpy()
    upper_af = upper_bounds[:, :, 0].numpy()
    target_af = targets[:, :, 0].numpy()
    mask_np = masks.numpy()

    # Flatten and filter by mask
    pred_flat = pred_af[mask_np]
    lower_flat = lower_af[mask_np]
    upper_flat = upper_af[mask_np]
    target_flat = target_af[mask_np]

    logger.info(f"Valid observations: {len(pred_flat)}")

    # =========================================================================
    # STEP 3: Evaluate conformal coverage
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("CONFORMAL PREDICTION COVERAGE")
    logger.info("=" * 60)

    # Check if targets fall within intervals
    in_interval = (target_flat >= lower_flat) & (target_flat <= upper_flat)
    empirical_coverage = in_interval.mean()

    logger.info(f"Target coverage: {CONFORMAL_COVERAGE*100:.0f}%")
    logger.info(f"Empirical coverage: {empirical_coverage*100:.1f}%")
    logger.info(f"Mean interval width: {(upper_flat - lower_flat).mean():.4f}")
    logger.info(f"Point prediction range: [{pred_flat.min():.3f}, {pred_flat.max():.3f}]")
    logger.info(f"Upper bound range: [{upper_flat.min():.3f}, {upper_flat.max():.3f}]")
    logger.info(f"Target range: [{target_flat.min():.3f}, {target_flat.max():.3f}]")

    # =========================================================================
    # STEP 4: Regression metrics
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("REGRESSION METRICS")
    logger.info("=" * 60)

    rmse = np.sqrt(np.mean((pred_flat - target_flat) ** 2))
    mae = np.mean(np.abs(pred_flat - target_flat))

    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE:  {mae:.4f}")

    # =========================================================================
    # STEP 5: Outbreak detection - POINT PREDICTIONS
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("OUTBREAK DETECTION - POINT PREDICTIONS")
    logger.info(f"(Outbreak = adult_female > {OUTBREAK_THRESHOLD})")
    logger.info("=" * 60)

    target_binary = (target_flat > OUTBREAK_THRESHOLD).astype(int)
    n_outbreaks = target_binary.sum()
    n_normal = len(target_binary) - n_outbreaks
    logger.info(f"Actual outbreaks: {n_outbreaks} ({100*n_outbreaks/len(target_binary):.1f}%)")
    logger.info(f"Normal cases:     {n_normal} ({100*n_normal/len(target_binary):.1f}%)")

    if n_outbreaks == 0:
        logger.warning("No outbreaks in test set!")
        return

    # At default threshold
    pred_binary_point = (pred_flat > OUTBREAK_THRESHOLD).astype(int)
    precision_point = precision_score(target_binary, pred_binary_point, zero_division=0)
    recall_point = recall_score(target_binary, pred_binary_point, zero_division=0)
    f1_point = f1_score(target_binary, pred_binary_point, zero_division=0)

    logger.info(f"\nAt threshold = {OUTBREAK_THRESHOLD}:")
    logger.info(f"  Precision: {precision_point:.2%}")
    logger.info(f"  Recall:    {recall_point:.2%}")
    logger.info(f"  F1 Score:  {f1_point:.4f}")

    # =========================================================================
    # STEP 6: Outbreak detection - RISK-AWARE (using upper bounds)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("OUTBREAK DETECTION - RISK-AWARE (Upper Bounds)")
    logger.info(f"(Risk = upper_bound > {OUTBREAK_THRESHOLD})")
    logger.info("=" * 60)

    # Use upper bound for conservative risk assessment
    # If upper bound > threshold, flag as potential outbreak
    pred_binary_risk = (upper_flat > OUTBREAK_THRESHOLD).astype(int)

    precision_risk = precision_score(target_binary, pred_binary_risk, zero_division=0)
    recall_risk = recall_score(target_binary, pred_binary_risk, zero_division=0)
    f1_risk = f1_score(target_binary, pred_binary_risk, zero_division=0)

    logger.info(f"At threshold = {OUTBREAK_THRESHOLD} (using upper bound):")
    logger.info(f"  Precision: {precision_risk:.2%}")
    logger.info(f"  Recall:    {recall_risk:.2%}")
    logger.info(f"  F1 Score:  {f1_risk:.4f}")

    cm_risk = confusion_matrix(target_binary, pred_binary_risk)
    logger.info(f"\nConfusion Matrix (Risk-Aware):")
    logger.info(f"                 Predicted")
    logger.info(f"              Normal  At-Risk")
    logger.info(f"  Actual Normal   {cm_risk[0,0]:5d}    {cm_risk[0,1]:5d}")
    logger.info(f"  Actual Outbreak {cm_risk[1,0]:5d}    {cm_risk[1,1]:5d}")

    # =========================================================================
    # STEP 7: Risk assessment summary
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("RISK ASSESSMENT SUMMARY")
    logger.info("=" * 60)

    # Calculate risk metrics
    risk_flagged = (upper_flat > OUTBREAK_THRESHOLD).sum()
    true_outbreaks_caught = ((upper_flat > OUTBREAK_THRESHOLD) & (target_flat > OUTBREAK_THRESHOLD)).sum()
    false_alarms = ((upper_flat > OUTBREAK_THRESHOLD) & (target_flat <= OUTBREAK_THRESHOLD)).sum()
    missed_outbreaks = ((upper_flat <= OUTBREAK_THRESHOLD) & (target_flat > OUTBREAK_THRESHOLD)).sum()

    logger.info(f"Total observations: {len(target_flat)}")
    logger.info(f"Flagged as at-risk: {risk_flagged} ({100*risk_flagged/len(target_flat):.1f}%)")
    logger.info(f"True outbreaks caught: {true_outbreaks_caught}/{n_outbreaks} ({100*true_outbreaks_caught/n_outbreaks:.1f}%)")
    logger.info(f"False alarms: {false_alarms}")
    logger.info(f"Missed outbreaks: {missed_outbreaks}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)

    logger.info("\n--- Point Prediction Performance ---")
    logger.info(f"  Recall:    {recall_point:.1%} (target: 90%)")
    logger.info(f"  Precision: {precision_point:.1%} (target: 80%)")
    logger.info(f"  F1:        {f1_point:.4f} (target: 0.85)")

    logger.info("\n--- Risk-Aware Performance (Upper Bound) ---")
    logger.info(f"  Recall:    {recall_risk:.1%} (target: 90%)")
    logger.info(f"  Precision: {precision_risk:.1%} (target: 80%)")
    logger.info(f"  F1:        {f1_risk:.4f} (target: 0.85)")

    logger.info("\n--- Conformal Coverage ---")
    logger.info(f"  Target: {CONFORMAL_COVERAGE*100:.0f}%, Achieved: {empirical_coverage*100:.1f}%")

    # Check if targets met
    meets_recall_risk = recall_risk >= 0.90
    meets_precision_risk = precision_risk >= 0.80

    if meets_recall_risk and meets_precision_risk:
        logger.info("\n[SUCCESS] Risk-aware detection meets targets!")
    elif meets_recall_risk:
        logger.info("\n[PARTIAL] High recall but low precision - too many false alarms")
        logger.info("  -> Model is conservative (good for safety, train more for precision)")
    else:
        logger.info("\n[NEEDS TRAINING] Model does not yet meet targets.")
        logger.info("Recommendations:")
        logger.info("  - Train for more epochs (100+)")
        logger.info("  - Use full dataset (1777 nodes)")
        logger.info("  - Higher learning rate (1e-2) for Tweedie loss")

if __name__ == "__main__":
    main()
