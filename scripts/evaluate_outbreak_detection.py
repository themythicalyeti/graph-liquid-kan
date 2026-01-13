#!/usr/bin/env python3
"""
Evaluate model for outbreak detection metrics.

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

def main():
    logger.info("=" * 60)
    logger.info("OUTBREAK DETECTION EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Outbreak threshold: {OUTBREAK_THRESHOLD} lice/fish")

    # Load data
    logger.info("\nLoading data...")
    data = np.load("data/processed/tensors.npz", allow_pickle=True)
    graph_data = torch.load("data/processed/spatial_graph.pt", weights_only=False)

    X = torch.from_numpy(data["X"]).float()
    Y = torch.from_numpy(data["Y"]).float()  # (T, N, 3) - adult_female is index 0
    mask = torch.from_numpy(data["mask"]).bool()
    edge_index = graph_data["edge_index"]

    # Subset to 200 nodes (matching training)
    N_subset = 200
    X = X[:, :N_subset, :]
    Y = Y[:, :N_subset, :]
    mask = mask[:, :N_subset]

    valid_edges = (edge_index[0] < N_subset) & (edge_index[1] < N_subset)
    edge_index = edge_index[:, valid_edges]

    logger.info(f"Data shape: X={X.shape}, Y={Y.shape}")

    # Load model
    logger.info("\nLoading model...")
    from src.models.network import GLKANPredictor

    model = GLKANPredictor(
        input_dim=X.shape[-1],
        hidden_dim=64,
        output_dim=3,
        n_bases=8,
        n_layers=1,
    )

    checkpoint_path = Path("checkpoints/best_model.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded trained model")
    else:
        logger.warning("No checkpoint found - using untrained model")

    model.eval()

    # Make predictions on validation set (where we have observations)
    # Observations are weekly, so we need to find where they exist
    logger.info("\nMaking predictions on validation set...")
    T_total = X.shape[0]

    # Find time indices with observations
    obs_per_time = mask.sum(dim=1).numpy()
    valid_times = np.where(obs_per_time > 0)[0]
    logger.info(f"Time steps with observations: {len(valid_times)}")

    if len(valid_times) == 0:
        logger.error("No observations in data!")
        return

    # Use second half of observed data for evaluation
    eval_times = valid_times[len(valid_times)//2:]
    logger.info(f"Evaluation time steps: {len(eval_times)}")

    window = 14  # Smaller window to get more sequences
    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for t_center in eval_times:
            t_start = max(0, t_center - window // 2)
            t_end = min(T_total, t_start + window)

            if t_end - t_start < window:
                continue

            batch = {
                'x': X[t_start:t_end].unsqueeze(0),
                'y': Y[t_start:t_end].unsqueeze(0),
                'mask': mask[t_start:t_end].unsqueeze(0),
                'edge_index': edge_index,
            }

            output = model(batch)
            pred = output['predictions'].squeeze(0)  # (T, N, 3)

            all_preds.append(pred)
            all_targets.append(Y[t_start:t_end])
            all_masks.append(mask[t_start:t_end])

    if len(all_preds) == 0:
        logger.error("No test sequences available!")
        return

    # Concatenate predictions
    preds = torch.cat(all_preds, dim=0)  # (total_T, N, 3)
    targets = torch.cat(all_targets, dim=0)
    masks = torch.cat(all_masks, dim=0)

    logger.info(f"Test predictions shape: {preds.shape}")

    # Extract adult female lice (index 0)
    pred_af = preds[:, :, 0].numpy()  # adult female predictions
    target_af = targets[:, :, 0].numpy()
    mask_np = masks.numpy()

    # Flatten and filter by mask
    pred_flat = pred_af[mask_np]
    target_flat = target_af[mask_np]

    logger.info(f"Valid observations: {len(pred_flat)}")

    # Regression metrics
    logger.info("\n" + "=" * 60)
    logger.info("REGRESSION METRICS")
    logger.info("=" * 60)

    rmse = np.sqrt(np.mean((pred_flat - target_flat) ** 2))
    mae = np.mean(np.abs(pred_flat - target_flat))

    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE:  {mae:.4f}")
    logger.info(f"Prediction range: [{pred_flat.min():.3f}, {pred_flat.max():.3f}]")
    logger.info(f"Target range:     [{target_flat.min():.3f}, {target_flat.max():.3f}]")

    # Binary classification: outbreak = adult_female > threshold
    logger.info("\n" + "=" * 60)
    logger.info("OUTBREAK DETECTION METRICS")
    logger.info(f"(Outbreak = adult_female > {OUTBREAK_THRESHOLD})")
    logger.info("=" * 60)

    target_binary = (target_flat > OUTBREAK_THRESHOLD).astype(int)

    # Check if we have both classes
    n_outbreaks = target_binary.sum()
    n_normal = len(target_binary) - n_outbreaks
    logger.info(f"Actual outbreaks: {n_outbreaks} ({100*n_outbreaks/len(target_binary):.1f}%)")
    logger.info(f"Normal cases:     {n_normal} ({100*n_normal/len(target_binary):.1f}%)")

    if n_outbreaks == 0:
        logger.warning("No outbreaks in test set - cannot compute detection metrics")
        return

    # Find optimal threshold for target recall
    logger.info("\n--- Threshold Analysis ---")

    precisions, recalls, thresholds = precision_recall_curve(target_binary, pred_flat)

    # Find threshold for 90% recall
    target_recall = 0.90
    best_threshold = None
    best_precision = 0
    best_f1 = 0

    for i, (p, r, t) in enumerate(zip(precisions[:-1], recalls[:-1], thresholds)):
        if r >= target_recall:
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_precision = p
                best_threshold = t

    if best_threshold is not None:
        logger.info(f"\nOptimal threshold for {target_recall*100:.0f}% recall: {best_threshold:.4f}")

        pred_binary = (pred_flat > best_threshold).astype(int)

        precision = precision_score(target_binary, pred_binary, zero_division=0)
        recall = recall_score(target_binary, pred_binary, zero_division=0)
        f1 = f1_score(target_binary, pred_binary, zero_division=0)

        cm = confusion_matrix(target_binary, pred_binary)

        logger.info(f"\nAt threshold = {best_threshold:.4f}:")
        logger.info(f"  Precision: {precision:.2%} (target: 80%)")
        logger.info(f"  Recall:    {recall:.2%} (target: 90%)")
        logger.info(f"  F1 Score:  {f1:.4f} (target: 0.85)")

        logger.info(f"\nConfusion Matrix:")
        logger.info(f"                 Predicted")
        logger.info(f"              Normal  Outbreak")
        logger.info(f"  Actual Normal   {cm[0,0]:5d}    {cm[0,1]:5d}")
        logger.info(f"  Actual Outbreak {cm[1,0]:5d}    {cm[1,1]:5d}")
    else:
        logger.warning(f"Could not achieve {target_recall*100:.0f}% recall with any threshold")

    # Show metrics at default threshold (0.5)
    logger.info(f"\n--- At Default Threshold ({OUTBREAK_THRESHOLD}) ---")
    pred_binary_default = (pred_flat > OUTBREAK_THRESHOLD).astype(int)

    precision_default = precision_score(target_binary, pred_binary_default, zero_division=0)
    recall_default = recall_score(target_binary, pred_binary_default, zero_division=0)
    f1_default = f1_score(target_binary, pred_binary_default, zero_division=0)

    logger.info(f"  Precision: {precision_default:.2%}")
    logger.info(f"  Recall:    {recall_default:.2%}")
    logger.info(f"  F1 Score:  {f1_default:.4f}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    meets_recall = recall >= 0.90 if best_threshold else False
    meets_precision = precision >= 0.80 if best_threshold else False
    meets_f1 = f1 >= 0.85 if best_threshold else False

    logger.info(f"Target: 90% Recall, 80% Precision, F1 ≥ 0.85")
    logger.info(f"")
    if best_threshold:
        logger.info(f"  Recall ≥ 90%:    {'PASS' if meets_recall else 'FAIL'} ({recall:.1%})")
        logger.info(f"  Precision ≥ 80%: {'PASS' if meets_precision else 'FAIL'} ({precision:.1%})")
        logger.info(f"  F1 ≥ 0.85:       {'PASS' if meets_f1 else 'FAIL'} ({f1:.4f})")

    if meets_recall and meets_precision:
        logger.info("\n[SUCCESS] Model meets outbreak detection targets!")
    else:
        logger.info("\n[NEEDS TRAINING] Model does not yet meet targets.")
        logger.info("Recommendations:")
        logger.info("  - Train for more epochs (100+)")
        logger.info("  - Use full dataset (1777 nodes)")
        logger.info("  - Consider GPU for faster training")

if __name__ == "__main__":
    main()
