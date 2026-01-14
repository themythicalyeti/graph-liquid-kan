#!/usr/bin/env python3
"""
Training script with reduced data for debugging memory issues.

Subsets the graph to a smaller number of nodes.
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
    logger.info("REDUCED DATA TRAINING TEST")
    logger.info("=" * 60)

    # Load data
    logger.info("\nLoading data...")
    data = np.load("data/processed/tensors.npz", allow_pickle=True)
    graph_data = torch.load("data/processed/spatial_graph.pt", weights_only=False)

    X = torch.from_numpy(data["X"]).float()  # (366, 1777, 8)
    Y = torch.from_numpy(data["Y"]).float()  # (366, 1777, 3)
    mask = torch.from_numpy(data["mask"]).bool()  # (366, 1777)
    edge_index = graph_data["edge_index"]

    # Load feature indices for SeaLiceGLKAN
    if "feature_indices" in data:
        feature_indices = data["feature_indices"].item()
        logger.info(f"Loaded feature_indices: {list(feature_indices.keys())}")
    else:
        feature_indices = None
        logger.warning("No feature_indices found - using defaults")

    logger.info(f"Original: X={X.shape}, edges={edge_index.shape[1]}")

    # Subset to first N_subset nodes
    N_subset = 200  # Reduced from 1777
    logger.info(f"\nSubsetting to {N_subset} nodes...")

    X = X[:, :N_subset, :]
    Y = Y[:, :N_subset, :]
    mask = mask[:, :N_subset]

    # Filter edges to only include nodes in subset
    valid_edges = (edge_index[0] < N_subset) & (edge_index[1] < N_subset)
    edge_index = edge_index[:, valid_edges]

    logger.info(f"Reduced: X={X.shape}, edges={edge_index.shape[1]}")

    # Create training batch (use first 30 time steps)
    window = 30
    B, T, N, F = 1, window, N_subset, X.shape[-1]

    x_batch = X[:window].unsqueeze(0)  # (1, 30, N, F)
    y_batch = Y[:window].unsqueeze(0)  # (1, 30, N, 3)
    mask_batch = mask[:window].unsqueeze(0)  # (1, 30, N)

    logger.info(f"\nBatch shapes:")
    logger.info(f"  x: {x_batch.shape}")
    logger.info(f"  y: {y_batch.shape}")
    logger.info(f"  mask: {mask_batch.shape}")

    # Create model (using SeaLicePredictor with biology modules)
    logger.info("\nCreating SeaLicePredictor model...")
    from src.models.sea_lice_network import SeaLicePredictor

    model = SeaLicePredictor(
        input_dim=F,
        hidden_dim=64,
        output_dim=3,
        n_bases=8,
        k_hops=3,
        dropout=0.1,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {total_params:,}")

    # Create batch dict
    batch = {
        'x': x_batch,
        'y': y_batch,
        'mask': mask_batch,
        'edge_index': edge_index,
        'feature_indices': feature_indices,
    }

    # Test forward pass
    logger.info("\nTesting forward pass...")
    try:
        output = model(batch)
        logger.info(f"  Predictions: {output['predictions'].shape}")
    except Exception as e:
        logger.error(f"  Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Overfit test
    logger.info("\n" + "=" * 60)
    logger.info("OVERFIT-ON-ONE-BATCH TEST (500 steps)")
    logger.info("=" * 60)

    from src.training.losses import GLKANLoss

    model = SeaLicePredictor(
        input_dim=F,
        hidden_dim=64,
        output_dim=3,
        n_bases=8,
        k_hops=3,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = GLKANLoss()

    losses = []
    for step in range(500):
        optimizer.zero_grad()

        output = model(batch)
        loss, metrics = criterion(output, batch)

        if torch.isnan(loss):
            logger.error(f"NaN loss at step {step}")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())

        if step % 100 == 0:
            logger.info(f"Step {step}: Loss = {loss.item():.6f}")

    logger.info(f"\nInitial loss: {losses[0]:.4f}")
    logger.info(f"Final loss: {losses[-1]:.4f}")

    if losses[-1] < 0.1:
        logger.info("[PASS] Overfit test passed - architecture is working")
    elif losses[-1] < losses[0] * 0.5:
        logger.info("[WARN] Model learns but didn't fully converge")
    else:
        logger.warning("[FAIL] Model did not overfit - check architecture")

    # Now test with full data subset
    logger.info("\n" + "=" * 60)
    logger.info("MINI TRAINING RUN (10 epochs)")
    logger.info("=" * 60)

    from src.training.trainer import GLKANTrainer, TrainingConfig
    from src.training.losses import LossConfig

    # Create simple dataset from numpy arrays
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, X, Y, mask, edge_index, feature_indices=None, window_size=30, stride=7):
            self.X = X
            self.Y = Y
            self.mask = mask
            self.edge_index = edge_index
            self.feature_indices = feature_indices
            self.sequences = []

            T = X.shape[0]
            for t in range(0, T - window_size, stride):
                self.sequences.append((t, t + window_size))

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            t_start, t_end = self.sequences[idx]
            return {
                'x': self.X[t_start:t_end],
                'y': self.Y[t_start:t_end],
                'mask': self.mask[t_start:t_end],
                'edge_index': self.edge_index,
                'time_points': torch.linspace(0, 1, t_end - t_start),
                'feature_indices': self.feature_indices,
            }

    def collate_fn(batch):
        return {
            'x': torch.stack([b['x'] for b in batch]),
            'y': torch.stack([b['y'] for b in batch]),
            'mask': torch.stack([b['mask'] for b in batch]),
            'edge_index': batch[0]['edge_index'],
            'time_points': batch[0]['time_points'],
            'feature_indices': batch[0]['feature_indices'],
        }

    # Reload and subset full data
    X_full = torch.from_numpy(data["X"]).float()[:, :N_subset, :]
    Y_full = torch.from_numpy(data["Y"]).float()[:, :N_subset, :]
    mask_full = torch.from_numpy(data["mask"]).bool()[:, :N_subset]

    # Split
    T_total = X_full.shape[0]
    T_train = int(T_total * 0.8)

    train_ds = SimpleDataset(X_full[:T_train], Y_full[:T_train], mask_full[:T_train], edge_index, feature_indices)
    val_ds = SimpleDataset(X_full[T_train:], Y_full[T_train:], mask_full[T_train:], edge_index, feature_indices)

    logger.info(f"Train sequences: {len(train_ds)}")
    logger.info(f"Val sequences: {len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # Create fresh model
    model = SeaLicePredictor(
        input_dim=F,
        hidden_dim=64,
        output_dim=3,
        n_bases=8,
        k_hops=3,
    )

    # Create trainer
    config = TrainingConfig(
        learning_rate=1e-4,
        n_epochs=10,
        device="cpu",
        checkpoint_dir="checkpoints",
        validate_every=1,
        log_every=5,
        early_stopping_patience=50,  # Don't early stop
    )

    loss_config = LossConfig(
        lambda_bio=0.1,
        lambda_stability=0.01,
    )

    trainer = GLKANTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        loss_config=loss_config,
    )

    # Train
    history = trainer.train(n_epochs=10)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best val loss: {trainer.best_val_loss:.6f}")

if __name__ == "__main__":
    main()
