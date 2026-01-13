#!/usr/bin/env python3
"""
Graph-Liquid-KAN Training Script

Phase 4: Optimization & Training

Usage:
    # Quick test (overfit on one batch)
    python scripts/train_glkan.py --test-overfit

    # Full training
    python scripts/train_glkan.py --epochs 100

    # Resume from checkpoint
    python scripts/train_glkan.py --resume checkpoints/best_model.pt
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)
logger.add(
    "logs/training_{time}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG",
    rotation="100 MB",
)


def main():
    parser = argparse.ArgumentParser(description="Train Graph-Liquid-KAN")

    # Data paths
    parser.add_argument("--tensor-path", type=str, default="data/processed/tensors.npz",
                        help="Path to tensor dataset")
    parser.add_argument("--graph-path", type=str, default="data/processed/spatial_graph.pt",
                        help="Path to graph topology")

    # Model architecture
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Hidden state dimension")
    parser.add_argument("--n-bases", type=int, default=8,
                        help="Number of RBF basis functions")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="Number of GLKAN layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping norm")

    # Dataset
    parser.add_argument("--window-size", type=int, default=30,
                        help="Sequence window size (days)")
    parser.add_argument("--stride", type=int, default=7,
                        help="Window stride (days)")

    # Loss weights
    parser.add_argument("--lambda-bio", type=float, default=0.1,
                        help="Biological constraint weight")
    parser.add_argument("--lambda-stability", type=float, default=0.01,
                        help="Stability regularization weight")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")

    # Testing
    parser.add_argument("--test-overfit", action="store_true",
                        help="Run overfit-on-one-batch test")
    parser.add_argument("--run-audit", action="store_true",
                        help="Run scientific validation after training")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cuda/cpu/auto)")

    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info("=" * 60)
    logger.info("GRAPH-LIQUID-KAN TRAINING")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Tensor path: {args.tensor_path}")
    logger.info(f"Graph path: {args.graph_path}")

    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    # Import modules
    from src.data.dataset import (
        SeaLiceGraphDataset,
        DatasetConfig,
        collate_graph_sequences,
        create_dataloaders,
    )
    from src.models.network import GLKANPredictor
    from src.training.losses import LossConfig
    from src.training.trainer import GLKANTrainer, TrainingConfig, overfit_one_batch
    from src.training.audit import run_full_audit, AuditConfig

    # Create dataset config
    dataset_config = DatasetConfig(
        window_size=args.window_size,
        stride=args.stride,
    )

    # Create dataloaders
    logger.info("\nLoading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        tensor_path=args.tensor_path,
        graph_path=args.graph_path,
        batch_size=args.batch_size,
        config=dataset_config,
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    # Get input dimension from data
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch['x'].shape[-1]
    logger.info(f"Input dimension: {input_dim}")

    # Create model
    logger.info("\nCreating model...")
    model = GLKANPredictor(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=3,  # adult_female, mobile, stationary
        n_bases=args.n_bases,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Overfit test
    if args.test_overfit:
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING OVERFIT-ON-ONE-BATCH TEST")
        logger.info("=" * 60)

        losses = overfit_one_batch(
            model=model,
            batch=sample_batch,
            n_steps=500,
            lr=1e-3,
            device=device,
        )

        if losses[-1] < 0.1:
            logger.info("Overfit test PASSED - architecture is working")
        else:
            logger.warning("Overfit test FAILED - check architecture")
            logger.warning("Proceeding with training anyway...")

        # Reset model for actual training
        model = GLKANPredictor(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=3,
            n_bases=args.n_bases,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )

    # Create training config
    training_config = TrainingConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.grad_clip,
        n_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
    )

    # Create loss config
    loss_config = LossConfig(
        lambda_bio=args.lambda_bio,
        lambda_stability=args.lambda_stability,
    )

    # Create trainer
    trainer = GLKANTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        loss_config=loss_config,
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    logger.info("\n" + "=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)

    history = trainer.train(n_epochs=args.epochs)

    # Save final model
    trainer.save_checkpoint("final_model.pt")

    # Run scientific audit
    if args.run_audit:
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING SCIENTIFIC VALIDATION")
        logger.info("=" * 60)

        # Load best model for audit
        trainer.load_checkpoint("best_model.pt")
        model.eval()

        audit_results = run_full_audit(
            model=model,
            dataloader=test_loader,
            config=AuditConfig(),
        )

        all_passed = all(audit_results.values())
        if all_passed:
            logger.info("\n[SUCCESS] Model passed all scientific validation tests!")
        else:
            logger.warning("\n[WARNING] Some validation tests failed")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best validation loss: {trainer.best_val_loss:.6f}")
    logger.info(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
