"""
Phase 4.3: Training Executive (The Loop)

Implements the training loop for Graph-Liquid-KAN with:
1. AdamW optimizer with weight decay (prevents spline oscillation)
2. ReduceLROnPlateau scheduler
3. Gradient clipping (max_norm=1.0) - non-negotiable for ODE stability
4. Teacher forcing schedule (decays from 1.0 to 0.0)
5. Checkpointing and logging

Warning: This is a Hybrid Dynamic System, not a static classifier.
Standard training loops will fail without these stabilization mechanisms.
"""

import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from loguru import logger

from .losses import GLKANLoss, LossConfig, compute_rmse, compute_mae
from .pretrain import BiologyPretrainer, BiologyPretrainConfig, get_biology_param_names


@dataclass
class TrainingConfig:
    """Configuration for GLKAN training."""
    # Optimizer
    learning_rate: float = 1e-2  # High LR needed for Tweedie loss to escape mean-collapse
    weight_decay: float = 1e-4  # Prevents spline coefficient oscillation
    betas: Tuple[float, float] = (0.9, 0.999)

    # Scheduler - ReduceLROnPlateau
    scheduler_patience: int = 15  # Epochs without improvement before reducing LR
    scheduler_factor: float = 0.5  # Multiply LR by this factor when reducing
    min_lr: float = 1e-6  # Don't reduce below this

    # Gradient clipping - NON-NEGOTIABLE for ODE stability
    max_grad_norm: float = 1.0

    # Teacher forcing
    teacher_forcing_start: float = 0.5  # Start with 50% teacher forcing
    teacher_forcing_end: float = 0.0
    teacher_forcing_epochs: int = 50  # Decay over 50 epochs

    # Training
    n_epochs: int = 100
    validate_every: int = 1
    log_every: int = 10  # Log every N batches

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best: bool = True
    save_every: int = 10  # Save every N epochs

    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-6

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Biological module pre-training
    pretrain_biology: bool = True  # Whether to pre-train biology modules
    pretrained_biology_path: Optional[str] = None  # Path to load pre-trained weights
    bio_lr_scale: float = 0.5  # Learning rate multiplier for biology modules (0.5 = half)


class GLKANTrainer:
    """
    Training Executive for Graph-Liquid-KAN.

    Manages the complete training pipeline including:
    - Physics-informed loss computation
    - Gradient clipping for ODE stability
    - Teacher forcing schedule
    - Learning rate scheduling
    - Checkpointing and early stopping
    - Validation and metrics logging

    Args:
        model: The GLKANPredictor model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        loss_config: Loss function configuration
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        loss_config: Optional[LossConfig] = None,
    ):
        self.config = config or TrainingConfig()
        self.model = model.to(self.config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss function
        self.criterion = GLKANLoss(loss_config)

        # Create checkpoint directory (needed for pre-training)
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Handle biological module pre-training
        self._setup_biology_pretraining()

        # Optimizer - AdamW with weight decay and differential LR
        self.optimizer = self._create_optimizer()

        # Scheduler - ReduceLROnPlateau
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            min_lr=self.config.min_lr,
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # History
        self.train_history: List[Dict] = []
        self.val_history: List[Dict] = []

    def _setup_biology_pretraining(self) -> None:
        """Handle pre-training or loading of biological modules."""
        # Check if model has biology loading capability
        if not hasattr(self.model, 'load_pretrained_biology') and not hasattr(self.model, 'network'):
            logger.info("Model does not support biology pre-training, skipping.")
            return

        # Get the actual network (handle SeaLicePredictor wrapper)
        network = getattr(self.model, 'network', self.model)

        if not hasattr(network, 'load_pretrained_biology'):
            logger.info("Network does not support biology pre-training, skipping.")
            return

        if self.config.pretrained_biology_path:
            # Load from existing checkpoint
            pretrained = BiologyPretrainer.load_pretrained(
                self.config.pretrained_biology_path,
                device=self.config.device,
            )
            network.load_pretrained_biology(pretrained)
        elif self.config.pretrain_biology:
            # Run pre-training
            pretrainer = BiologyPretrainer(BiologyPretrainConfig(
                device=self.config.device,
                checkpoint_dir=self.config.checkpoint_dir,
            ))
            pretrained = pretrainer.pretrain_all(verbose=True, save=True)
            network.load_pretrained_biology(pretrained)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer with differential learning rates.

        Biology modules get bio_lr_scale * learning_rate (default 0.5x)
        to preserve pre-trained knowledge while allowing controlled drift.
        """
        # Get biology parameter names
        bio_param_patterns = get_biology_param_names()

        # Separate parameters
        bio_params = []
        other_params = []
        bio_param_ids = set()

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            is_bio = any(pattern in name for pattern in bio_param_patterns)

            if is_bio:
                # Avoid duplicates (some modules may share weights)
                if id(param) not in bio_param_ids:
                    bio_params.append(param)
                    bio_param_ids.add(id(param))
            else:
                other_params.append(param)

        # Create parameter groups
        bio_lr = self.config.learning_rate * self.config.bio_lr_scale

        param_groups = []

        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.config.learning_rate,
                'name': 'other',
            })

        if bio_params:
            param_groups.append({
                'params': bio_params,
                'lr': bio_lr,
                'name': 'biology',
            })

        logger.info(f"Optimizer parameter groups:")
        logger.info(f"  Other params: {len(other_params)} tensors, LR = {self.config.learning_rate}")
        logger.info(f"  Biology params: {len(bio_params)} tensors, LR = {bio_lr} ({self.config.bio_lr_scale}x)")

        return torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
        )

    def get_teacher_forcing_ratio(self) -> float:
        """
        Get current teacher forcing ratio based on epoch.

        Linearly decays from teacher_forcing_start to teacher_forcing_end
        over teacher_forcing_epochs.
        """
        if self.current_epoch >= self.config.teacher_forcing_epochs:
            return self.config.teacher_forcing_end

        progress = self.current_epoch / self.config.teacher_forcing_epochs
        ratio = self.config.teacher_forcing_start + progress * (
            self.config.teacher_forcing_end - self.config.teacher_forcing_start
        )
        return ratio

    def train_epoch(self) -> Dict[str, float]:
        """
        Run one training epoch.

        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'l_data': 0.0,
            'l_bio': 0.0,
            'l_stability': 0.0,
            'rmse': 0.0,
            'mae': 0.0,
        }
        n_batches = 0

        teacher_forcing_ratio = self.get_teacher_forcing_ratio()
        logger.info(f"Teacher forcing ratio: {teacher_forcing_ratio:.3f}")

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._to_device(batch)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(batch)

            # Compute loss
            loss, metrics = self.criterion(output, batch)

            # Check for NaN
            if torch.isnan(loss):
                logger.error("NaN loss detected! Skipping batch.")
                continue

            # Backward pass
            loss.backward()

            # Gradient clipping - NON-NEGOTIABLE
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.max_grad_norm
            )

            # Optimizer step
            self.optimizer.step()

            # Update metrics
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key] += metrics[key]
            # Handle 'loss' from 'total_loss'
            if 'total_loss' in metrics:
                epoch_metrics['loss'] += metrics['total_loss']

            # Compute additional metrics
            with torch.no_grad():
                rmse = compute_rmse(output['predictions'], batch['y'], batch['mask'])
                mae = compute_mae(output['predictions'], batch['y'], batch['mask'])
                epoch_metrics['rmse'] += rmse.item()
                epoch_metrics['mae'] += mae.item()

            n_batches += 1
            self.global_step += 1

            # Logging
            if batch_idx % self.config.log_every == 0:
                logger.info(
                    f"  Batch {batch_idx}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Grad: {grad_norm:.4f}"
                )

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= max(n_batches, 1)

        return epoch_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.

        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_metrics = {
            'val_loss': 0.0,
            'val_rmse': 0.0,
            'val_mae': 0.0,
        }
        n_batches = 0

        for batch in self.val_loader:
            batch = self._to_device(batch)

            output = self.model(batch)
            loss, metrics = self.criterion(output, batch)

            val_metrics['val_loss'] += loss.item()

            rmse = compute_rmse(output['predictions'], batch['y'], batch['mask'])
            mae = compute_mae(output['predictions'], batch['y'], batch['mask'])
            val_metrics['val_rmse'] += rmse.item()
            val_metrics['val_mae'] += mae.item()

            n_batches += 1

        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= max(n_batches, 1)

        return val_metrics

    def train(self, n_epochs: Optional[int] = None) -> Dict[str, List]:
        """
        Full training loop.

        Args:
            n_epochs: Number of epochs (defaults to config)

        Returns:
            Dictionary containing training history
        """
        n_epochs = n_epochs or self.config.n_epochs

        logger.info("=" * 60)
        logger.info("PHASE 4: GRAPH-LIQUID-KAN TRAINING")
        logger.info("=" * 60)
        logger.info(f"Device: {self.config.device}")
        logger.info(f"Epochs: {n_epochs}")
        logger.info(f"Learning rate: {self.config.learning_rate} (reduces by {self.config.scheduler_factor}x after {self.config.scheduler_patience} epochs w/o improvement)")
        logger.info(f"Min LR: {self.config.min_lr}")
        logger.info(f"Weight decay: {self.config.weight_decay}")
        logger.info(f"Grad clip norm: {self.config.max_grad_norm}")

        start_time = time.time()

        for epoch in range(n_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{n_epochs}")
            logger.info(f"{'='*60}")

            # Train
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)

            # Validate
            if self.val_loader is not None and (epoch + 1) % self.config.validate_every == 0:
                val_metrics = self.validate()
                self.val_history.append(val_metrics)

                # Update scheduler (track LR changes)
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(val_metrics['val_loss'])
                new_lr = self.optimizer.param_groups[0]['lr']

                if new_lr < old_lr:
                    logger.info(f">>> LR REDUCED: {old_lr:.2e} -> {new_lr:.2e} <<<")

                # Check for improvement
                if val_metrics['val_loss'] < self.best_val_loss - self.config.early_stopping_min_delta:
                    self.best_val_loss = val_metrics['val_loss']
                    self.epochs_without_improvement = 0

                    if self.config.save_best:
                        self.save_checkpoint('best_model.pt')
                else:
                    self.epochs_without_improvement += 1
            else:
                val_metrics = {}

            # Log epoch summary
            epoch_time = time.time() - epoch_start
            self._log_epoch_summary(epoch, train_metrics, val_metrics, epoch_time)

            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')

            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        total_time = time.time() - start_time
        logger.info(f"\nTraining completed in {total_time / 60:.2f} minutes")

        return {
            'train': self.train_history,
            'val': self.val_history,
        }

    def _log_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        epoch_time: float,
    ):
        """Log epoch summary."""
        lr = self.optimizer.param_groups[0]['lr']

        summary = (
            f"\nEpoch {epoch + 1} Summary:\n"
            f"  Train Loss: {train_metrics['loss']:.4f}\n"
            f"  Train RMSE: {train_metrics['rmse']:.4f}\n"
            f"  L_data: {train_metrics['l_data']:.4f}, "
            f"L_bio: {train_metrics['l_bio']:.4f}, "
            f"L_stability: {train_metrics['l_stability']:.4f}\n"
        )

        if val_metrics:
            summary += (
                f"  Val Loss: {val_metrics['val_loss']:.4f}\n"
                f"  Val RMSE: {val_metrics['val_rmse']:.4f}\n"
            )

        summary += (
            f"  LR: {lr:.2e}\n"
            f"  Time: {epoch_time:.1f}s"
        )

        logger.info(summary)

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        path = Path(self.config.checkpoint_dir) / filename

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config,
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, filename: str):
        """Load training checkpoint."""
        path = Path(self.config.checkpoint_dir) / filename

        checkpoint = torch.load(path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']

        logger.info(f"Loaded checkpoint: {path} (epoch {self.current_epoch})")

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        return {
            k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }


def overfit_one_batch(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    n_steps: int = 1000,
    lr: float = 1e-3,
    device: str = "cpu",
) -> List[float]:
    """
    Overfit on a single batch to verify architecture is working.

    The model should reach Loss ~ 0.0. If not, the architecture is broken.

    Args:
        model: The model to train
        batch: Single batch of data
        n_steps: Number of optimization steps
        lr: Learning rate
        device: Device to use

    Returns:
        List of loss values
    """
    logger.info("=" * 60)
    logger.info("OVERFIT-ONE-BATCH TEST")
    logger.info("=" * 60)

    model = model.to(device)
    model.train()

    # Move batch to device
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = GLKANLoss()

    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()

        output = model(batch)
        loss, metrics = criterion(output, batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())

        if step % 100 == 0:
            logger.info(f"Step {step}: Loss = {loss.item():.6f}")

    final_loss = losses[-1]
    logger.info(f"\nFinal Loss: {final_loss:.6f}")

    if final_loss < 0.01:
        logger.info("[PASS] Model successfully overfit on single batch")
    else:
        logger.warning("[WARN] Model did not fully overfit - check architecture")

    return losses
