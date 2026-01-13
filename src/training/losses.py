"""
Phase 4.1: Physics-Informed Loss "Legislator"

We do not minimize error; we minimize ENERGY. The loss function represents
deviation from observed data PLUS deviation from physical and biological laws.

Loss Components:
1. L_data: Huber loss on masked observations (robust to outliers)
2. L_bio: Biological constraints (non-negativity, growth rate bounds)
3. L_stability: System stability (tau regularization)

Total: L = L_data + λ_bio * L_bio + λ_stability * L_stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LossConfig:
    """Configuration for Physics-Informed Loss."""
    # Huber loss delta (transition from quadratic to linear)
    huber_delta: float = 1.0

    # Biological constraint weights
    lambda_bio: float = 0.1

    # Non-negativity penalty weight
    lambda_nonneg: float = 1.0

    # Growth rate bound (max daily change, e.g., 20% = 0.2)
    max_daily_change: float = 0.2
    lambda_growth: float = 0.1

    # System stability weight
    lambda_stability: float = 0.01

    # Target tau for stability (pushes toward 1.0)
    tau_target: float = 1.0


class PhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss for Graph-Liquid-KAN.

    Combines data fidelity with physical constraints to ensure the model
    learns biologically plausible dynamics, not just curve fitting.

    Components:
    1. Data Fidelity (L_data): Huber loss on masked observations
       - Robust to outliers in biological measurements
       - Only computed where ground-truth exists (mask=True)

    2. Biological Constraints (L_bio):
       - Non-negativity: ReLU(-prediction) penalizes negative lice counts
       - Growth rate bound: Penalizes unrealistic daily changes

    3. System Stability (L_stability):
       - Tau regularization: (tau - 1)^2 + (1/tau - 1)^2
       - Prevents tau from going to 0 (stiff) or infinity (frozen)

    Args:
        config: Loss configuration
    """

    def __init__(self, config: Optional[LossConfig] = None):
        super().__init__()
        self.config = config or LossConfig()

        # Huber loss for robustness
        self.huber = nn.SmoothL1Loss(reduction='none', beta=self.config.huber_delta)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        tau: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute physics-informed loss.

        Args:
            predictions: Model predictions, shape (B, T, N, output_dim)
            targets: Ground truth, shape (B, T, N, output_dim)
            mask: Observation mask, shape (B, T, N)
            tau: Optional time constants for stability regularization

        Returns:
            total_loss: Scalar loss value
            metrics: Dictionary of component losses for monitoring
        """
        metrics = {}

        # =========================================================
        # 1. Data Fidelity Term (L_data)
        # =========================================================
        l_data = self._compute_data_loss(predictions, targets, mask)
        metrics['l_data'] = l_data.item()

        # =========================================================
        # 2. Biological Constraint Term (L_bio)
        # =========================================================
        l_bio, bio_metrics = self._compute_biological_loss(predictions)
        metrics.update(bio_metrics)
        metrics['l_bio'] = l_bio.item()

        # =========================================================
        # 3. System Stability Term (L_stability)
        # =========================================================
        if tau is not None:
            l_stability = self._compute_stability_loss(tau)
            metrics['l_stability'] = l_stability.item()
        else:
            l_stability = torch.tensor(0.0, device=predictions.device)
            metrics['l_stability'] = 0.0

        # =========================================================
        # Total Loss Aggregation
        # =========================================================
        total_loss = (
            l_data
            + self.config.lambda_bio * l_bio
            + self.config.lambda_stability * l_stability
        )

        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics

    def _compute_data_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Huber loss on masked observations.

        Only days with ground-truth observations contribute to the loss.
        The ODE fills in the gaps, but we don't penalize interpolated values.
        """
        # Expand mask to match output dimension
        if mask.dim() == predictions.dim() - 1:
            mask_expanded = mask.unsqueeze(-1).expand_as(predictions)
        else:
            mask_expanded = mask

        # Compute Huber loss (element-wise)
        huber_loss = self.huber(predictions, targets)

        # Apply mask
        masked_loss = huber_loss * mask_expanded.float()

        # Average over valid observations
        n_valid = mask_expanded.float().sum()
        if n_valid > 0:
            return masked_loss.sum() / n_valid
        else:
            return torch.tensor(0.0, device=predictions.device)

    def _compute_biological_loss(
        self,
        predictions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute biological constraint losses.

        1. Non-negativity: Lice counts cannot be negative
        2. Growth rate bound: Daily changes cannot exceed biological limits
        """
        metrics = {}

        # ---------------------------------------------------------
        # Non-negativity Constraint
        # ReLU(-prediction) is positive when prediction is negative
        # ---------------------------------------------------------
        negative_penalty = F.relu(-predictions)
        l_nonneg = negative_penalty.mean()
        metrics['l_nonneg'] = l_nonneg.item()

        # ---------------------------------------------------------
        # Growth Rate Bound
        # |H[t+1] - H[t]| should not exceed max_daily_change * H[t]
        # ---------------------------------------------------------
        if predictions.shape[1] > 1:  # Need at least 2 time steps
            # Compute daily differences
            delta_h = predictions[:, 1:] - predictions[:, :-1]  # (B, T-1, N, D)

            # Compute relative change threshold
            # Use abs(predictions) to handle near-zero values
            threshold = self.config.max_daily_change * (predictions[:, :-1].abs() + 0.1)

            # Penalize changes exceeding threshold
            excess_change = F.relu(delta_h.abs() - threshold)
            l_growth = excess_change.mean()
        else:
            l_growth = torch.tensor(0.0, device=predictions.device)

        metrics['l_growth'] = l_growth.item()

        # Total biological loss
        l_bio = (
            self.config.lambda_nonneg * l_nonneg
            + self.config.lambda_growth * l_growth
        )

        return l_bio, metrics

    def _compute_stability_loss(
        self,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute system stability loss on time constants.

        The "hook" function (tau - 1)^2 + (1/tau - 1)^2 creates a convex
        penalty that pushes tau toward 1.0, preventing:
        - tau -> 0 (stiffness, numerical instability)
        - tau -> inf (frozen dynamics, no learning)
        """
        # Avoid division by zero
        tau_safe = tau.clamp(min=1e-6)

        # Convex hook penalty
        l_tau = (tau_safe - self.config.tau_target) ** 2
        l_inv_tau = (1.0 / tau_safe - 1.0 / self.config.tau_target) ** 2

        stability_loss = (l_tau + l_inv_tau).mean()

        return stability_loss


class GLKANLoss(nn.Module):
    """
    Complete loss function for GLKAN training.

    Wraps PhysicsInformedLoss with additional utilities:
    - Automatic tau extraction from model
    - Per-output weighting
    - Logging-friendly interface

    Args:
        config: Loss configuration
        output_weights: Optional per-output weights (e.g., prioritize adult_female)
    """

    def __init__(
        self,
        config: Optional[LossConfig] = None,
        output_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.physics_loss = PhysicsInformedLoss(config)

        if output_weights is not None:
            self.register_buffer('output_weights', output_weights)
        else:
            self.output_weights = None

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss from model output and batch.

        Args:
            model_output: Dict with 'predictions', optionally 'tau'
            batch: Dict with 'y' (targets), 'mask'

        Returns:
            loss: Scalar loss
            metrics: Component metrics
        """
        predictions = model_output['predictions']
        targets = batch['y']
        mask = batch['mask']
        tau = model_output.get('tau', None)

        # Apply output weights if provided
        if self.output_weights is not None:
            # Weight predictions and targets before loss
            predictions = predictions * self.output_weights
            targets = targets * self.output_weights

        return self.physics_loss(predictions, targets, mask, tau)


# =========================================================================
# Utility Functions
# =========================================================================

def compute_rmse(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute masked RMSE for monitoring."""
    if mask.dim() == predictions.dim() - 1:
        mask = mask.unsqueeze(-1).expand_as(predictions)

    sq_error = (predictions - targets) ** 2
    masked_sq_error = sq_error * mask.float()

    n_valid = mask.float().sum()
    if n_valid > 0:
        mse = masked_sq_error.sum() / n_valid
        return torch.sqrt(mse)
    return torch.tensor(0.0, device=predictions.device)


def compute_mae(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute masked MAE for monitoring."""
    if mask.dim() == predictions.dim() - 1:
        mask = mask.unsqueeze(-1).expand_as(predictions)

    abs_error = (predictions - targets).abs()
    masked_abs_error = abs_error * mask.float()

    n_valid = mask.float().sum()
    if n_valid > 0:
        return masked_abs_error.sum() / n_valid
    return torch.tensor(0.0, device=predictions.device)
