"""
Phase 3.4: GLKAN Network (Temporal Unroller)

This module assembles the complete Graph-Liquid-KAN network that processes
spatiotemporal sequences from Phase 2.

Architecture:
1. Input Encoder: Project raw features to hidden dimension
2. Temporal Loop: Unroll GraphLiquidKANCell over time
3. Output Decoder: Project hidden states to predictions

The network processes sequences of shape (T, N, F) where:
- T: Time steps
- N: Number of nodes (farms)
- F: Feature dimension

Output is (T, N, output_dim) for the prediction targets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from .layers import FastKAN, GraphonAggregator
from .cells import GraphLiquidKANCell


class GLKANNetwork(nn.Module):
    """
    Graph-Liquid-KAN Network for Spatiotemporal Prediction.

    This is the complete model that:
    1. Encodes input features
    2. Unrolls the GraphLiquidKANCell over time
    3. Decodes hidden states to predictions

    The model is designed for sea lice prediction but generalizes to
    any spatiotemporal graph regression task.

    Args:
        input_dim: Dimension of input features (from Phase 2)
        hidden_dim: Dimension of hidden state
        output_dim: Dimension of output (e.g., 3 for lice counts)
        n_bases: Number of RBF bases for KAN layers
        n_layers: Number of stacked GLKAN cells (depth)
        dropout: Dropout probability
        tau_min: Minimum time constant
        tau_max: Maximum time constant
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 3,
        n_bases: int = 8,
        n_layers: int = 1,
        dropout: float = 0.1,
        tau_min: float = 0.01,
        tau_max: float = 10.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        # Input encoder: project features to hidden dimension
        self.input_encoder = FastKAN(
            in_features=input_dim,
            out_features=hidden_dim,
            n_bases=n_bases,
        )

        # Stack of Graph-Liquid-KAN cells
        self.cells = nn.ModuleList([
            GraphLiquidKANCell(
                input_dim=hidden_dim,  # Takes encoded input
                hidden_dim=hidden_dim,
                n_bases=n_bases,
                tau_min=tau_min,
                tau_max=tau_max,
            )
            for _ in range(n_layers)
        ])

        # Layer normalization between cells
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(n_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output decoder: project hidden to predictions
        self.output_decoder = FastKAN(
            in_features=hidden_dim,
            out_features=output_dim,
            n_bases=n_bases,
        )

        # Learnable initial hidden state (optional)
        self.h0 = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        time_points: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the network.

        Args:
            x: Input features, shape (B, T, N, F) or (T, N, F)
            edge_index: Graph edges, shape (2, E)
            time_points: Normalized time points, shape (T,)
            edge_weight: Optional edge weights, shape (E,)
            return_hidden: Whether to return hidden states

        Returns:
            predictions: Output predictions, shape (B, T, N, output_dim) or (T, N, output_dim)
            hidden_states: Optional hidden trajectory, same shape as predictions with hidden_dim
        """
        # Handle batched vs non-batched input
        if x.dim() == 3:
            # Non-batched: (T, N, F) -> add batch dimension
            x = x.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        B, T, N, F = x.shape

        # Compute time steps (dt between consecutive points)
        if time_points is None:
            # Assume uniform time steps
            dt = torch.ones(T, device=x.device) / T
        else:
            # Compute differences between consecutive time points
            dt = torch.zeros(T, device=x.device)
            dt[1:] = time_points[1:] - time_points[:-1]
            dt[0] = dt[1] if T > 1 else 1.0 / T

        # Initialize hidden states for each layer
        # Shape: (B, N, hidden_dim)
        h_list = [
            self.h0.expand(B, N, -1).clone()
            for _ in range(self.n_layers)
        ]

        # Storage for trajectory (if needed)
        if return_hidden:
            hidden_trajectory = []

        # Storage for outputs
        outputs = []

        # Temporal unrolling
        for t in range(T):
            # Get input at time t
            x_t = x[:, t, :, :]  # (B, N, F)

            # Encode input
            u_t = self.input_encoder(x_t)  # (B, N, hidden_dim)
            u_t = self.dropout(u_t)

            # Get time step
            dt_t = dt[t]

            # Process through each layer
            for layer_idx in range(self.n_layers):
                # Get current hidden state
                h = h_list[layer_idx]

                # Process through cell
                # Note: for multi-layer, we pass u_t only to first layer
                # and the output of previous layer to subsequent layers
                if layer_idx == 0:
                    cell_input = u_t
                else:
                    cell_input = h_list[layer_idx - 1]

                h_new = self.cells[layer_idx](
                    h=h,
                    u=cell_input,
                    edge_index=edge_index,
                    dt=dt_t,
                    edge_weight=edge_weight,
                )

                # Layer norm and residual connection
                h_new = self.layer_norms[layer_idx](h_new)
                if layer_idx > 0:
                    h_new = h_new + h  # Residual

                h_new = self.dropout(h_new)

                # Update hidden state
                h_list[layer_idx] = h_new

            # Decode output from final layer hidden state
            h_final = h_list[-1]  # (B, N, hidden_dim)
            y_t = self.output_decoder(h_final)  # (B, N, output_dim)

            outputs.append(y_t)

            if return_hidden:
                hidden_trajectory.append(h_final.clone())

        # Stack outputs: (B, T, N, output_dim)
        predictions = torch.stack(outputs, dim=1)

        if return_hidden:
            hidden_states = torch.stack(hidden_trajectory, dim=1)
        else:
            hidden_states = None

        # Remove batch dimension if it was added
        if squeeze_batch:
            predictions = predictions.squeeze(0)
            if hidden_states is not None:
                hidden_states = hidden_states.squeeze(0)

        return predictions, hidden_states

    def reset_cache(self):
        """Reset all aggregator caches."""
        for cell in self.cells:
            cell.reset_cache()

    def get_tau_trajectory(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        time_points: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get the time constant trajectory for inspection.

        Useful for debugging and understanding learned dynamics.

        Returns:
            tau: Time constants, shape (T, N, hidden_dim)
        """
        # Run forward pass to get hidden states
        _, hidden = self.forward(
            x, edge_index, time_points, edge_weight, return_hidden=True
        )

        # This is approximate - for exact tau we'd need to track during forward
        # For now, return the hidden states magnitude as proxy
        return hidden


class GLKANPredictor(nn.Module):
    """
    Complete predictor model with loss computation.

    Wraps GLKANNetwork and adds:
    - Loss function (masked MSE for sparse observations)
    - Prediction head options
    - Training/inference modes

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden state dimension
        output_dim: Output dimension (e.g., 3 for lice counts)
        n_bases: Number of RBF bases
        n_layers: Number of GLKAN layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 3,
        n_bases: int = 8,
        n_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.network = GLKANNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_bases=n_bases,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.output_dim = output_dim

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with loss computation.

        Args:
            batch: Dictionary containing:
                - x: Input features (B, T, N, F)
                - y: Target values (B, T, N, output_dim)
                - mask: Observation mask (B, T, N)
                - edge_index: Graph edges (2, E)
                - time_points: Normalized times (T,)

        Returns:
            Dictionary containing:
                - predictions: Model predictions
                - loss: Masked loss value
                - metrics: Additional metrics
        """
        x = batch["x"]
        y = batch["y"]
        mask = batch["mask"]
        edge_index = batch["edge_index"]
        time_points = batch.get("time_points", None)
        edge_weight = batch.get("edge_weight", None)

        # Forward pass
        predictions, _ = self.network(
            x=x,
            edge_index=edge_index,
            time_points=time_points,
            edge_weight=edge_weight,
        )

        # Compute masked loss
        loss, metrics = self._compute_loss(predictions, y, mask)

        return {
            "predictions": predictions,
            "loss": loss,
            "metrics": metrics,
        }

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute masked MSE loss.

        Only observations where mask=True contribute to the loss.
        This handles the sparse weekly observations.

        Args:
            predictions: Model predictions (B, T, N, output_dim)
            targets: Ground truth (B, T, N, output_dim)
            mask: Observation mask (B, T, N)

        Returns:
            loss: Scalar loss value
            metrics: Dictionary of additional metrics
        """
        # Expand mask to match output dimension
        if mask.dim() == predictions.dim() - 1:
            mask = mask.unsqueeze(-1).expand_as(predictions)

        # Compute squared errors
        sq_errors = (predictions - targets) ** 2

        # Apply mask
        masked_errors = sq_errors * mask.float()

        # Compute mean over valid observations
        n_valid = mask.float().sum()
        if n_valid > 0:
            loss = masked_errors.sum() / n_valid
        else:
            loss = torch.tensor(0.0, device=predictions.device)

        # Compute per-output metrics
        metrics = {}
        for i in range(self.output_dim):
            mask_i = mask[..., i] if mask.dim() == predictions.dim() else mask
            n_valid_i = mask_i.float().sum()
            if n_valid_i > 0:
                rmse_i = torch.sqrt(
                    (sq_errors[..., i] * mask_i.float()).sum() / n_valid_i
                )
                metrics[f"rmse_{i}"] = rmse_i.item()

        metrics["n_valid"] = n_valid.item()

        return loss, metrics

    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        time_points: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Make predictions without loss computation.

        Args:
            x: Input features
            edge_index: Graph edges
            time_points: Optional time points

        Returns:
            predictions: Model predictions
        """
        self.eval()
        with torch.no_grad():
            predictions, _ = self.network(x, edge_index, time_points)
        return predictions
