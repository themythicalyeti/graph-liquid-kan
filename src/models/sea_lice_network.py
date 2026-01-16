"""
Sea Lice Graph-Liquid-KAN Network

This module implements the complete Graph-Liquid-KAN network specialized for
Lepeophtheirus salmonis (sea lice) population dynamics prediction.

Key Innovations:
1. Temperature-dependent development rate (BelehradekKAN instead of fixed equation)
2. Salinity-dependent survival factor
3. Multi-hop graph convolution for long-range spatial dependencies
4. Larval transport module for current-driven infection spread
5. Physics-informed loss with mass conservation constraint

Architecture:
    Input: (T, N, F) - Environmental features over time for N farms
    -> Input Encoder: FastKAN
    -> Temporal Loop:
        -> K-hop Graph Convolution (spatial aggregation)
        -> Larval Transport (cross-infection pressure)
        -> Sea Lice Dynamics Cell (biological ODE)
    -> Output Decoder: FastKAN -> softplus
    Output: (T, N, 3) - Predicted lice counts (adult_female, mobile, attached)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from .layers import FastKAN, GraphonAggregator
from .biology import (
    BelehradekKAN,
    SalinityMortalityKAN,
    LarvalTransportModule,
    KHopGraphConv,
    SeaLiceDynamicsCell,
    PopulationMassConservation,
)


class SeaLiceGLKAN(nn.Module):
    """
    Complete Sea Lice Graph-Liquid-KAN Network.

    This network integrates domain-specific biological knowledge with the
    Graph-Liquid-KAN architecture for accurate sea lice prediction.

    The model learns:
    1. How temperature affects development rate (symbolic regression possible)
    2. How salinity affects survival
    3. How infection spreads between farms via currents
    4. The underlying population dynamics ODE

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden state dimension
        output_dim: Output dimension (default: 3 for lice types)
        n_bases: Number of RBF bases for KAN layers
        k_hops: Number of hops for spatial aggregation
        tau_min: Minimum time constant (must be >= 1.0)
        tau_max: Maximum time constant
        dropout: Dropout probability
        use_larval_transport: Whether to use explicit larval transport modeling
        use_mass_conservation: Whether to use mass conservation regularization
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 3,
        n_bases: int = 8,
        k_hops: int = 3,
        tau_min: float = 1.0,
        tau_max: float = 10.0,
        dropout: float = 0.1,
        use_larval_transport: bool = True,
        use_mass_conservation: bool = True,
        # Input denormalization for biological modules (when data is z-scored)
        temp_mean: Optional[float] = None,
        temp_std: Optional[float] = None,
        sal_mean: Optional[float] = None,
        sal_std: Optional[float] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.k_hops = k_hops
        self.use_larval_transport = use_larval_transport
        self.use_mass_conservation = use_mass_conservation

        # Store normalization params for biological modules
        self.temp_mean = temp_mean
        self.temp_std = temp_std
        self.sal_mean = sal_mean
        self.sal_std = sal_std

        # ===================================================================
        # Input Processing
        # ===================================================================

        # Input encoder: raw features -> hidden
        self.input_encoder = FastKAN(
            in_features=input_dim,
            out_features=hidden_dim,
            n_bases=n_bases,
        )

        # Feature extraction for environmental variables
        # We need to identify which features are temp, salinity, etc.
        # This is handled by explicit indexing in forward()

        # ===================================================================
        # Biological Modules
        # ===================================================================

        # Temperature-dependent development rate (with input denormalization)
        self.belehradek = BelehradekKAN(
            n_bases=n_bases,
            input_mean=temp_mean,
            input_std=temp_std,
        )

        # Salinity-dependent survival (with input denormalization)
        self.salinity_survival = SalinityMortalityKAN(
            n_bases=n_bases,
            input_mean=sal_mean,
            input_std=sal_std,
        )

        # ===================================================================
        # Spatial Aggregation
        # ===================================================================

        # K-hop graph convolution
        self.khop_conv = KHopGraphConv(
            in_features=hidden_dim,
            out_features=hidden_dim,
            k_hops=k_hops,
            n_bases=n_bases,
            aggregation='attention',
        )

        # Standard graphon aggregator (for comparison)
        self.graphon_agg = GraphonAggregator(
            add_self_loops=True,
            normalize=True,
        )

        # Larval transport module (optional)
        if use_larval_transport:
            self.larval_transport = LarvalTransportModule(
                hidden_dim=hidden_dim,
                n_bases=n_bases,
            )
        else:
            self.larval_transport = None

        # ===================================================================
        # Temporal Dynamics (Liquid Cell)
        # ===================================================================

        # Sea lice-specific dynamics cell (with input denormalization)
        self.dynamics_cell = SeaLiceDynamicsCell(
            env_dim=5,  # Will extract from input
            hidden_dim=hidden_dim,
            n_bases=n_bases,
            tau_min=tau_min,
            tau_max=tau_max,
            temp_mean=temp_mean,
            temp_std=temp_std,
            sal_mean=sal_mean,
            sal_std=sal_std,
        )

        # ===================================================================
        # Physics Constraints
        # ===================================================================

        if use_mass_conservation:
            self.mass_conservation = PopulationMassConservation(
                hidden_dim=hidden_dim,
                n_bases=n_bases,
            )
        else:
            self.mass_conservation = None

        # ===================================================================
        # Output Processing
        # ===================================================================

        # Output decoder: hidden -> predictions
        self.output_decoder = FastKAN(
            in_features=hidden_dim,
            out_features=output_dim,
            n_bases=n_bases,
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Learnable initial hidden state
        self.h0 = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        time_points: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        feature_indices: Optional[Dict[str, int]] = None,
        return_auxiliary: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through the network.

        Args:
            x: Input features, shape (B, T, N, F) or (T, N, F)
            edge_index: Graph edges, shape (2, E)
            time_points: Normalized time points, shape (T,)
            edge_weight: Optional edge weights, shape (E,)
            edge_attr: Optional edge attributes for larval transport, shape (E, 4)
            feature_indices: Dict mapping feature names to indices
                Expected keys: 'temperature', 'salinity', 'current_u', 'current_v', 'treatment'
            return_auxiliary: Whether to return auxiliary outputs

        Returns:
            predictions: Output predictions, shape (B, T, N, output_dim) or (T, N, output_dim)
            auxiliary: Optional dict with biological factors, tau, etc.
        """
        # Handle batched vs non-batched input
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        B, T, N, D = x.shape  # D=feature dimension (avoid shadowing F=torch.nn.functional)
        device = x.device

        # Default feature indices if not provided
        if feature_indices is None:
            # Assume autoregressive features first, then environmental
            # AR features: 8 (2 lags * 4)
            # Then: temp (8), salinity (9), current_u (10), current_v (11), treatment (12-14)
            feature_indices = {
                'temperature': 8,
                'salinity': 9,
                'current_u': 10,
                'current_v': 11,
                'treatment': 12,  # First treatment feature
            }

        # Compute time steps
        if time_points is None:
            dt = torch.ones(T, device=device) / T
        else:
            dt = torch.zeros(T, device=device)
            dt[1:] = time_points[1:] - time_points[:-1]
            dt[0] = dt[1] if T > 1 else 1.0 / T

        # Initialize hidden states
        h = self.h0.expand(B, N, -1).clone()

        # Storage for outputs and auxiliary
        outputs = []
        auxiliary = {
            'tau': [],
            'dev_rate': [],
            'survival': [],
            'mass_residual': [],
        }

        # Temporal unrolling
        for t in range(T):
            x_t = x[:, t, :, :]  # (B, N, F)

            # ===============================================================
            # Step 1: Extract environmental features
            # ===============================================================
            temp = x_t[..., feature_indices['temperature']]  # (B, N)
            sal = x_t[..., feature_indices['salinity']]  # (B, N)

            # Handle multi-batch (use reshape for non-contiguous tensors from DataLoader)
            temp_flat = temp.reshape(-1)  # (B*N,)
            sal_flat = sal.reshape(-1)

            # Compute biological factors
            dev_rate = self.belehradek(temp_flat).reshape(B, N)  # (B, N)
            survival = self.salinity_survival(sal_flat).reshape(B, N)  # (B, N)

            # ===============================================================
            # Step 2: Encode input
            # ===============================================================
            u_t = self.input_encoder(x_t)  # (B, N, hidden_dim)
            u_t = self.dropout(u_t)

            # ===============================================================
            # Step 3: Spatial aggregation (per batch)
            # ===============================================================
            h_spatial = []
            for b in range(B):
                h_b = h[b]  # (N, hidden_dim)

                # K-hop convolution
                h_khop = self.khop_conv(h_b, edge_index, edge_weight)

                # Larval transport (if enabled and edge_attr provided)
                if self.larval_transport is not None and edge_attr is not None:
                    pressure = self.larval_transport(h_b, edge_index, edge_attr)
                else:
                    # Fall back to standard aggregation
                    pressure = self.graphon_agg(h_b, edge_index, edge_weight)

                # Combine
                h_agg = h_khop + pressure
                h_spatial.append(h_agg)

            h_spatial = torch.stack(h_spatial, dim=0)  # (B, N, hidden_dim)

            # ===============================================================
            # Step 4: Extract environment for dynamics cell
            # ===============================================================
            # Build environment tensor
            env = torch.stack([
                temp,
                sal,
                x_t[..., feature_indices['current_u']],
                x_t[..., feature_indices['current_v']],
                x_t[..., feature_indices['treatment']],
            ], dim=-1)  # (B, N, 5)

            # ===============================================================
            # Step 5: Update dynamics (per batch)
            # ===============================================================
            h_new_list = []
            tau_list = []

            for b in range(B):
                h_b = h[b]  # (N, hidden_dim)
                env_b = env[b]  # (N, 5)
                pressure_b = h_spatial[b]  # (N, hidden_dim)

                h_new_b, aux_b = self.dynamics_cell(
                    h=h_b,
                    env=env_b,
                    pressure=pressure_b,
                    dt=dt[t],
                )

                h_new_list.append(h_new_b)
                tau_list.append(aux_b['tau'])

            h_new = torch.stack(h_new_list, dim=0)  # (B, N, hidden_dim)

            # Layer norm and residual
            h_new = self.layer_norm(h_new)
            h_new = h_new + u_t  # Residual connection

            # Mass conservation residual (if enabled)
            if self.mass_conservation is not None and return_auxiliary:
                for b in range(B):
                    pressure_in = h_spatial[b].norm(dim=-1).mean()
                    pressure_out = h[b].norm(dim=-1).mean()
                    residual = self.mass_conservation(
                        h[b], h_new[b], pressure_in, pressure_out, dt[t].item()
                    )
                    auxiliary['mass_residual'].append(residual)

            # Update hidden state
            h = h_new

            # ===============================================================
            # Step 6: Decode output
            # ===============================================================
            y_t = self.output_decoder(h)  # (B, N, output_dim)
            y_t = F.softplus(y_t)  # Non-negative outputs

            outputs.append(y_t)

            # Store auxiliary
            if return_auxiliary:
                auxiliary['tau'].append(torch.stack(tau_list, dim=0))
                auxiliary['dev_rate'].append(dev_rate)
                auxiliary['survival'].append(survival)

        # Stack outputs
        predictions = torch.stack(outputs, dim=1)  # (B, T, N, output_dim)

        if squeeze_batch:
            predictions = predictions.squeeze(0)

        # Process auxiliary outputs
        if return_auxiliary:
            auxiliary['tau'] = torch.stack(auxiliary['tau'], dim=1)
            auxiliary['dev_rate'] = torch.stack(auxiliary['dev_rate'], dim=1)
            auxiliary['survival'] = torch.stack(auxiliary['survival'], dim=1)
            if auxiliary['mass_residual']:
                auxiliary['mass_residual'] = torch.stack(auxiliary['mass_residual'])
            return predictions, auxiliary
        else:
            return predictions, None

    def get_biological_curves(
        self,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract learned biological curves for symbolic regression.

        Returns:
            Dict with 'temperature_development' and 'salinity_survival' curves
        """
        if device is None:
            device = next(self.parameters()).device

        temp_curve = self.belehradek.symbolic_extraction(device=device)
        sal_curve = self.salinity_survival.symbolic_extraction(device=device)

        return {
            'temperature_development': temp_curve,
            'salinity_survival': sal_curve,
        }

    def get_learned_biological_zero(self) -> float:
        """Return learned biological zero temperature from BelehradekKAN."""
        return self.belehradek.get_biological_zero()

    def reset_cache(self):
        """Reset all aggregator caches."""
        self.graphon_agg.reset_cache()


class SeaLicePINNLoss(nn.Module):
    """
    Physics-Informed Neural Network Loss for Sea Lice Prediction.

    Combines:
    1. Data fidelity (Tweedie loss for zero-inflated counts)
    2. Biological constraints (non-negativity, growth bounds)
    3. Mass conservation residual (PDE regularization)
    4. Tau stability regularization

    Args:
        tweedie_p: Tweedie power parameter (1.5 recommended)
        lambda_bio: Weight for biological constraints
        lambda_mass: Weight for mass conservation
        lambda_stability: Weight for tau stability
    """

    def __init__(
        self,
        tweedie_p: float = 1.5,
        lambda_bio: float = 0.1,
        lambda_mass: float = 0.01,
        lambda_stability: float = 0.01,
        max_daily_change: float = 0.3,
    ):
        super().__init__()

        self.tweedie_p = tweedie_p
        self.lambda_bio = lambda_bio
        self.lambda_mass = lambda_mass
        self.lambda_stability = lambda_stability
        self.max_daily_change = max_daily_change

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        auxiliary: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute physics-informed loss.

        Args:
            predictions: Model predictions, shape (B, T, N, output_dim)
            targets: Ground truth, shape (B, T, N, output_dim)
            mask: Observation mask, shape (B, T, N)
            auxiliary: Optional auxiliary outputs from model (tau, mass_residual)

        Returns:
            total_loss: Scalar loss
            metrics: Component metrics dict
        """
        metrics = {}

        # ===================================================================
        # 1. Data Fidelity: Tweedie Loss
        # ===================================================================
        l_data = self._tweedie_loss(predictions, targets, mask)
        metrics['l_data'] = l_data.item()

        # ===================================================================
        # 2. Biological Constraints
        # ===================================================================
        l_bio, bio_metrics = self._biological_loss(predictions)
        metrics.update(bio_metrics)
        metrics['l_bio'] = l_bio.item()

        # ===================================================================
        # 3. Mass Conservation (if available)
        # ===================================================================
        if auxiliary is not None and 'mass_residual' in auxiliary and len(auxiliary['mass_residual']) > 0:
            l_mass = auxiliary['mass_residual'].mean()
            metrics['l_mass'] = l_mass.item()
        else:
            l_mass = torch.tensor(0.0, device=predictions.device)
            metrics['l_mass'] = 0.0

        # ===================================================================
        # 4. Tau Stability (if available)
        # ===================================================================
        if auxiliary is not None and 'tau' in auxiliary:
            tau = auxiliary['tau']
            l_stability = self._stability_loss(tau)
            metrics['l_stability'] = l_stability.item()
        else:
            l_stability = torch.tensor(0.0, device=predictions.device)
            metrics['l_stability'] = 0.0

        # ===================================================================
        # Total Loss
        # ===================================================================
        total_loss = (
            l_data
            + self.lambda_bio * l_bio
            + self.lambda_mass * l_mass
            + self.lambda_stability * l_stability
        )

        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics

    def _tweedie_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Tweedie deviance loss for zero-inflated counts."""
        if mask.dim() == predictions.dim() - 1:
            mask_expanded = mask.unsqueeze(-1).expand_as(predictions)
        else:
            mask_expanded = mask

        p = self.tweedie_p
        mu = predictions.clamp(min=1e-6)

        term1 = torch.pow(mu, 2 - p) / (2 - p)
        term2 = targets * torch.pow(mu, 1 - p) / (1 - p)
        loss = term1 - term2

        masked_loss = loss * mask_expanded.float()
        n_valid = mask_expanded.float().sum()

        if n_valid > 0:
            return masked_loss.sum() / n_valid
        return torch.tensor(0.0, device=predictions.device)

    def _biological_loss(
        self,
        predictions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Biological constraint losses."""
        metrics = {}

        # Non-negativity
        l_nonneg = F.relu(-predictions).mean()
        metrics['l_nonneg'] = l_nonneg.item()

        # Growth rate bounds
        if predictions.shape[1] > 1:
            delta_h = predictions[:, 1:] - predictions[:, :-1]
            threshold = self.max_daily_change * (predictions[:, :-1].abs() + 0.1)
            l_growth = F.relu(delta_h.abs() - threshold).mean()
        else:
            l_growth = torch.tensor(0.0, device=predictions.device)
        metrics['l_growth'] = l_growth.item()

        l_bio = l_nonneg + 0.1 * l_growth
        return l_bio, metrics

    def _stability_loss(self, tau: torch.Tensor) -> torch.Tensor:
        """Tau stability regularization."""
        tau_safe = tau.clamp(min=1e-6)
        l_tau = (tau_safe - 1.0) ** 2
        l_inv_tau = (1.0 / tau_safe - 1.0) ** 2
        return (l_tau + l_inv_tau).mean()


# ============================================================================
# Complete Training Wrapper
# ============================================================================

class SeaLicePredictor(nn.Module):
    """
    Complete Sea Lice Predictor with training utilities.

    Wraps SeaLiceGLKAN with loss computation and prediction interface.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 3,
        n_bases: int = 8,
        k_hops: int = 3,
        # Input denormalization for biological modules (when data is z-scored)
        temp_mean: Optional[float] = None,
        temp_std: Optional[float] = None,
        sal_mean: Optional[float] = None,
        sal_std: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()

        self.network = SeaLiceGLKAN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_bases=n_bases,
            k_hops=k_hops,
            temp_mean=temp_mean,
            temp_std=temp_std,
            sal_mean=sal_mean,
            sal_std=sal_std,
            **kwargs,
        )

        self.loss_fn = SeaLicePINNLoss()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with loss computation.

        Args:
            batch: Dict with 'x', 'y', 'mask', 'edge_index', optional 'edge_attr', 'feature_indices'

        Returns:
            Dict with 'predictions', 'loss', 'metrics'
        """
        x = batch['x']
        y = batch['y']
        mask = batch['mask']
        edge_index = batch['edge_index']
        edge_attr = batch.get('edge_attr', None)
        time_points = batch.get('time_points', None)
        edge_weight = batch.get('edge_weight', None)
        feature_indices = batch.get('feature_indices', None)  # For biological modules

        predictions, auxiliary = self.network(
            x=x,
            edge_index=edge_index,
            time_points=time_points,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            feature_indices=feature_indices,
            return_auxiliary=True,
        )

        loss, metrics = self.loss_fn(predictions, y, mask, auxiliary)

        return {
            'predictions': predictions,
            'loss': loss,
            'metrics': metrics,
            'auxiliary': auxiliary,
        }

    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        feature_indices: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Make predictions without loss computation."""
        self.eval()
        with torch.no_grad():
            predictions, _ = self.network(
                x, edge_index, feature_indices=feature_indices, **kwargs
            )
        return predictions

    def get_biological_insights(self) -> Dict:
        """Extract learned biological relationships."""
        return self.network.get_biological_curves()
