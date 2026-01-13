"""
Phase 3.3: Liquid-KAN Cell (The Core Engine)

This module implements the Liquid Time-Constant dynamics using the
Closed-form Continuous (CfC) update rule, parameterized by KAN layers.

The "Liquid" Concept:
In standard ODEs, the time constant tau is fixed. In Liquid networks,
tau varies based on input - the system becomes more "liquid" (reactive)
when inputs are intense, and more "frozen" (retentive) when inputs are stable.

Key Equations:
    tau(context) = softplus(KAN_tau(context)) + epsilon
    x_equilibrium = KAN_eq(context)
    decay = exp(-dt / tau)
    h_new = decay * h_old + (1 - decay) * x_equilibrium

This closed-form solution is numerically stable even for large or
irregular time steps, unlike Euler integration.

Physics Interpretation:
- tau: How fast the system reacts (e.g., high temp -> low tau -> fast growth)
- x_equilibrium: Steady-state / carrying capacity given current environment
- decay: Memory retention factor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .layers import FastKAN, GraphonAggregator


class LiquidKANCell(nn.Module):
    """
    Liquid Time-Constant Cell with KAN-parameterized dynamics.

    This is the heart of the Graph-Liquid-KAN architecture.
    It replaces standard RNN update rules with a CfC (Closed-form Continuous)
    update rule where the time constants are learned via KAN layers.

    The cell receives:
    - Current hidden state h (previous biological state)
    - Environmental input u (temperature, salinity, currents)
    - Ambient infection pressure p (aggregated from graph neighbors)
    - Time step dt (for continuous-time integration)

    And computes:
    - New hidden state h_new using exact ODE integration

    Args:
        input_dim: Dimension of environmental input u
        hidden_dim: Dimension of hidden state h
        pressure_dim: Dimension of infection pressure p (usually = hidden_dim)
        n_bases: Number of RBF bases for KAN layers
        tau_min: Minimum time constant (MUST be >= 1.0 to prevent saturation)
        tau_max: Maximum time constant (prevents frozen dynamics)

    IMPORTANT: tau_min MUST be >= 1.0 to prevent LTC saturation.
    If tau is too small (e.g., 0.01), the decay term exp(-dt/tau) ≈ 0,
    causing the hidden state to be wiped out and converge to dataset mean.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        pressure_dim: Optional[int] = None,
        n_bases: int = 8,
        tau_min: float = 1.0,  # CRITICAL: Must be >= 1.0 to prevent saturation
        tau_max: float = 10.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pressure_dim = pressure_dim or hidden_dim
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Context vector dimension: u + p
        context_dim = input_dim + self.pressure_dim

        # KAN layer for time constant tau
        # Output is scalar per hidden dimension
        self.kan_tau = FastKAN(
            in_features=context_dim,
            out_features=hidden_dim,
            n_bases=n_bases,
        )

        # KAN layer for equilibrium/target state
        self.kan_equilibrium = FastKAN(
            in_features=context_dim,
            out_features=hidden_dim,
            n_bases=n_bases,
        )

        # Optional: KAN layer for input gating
        self.kan_gate = FastKAN(
            in_features=context_dim,
            out_features=hidden_dim,
            n_bases=n_bases,
        )

    def forward(
        self,
        h: torch.Tensor,
        u: torch.Tensor,
        p: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single time step update using CfC dynamics.

        Args:
            h: Current hidden state, shape (N, hidden_dim) or (B, N, hidden_dim)
            u: Environmental input, shape (N, input_dim) or (B, N, input_dim)
            p: Infection pressure from neighbors, shape (N, pressure_dim) or (B, N, pressure_dim)
            dt: Time step, scalar or shape matching h

        Returns:
            h_new: Updated hidden state, same shape as h
        """
        # Step 1: Construct context vector
        # Concatenate environmental input and infection pressure
        context = torch.cat([u, p], dim=-1)  # (..., context_dim)

        # Step 2: Calculate Liquid Time Constant tau
        # KAN output -> Softplus (ensures positive) -> clamp to [tau_min, tau_max]
        tau_raw = self.kan_tau(context)
        tau = F.softplus(tau_raw) + self.tau_min
        tau = torch.clamp(tau, min=self.tau_min, max=self.tau_max)

        # Step 3: Calculate Equilibrium Target
        # This represents the steady-state the system would reach
        # given infinite time at current environmental conditions
        x_equilibrium = self.kan_equilibrium(context)

        # Step 4: Calculate Input Gate (optional modulation)
        gate = torch.sigmoid(self.kan_gate(context))

        # Step 5: Closed-form Continuous (CfC) Integration
        # Exact solution to: dh/dt = (1/tau) * (x_eq - h)
        # h(t+dt) = h(t) * exp(-dt/tau) + x_eq * (1 - exp(-dt/tau))

        # Handle scalar or tensor dt
        if isinstance(dt, (int, float)):
            dt = torch.tensor(dt, device=h.device, dtype=h.dtype)

        # Ensure dt has correct shape for broadcasting
        while dt.dim() < tau.dim():
            dt = dt.unsqueeze(-1)

        # Compute decay factor
        decay = torch.exp(-dt / tau)

        # Update state
        h_new = decay * h + (1 - decay) * x_equilibrium * gate

        return h_new

    def get_tau(
        self,
        u: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the time constant for given inputs (for inspection/debugging).

        Args:
            u: Environmental input
            p: Infection pressure

        Returns:
            tau: Time constants
        """
        context = torch.cat([u, p], dim=-1)
        tau_raw = self.kan_tau(context)
        tau = F.softplus(tau_raw) + self.tau_min
        tau = torch.clamp(tau, min=self.tau_min, max=self.tau_max)
        return tau


class GraphLiquidKANCell(nn.Module):
    """
    Graph-aware Liquid-KAN Cell that combines spatial and temporal dynamics.

    This cell integrates:
    1. Graphon Aggregation: Compute infection pressure from neighbors
    2. Liquid-KAN Dynamics: Update state using CfC rule

    The separation allows for different graph topologies without
    changing the core dynamics.

    Args:
        input_dim: Dimension of environmental input
        hidden_dim: Dimension of hidden state
        n_bases: Number of RBF bases for KAN layers
        add_self_loops: Whether to add self-loops in graph aggregation
        tau_min: Minimum time constant (MUST be >= 1.0)
        tau_max: Maximum time constant
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_bases: int = 8,
        add_self_loops: bool = True,
        tau_min: float = 1.0,  # CRITICAL: Must be >= 1.0 to prevent saturation
        tau_max: float = 10.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Graph aggregator for infection pressure
        self.aggregator = GraphonAggregator(
            add_self_loops=add_self_loops,
            normalize=True,
        )

        # Projection from hidden to pressure space
        # (in case they have different dimensions)
        self.pressure_proj = FastKAN(
            in_features=hidden_dim,
            out_features=hidden_dim,
            n_bases=n_bases,
        )

        # Core liquid dynamics cell
        self.liquid_cell = LiquidKANCell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            pressure_dim=hidden_dim,
            n_bases=n_bases,
            tau_min=tau_min,
            tau_max=tau_max,
        )

    def forward(
        self,
        h: torch.Tensor,
        u: torch.Tensor,
        edge_index: torch.Tensor,
        dt: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single time step with graph-based spatial mixing.

        Args:
            h: Current hidden state, shape (N, hidden_dim) or (B, N, hidden_dim)
            u: Environmental input, shape (N, input_dim) or (B, N, input_dim)
            edge_index: Graph edges, shape (2, E)
            dt: Time step
            edge_weight: Optional edge weights, shape (E,)

        Returns:
            h_new: Updated hidden state
        """
        # Step A: Compute infection pressure via graph aggregation
        # Uses previous state to compute "ambient" pressure from neighbors
        h_aggregated = self.aggregator(h, edge_index, edge_weight)

        # Project aggregated signal to pressure space
        p = self.pressure_proj(h_aggregated)

        # Step B: Update state using liquid dynamics
        h_new = self.liquid_cell(h, u, p, dt)

        return h_new

    def reset_cache(self):
        """Reset aggregator cache."""
        self.aggregator.reset_cache()

    def get_tau(
        self,
        h: torch.Tensor,
        u: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get time constants for debugging."""
        h_aggregated = self.aggregator(h, edge_index, edge_weight)
        p = self.pressure_proj(h_aggregated)
        return self.liquid_cell.get_tau(u, p)
