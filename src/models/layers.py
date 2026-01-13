"""
Phase 3.1 & 3.2: Atomic Layers for Graph-Liquid-KAN

This module contains the fundamental building blocks:
1. FastKAN: Kolmogorov-Arnold Network using Gaussian RBF basis functions
2. GraphonAggregator: Graphon-compliant message passing with 1/N normalization

Design Principles:
- No "black box" layers (nn.Linear only for final projection)
- All transformations mathematically explicit
- Layer normalization as "safety interlock" for RBF activation

Warning (Memory Hazard):
The RBF expansion increases dimensionality: (batch, features) -> (batch, features, n_bases)
Apply KAN expansion AFTER graph aggregation, not before, to avoid VRAM explosion.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FastKAN(nn.Module):
    """
    Fast Kolmogorov-Arnold Network Layer using Gaussian RBF basis functions.

    Unlike MLPs which apply fixed non-linearities after linear transforms,
    KANs learn the non-linearity itself via basis function expansion.

    Architecture:
    1. Input normalization (LayerNorm) - critical safety interlock
    2. RBF basis expansion: x -> phi(x) where phi are Gaussians
    3. Learnable weight aggregation: sum over basis functions

    The grid centers are fixed (non-learnable) and evenly spaced in [-1, 1].
    This defines the "resolution" of learnable functions.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        n_bases: Number of RBF basis functions (grid resolution)
        grid_range: Range for grid centers (default: [-1, 1])
        sigma: Width of Gaussian RBFs (default: auto from grid spacing)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_bases: int = 8,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        sigma: Optional[float] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_bases = n_bases
        self.grid_range = grid_range

        # Layer normalization - CRITICAL safety interlock
        # Without this, inputs outside [-1, 1] will have zero gradients
        self.layer_norm = nn.LayerNorm(in_features)

        # Create fixed grid centers (non-learnable)
        # Evenly spaced between grid_range[0] and grid_range[1]
        grid_centers = torch.linspace(grid_range[0], grid_range[1], n_bases)
        self.register_buffer("grid_centers", grid_centers)

        # Compute sigma from grid spacing if not provided
        if sigma is None:
            # Use half the grid spacing for good overlap
            grid_spacing = (grid_range[1] - grid_range[0]) / (n_bases - 1)
            sigma = grid_spacing / 2.0
        self.sigma = sigma

        # Precompute denominator for Gaussian: 2 * sigma^2
        self.register_buffer(
            "gaussian_denom",
            torch.tensor(2.0 * sigma * sigma)
        )

        # Learnable weights for basis aggregation
        # Shape: (in_features, n_bases, out_features)
        # This learns the "shape" of the curve for each input->output connection
        self.weights = nn.Parameter(
            torch.empty(in_features, n_bases, out_features)
        )

        # Learnable bias (optional but helps with output centering)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using scaled uniform distribution."""
        # Scale by sqrt(1 / (in_features * n_bases)) for stable gradients
        scale = 1.0 / math.sqrt(self.in_features * self.n_bases)
        nn.init.uniform_(self.weights, -scale, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through FastKAN layer.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Store original shape for reshaping
        original_shape = x.shape[:-1]

        # Step 1: Layer Normalization (safety interlock)
        # Forces input into active range of RBF grid
        x = self.layer_norm(x)

        # Step 2: RBF Basis Expansion
        # x shape: (..., in_features)
        # grid_centers shape: (n_bases,)
        # We want: (..., in_features, n_bases)

        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(-1)  # (..., in_features, 1)
        centers = self.grid_centers.view(1, 1, -1)  # (1, 1, n_bases)

        # Compute squared distances to each grid center
        # Broadcasting: (..., in_features, 1) - (1, 1, n_bases) -> (..., in_features, n_bases)
        distances_sq = (x_expanded - centers) ** 2

        # Apply Gaussian: exp(-d^2 / (2*sigma^2))
        basis_activations = torch.exp(-distances_sq / self.gaussian_denom)
        # Shape: (..., in_features, n_bases)

        # Step 3: Weight Aggregation
        # basis_activations: (..., in_features, n_bases)
        # weights: (in_features, n_bases, out_features)
        # Output: (..., out_features)

        # Use einsum for efficient batched matrix multiplication
        # 'bif,ifo->bo' where b=batch (flattened), i=in_features, f=n_bases, o=out_features
        output = torch.einsum(
            "...if,ifo->...o",
            basis_activations,
            self.weights
        )

        # Add bias
        output = output + self.bias

        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"n_bases={self.n_bases}, "
            f"sigma={self.sigma:.4f}"
        )


class GraphonAggregator(nn.Module):
    """
    Graphon-Compliant Graph Aggregator with 1/N normalization.

    For physical systems (like ocean-based infection spread), as the number
    of nodes N increases, the total signal should converge to a mean-field
    integral, not explode. This requires careful normalization.

    Normalization Strategy:
    - Row normalization: A_norm[i,j] = A[i,j] / degree[i]
    - This ensures a farm with 50 neighbors receives the AVERAGE infection
      pressure, not 50x the pressure of a farm with 1 neighbor.

    The aggregator computes:
        h_aggregated[i] = (1/d_i) * sum_j A[i,j] * h[j]

    where d_i is the degree of node i.

    Args:
        add_self_loops: Whether to add self-loops before aggregation
        normalize: Whether to apply row normalization
    """

    def __init__(
        self,
        add_self_loops: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        # Cache for normalized adjacency (computed once per graph)
        self._cached_adj_norm: Optional[torch.Tensor] = None
        self._cached_n_nodes: Optional[int] = None

    def _get_normalized_adjacency(
        self,
        edge_index: torch.Tensor,
        n_nodes: int,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute row-normalized sparse adjacency matrix.

        Args:
            edge_index: (2, E) tensor of [src, dst] edges
            n_nodes: Number of nodes
            edge_weight: Optional (E,) tensor of edge weights

        Returns:
            Normalized sparse adjacency matrix
        """
        # Check cache
        if (self._cached_adj_norm is not None and
            self._cached_n_nodes == n_nodes and
            self._cached_adj_norm.device == edge_index.device):
            return self._cached_adj_norm

        src, dst = edge_index[0], edge_index[1]

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=edge_index.device)

        # Add self-loops if requested
        if self.add_self_loops:
            loop_index = torch.arange(n_nodes, device=edge_index.device)
            loop_weight = torch.ones(n_nodes, device=edge_index.device)

            src = torch.cat([src, loop_index])
            dst = torch.cat([dst, loop_index])
            edge_weight = torch.cat([edge_weight, loop_weight])

        # Compute degree for normalization
        if self.normalize:
            # Row-sum normalization (degree normalization)
            degree = torch.zeros(n_nodes, device=edge_index.device)
            degree.scatter_add_(0, src, edge_weight)

            # Avoid division by zero
            degree = degree.clamp(min=1.0)

            # Normalize weights by source degree
            edge_weight = edge_weight / degree[src]

        # Create sparse adjacency matrix
        indices = torch.stack([src, dst], dim=0)
        adj_sparse = torch.sparse_coo_tensor(
            indices,
            edge_weight,
            size=(n_nodes, n_nodes),
        ).coalesce()

        # Cache for reuse
        self._cached_adj_norm = adj_sparse
        self._cached_n_nodes = n_nodes

        return adj_sparse

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        n_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Aggregate node features using graph topology.

        Args:
            x: Node features of shape (N, F) or (B, N, F)
            edge_index: Edge indices of shape (2, E)
            edge_weight: Optional edge weights of shape (E,)
            n_nodes: Number of nodes (inferred from x if not provided)

        Returns:
            Aggregated features of same shape as input
        """
        # Handle batched input
        if x.dim() == 3:
            # Batched: (B, N, F) -> process each batch
            batch_size = x.shape[0]
            results = []
            for b in range(batch_size):
                result = self._aggregate_single(
                    x[b], edge_index, edge_weight, n_nodes
                )
                results.append(result)
            return torch.stack(results, dim=0)
        else:
            return self._aggregate_single(x, edge_index, edge_weight, n_nodes)

    def _aggregate_single(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        n_nodes: Optional[int],
    ) -> torch.Tensor:
        """Aggregate for a single graph (non-batched)."""
        if n_nodes is None:
            n_nodes = x.shape[0]

        # Get normalized adjacency matrix
        adj_norm = self._get_normalized_adjacency(
            edge_index, n_nodes, edge_weight
        )

        # Sparse matrix multiplication: A_norm @ x
        # adj_norm: (N, N) sparse, x: (N, F) dense -> (N, F) dense
        output = torch.sparse.mm(adj_norm, x)

        return output

    def reset_cache(self):
        """Reset the adjacency cache (call when graph changes)."""
        self._cached_adj_norm = None
        self._cached_n_nodes = None


class GraphKAN(nn.Module):
    """
    Graph KAN Layer: Combines graph aggregation with KAN transformation.

    This is a convenience module that chains:
    1. GraphonAggregator (spatial mixing)
    2. FastKAN (non-linear transformation)

    The order is critical for memory efficiency:
    - Aggregate FIRST (reduces to mean over neighbors)
    - KAN transform SECOND (basis expansion on aggregated features)

    Reversing the order would expand every neighbor's features to n_bases
    dimensions before aggregation, causing VRAM explosion.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        n_bases: Number of RBF basis functions
        add_self_loops: Whether to add self-loops in aggregation
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_bases: int = 8,
        add_self_loops: bool = True,
    ):
        super().__init__()

        self.aggregator = GraphonAggregator(
            add_self_loops=add_self_loops,
            normalize=True,
        )
        self.kan = FastKAN(
            in_features=in_features,
            out_features=out_features,
            n_bases=n_bases,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: aggregate then transform.

        Args:
            x: Node features (N, F) or (B, N, F)
            edge_index: Edge indices (2, E)
            edge_weight: Optional edge weights (E,)

        Returns:
            Transformed features of shape (..., out_features)
        """
        # Step 1: Spatial aggregation (graphon-compliant)
        x_agg = self.aggregator(x, edge_index, edge_weight)

        # Step 2: KAN transformation
        output = self.kan(x_agg)

        return output

    def reset_cache(self):
        """Reset aggregator cache."""
        self.aggregator.reset_cache()
