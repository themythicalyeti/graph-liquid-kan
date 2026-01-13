"""
Phase 2.3: Topological Constructor

Builds the spatial adjacency matrix for the Graph-Liquid-KAN architecture.

Key Considerations:
1. The graph should NOT be fully connected (too expensive, O(N^2))
2. Use distance-based threshold (geometric graph)
3. Sea lice larvae drift ~30km on average - use this for edge threshold
4. For Graphon 1/N scaling, we need carefully defined edges

Edge Types:
1. Distance-based: Farms within threshold distance
2. Production Area: Farms in same regulatory zone (optional)
3. Current-weighted: Edges weighted by larval transport probability (advanced)
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from loguru import logger


def haversine_distance(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Calculate Haversine distance between coordinate pairs.

    Args:
        lat1, lon1: First coordinates (degrees)
        lat2, lon2: Second coordinates (degrees)

    Returns:
        Distance in kilometers
    """
    R = 6371.0  # Earth radius in km

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def compute_distance_matrix(
    coords: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise Haversine distance matrix.

    Args:
        coords: (N, 2) array of [lat, lon] coordinates

    Returns:
        (N, N) distance matrix in kilometers
    """
    N = len(coords)
    lat = coords[:, 0]
    lon = coords[:, 1]

    # Broadcast to compute all pairs
    lat1 = lat[:, np.newaxis]
    lat2 = lat[np.newaxis, :]
    lon1 = lon[:, np.newaxis]
    lon2 = lon[np.newaxis, :]

    return haversine_distance(lat1, lon1, lat2, lon2)


def build_topology(
    nodes_path: Union[str, Path],
    distance_threshold_km: float = 30.0,
    include_self_loops: bool = False,
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Build spatial graph topology from node coordinates.

    Creates edges between farms within the distance threshold.
    Sea lice larvae typically drift ~30km, so this is the biologically
    motivated default threshold.

    Args:
        nodes_path: Path to graph_nodes_metadata.csv
        distance_threshold_km: Maximum distance for edge creation
        include_self_loops: Whether to include self-loops
        output_path: Optional path to save the graph

    Returns:
        Tuple of:
        - edge_index: (2, E) tensor of [src, dst] edges
        - degree: (N,) tensor of node degrees
        - dist_matrix: (N, N) distance matrix
    """
    logger.info("=" * 60)
    logger.info("PHASE 2.3: BUILDING SPATIAL TOPOLOGY")
    logger.info("=" * 60)

    # Load node coordinates
    nodes_df = pd.read_csv(nodes_path)
    coords = nodes_df[["latitude", "longitude"]].values
    N = len(coords)

    logger.info(f"Nodes: {N}")
    logger.info(f"Distance threshold: {distance_threshold_km} km")

    # Compute distance matrix
    logger.info("Computing pairwise distances...")
    dist_matrix = compute_distance_matrix(coords)

    logger.info(f"Distance matrix shape: {dist_matrix.shape}")
    logger.info(f"Distance range: {dist_matrix.min():.1f} - {dist_matrix.max():.1f} km")
    logger.info(f"Mean distance: {dist_matrix.mean():.1f} km")

    # Create adjacency matrix
    adj = (dist_matrix < distance_threshold_km).astype(np.float32)

    if not include_self_loops:
        np.fill_diagonal(adj, 0)

    # Convert to COO format (edge_index) for PyTorch Geometric
    src, dst = np.where(adj > 0)
    edge_index = torch.from_numpy(np.array([src, dst])).long()

    # Compute node degrees
    degree = torch.tensor(adj.sum(axis=1), dtype=torch.float32)

    # Statistics
    n_edges = edge_index.shape[1]
    avg_degree = degree.mean().item()
    max_degree = degree.max().item()
    min_degree = degree.min().item()
    isolated = (degree == 0).sum().item()

    logger.info(f"\nGraph Statistics:")
    logger.info(f"  Nodes: {N}")
    logger.info(f"  Edges: {n_edges}")
    logger.info(f"  Density: {n_edges / (N * (N - 1)):.4f}")
    logger.info(f"  Average degree: {avg_degree:.1f}")
    logger.info(f"  Degree range: {min_degree:.0f} - {max_degree:.0f}")
    logger.info(f"  Isolated nodes: {isolated}")

    if isolated > 0:
        logger.warning(f"Found {isolated} isolated nodes (no neighbors within {distance_threshold_km}km)")

    # Save if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "edge_index": edge_index,
            "degree": degree,
            "distance_matrix": dist_matrix,
            "distance_threshold_km": distance_threshold_km,
            "n_nodes": N,
            "n_edges": n_edges,
        }, output_path)

        logger.info(f"\nSaved graph to: {output_path}")

    logger.info("=" * 60)

    return edge_index, degree, dist_matrix


def compute_edge_weights(
    edge_index: torch.Tensor,
    dist_matrix: np.ndarray,
    weight_type: str = "inverse",
    sigma: float = 10.0,
) -> torch.Tensor:
    """
    Compute edge weights based on distance.

    Weight types:
    - "inverse": 1 / (1 + dist)
    - "gaussian": exp(-dist^2 / (2 * sigma^2))
    - "uniform": All edges have weight 1

    Args:
        edge_index: (2, E) edge indices
        dist_matrix: (N, N) distance matrix
        weight_type: Type of weighting
        sigma: Scale parameter for Gaussian weights

    Returns:
        (E,) tensor of edge weights
    """
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    distances = dist_matrix[src, dst]

    if weight_type == "inverse":
        weights = 1.0 / (1.0 + distances)
    elif weight_type == "gaussian":
        weights = np.exp(-distances ** 2 / (2 * sigma ** 2))
    elif weight_type == "uniform":
        weights = np.ones_like(distances)
    else:
        raise ValueError(f"Unknown weight type: {weight_type}")

    return torch.tensor(weights, dtype=torch.float32)


def build_production_area_edges(
    nodes_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edges based on production area membership.

    Farms in the same production area (PO1-PO13) are connected.
    This creates a hierarchical structure useful for regulatory modeling.

    Args:
        nodes_path: Path to graph_nodes_metadata.csv
        output_path: Optional path to save the graph

    Returns:
        Tuple of (edge_index, degree)
    """
    logger.info("Building production area graph...")

    nodes_df = pd.read_csv(nodes_path)

    if "production_area" not in nodes_df.columns:
        logger.warning("No production_area column found, returning empty graph")
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(len(nodes_df))

    N = len(nodes_df)
    adj = np.zeros((N, N), dtype=np.float32)

    # Group by production area
    for pa, group in nodes_df.groupby("production_area"):
        if pd.isna(pa):
            continue

        indices = group.index.tolist()

        # Connect all farms in same production area
        for i in indices:
            for j in indices:
                if i != j:
                    adj[i, j] = 1.0

    # Convert to edge_index
    src, dst = np.where(adj > 0)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    degree = torch.tensor(adj.sum(axis=1), dtype=torch.float32)

    logger.info(f"Production area edges: {edge_index.shape[1]}")

    if output_path is not None:
        torch.save({
            "edge_index": edge_index,
            "degree": degree,
        }, output_path)

    return edge_index, degree


def merge_graphs(
    edge_indices: list,
    n_nodes: int,
) -> torch.Tensor:
    """
    Merge multiple edge index tensors into a single graph.

    Useful for combining distance-based and production area edges.

    Args:
        edge_indices: List of (2, E_i) edge index tensors
        n_nodes: Total number of nodes

    Returns:
        Merged (2, E_total) edge index with duplicates removed
    """
    # Concatenate all edges
    merged = torch.cat(edge_indices, dim=1)

    # Remove duplicates
    edge_set = set()
    unique_edges = []

    for i in range(merged.shape[1]):
        edge = (merged[0, i].item(), merged[1, i].item())
        if edge not in edge_set:
            edge_set.add(edge)
            unique_edges.append(edge)

    if len(unique_edges) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    unique_tensor = torch.tensor(unique_edges, dtype=torch.long).T

    return unique_tensor
