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


def compute_flux_weights(
    edge_index: torch.Tensor,
    coords: np.ndarray,
    current_u: np.ndarray,
    current_v: np.ndarray,
    dist_matrix: np.ndarray,
    decay_km: float = 10.0,
) -> torch.Tensor:
    """
    Compute edge weights based on ocean current flux (larval transport).

    This is the KEY function for biologically-relevant graph weighting.
    It computes how much current flows from farm A toward farm B.

    Physics:
        flux(A→B) = current_A · direction(A→B) / |direction(A→B)|

    If flux > 0: water flows from A toward B (A is upstream, larvae drift to B)
    If flux < 0: water flows from B toward A (ignore or set to 0)

    The edge weight combines:
    1. Flux magnitude (how fast water moves toward B)
    2. Distance decay (larvae don't survive long transport)

    Args:
        edge_index: (2, E) edge indices [src, dst]
        coords: (N, 2) array of [lat, lon] coordinates
        current_u: (N,) or (T, N) eastward current at each farm (m/s)
        current_v: (N,) or (T, N) northward current at each farm (m/s)
        dist_matrix: (N, N) distance matrix in km
        decay_km: Distance decay scale for larval survival

    Returns:
        (E,) tensor of flux-based edge weights
    """
    logger.info("Computing flux-based edge weights from ocean currents...")

    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    E = len(src)

    # Handle time-averaged currents
    if current_u.ndim == 2:
        # Average over time
        u_mean = np.nanmean(current_u, axis=0)  # (N,)
        v_mean = np.nanmean(current_v, axis=0)  # (N,)
    else:
        u_mean = current_u
        v_mean = current_v

    # Get coordinates
    lat = coords[:, 0]
    lon = coords[:, 1]

    # Compute direction vectors from src to dst
    # Note: At Norwegian latitudes, 1 degree lat ≈ 111 km, 1 degree lon ≈ 55 km
    lat_scale = 111.0  # km per degree latitude
    lon_scale = 55.0   # km per degree longitude (approximate at 60°N)

    # Direction vector from src to dst (in km)
    dir_east = (lon[dst] - lon[src]) * lon_scale   # (E,) eastward component
    dir_north = (lat[dst] - lat[src]) * lat_scale  # (E,) northward component

    # Distance
    distances = dist_matrix[src, dst]  # (E,)
    distances = np.maximum(distances, 0.1)  # Avoid division by zero

    # Normalize direction
    dir_magnitude = np.sqrt(dir_east**2 + dir_north**2)
    dir_magnitude = np.maximum(dir_magnitude, 0.1)
    dir_east_norm = dir_east / dir_magnitude
    dir_north_norm = dir_north / dir_magnitude

    # Get currents at source farms
    u_src = u_mean[src]  # (E,) eastward current at source
    v_src = v_mean[src]  # (E,) northward current at source

    # Compute flux: dot product of current and direction
    # flux > 0 means current flows from src toward dst
    flux = u_src * dir_east_norm + v_src * dir_north_norm  # (E,)

    # Only keep positive flux (water flowing toward dst)
    flux_positive = np.maximum(flux, 0)

    # Apply distance decay (larvae die during transport)
    distance_factor = np.exp(-distances / decay_km)

    # Final weight: flux magnitude * distance decay
    weights = flux_positive * distance_factor

    # Normalize to [0, 1] range
    if weights.max() > 0:
        weights = weights / weights.max()

    logger.info(f"  Flux weights computed for {E} edges")
    logger.info(f"  Non-zero flux edges: {(weights > 0.01).sum()} ({100*(weights > 0.01).sum()/E:.1f}%)")
    logger.info(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")

    return torch.tensor(weights, dtype=torch.float32)


def build_flux_weighted_graph(
    nodes_path: Union[str, Path],
    hydro_path: Union[str, Path],
    distance_threshold_km: float = 30.0,
    decay_km: float = 10.0,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict:
    """
    Build a complete flux-weighted graph for larval transport.

    This creates edges based on distance threshold, then weights them
    by ocean current flux (how much water flows from source to destination).

    Args:
        nodes_path: Path to graph_nodes_metadata.csv
        hydro_path: Path to hydrography NetCDF file with u, v currents
        distance_threshold_km: Maximum distance for edge creation
        decay_km: Distance decay scale for larval survival
        output_path: Optional path to save the graph

    Returns:
        Dict with edge_index, edge_weights, degree, etc.
    """
    import xarray as xr

    logger.info("=" * 60)
    logger.info("BUILDING FLUX-WEIGHTED SPATIAL GRAPH")
    logger.info("=" * 60)

    # Build basic topology
    edge_index, degree, dist_matrix = build_topology(
        nodes_path, distance_threshold_km, include_self_loops=False
    )

    # Load coordinates and locality IDs
    nodes_df = pd.read_csv(nodes_path)
    coords = nodes_df[["latitude", "longitude"]].values
    locality_ids = nodes_df["locality_id"].values
    N = len(locality_ids)

    # Load hydrography
    logger.info(f"Loading hydrography from: {hydro_path}")
    ds = xr.open_dataset(hydro_path, engine='scipy')

    # Get currents from hydrography (indexed by farm_id)
    hydro_farm_ids = ds['farm_id'].values  # Farm IDs in hydro data
    u_raw = ds['u'].values  # (T, N_hydro)
    v_raw = ds['v'].values  # (T, N_hydro)

    logger.info(f"  Hydro data shape: {u_raw.shape}")
    logger.info(f"  Matching {N} graph nodes to {len(hydro_farm_ids)} hydro farms...")

    # Create mapping from hydro farm_id to index
    hydro_id_to_idx = {fid: i for i, fid in enumerate(hydro_farm_ids)}

    # Map graph nodes to hydro data indices
    u_aligned = np.full((u_raw.shape[0], N), np.nan)
    v_aligned = np.full((v_raw.shape[0], N), np.nan)
    matched = 0

    for i, loc_id in enumerate(locality_ids):
        if loc_id in hydro_id_to_idx:
            hydro_idx = hydro_id_to_idx[loc_id]
            u_aligned[:, i] = u_raw[:, hydro_idx]
            v_aligned[:, i] = v_raw[:, hydro_idx]
            matched += 1

    logger.info(f"  Matched {matched}/{N} nodes to hydrography data")
    logger.info(f"  NaN fraction in u: {np.isnan(u_aligned).mean():.1%}")

    # Compute flux weights using aligned currents
    flux_weights = compute_flux_weights(
        edge_index, coords, u_aligned, v_aligned, dist_matrix, decay_km
    )

    # Also compute distance weights for comparison
    dist_weights = compute_edge_weights(edge_index, dist_matrix, "inverse")

    # Handle NaN values in flux weights - replace with 0
    flux_weights_clean = flux_weights.clone()
    nan_mask = torch.isnan(flux_weights_clean)
    flux_weights_clean[nan_mask] = 0.0

    # Combine: use flux where available, fall back to distance
    # This ensures isolated farms still have some connectivity
    combined_weights = flux_weights_clean.clone()
    low_flux = (flux_weights_clean < 0.01) | nan_mask
    combined_weights[low_flux] = dist_weights[low_flux] * 0.1  # Reduced weight for distance-only

    # Build output dict
    graph_data = {
        "edge_index": edge_index,
        "edge_weights": combined_weights,
        "flux_weights": flux_weights_clean,
        "distance_weights": dist_weights,
        "degree": degree,
        "distance_matrix": dist_matrix,
        "distance_threshold_km": distance_threshold_km,
        "decay_km": decay_km,
        "n_nodes": len(coords),
        "n_edges": edge_index.shape[1],
    }

    # Statistics
    valid_flux = flux_weights_clean[~nan_mask]
    n_flux_edges = (flux_weights_clean > 0.01).sum().item()
    logger.info(f"\nFlux-weighted graph statistics:")
    logger.info(f"  Total edges: {edge_index.shape[1]}")
    logger.info(f"  NaN flux edges: {nan_mask.sum().item()}")
    logger.info(f"  Flux-active edges: {n_flux_edges} ({100*n_flux_edges/edge_index.shape[1]:.1f}%)")
    logger.info(f"  Mean flux weight (valid): {valid_flux.mean():.4f}" if len(valid_flux) > 0 else "  No valid flux weights")

    # Save if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph_data, output_path)
        logger.info(f"\nSaved flux-weighted graph to: {output_path}")

    return graph_data


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


def compute_static_edge_features(
    edge_index: torch.Tensor,
    coords: np.ndarray,
    dist_matrix: np.ndarray,
) -> Dict[str, torch.Tensor]:
    """
    Compute STATIC edge features that don't change over time.

    These are geometric properties of the farm-to-farm connections:
    - Distance between farms
    - Direction unit vector (for computing flux with currents)

    Args:
        edge_index: (2, E) edge indices [src, dst]
        coords: (N, 2) array of [lat, lon] coordinates
        dist_matrix: (N, N) distance matrix in km

    Returns:
        Dict with:
        - edge_distance: (E,) distance in km
        - edge_direction: (E, 2) unit vector [east, north] from src to dst
    """
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    E = len(src)

    # Get coordinates
    lat = coords[:, 0]
    lon = coords[:, 1]

    # Distance per edge
    edge_distance = dist_matrix[src, dst]

    # Direction vectors from src to dst
    # At Norwegian latitudes (~60-70°N):
    # 1 degree lat ≈ 111 km
    # 1 degree lon ≈ 55 km (varies with latitude)
    lat_scale = 111.0  # km per degree latitude
    lon_scale = 55.0   # km per degree longitude (approximate)

    # Direction in km
    dir_east = (lon[dst] - lon[src]) * lon_scale   # (E,) eastward component
    dir_north = (lat[dst] - lat[src]) * lat_scale  # (E,) northward component

    # Normalize to unit vector
    dir_magnitude = np.sqrt(dir_east**2 + dir_north**2)
    dir_magnitude = np.maximum(dir_magnitude, 0.1)  # Avoid division by zero

    edge_direction = np.stack([
        dir_east / dir_magnitude,
        dir_north / dir_magnitude,
    ], axis=1)  # (E, 2)

    return {
        'edge_distance': torch.tensor(edge_distance, dtype=torch.float32),
        'edge_direction': torch.tensor(edge_direction, dtype=torch.float32),
    }


def compute_dynamic_edge_attr(
    edge_index: torch.Tensor,
    edge_direction: torch.Tensor,
    edge_distance: torch.Tensor,
    node_features: torch.Tensor,
    feature_indices: Dict[str, int],
    decay_km: float = 15.0,
) -> torch.Tensor:
    """
    Compute DYNAMIC edge attributes from current node features.

    This is the key function for physics-informed larval transport.
    Called per-timestep to get current-dependent edge weights.

    Physics:
        flux(A→B) = current_A · direction(A→B)

        If flux > 0: water flows from A toward B (larvae transport)
        If flux < 0: water flows away (no transport)

    Args:
        edge_index: (2, E) edge indices [src, dst]
        edge_direction: (E, 2) unit direction vectors [east, north]
        edge_distance: (E,) distances in km
        node_features: (N, F) current node features
        feature_indices: Dict mapping feature names to indices
        decay_km: Distance decay scale for larval survival

    Returns:
        edge_attr: (E, 4) tensor with [distance, flux, temp_src, sal_src]
    """
    src = edge_index[0]
    dst = edge_index[1]

    # Extract relevant features from nodes
    temp_idx = feature_indices.get('temperature', 11)
    sal_idx = feature_indices.get('salinity', 12)
    u_idx = feature_indices.get('current_u', 13)
    v_idx = feature_indices.get('current_v', 14)

    # Get currents at source farms
    current_u = node_features[src, u_idx]  # (E,) eastward current
    current_v = node_features[src, v_idx]  # (E,) northward current

    # Compute flux: dot product of current and direction
    # flux > 0 means water flows from src toward dst
    flux = (current_u * edge_direction[:, 0] +
            current_v * edge_direction[:, 1])  # (E,)

    # Only keep positive flux (water flowing toward dst)
    flux_positive = torch.clamp(flux, min=0)

    # Apply distance decay (larvae mortality during transport)
    distance_factor = torch.exp(-edge_distance / decay_km)

    # Weighted flux
    weighted_flux = flux_positive * distance_factor

    # Get environmental conditions at source
    temp_src = node_features[src, temp_idx]
    sal_src = node_features[src, sal_idx]

    # Stack into edge_attr: [distance, flux, temp, salinity]
    edge_attr = torch.stack([
        edge_distance,
        weighted_flux,
        temp_src,
        sal_src,
    ], dim=1)  # (E, 4)

    return edge_attr


def compute_dynamic_edge_attr_batched(
    edge_index: torch.Tensor,
    edge_direction: torch.Tensor,
    edge_distance: torch.Tensor,
    node_features: torch.Tensor,
    feature_indices: Dict[str, int],
    decay_km: float = 15.0,
) -> torch.Tensor:
    """
    Batched version of compute_dynamic_edge_attr.

    Handles input with batch and time dimensions.

    Args:
        edge_index: (2, E) edge indices
        edge_direction: (E, 2) unit direction vectors
        edge_distance: (E,) distances in km
        node_features: (B, T, N, F) or (T, N, F) node features
        feature_indices: Dict mapping feature names to indices
        decay_km: Distance decay scale

    Returns:
        edge_attr: (B, T, E, 4) or (T, E, 4) dynamic edge attributes
    """
    # Handle different input shapes
    if node_features.dim() == 3:
        # (T, N, F) -> add batch dim
        node_features = node_features.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    B, T, N, F = node_features.shape
    E = edge_index.shape[1]
    device = node_features.device

    # Move static features to same device
    edge_direction = edge_direction.to(device)
    edge_distance = edge_distance.to(device)

    # Extract feature indices
    temp_idx = feature_indices.get('temperature', 11)
    sal_idx = feature_indices.get('salinity', 12)
    u_idx = feature_indices.get('current_u', 13)
    v_idx = feature_indices.get('current_v', 14)

    src = edge_index[0]  # (E,)

    # Get source node features for all edges: (B, T, E, F)
    # node_features[:, :, src] -> (B, T, E, F)
    src_features = node_features[:, :, src, :]

    # Extract currents at source: (B, T, E)
    current_u = src_features[:, :, :, u_idx]
    current_v = src_features[:, :, :, v_idx]

    # Compute flux: (B, T, E)
    # edge_direction is (E, 2), broadcast over B, T
    flux = (current_u * edge_direction[:, 0].view(1, 1, E) +
            current_v * edge_direction[:, 1].view(1, 1, E))

    # Positive flux only
    flux_positive = torch.clamp(flux, min=0)

    # Distance decay: (E,) -> (1, 1, E)
    distance_factor = torch.exp(-edge_distance / decay_km).view(1, 1, E)
    weighted_flux = flux_positive * distance_factor

    # Environmental at source: (B, T, E)
    temp_src = src_features[:, :, :, temp_idx]
    sal_src = src_features[:, :, :, sal_idx]

    # Stack: (B, T, E, 4)
    edge_attr = torch.stack([
        edge_distance.view(1, 1, E).expand(B, T, E),
        weighted_flux,
        temp_src,
        sal_src,
    ], dim=-1)

    if squeeze_batch:
        edge_attr = edge_attr.squeeze(0)

    return edge_attr


def add_static_edge_features_to_graph(
    graph_path: Union[str, Path],
    nodes_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> Dict:
    """
    Add static edge features (direction, distance) to existing graph.

    This updates spatial_graph.pt with the geometric features needed
    for dynamic flux computation.

    Args:
        graph_path: Path to existing spatial_graph.pt
        nodes_path: Path to graph_nodes_metadata.csv
        output_path: Path to save updated graph (default: overwrite)

    Returns:
        Updated graph dict
    """
    logger.info("Adding static edge features to graph...")

    # Load existing graph
    graph = torch.load(graph_path, map_location='cpu', weights_only=False)
    edge_index = graph['edge_index']
    dist_matrix = graph['distance_matrix']

    # Load coordinates
    nodes_df = pd.read_csv(nodes_path)
    coords = nodes_df[["latitude", "longitude"]].values

    # Compute static edge features
    static_features = compute_static_edge_features(edge_index, coords, dist_matrix)

    # Add to graph
    graph['edge_distance'] = static_features['edge_distance']
    graph['edge_direction'] = static_features['edge_direction']

    logger.info(f"  Added edge_distance: {graph['edge_distance'].shape}")
    logger.info(f"  Added edge_direction: {graph['edge_direction'].shape}")

    # Save
    output_path = output_path or graph_path
    torch.save(graph, output_path)
    logger.info(f"  Saved to: {output_path}")

    return graph
