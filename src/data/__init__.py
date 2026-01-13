"""
Phase 2: Tensor Construction and Temporal Alignment

Modules:
- builder: GLKANDatasetBuilder for temporal alignment of ocean/lice data
- graph: Topological constructor for spatial adjacency matrix
- dataset: SeaLiceGraphDataset for PyTorch Geometric
"""

from .builder import GLKANDatasetBuilder
from .graph import build_topology, compute_edge_weights
from .dataset import SeaLiceGraphDataset

__all__ = [
    "GLKANDatasetBuilder",
    "build_topology",
    "compute_edge_weights",
    "SeaLiceGraphDataset",
]
