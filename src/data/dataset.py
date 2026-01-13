"""
Phase 2.4: SeaLiceGraphDataset for PyTorch Geometric

The final assembly that defines how the Graph-Liquid-KAN sees the data.
This class creates sliding window sequences over the spatiotemporal tensor
and wraps them in PyTorch Geometric Data objects.

Key Design:
- Sliding window over time dimension
- Each sample contains (window_size) days of features and labels
- Graph structure (edge_index) is static across time
- Mask indicates which observations are ground truth vs interpolated
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from loguru import logger

# Try to import PyTorch Geometric
try:
    from torch_geometric.data import Data, Dataset as PyGDataset
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    PyGDataset = Dataset  # Fallback to standard PyTorch Dataset


@dataclass
class DatasetConfig:
    """Configuration for SeaLiceGraphDataset."""
    window_size: int = 30  # Days of input context
    prediction_horizon: int = 7  # Days ahead to predict
    stride: int = 7  # Days between windows (weekly sampling)
    min_observations: int = 1  # Minimum valid obs in window


class SeaLiceGraphDataset(Dataset):
    """
    PyTorch Dataset for Sea Lice Graph Temporal Prediction.

    This dataset:
    1. Loads pre-processed tensors from Phase 2.2
    2. Loads graph topology from Phase 2.3
    3. Creates sliding window sequences for Neural ODE training
    4. Returns PyTorch Geometric Data objects (or standard tensors)

    The Graph-Liquid-KAN will process the temporal dimension internally,
    so we pass the full window as a feature tensor.

    Usage:
        dataset = SeaLiceGraphDataset(
            tensor_path="data/processed/tensors.npz",
            graph_path="data/processed/spatial_graph.pt",
            window_size=30,
        )
        sample = dataset[0]
        # sample.x: (T, N, F) features
        # sample.y: (T, N, 3) labels
        # sample.mask: (T, N) observation mask
        # sample.edge_index: (2, E) graph edges
    """

    def __init__(
        self,
        tensor_path: Union[str, Path],
        graph_path: Union[str, Path],
        config: Optional[DatasetConfig] = None,
        split: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ):
        """
        Initialize the dataset.

        Args:
            tensor_path: Path to tensors.npz from Phase 2.2
            graph_path: Path to spatial_graph.pt from Phase 2.3
            config: Dataset configuration
            split: 'train', 'val', 'test', or None for all data
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
        """
        self.config = config or DatasetConfig()
        self.split = split

        # Load tensors
        logger.info(f"Loading tensors from: {tensor_path}")
        data = np.load(tensor_path, allow_pickle=True)

        self.X = torch.from_numpy(data["X"]).float()  # (T, N, F)
        self.Y = torch.from_numpy(data["Y"]).float()  # (T, N, 3)
        self.mask = torch.from_numpy(data["mask"]).bool()  # (T, N)
        self.times = data["times"]  # (T,) datetime array
        self.site_ids = torch.from_numpy(data["site_ids"]).long()  # (N,)

        # Load normalization stats if available
        if "feature_mean" in data and data["feature_mean"] is not None:
            self.feature_mean = torch.from_numpy(data["feature_mean"]).float()
            self.feature_std = torch.from_numpy(data["feature_std"]).float()
        else:
            self.feature_mean = None
            self.feature_std = None

        self.total_time, self.num_nodes, self.num_features = self.X.shape
        self.num_targets = self.Y.shape[-1]

        logger.info(f"Loaded X: {self.X.shape}, Y: {self.Y.shape}")

        # Load graph
        logger.info(f"Loading graph from: {graph_path}")
        graph_data = torch.load(graph_path, weights_only=False)

        self.edge_index = graph_data["edge_index"]
        self.degree = graph_data["degree"]

        logger.info(f"Graph: {self.num_nodes} nodes, {self.edge_index.shape[1]} edges")

        # Create time indices for split
        self._create_split_indices(train_ratio, val_ratio)

        # Create sequence indices
        self._create_sequences()

        logger.info(f"Dataset initialized: {len(self)} sequences")

    def _create_split_indices(self, train_ratio: float, val_ratio: float):
        """Create time indices for train/val/test splits."""
        T = self.total_time

        train_end = int(T * train_ratio)
        val_end = int(T * (train_ratio + val_ratio))

        if self.split == "train":
            self.time_start = 0
            self.time_end = train_end
        elif self.split == "val":
            self.time_start = train_end
            self.time_end = val_end
        elif self.split == "test":
            self.time_start = val_end
            self.time_end = T
        else:
            self.time_start = 0
            self.time_end = T

        logger.info(f"Split '{self.split}': time indices [{self.time_start}, {self.time_end})")

    def _create_sequences(self):
        """Create list of valid sequence start indices."""
        self.sequences = []

        window = self.config.window_size
        horizon = self.config.prediction_horizon
        stride = self.config.stride
        min_obs = self.config.min_observations

        # Iterate through time with stride
        for t_start in range(self.time_start, self.time_end - window - horizon + 1, stride):
            t_end = t_start + window

            # Check if window has minimum observations
            obs_in_window = self.mask[t_start:t_end].sum().item()

            if obs_in_window >= min_obs:
                self.sequences.append({
                    "t_start": t_start,
                    "t_end": t_end,
                    "t_target": t_end + horizon - 1,
                })

        logger.info(f"Created {len(self.sequences)} sequences (stride={stride})")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        """
        Get a single sequence.

        Returns either a PyTorch Geometric Data object (if available)
        or a dictionary of tensors.
        """
        seq = self.sequences[idx]
        t_start = seq["t_start"]
        t_end = seq["t_end"]

        # Extract window
        x = self.X[t_start:t_end]  # (window, N, F)
        y = self.Y[t_start:t_end]  # (window, N, 3)
        mask = self.mask[t_start:t_end]  # (window, N)

        # Create time points normalized to [0, 1]
        time_points = torch.linspace(0, 1, t_end - t_start)

        if HAS_PYG:
            # Return PyTorch Geometric Data object
            data = Data(
                x=x,  # (T, N, F)
                y=y,  # (T, N, 3)
                mask=mask,  # (T, N)
                edge_index=self.edge_index,
                degree=self.degree,
                time_points=time_points,
                num_nodes=self.num_nodes,
            )
            return data
        else:
            # Return dictionary of tensors
            return {
                "x": x,
                "y": y,
                "mask": mask,
                "edge_index": self.edge_index,
                "degree": self.degree,
                "time_points": time_points,
            }

    def get_full_sequence(self) -> Dict[str, torch.Tensor]:
        """
        Get the full time series (all windows concatenated).

        Useful for visualization and analysis.
        """
        return {
            "X": self.X[self.time_start:self.time_end],
            "Y": self.Y[self.time_start:self.time_end],
            "mask": self.mask[self.time_start:self.time_end],
            "times": self.times[self.time_start:self.time_end],
            "edge_index": self.edge_index,
            "site_ids": self.site_ids,
        }


def collate_graph_sequences(batch: List) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching graph sequences.

    Stacks temporal sequences into batch dimension.

    Args:
        batch: List of Data objects or dicts

    Returns:
        Batched dictionary of tensors
    """
    if HAS_PYG and hasattr(batch[0], "x"):
        # PyG Data objects
        x_list = [b.x for b in batch]
        y_list = [b.y for b in batch]
        mask_list = [b.mask for b in batch]
        time_list = [b.time_points for b in batch]

        # Edge index is shared across batch
        edge_index = batch[0].edge_index
        degree = batch[0].degree
    else:
        # Dictionary format
        x_list = [b["x"] for b in batch]
        y_list = [b["y"] for b in batch]
        mask_list = [b["mask"] for b in batch]
        time_list = [b["time_points"] for b in batch]
        edge_index = batch[0]["edge_index"]
        degree = batch[0]["degree"]

    # Stack into batch: (B, T, N, F)
    x_batch = torch.stack(x_list, dim=0)
    y_batch = torch.stack(y_list, dim=0)
    mask_batch = torch.stack(mask_list, dim=0)
    time_batch = time_list[0]  # All same

    return {
        "x": x_batch,  # (B, T, N, F)
        "y": y_batch,  # (B, T, N, 3)
        "mask": mask_batch,  # (B, T, N)
        "time_points": time_batch,  # (T,)
        "edge_index": edge_index,  # (2, E)
        "degree": degree,  # (N,)
    }


def create_dataloaders(
    tensor_path: Union[str, Path],
    graph_path: Union[str, Path],
    batch_size: int = 16,
    config: Optional[DatasetConfig] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders.

    Args:
        tensor_path: Path to tensors.npz
        graph_path: Path to spatial_graph.pt
        batch_size: Batch size
        config: Dataset configuration
        num_workers: DataLoader workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = SeaLiceGraphDataset(
        tensor_path, graph_path, config=config, split="train"
    )
    val_dataset = SeaLiceGraphDataset(
        tensor_path, graph_path, config=config, split="val"
    )
    test_dataset = SeaLiceGraphDataset(
        tensor_path, graph_path, config=config, split="test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_graph_sequences,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graph_sequences,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graph_sequences,
    )

    return train_loader, val_loader, test_loader


# =========================================================================
# Verification Tests
# =========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SeaLiceGraphDataset Verification Test")
    print("=" * 60)

    # Create synthetic test data
    print("\n1. Creating synthetic test data...")
    np.random.seed(42)

    T, N, F = 365, 50, 8  # 1 year, 50 farms, 8 features
    X = np.random.randn(T, N, F).astype(np.float32)
    Y = np.abs(np.random.randn(T, N, 3).astype(np.float32))
    mask = (np.random.rand(T, N) > 0.7).astype(bool)  # 30% observation rate
    times = np.array([f"2023-{i//30+1:02d}-{i%30+1:02d}" for i in range(T)])
    site_ids = np.arange(N)

    # Save test tensors
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        tensor_path = os.path.join(tmpdir, "tensors.npz")
        graph_path = os.path.join(tmpdir, "graph.pt")

        np.savez(tensor_path, X=X, Y=Y, mask=mask, times=times, site_ids=site_ids)

        # Create test graph (random edges)
        edge_src = np.random.randint(0, N, size=200)
        edge_dst = np.random.randint(0, N, size=200)
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        degree = torch.zeros(N)

        torch.save({"edge_index": edge_index, "degree": degree}, graph_path)

        print(f"   X: {X.shape}, Y: {Y.shape}")

        # Test dataset creation
        print("\n2. Testing dataset creation...")
        config = DatasetConfig(window_size=30, stride=7)
        dataset = SeaLiceGraphDataset(tensor_path, graph_path, config=config)

        print(f"   Dataset size: {len(dataset)}")

        # Test __getitem__
        print("\n3. Testing sample retrieval...")
        sample = dataset[0]

        if HAS_PYG:
            print(f"   sample.x: {sample.x.shape}")
            print(f"   sample.y: {sample.y.shape}")
            print(f"   sample.mask: {sample.mask.shape}")
            print(f"   sample.edge_index: {sample.edge_index.shape}")
        else:
            print(f"   sample['x']: {sample['x'].shape}")
            print(f"   sample['y']: {sample['y'].shape}")

        # Test DataLoader
        print("\n4. Testing DataLoader...")
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_graph_sequences,
        )

        batch = next(iter(loader))
        print(f"   Batch x: {batch['x'].shape}")
        print(f"   Batch y: {batch['y'].shape}")
        print(f"   Batch edge_index: {batch['edge_index'].shape}")

        # Test splits
        print("\n5. Testing train/val/test splits...")
        train_ds = SeaLiceGraphDataset(tensor_path, graph_path, config=config, split="train")
        val_ds = SeaLiceGraphDataset(tensor_path, graph_path, config=config, split="val")
        test_ds = SeaLiceGraphDataset(tensor_path, graph_path, config=config, split="test")

        print(f"   Train: {len(train_ds)} sequences")
        print(f"   Val: {len(val_ds)} sequences")
        print(f"   Test: {len(test_ds)} sequences")

    print("\n" + "=" * 60)
    print("SeaLiceGraphDataset test: PASSED")
    print("=" * 60)
