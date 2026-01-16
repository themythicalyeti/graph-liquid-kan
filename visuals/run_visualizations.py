#!/usr/bin/env python3
"""
Sea Lice Model Visualization Suite - Main Runner

Generates all four categories of visualizations:
1. Early Warning Dashboard - For farm managers
2. Interpretability Suite - For biologists
3. Spatial Network - For regulators
4. What-If Simulator - For planners

Usage:
    python visuals/run_visualizations.py
    python visuals/run_visualizations.py --checkpoint path/to/model.pt
    python visuals/run_visualizations.py --category interpretability
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
from loguru import logger

# Set up logging
logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")


def load_data(data_dir: str = "data/processed"):
    """Load model data from processed directory."""
    data_path = Path(data_dir)

    logger.info(f"Loading data from {data_path}")

    # Load tensors
    data = np.load(data_path / "tensors.npz", allow_pickle=True)
    X = torch.from_numpy(data["X"]).float()
    Y = torch.from_numpy(data["Y"]).float()
    mask = torch.from_numpy(data["mask"]).bool()

    # Load feature indices
    if "feature_indices" in data:
        feature_indices = data["feature_indices"].item()
    else:
        feature_indices = {
            'temperature': 8,
            'salinity': 9,
            'current_u': 10,
            'current_v': 11,
            'treatment': 12,
        }

    # Load graph
    graph_data = torch.load(data_path / "spatial_graph.pt", weights_only=False)
    edge_index = graph_data["edge_index"]

    # Subset to manageable size
    N_subset = min(200, X.shape[1])
    X = X[:, :N_subset, :]
    Y = Y[:, :N_subset, :]
    mask = mask[:, :N_subset]

    # Filter edges
    valid_edges = (edge_index[0] < N_subset) & (edge_index[1] < N_subset)
    edge_index = edge_index[:, valid_edges]

    logger.info(f"Data loaded: X={X.shape}, Y={Y.shape}")

    return {
        'X': X,
        'Y': Y,
        'mask': mask,
        'edge_index': edge_index,
        'graph_data': graph_data,
        'feature_indices': feature_indices,
    }


def load_model(checkpoint_path: str, input_dim: int):
    """Load trained model from checkpoint."""
    from src.models.sea_lice_network import SeaLicePredictor

    logger.info(f"Loading model from {checkpoint_path}")

    model = SeaLicePredictor(
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=3,
        n_bases=8,
        k_hops=3,
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")

    return model


def run_early_warning(model, data, output_dir: Path, farm_idx: int = 0):
    """Generate Early Warning Dashboard visualizations."""
    from visuals.early_warning import EarlyWarningDashboard

    logger.info("=" * 60)
    logger.info("GENERATING EARLY WARNING DASHBOARD")
    logger.info("=" * 60)

    # Create dashboard
    dashboard = EarlyWarningDashboard(model, coverage=0.90)

    # Split data for calibration
    T = data['X'].shape[0]
    calib_end = T // 2

    # Calibrate
    logger.info("Calibrating conformal predictor...")
    dashboard.calibrate(
        data['X'][:calib_end],
        data['Y'][:calib_end],
        data['edge_index'],
        data['mask'][:calib_end],
        data['feature_indices'],
    )

    # Farm indices (include the specified farm_idx and a couple others)
    farm_indices = [farm_idx, (farm_idx + 1) % data['X'].shape[1], (farm_idx + 2) % data['X'].shape[1]]

    # Generate test data (last 14 days)
    X_test = data['X'][-14:]

    # Create dashboard
    fig = dashboard.create_dashboard(
        X_test,
        data['edge_index'],
        farm_indices,
        data['feature_indices'],
        start_date="2024-01-01",
        save_path=str(output_dir / "early_warning_dashboard.png"),
    )
    plt.close(fig)

    # Create risk matrix for more farms
    fig_risk = dashboard.plot_multi_farm_risk_matrix(
        X_test,
        data['edge_index'],
        list(range(min(20, data['X'].shape[1]))),
        data['feature_indices'],
        save_path=str(output_dir / "farm_risk_matrix.png"),
    )
    plt.close(fig_risk)

    logger.info(f"Early warning visualizations saved to {output_dir}")


def run_interpretability(model, data, output_dir: Path):
    """Generate Interpretability Suite visualizations."""
    from visuals.interpretability import InterpretabilitySuite

    logger.info("=" * 60)
    logger.info("GENERATING INTERPRETABILITY SUITE")
    logger.info("=" * 60)

    suite = InterpretabilitySuite(model)

    # Create comprehensive report
    fig = suite.create_interpretability_report(
        data['X'],
        data['edge_index'],
        data['feature_indices'],
        save_dir=str(output_dir),
    )
    plt.close(fig)

    # Compute curve similarity metrics
    try:
        similarity = suite.compute_curve_similarity()
        logger.info("Curve Similarity Metrics:")
        logger.info(f"  Temperature correlation: {similarity['temperature_correlation']:.3f}")
        logger.info(f"  Salinity correlation: {similarity['salinity_correlation']:.3f}")
    except Exception as e:
        logger.warning(f"Could not compute similarity metrics: {e}")

    logger.info(f"Interpretability visualizations saved to {output_dir}")


def run_spatial_network(model, data, output_dir: Path):
    """Generate Spatial Network visualizations."""
    from visuals.spatial_network import SpatialNetworkVisualizer

    logger.info("=" * 60)
    logger.info("GENERATING SPATIAL NETWORK VISUALIZATIONS")
    logger.info("=" * 60)

    viz = SpatialNetworkVisualizer(model)

    # Create edge attributes if not present
    if 'edge_attr' not in data['graph_data']:
        # Generate synthetic edge attributes for demo
        E = data['edge_index'].shape[1]
        edge_attr = torch.zeros(E, 4)
        edge_attr[:, 0] = torch.rand(E) * 30  # Distance (km)
        edge_attr[:, 1] = torch.randn(E) * 0.2  # Current u
        edge_attr[:, 2] = torch.randn(E) * 0.2  # Current v
        edge_attr[:, 3] = 10 + torch.rand(E) * 10  # Temperature
    else:
        edge_attr = data['graph_data']['edge_attr']

    # Create spatial report
    fig = viz.create_spatial_report(
        data['X'],
        data['edge_index'],
        data['graph_data'],
        edge_attr=edge_attr,
        feature_indices=data['feature_indices'],
        save_dir=str(output_dir),
    )
    plt.close(fig)

    # Try to create animation (may fail without proper setup)
    try:
        x_coords, y_coords = viz.load_farm_positions(data['graph_data'])
        anim = viz.create_animated_infection(
            x_coords, y_coords,
            data['edge_index'],
            data['X'][:20],  # First 20 time steps
            edge_attr=edge_attr,
            feature_indices=data['feature_indices'],
            n_frames=20,
            save_path=str(output_dir / "infection_spread.gif"),
            fps=2,
        )
        plt.close()
    except Exception as e:
        logger.warning(f"Could not create animation: {e}")

    logger.info(f"Spatial network visualizations saved to {output_dir}")


def run_what_if(model, data, output_dir: Path, farm_idx: int = 0):
    """Generate What-If Simulator visualizations."""
    from visuals.counterfactual import WhatIfSimulator

    logger.info("=" * 60)
    logger.info("GENERATING WHAT-IF SIMULATOR VISUALIZATIONS")
    logger.info("=" * 60)

    sim = WhatIfSimulator(model, threshold=0.5)

    # Use a 14-day window for simulation
    X_test = data['X'][:14]

    # Create comprehensive report
    fig = sim.create_what_if_report(
        X_test,
        data['edge_index'],
        farm_idx=farm_idx,
        feature_indices=data['feature_indices'],
        save_dir=str(output_dir),
    )
    plt.close(fig)

    logger.info(f"What-If visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Sea Lice Model Visualizations"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to processed data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visuals/output",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["all", "early_warning", "interpretability", "spatial", "what_if"],
        default="all",
        help="Which visualization category to generate"
    )
    parser.add_argument(
        "--farm-idx",
        type=int,
        default=0,
        help="Farm index to focus on"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SEA LICE MODEL VISUALIZATION SUITE")
    logger.info("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        data = load_data(args.data_dir)
    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}")
        logger.info("Please ensure data/processed/tensors.npz and spatial_graph.pt exist")
        return 1

    # Load model
    try:
        model = load_model(args.checkpoint, data['X'].shape[-1])
    except FileNotFoundError:
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        logger.info("Please train the model first or specify a valid checkpoint path")
        return 1

    # Generate visualizations
    categories = {
        "early_warning": run_early_warning,
        "interpretability": run_interpretability,
        "spatial": run_spatial_network,
        "what_if": run_what_if,
    }

    if args.category == "all":
        for name, func in categories.items():
            try:
                if name in ["early_warning", "what_if"]:
                    func(model, data, output_dir, farm_idx=args.farm_idx)
                else:
                    func(model, data, output_dir)
            except Exception as e:
                logger.error(f"Error generating {name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        func = categories[args.category]
        try:
            if args.category in ["early_warning", "what_if"]:
                func(model, data, output_dir, farm_idx=args.farm_idx)
            else:
                func(model, data, output_dir)
        except Exception as e:
            logger.error(f"Error generating {args.category}: {e}")
            import traceback
            traceback.print_exc()
            return 1

    logger.info("=" * 60)
    logger.info("VISUALIZATION COMPLETE")
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
