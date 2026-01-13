#!/usr/bin/env python3
"""
Phase 2: Tensor Construction and Temporal Alignment

Master script that transforms Phase 1 raw data into GraphTemporalSignal tensors.

Steps:
1. Build spatial graph topology (distance-based adjacency)
2. Build aligned tensor dataset (X, Y, mask)
3. Verify dataset with PyTorch DataLoader

Prerequisites:
    - Phase 1 complete (graph_nodes_metadata.csv, lice_reports.csv, treatments.csv)
    - Optional: hydrography_*.nc from Phase 1

Usage:
    python scripts/run_phase2.py --year 2023

    # With custom paths:
    python scripts/run_phase2.py \\
        --nodes-path data/raw/barentswatch/graph_nodes_metadata.csv \\
        --lice-path data/raw/barentswatch/lice_reports.csv \\
        --hydro-path data/raw/norkyst800/hydrography_2023.nc \\
        --output-dir data/processed
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)
logger.add(
    "data/logs/phase2_{time}.log",
    rotation="10 MB",
    level="DEBUG",
)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Tensor Construction and Temporal Alignment"
    )

    # Input paths
    parser.add_argument(
        "--nodes-path",
        type=str,
        default="data/raw/barentswatch/graph_nodes_metadata.csv",
        help="Path to graph nodes CSV",
    )
    parser.add_argument(
        "--lice-path",
        type=str,
        default="data/raw/barentswatch/lice_reports.csv",
        help="Path to lice reports CSV",
    )
    parser.add_argument(
        "--treatments-path",
        type=str,
        default="data/raw/barentswatch/treatments.csv",
        help="Path to treatments CSV",
    )
    parser.add_argument(
        "--hydro-path",
        type=str,
        default=None,
        help="Path to hydrography NetCDF (optional)",
    )

    # Date range
    parser.add_argument(
        "--year",
        type=int,
        help="Year to process (sets start-date and end-date)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD)",
    )

    # Graph parameters
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=30.0,
        help="Distance threshold for edges in km (default: 30)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory",
    )

    # Flags
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip graph construction (use existing)",
    )
    parser.add_argument(
        "--skip-tensors",
        action="store_true",
        help="Skip tensor construction (use existing)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing dataset",
    )

    args = parser.parse_args()

    # Handle year shortcut
    if args.year:
        args.start_date = f"{args.year}-01-01"
        args.end_date = f"{args.year}-12-31"

        # Auto-detect hydrography file if not specified
        if args.hydro_path is None:
            hydro_path = Path(f"data/raw/norkyst800/hydrography_{args.year}.nc")
            if hydro_path.exists():
                args.hydro_path = str(hydro_path)

    # Validate dates
    if not args.verify_only and (not args.start_date or not args.end_date):
        logger.error("Must specify --year or both --start-date and --end-date")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("PHASE 2: TENSOR CONSTRUCTION & TEMPORAL ALIGNMENT")
    logger.info("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("data/logs").mkdir(parents=True, exist_ok=True)

    # Check prerequisites
    nodes_path = Path(args.nodes_path)
    lice_path = Path(args.lice_path)
    treatments_path = Path(args.treatments_path)

    missing = []
    if not nodes_path.exists():
        missing.append(str(nodes_path))
    if not lice_path.exists():
        missing.append(str(lice_path))
    if not treatments_path.exists():
        missing.append(str(treatments_path))

    if missing:
        logger.error("Missing prerequisite files:")
        for f in missing:
            logger.error(f"  - {f}")
        logger.error("\nRun Phase 1 first: python scripts/run_phase1.py --year 2023")
        sys.exit(1)

    # Import modules
    try:
        from src.data.builder import GLKANDatasetBuilder
        from src.data.graph import build_topology, compute_edge_weights
        from src.data.dataset import SeaLiceGraphDataset, DatasetConfig, create_dataloaders
    except ImportError as e:
        logger.error(f"Import error: {e}")
        sys.exit(1)

    # Output paths
    graph_path = output_dir / "spatial_graph.pt"
    tensor_path = output_dir / "tensors.npz"

    # =========================================================================
    # Step 1: Build Spatial Graph Topology
    # =========================================================================
    if not args.skip_graph and not args.verify_only:
        logger.info("\n[1/3] Building spatial graph topology...")

        try:
            edge_index, degree, dist_matrix = build_topology(
                nodes_path=args.nodes_path,
                distance_threshold_km=args.distance_threshold,
                output_path=graph_path,
            )

            # Also compute edge weights
            edge_weights = compute_edge_weights(
                edge_index, dist_matrix, weight_type="gaussian", sigma=10.0
            )

            # Update saved graph with weights
            import torch
            graph_data = torch.load(graph_path)
            graph_data["edge_weights"] = edge_weights
            torch.save(graph_data, graph_path)

            logger.info(f"Graph saved to: {graph_path}")

        except Exception as e:
            logger.error(f"Failed to build graph: {e}")
            raise
    else:
        logger.info("\n[1/3] Skipping graph construction")

    # =========================================================================
    # Step 2: Build Aligned Tensor Dataset
    # =========================================================================
    if not args.skip_tensors and not args.verify_only:
        logger.info(f"\n[2/3] Building tensor dataset ({args.start_date} to {args.end_date})...")

        try:
            builder = GLKANDatasetBuilder(
                nodes_path=args.nodes_path,
                lice_path=args.lice_path,
                treatments_path=args.treatments_path,
                hydro_path=args.hydro_path,
            )

            dataset = builder.build_dataset(
                start_date=args.start_date,
                end_date=args.end_date,
                output_path=tensor_path,
            )

            logger.info(f"Tensors saved to: {tensor_path}")

        except Exception as e:
            logger.error(f"Failed to build tensors: {e}")
            raise
    else:
        logger.info("\n[2/3] Skipping tensor construction")

    # =========================================================================
    # Step 3: Verify Dataset
    # =========================================================================
    logger.info("\n[3/3] Verifying dataset...")

    if not tensor_path.exists():
        logger.error(f"Tensor file not found: {tensor_path}")
        sys.exit(1)

    if not graph_path.exists():
        logger.error(f"Graph file not found: {graph_path}")
        sys.exit(1)

    try:
        import torch
        import numpy as np

        # Load and verify graph
        graph_data = torch.load(graph_path)
        n_edges = graph_data["edge_index"].shape[1]
        n_nodes = graph_data.get("n_nodes", graph_data["degree"].shape[0])

        logger.info(f"\nGraph verification:")
        logger.info(f"  Nodes: {n_nodes}")
        logger.info(f"  Edges: {n_edges}")
        logger.info(f"  Avg degree: {graph_data['degree'].mean():.1f}")

        # Load and verify tensors
        tensors = np.load(tensor_path, allow_pickle=True)

        X = tensors["X"]
        Y = tensors["Y"]
        mask = tensors["mask"]

        logger.info(f"\nTensor verification:")
        logger.info(f"  X (features): {X.shape}")
        logger.info(f"  Y (labels): {Y.shape}")
        logger.info(f"  mask: {mask.shape}")
        logger.info(f"  Observation coverage: {100 * mask.mean():.2f}%")

        # Feature statistics
        logger.info(f"\nFeature statistics:")
        for i in range(X.shape[-1]):
            feat = X[:, :, i]
            valid = feat != 0
            if valid.sum() > 0:
                logger.info(f"  Feature {i}: mean={feat[valid].mean():.3f}, std={feat[valid].std():.3f}")

        # Label statistics
        logger.info(f"\nLabel statistics (adult female lice):")
        y_adult = Y[:, :, 0]
        y_valid = y_adult[mask]
        if len(y_valid) > 0:
            logger.info(f"  Mean: {y_valid.mean():.3f}")
            logger.info(f"  Std: {y_valid.std():.3f}")
            logger.info(f"  Max: {y_valid.max():.3f}")
            logger.info(f"  % above 0.5: {100 * (y_valid > 0.5).mean():.1f}%")

        # Test dataset class
        logger.info(f"\nTesting SeaLiceGraphDataset...")
        config = DatasetConfig(window_size=30, stride=7)
        test_ds = SeaLiceGraphDataset(
            tensor_path=str(tensor_path),
            graph_path=str(graph_path),
            config=config,
        )

        logger.info(f"  Total sequences: {len(test_ds)}")

        # Test a sample
        sample = test_ds[0]
        if hasattr(sample, "x"):
            logger.info(f"  Sample x shape: {sample.x.shape}")
            logger.info(f"  Sample y shape: {sample.y.shape}")
        else:
            logger.info(f"  Sample x shape: {sample['x'].shape}")
            logger.info(f"  Sample y shape: {sample['y'].shape}")

        logger.info("\nDataset verification: PASSED")

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 COMPLETE: TENSORS CONSTRUCTED")
    logger.info("=" * 60)

    logger.info(f"\nOutput directory: {output_dir}")
    logger.info("\nOutput files:")
    logger.info(f"  - {graph_path}")
    logger.info(f"  - {tensor_path}")

    logger.info("\nNext steps:")
    logger.info("  1. Review tensor statistics above")
    logger.info("  2. Proceed to Phase 3: Graph-Liquid-KAN Architecture")
    logger.info("\nExample usage for Phase 3:")
    logger.info("  from src.data.dataset import create_dataloaders")
    logger.info("  train_loader, val_loader, test_loader = create_dataloaders(")
    logger.info(f'      "{tensor_path}",')
    logger.info(f'      "{graph_path}",')
    logger.info("      batch_size=16,")
    logger.info("  )")


if __name__ == "__main__":
    main()
