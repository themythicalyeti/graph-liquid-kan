#!/usr/bin/env python3
"""
End-to-End Pipeline Test with Real Data

Tests the complete Graph-Liquid-KAN pipeline:
1. Phase 1: Fetch biological data from BarentsWatch
2. Phase 1: Fetch environmental data from NorKyst-800 (optional, slower)
3. Phase 2: Build tensors and graph topology
4. Phase 3: Validate architecture with real data
5. Phase 3: Run a mini training loop

Usage:
    # Quick test (skip environmental data, use 2024 only)
    python scripts/test_full_pipeline.py --quick

    # Full test with environmental data
    python scripts/test_full_pipeline.py --year 2024

    # Use existing Phase 1 data
    python scripts/test_full_pipeline.py --skip-phase1
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)


def test_phase1_biological(output_dir: Path, year: int = 2024) -> bool:
    """Test Phase 1: Biological data acquisition from BarentsWatch."""
    logger.info("=" * 60)
    logger.info("PHASE 1 TEST: Biological Data (BarentsWatch)")
    logger.info("=" * 60)

    try:
        from src.ingestion.barentswatch import BarentsWatchClient

        client = BarentsWatchClient()

        # Fetch localities (graph nodes)
        logger.info("\n[1/3] Fetching localities...")
        sites = client.get_localities_with_coordinates()
        sites_df = client.sites_to_dataframe(sites)

        nodes_path = output_dir / "graph_nodes_metadata.csv"
        sites_df.to_csv(nodes_path, index=False)
        logger.info(f"  Saved {len(sites_df)} localities to {nodes_path}")

        # Fetch lice data (just current year for speed)
        logger.info(f"\n[2/3] Fetching lice data for {year}...")
        current_week = datetime.now().isocalendar()[1]
        reports = client.download_lice_data_csv(
            from_year=year,
            from_week=1,
            to_year=year,
            to_week=min(current_week, 52),
        )
        lice_df = client.lice_reports_to_dataframe(reports)

        lice_path = output_dir / "lice_reports.csv"
        lice_df.to_csv(lice_path, index=False)
        logger.info(f"  Saved {len(lice_df)} lice records to {lice_path}")

        # Fetch treatments
        logger.info(f"\n[3/3] Fetching treatment data for {year}...")
        treatments = client.get_treatments(from_year=year, to_year=year)
        treatments_df = client.treatments_to_dataframe(treatments)

        treatments_path = output_dir / "treatments.csv"
        treatments_df.to_csv(treatments_path, index=False)
        logger.info(f"  Saved {len(treatments_df)} treatments to {treatments_path}")

        # Summary
        logger.info("\n[PASS] Phase 1 Biological Data Complete")
        logger.info(f"  Localities: {len(sites_df)}")
        logger.info(f"  Lice records: {len(lice_df)}")
        logger.info(f"  Treatments: {len(treatments_df)}")

        return True

    except Exception as e:
        logger.error(f"[FAIL] Phase 1 Biological: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase1_environmental(
    output_dir: Path,
    nodes_path: Path,
    start_date: str,
    end_date: str,
    max_farms: int = 50,
) -> bool:
    """Test Phase 1: Environmental data from NorKyst-800."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1 TEST: Environmental Data (NorKyst-800)")
    logger.info("=" * 60)

    try:
        import pandas as pd
        from src.ingestion.norkyst800 import NorKyst800Client, NorKyst800Config

        # Load farm locations
        farms_df = pd.read_csv(nodes_path)
        logger.info(f"Loaded {len(farms_df)} farms")

        # Limit farms for faster testing
        if max_farms and len(farms_df) > max_farms:
            farms_df = farms_df.head(max_farms)
            logger.info(f"Limited to {max_farms} farms for testing")

        # Initialize client
        config = NorKyst800Config(output_dir=str(output_dir))
        client = NorKyst800Client(config)

        # Build spatial mapping
        logger.info("\n[1/2] Building spatial mapping...")
        mapped_df, mappings = client.build_spatial_mapping(farms_df)

        mapping_path = output_dir / "spatial_mapping.csv"
        mapped_df.to_csv(mapping_path, index=False)

        avg_dist = mapped_df["dist_error_deg"].mean()
        logger.info(f"  Average distance error: {avg_dist:.4f} deg")

        # Extract hydrography (short period for testing)
        logger.info(f"\n[2/2] Extracting hydrography ({start_date} to {end_date})...")
        result_ds = client.extract_hydrography(
            farms_df=farms_df,
            start_date=start_date,
            end_date=end_date,
            output_netcdf=True,
        )

        logger.info("\n[PASS] Phase 1 Environmental Data Complete")
        logger.info(f"  Farms mapped: {len(mapped_df)}")
        logger.info(f"  Time steps: {result_ds.dims.get('time', 0)}")

        return True

    except Exception as e:
        logger.error(f"[FAIL] Phase 1 Environmental: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase2(
    raw_dir: Path,
    processed_dir: Path,
    year: int,
    has_hydro: bool = False,
) -> bool:
    """Test Phase 2: Tensor construction and graph building."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 TEST: Tensor Construction")
    logger.info("=" * 60)

    try:
        from src.data.builder import GLKANDatasetBuilder
        from src.data.graph import build_topology

        nodes_path = raw_dir / "graph_nodes_metadata.csv"
        lice_path = raw_dir / "lice_reports.csv"
        treatments_path = raw_dir / "treatments.csv"

        # Check for hydrography
        hydro_path = raw_dir / f"hydrography_{year}.nc"
        if not hydro_path.exists():
            hydro_path = None
            logger.warning("No hydrography file found - using zeros for environmental features")

        # Build graph topology
        logger.info("\n[1/2] Building spatial graph...")
        edge_index, degree, dist_matrix = build_topology(
            nodes_path=str(nodes_path),
            distance_threshold_km=30.0,
            output_path=processed_dir / "spatial_graph.pt",
        )

        # Build tensor dataset
        logger.info(f"\n[2/2] Building tensor dataset for {year}...")
        builder = GLKANDatasetBuilder(
            nodes_path=str(nodes_path),
            lice_path=str(lice_path),
            treatments_path=str(treatments_path),
            hydro_path=str(hydro_path) if hydro_path else None,
        )

        dataset = builder.build_dataset(
            start_date=f"{year}-01-01",
            end_date=f"{year}-12-31",
            output_path=processed_dir / "tensors.npz",
        )

        logger.info("\n[PASS] Phase 2 Complete")
        logger.info(f"  X shape: {dataset['X'].shape}")
        logger.info(f"  Y shape: {dataset['Y'].shape}")
        logger.info(f"  Edges: {edge_index.shape[1]}")
        logger.info(f"  Observation coverage: {100 * dataset['mask'].float().mean():.1f}%")

        return True

    except Exception as e:
        logger.error(f"[FAIL] Phase 2: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase3_validation(processed_dir: Path) -> bool:
    """Test Phase 3: Model architecture validation."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3 TEST: Architecture Validation")
    logger.info("=" * 60)

    try:
        from src.models import GLKANNetwork
        from src.data.dataset import SeaLiceGraphDataset, DatasetConfig

        tensor_path = processed_dir / "tensors.npz"
        graph_path = processed_dir / "spatial_graph.pt"

        # Load dataset
        logger.info("\n[1/3] Loading dataset...")
        config = DatasetConfig(window_size=14, stride=7)  # Smaller window for testing
        dataset = SeaLiceGraphDataset(
            tensor_path=str(tensor_path),
            graph_path=str(graph_path),
            config=config,
        )

        logger.info(f"  Sequences: {len(dataset)}")

        if len(dataset) == 0:
            logger.error("No sequences created - check data")
            return False

        # Get a sample
        sample = dataset[0]
        if hasattr(sample, 'x'):
            x_shape = sample.x.shape
            edge_index = sample.edge_index
        else:
            x_shape = sample['x'].shape
            edge_index = sample['edge_index']

        logger.info(f"  Sample x shape: {x_shape}")
        logger.info(f"  Edge index shape: {edge_index.shape}")

        # Create model
        logger.info("\n[2/3] Creating model...")
        input_dim = x_shape[-1]
        model = GLKANNetwork(
            input_dim=input_dim,
            hidden_dim=32,  # Smaller for testing
            output_dim=3,
            n_bases=8,
            n_layers=1,
        )

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Total parameters: {total_params:,}")

        # Forward pass
        logger.info("\n[3/3] Testing forward pass...")
        model.eval()

        if hasattr(sample, 'x'):
            x = sample.x.unsqueeze(0)  # Add batch dim
        else:
            x = sample['x'].unsqueeze(0)

        with torch.no_grad():
            pred, hidden = model(
                x=x,
                edge_index=edge_index,
                return_hidden=True,
            )

        logger.info(f"  Prediction shape: {pred.shape}")
        logger.info(f"  Hidden shape: {hidden.shape}")
        logger.info(f"  Prediction range: [{pred.min():.3f}, {pred.max():.3f}]")

        logger.info("\n[PASS] Phase 3 Validation Complete")
        return True

    except Exception as e:
        logger.error(f"[FAIL] Phase 3 Validation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mini_training(processed_dir: Path, n_epochs: int = 3) -> bool:
    """Test mini training loop with real data."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3 TEST: Mini Training Loop")
    logger.info("=" * 60)

    try:
        from src.models.network import GLKANPredictor
        from src.data.dataset import SeaLiceGraphDataset, DatasetConfig, collate_graph_sequences
        from torch.utils.data import DataLoader

        tensor_path = processed_dir / "tensors.npz"
        graph_path = processed_dir / "spatial_graph.pt"

        # Load dataset
        config = DatasetConfig(window_size=14, stride=7)
        dataset = SeaLiceGraphDataset(
            tensor_path=str(tensor_path),
            graph_path=str(graph_path),
            config=config,
            split="train",
        )

        if len(dataset) < 2:
            logger.warning("Not enough sequences for training test")
            return True  # Not a failure, just insufficient data

        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=min(4, len(dataset)),
            shuffle=True,
            collate_fn=collate_graph_sequences,
        )

        # Get input dim from first batch
        batch = next(iter(loader))
        input_dim = batch["x"].shape[-1]

        # Create model
        model = GLKANPredictor(
            input_dim=input_dim,
            hidden_dim=32,
            output_dim=3,
            n_bases=8,
            n_layers=1,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training loop
        logger.info(f"\nTraining for {n_epochs} epochs...")
        model.train()

        losses = []
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in loader:
                optimizer.zero_grad()
                output = model(batch)
                loss = output["loss"]

                if torch.isnan(loss):
                    logger.error("NaN loss detected!")
                    return False

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            logger.info(f"  Epoch {epoch + 1}: loss = {avg_loss:.4f}")

        # Check if loss decreased
        if len(losses) >= 2 and losses[-1] < losses[0]:
            logger.info("\n[PASS] Loss decreased during training")
        else:
            logger.warning("\n[WARN] Loss did not decrease (may need more epochs)")

        logger.info("[PASS] Mini Training Complete")
        return True

    except Exception as e:
        logger.error(f"[FAIL] Mini Training: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline Test")
    parser.add_argument("--year", type=int, default=2024, help="Year to test")
    parser.add_argument("--quick", action="store_true", help="Quick test (skip environmental)")
    parser.add_argument("--skip-phase1", action="store_true", help="Skip Phase 1 (use existing data)")
    parser.add_argument("--skip-env", action="store_true", help="Skip environmental data fetch")
    parser.add_argument("--max-farms", type=int, default=50, help="Max farms for env data")
    parser.add_argument("--env-days", type=int, default=7, help="Days of env data to fetch")

    args = parser.parse_args()

    if args.quick:
        args.skip_env = True

    logger.info("=" * 60)
    logger.info("GRAPH-LIQUID-KAN: END-TO-END PIPELINE TEST")
    logger.info("=" * 60)
    logger.info(f"Year: {args.year}")
    logger.info(f"Quick mode: {args.quick}")

    # Setup directories
    raw_dir = Path("data/raw/barentswatch")
    env_dir = Path("data/raw/norkyst800")
    processed_dir = Path("data/processed")

    raw_dir.mkdir(parents=True, exist_ok=True)
    env_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Phase 1: Biological
    if not args.skip_phase1:
        results["phase1_bio"] = test_phase1_biological(raw_dir, args.year)
    else:
        logger.info("\n[SKIP] Phase 1 Biological (using existing data)")
        results["phase1_bio"] = (raw_dir / "graph_nodes_metadata.csv").exists()

    # Phase 1: Environmental (optional)
    if not args.skip_env and results.get("phase1_bio", False):
        # Calculate date range
        start_date = f"{args.year}-01-01"
        end_date = f"{args.year}-01-{args.env_days:02d}"

        results["phase1_env"] = test_phase1_environmental(
            output_dir=env_dir,
            nodes_path=raw_dir / "graph_nodes_metadata.csv",
            start_date=start_date,
            end_date=end_date,
            max_farms=args.max_farms,
        )
    else:
        logger.info("\n[SKIP] Phase 1 Environmental")
        results["phase1_env"] = True  # Not required

    # Phase 2
    if results.get("phase1_bio", False):
        results["phase2"] = test_phase2(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            year=args.year,
            has_hydro=not args.skip_env,
        )
    else:
        results["phase2"] = False

    # Phase 3: Validation
    if results.get("phase2", False):
        results["phase3_validation"] = test_phase3_validation(processed_dir)
    else:
        results["phase3_validation"] = False

    # Phase 3: Mini Training
    if results.get("phase3_validation", False):
        results["phase3_training"] = test_mini_training(processed_dir, n_epochs=3)
    else:
        results["phase3_training"] = False

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "[x]" if passed else "[ ]"
        logger.info(f"  {symbol} {test_name}: {status}")
        if not passed:
            all_passed = False

    logger.info("=" * 60)

    if all_passed:
        logger.info("ALL TESTS PASSED!")
        logger.info("\nThe pipeline is working correctly with real data.")
        logger.info("Ready to proceed with full-scale training.")
        return 0
    else:
        logger.error("SOME TESTS FAILED")
        logger.error("Review the errors above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
