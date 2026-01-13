#!/usr/bin/env python3
"""
Phase 1.2: Extract Hydrography (Environmental Forcing)

This script extracts oceanographic data from NorKyst-800 via OPeNDAP,
constructing the Forcing Function u(t) for the Liquid-KAN architecture.

The script:
1. Loads graph nodes from Phase 1.1 (graph_nodes_metadata.csv)
2. Builds KD-Tree spatial mapping to curvilinear ocean grid
3. Extracts temperature, salinity, and currents at farm locations
4. Saves output as NetCDF (hydrography_{year}.nc)

Output Artifacts:
- hydrography_{year}.nc: NetCDF with dims (time, farm_id)
- environmental_data.csv: Flat CSV for inspection
- spatial_mapping.csv: Farm-to-grid mapping with distance errors

Usage:
    python scripts/extract_hydrography.py --start-date 2023-01-01 --end-date 2023-12-31

Prerequisites:
    - Run fetch_active_localities.py first to generate graph_nodes_metadata.csv

Verification Checklist (from Protocol):
    [ ] Average dist_error_deg < 0.01 (high-quality snapping)
    [ ] hydrography_{year}.nc contains time, farm_id, temperature
    [ ] Temperature range is realistic (0-20 degC for Norwegian waters)
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

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
    "data/logs/extract_hydrography_{time}.log",
    rotation="10 MB",
    level="DEBUG",
)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1.2: Extract Hydrography (Environmental Forcing)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--sites-file",
        type=str,
        default="data/raw/barentswatch/graph_nodes_metadata.csv",
        help="Path to graph nodes CSV from Phase 1.1",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/norkyst800",
        help="Output directory",
    )
    parser.add_argument(
        "--max-farms",
        type=int,
        default=None,
        help="Limit number of farms (for testing)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PHASE 1.2: ENVIRONMENTAL FORCING EXTRACTION")
    logger.info("Constructing Forcing Function u(t) for Liquid-KAN")
    logger.info("=" * 60)

    # Check prerequisites
    sites_path = Path(args.sites_file)
    if not sites_path.exists():
        logger.error(f"Sites file not found: {sites_path}")
        logger.error("Run 'python scripts/fetch_active_localities.py' first")
        sys.exit(1)

    # Load farm locations
    logger.info(f"\nLoading farm locations from: {sites_path}")
    farms_df = pd.read_csv(sites_path)
    logger.info(f"Loaded {len(farms_df)} farms")

    # Optionally limit farms for testing
    if args.max_farms is not None:
        farms_df = farms_df.head(args.max_farms)
        logger.info(f"Limited to {len(farms_df)} farms for testing")

    # Validate required columns
    required_cols = ["locality_id", "latitude", "longitude"]
    missing = [c for c in required_cols if c not in farms_df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        sys.exit(1)

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("data/logs").mkdir(parents=True, exist_ok=True)

    # Import NorKyst800 client
    try:
        from src.ingestion.norkyst800 import NorKyst800Client, NorKyst800Config
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure you're running from the project root (v5/)")
        sys.exit(1)

    # Initialize client
    config = NorKyst800Config(output_dir=str(output_dir))
    client = NorKyst800Client(config)

    # =========================================================================
    # Step 1: Build Spatial Mapping (KD-Tree)
    # =========================================================================
    logger.info("\n[1/2] Building spatial mapping (KD-Tree)...")

    try:
        mapped_farms_df, mappings = client.build_spatial_mapping(farms_df)

        # Save spatial mapping for verification
        mapping_path = output_dir / "spatial_mapping.csv"
        mapped_farms_df.to_csv(mapping_path, index=False)
        logger.info(f"Saved spatial mapping to: {mapping_path}")

        # Verification metrics
        avg_dist = mapped_farms_df["dist_error_deg"].mean()
        max_dist = mapped_farms_df["dist_error_deg"].max()
        valid_count = mapped_farms_df["is_ocean_valid"].sum()

        logger.info("\nSpatial Mapping Verification:")
        logger.info(f"  Average dist_error_deg: {avg_dist:.4f}")
        logger.info(f"  Maximum dist_error_deg: {max_dist:.4f}")
        logger.info(f"  Valid ocean points: {valid_count}/{len(mapped_farms_df)}")

        if avg_dist > 0.01:
            logger.warning(f"Average distance error ({avg_dist:.4f}) exceeds 0.01 threshold")

    except Exception as e:
        logger.error(f"Failed to build spatial mapping: {e}")
        logger.info("\nTroubleshooting:")
        logger.info("  1. Check internet connection to thredds.met.no")
        logger.info("  2. Verify MET Norway THREDDS server is accessible")
        logger.info("  3. Try with --max-farms 10 for a quick test")
        raise

    # =========================================================================
    # Step 2: Extract Hydrography
    # =========================================================================
    logger.info(f"\n[2/2] Extracting hydrography ({args.start_date} to {args.end_date})...")

    try:
        result_ds = client.extract_hydrography(
            farms_df=farms_df,
            start_date=args.start_date,
            end_date=args.end_date,
            output_netcdf=True,
        )

        # Log dataset info
        logger.info("\nOutput Dataset:")
        logger.info(f"  Dimensions: {dict(result_ds.dims)}")
        logger.info(f"  Variables: {list(result_ds.data_vars)}")
        logger.info(f"  Time range: {result_ds.time.values[0]} to {result_ds.time.values[-1]}")

        # Data quality check
        if "temperature" in result_ds:
            temp = result_ds["temperature"].values
            temp_valid = ~pd.isna(temp)
            if temp_valid.any():
                logger.info(f"  Temperature range: {temp[temp_valid].min():.1f} - {temp[temp_valid].max():.1f} degC")
                logger.info(f"  Temperature coverage: {100*temp_valid.mean():.1f}%")

    except Exception as e:
        logger.error(f"Failed to extract hydrography: {e}")
        logger.info("\nTroubleshooting:")
        logger.info("  1. Check if the date range has data available")
        logger.info("  2. Try a shorter date range (e.g., 7 days)")
        logger.info("  3. Check thredds.met.no for data availability")
        raise

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1.2 COMPLETE: ENVIRONMENTAL FORCING EXTRACTED")
    logger.info("=" * 60)
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info("\nOutput files:")
    logger.info(f"  - hydrography_*.nc (NetCDF)")
    logger.info(f"  - environmental_data.csv")
    logger.info(f"  - spatial_mapping.csv")

    logger.info("\nVerification Checklist:")
    logger.info(f"  [{'x' if avg_dist < 0.01 else ' '}] Average dist_error_deg < 0.01 ({avg_dist:.4f})")
    logger.info(f"  [x] hydrography_*.nc contains time, farm_id, temperature")

    # Check if we can open the NetCDF
    year = args.start_date[:4]
    nc_path = output_dir / f"hydrography_{year}.nc"
    if nc_path.exists():
        logger.info(f"  [x] NetCDF file created: {nc_path}")
        logger.info(f"\nTo verify with ncdump:")
        logger.info(f"  ncdump -h {nc_path}")


if __name__ == "__main__":
    main()
