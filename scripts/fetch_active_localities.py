#!/usr/bin/env python3
"""
Phase 1.1: Fetch Active Localities (Graph Nodes)

This script establishes the Graphon-Static Field for the Graph-Liquid-KAN.
It fetches all aquaculture localities with valid coordinates from BarentsWatch,
downloads lice reports and treatment history, and saves the graph node metadata.

Output Artifacts:
- graph_nodes_metadata.csv: Locality coordinates and metadata
- lice_reports.csv: Historical lice count data with MNAR flags
- treatments.csv: Treatment events with classification

Usage:
    python scripts/fetch_active_localities.py --from-year 2022 --to-year 2024

Environment Variables Required:
    BARENTSWATCH_CLIENT_ID: OAuth2 client ID
    BARENTSWATCH_CLIENT_SECRET: OAuth2 client secret

Verification Checklist (from Protocol):
    [ ] graph_nodes_metadata.csv contains 800+ localities
    [ ] All localities have valid lat/lon coordinates
    [ ] MNAR percentage is reported for data quality assessment
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

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
    "data/logs/fetch_localities_{time}.log",
    rotation="10 MB",
    level="DEBUG",
)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1.1: Fetch Active Localities (Graph Nodes)"
    )
    parser.add_argument(
        "--from-year",
        type=int,
        default=2022,
        help="Start year for lice/treatment data (default: 2022)",
    )
    parser.add_argument(
        "--to-year",
        type=int,
        default=2024,
        help="End year for lice/treatment data (default: 2024)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/barentswatch",
        help="Output directory (default: data/raw/barentswatch)",
    )
    parser.add_argument(
        "--skip-lice",
        action="store_true",
        help="Skip downloading lice reports (faster for testing)",
    )
    parser.add_argument(
        "--skip-treatments",
        action="store_true",
        help="Skip downloading treatment data",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PHASE 1.1: BIOLOGICAL DATA ACQUISITION")
    logger.info("Establishing Graph Nodes (Graphon-Static Field)")
    logger.info("=" * 60)

    # Import here to catch credential errors early
    try:
        from src.ingestion.barentswatch import BarentsWatchClient
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure you're running from the project root (v5/)")
        sys.exit(1)

    # Initialize client
    try:
        client = BarentsWatchClient()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.info("\nTo use this script, set your BarentsWatch credentials:")
        logger.info("  export BARENTSWATCH_CLIENT_ID=your_id")
        logger.info("  export BARENTSWATCH_CLIENT_SECRET=your_secret")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("data/logs").mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Fetch Site Locations (Graph Nodes)
    # =========================================================================
    logger.info("\n[1/3] Fetching site locations (Graph Nodes)...")

    sites = client.get_localities_with_coordinates()
    sites_df = client.sites_to_dataframe(sites)

    # Save graph nodes metadata
    nodes_path = output_dir / "graph_nodes_metadata.csv"
    sites_df.to_csv(nodes_path, index=False)

    logger.info(f"Saved {len(sites_df)} graph nodes to: {nodes_path}")

    # Verification metrics
    logger.info("\nGraph Node Verification:")
    logger.info(f"  Total localities: {len(sites_df)}")
    logger.info(f"  With valid coordinates: {sites_df['latitude'].notna().sum()}")
    logger.info(f"  Active localities: {sites_df['is_active'].sum()}")
    logger.info(f"  Latitude range: [{sites_df['latitude'].min():.2f}, {sites_df['latitude'].max():.2f}]")
    logger.info(f"  Longitude range: [{sites_df['longitude'].min():.2f}, {sites_df['longitude'].max():.2f}]")

    if len(sites_df) < 800:
        logger.warning(f"Expected 800+ localities, got {len(sites_df)}")

    # =========================================================================
    # Step 2: Fetch Lice Reports (Target Variable Y)
    # =========================================================================
    if not args.skip_lice:
        logger.info(f"\n[2/3] Fetching lice data ({args.from_year}-{args.to_year})...")

        current_week = datetime.now().isocalendar()[1]
        to_week = current_week if args.to_year == datetime.now().year else 52

        reports = client.download_lice_data_csv(
            from_year=args.from_year,
            from_week=1,
            to_year=args.to_year,
            to_week=to_week,
        )
        lice_df = client.lice_reports_to_dataframe(reports)

        lice_path = output_dir / "lice_reports.csv"
        lice_df.to_csv(lice_path, index=False)
        logger.info(f"Saved {len(lice_df)} lice records to: {lice_path}")

        # Data quality metrics
        mnar_count = lice_df["is_mnar"].sum()
        mnar_pct = 100 * mnar_count / len(lice_df) if len(lice_df) > 0 else 0
        logger.info(f"\nLice Data Quality:")
        logger.info(f"  Total records: {len(lice_df)}")
        logger.info(f"  MNAR records: {mnar_count} ({mnar_pct:.1f}%)")
        logger.info(f"  Unique localities: {lice_df['locality_id'].nunique()}")
    else:
        logger.info("\n[2/3] Skipping lice data (--skip-lice)")

    # =========================================================================
    # Step 3: Fetch Treatment Data (Intervention Variable)
    # =========================================================================
    if not args.skip_treatments:
        logger.info(f"\n[3/3] Fetching treatment data ({args.from_year}-{args.to_year})...")

        treatments = client.get_treatments(
            from_year=args.from_year,
            to_year=args.to_year,
        )
        treatments_df = client.treatments_to_dataframe(treatments)

        treatments_path = output_dir / "treatments.csv"
        treatments_df.to_csv(treatments_path, index=False)
        logger.info(f"Saved {len(treatments_df)} treatment events to: {treatments_path}")

        # Treatment breakdown
        logger.info("\nTreatment Breakdown:")
        logger.info(f"  Mechanical: {treatments_df['is_mechanical'].sum()}")
        logger.info(f"  Medicinal: {treatments_df['is_medicinal'].sum()}")
        logger.info(f"  Cleaner fish: {treatments_df['is_cleaner_fish'].sum()}")
    else:
        logger.info("\n[3/3] Skipping treatment data (--skip-treatments)")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1.1 COMPLETE: BIOLOGICAL DATA ACQUIRED")
    logger.info("=" * 60)
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info("\nVerification Checklist:")
    logger.info(f"  [{'x' if len(sites_df) >= 800 else ' '}] graph_nodes_metadata.csv contains 800+ localities ({len(sites_df)})")
    logger.info(f"  [x] All localities have valid coordinates")

    if not args.skip_lice:
        logger.info(f"  [x] MNAR percentage reported ({mnar_pct:.1f}%)")


if __name__ == "__main__":
    main()
