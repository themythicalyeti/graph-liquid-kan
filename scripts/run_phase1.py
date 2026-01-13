#!/usr/bin/env python3
"""
Phase 1: Data Engineering Pipeline

Master script that runs the complete Phase 1 data acquisition:
1. Fetch biological graph nodes from BarentsWatch
2. Extract environmental forcing from NorKyst-800

This establishes the data backbone for the Graph-Liquid-KAN architecture.

Usage:
    python scripts/run_phase1.py --year 2023

    # Or with custom date range:
    python scripts/run_phase1.py --start-date 2023-01-01 --end-date 2023-12-31

Prerequisites:
    - Set BARENTSWATCH_CLIENT_ID and BARENTSWATCH_CLIENT_SECRET env vars
    - Internet connection to thredds.met.no
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Data Engineering Pipeline"
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Year to process (sets start-date to Jan 1 and end-date to Dec 31)",
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
    parser.add_argument(
        "--from-year",
        type=int,
        default=2022,
        help="Start year for biological data (default: 2022)",
    )
    parser.add_argument(
        "--to-year",
        type=int,
        default=None,
        help="End year for biological data (default: current year)",
    )
    parser.add_argument(
        "--skip-biological",
        action="store_true",
        help="Skip Step 1 (biological data)",
    )
    parser.add_argument(
        "--skip-environmental",
        action="store_true",
        help="Skip Step 2 (environmental data)",
    )
    parser.add_argument(
        "--max-farms",
        type=int,
        default=None,
        help="Limit farms for environmental extraction (testing)",
    )

    args = parser.parse_args()

    # Set default year if not specified
    if args.year:
        args.start_date = f"{args.year}-01-01"
        args.end_date = f"{args.year}-12-31"
        if args.to_year is None:
            args.to_year = args.year

    if args.to_year is None:
        args.to_year = datetime.now().year

    # Validate dates for environmental extraction
    if not args.skip_environmental and (not args.start_date or not args.end_date):
        logger.error("Must specify --year or both --start-date and --end-date")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("GRAPH-LIQUID-KAN: PHASE 1 DATA ENGINEERING")
    logger.info("=" * 60)
    logger.info(f"\nBiological data: {args.from_year} - {args.to_year}")
    if args.start_date:
        logger.info(f"Environmental data: {args.start_date} to {args.end_date}")

    success = True

    # =========================================================================
    # Step 1: Biological Data (BarentsWatch)
    # =========================================================================
    if not args.skip_biological:
        cmd = [
            sys.executable,
            "scripts/fetch_active_localities.py",
            "--from-year", str(args.from_year),
            "--to-year", str(args.to_year),
        ]
        if not run_command(cmd, "Step 1: Fetch Biological Data"):
            logger.error("Step 1 failed!")
            success = False

    # =========================================================================
    # Step 2: Environmental Data (NorKyst-800)
    # =========================================================================
    if not args.skip_environmental and success:
        cmd = [
            sys.executable,
            "scripts/extract_hydrography.py",
            "--start-date", args.start_date,
            "--end-date", args.end_date,
        ]
        if args.max_farms:
            cmd.extend(["--max-farms", str(args.max_farms)])

        if not run_command(cmd, "Step 2: Extract Environmental Forcing"):
            logger.error("Step 2 failed!")
            success = False

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("PHASE 1 COMPLETE: DATA BACKBONE ESTABLISHED")
    else:
        logger.error("PHASE 1 FAILED: Check logs for errors")
    logger.info("=" * 60)

    logger.info("\nOutput artifacts:")
    logger.info("  data/raw/barentswatch/")
    logger.info("    - graph_nodes_metadata.csv (Graph Nodes)")
    logger.info("    - lice_reports.csv (Target Variable Y)")
    logger.info("    - treatments.csv (Intervention Variable)")
    logger.info("  data/raw/norkyst800/")
    logger.info("    - hydrography_*.nc (Forcing Function u(t))")
    logger.info("    - spatial_mapping.csv (Grid Mapping)")

    logger.info("\nNext steps:")
    logger.info("  1. Verify graph_nodes_metadata.csv has 800+ localities")
    logger.info("  2. Check spatial_mapping.csv for dist_error_deg < 0.01")
    logger.info("  3. Use 'ncdump -h' to verify NetCDF structure")
    logger.info("  4. Proceed to Phase 2: Tensor Construction")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
