"""
NorKyst-800 Client for MET Norway THREDDS Server

Phase 1.2: Environmental Forcing Data Acquisition for Graph-Liquid-KAN

Retrieves oceanographic data from thredds.met.no using OPeNDAP:
- sea_water_temperature: Temperature at ~3m depth (for tau(T) time constant)
- salinity: Sea water salinity (copepodid survival)
- u_eastward, v_northward: Current vectors (larval transport)

Key Design Decisions:
1. OPeNDAP server-side subsetting avoids the "Memory Wall" of downloading
   terabytes of unnecessary ocean data.

2. KD-Tree (cKDTree) for curvilinear grid mapping - the NorKyst-800 grid is
   curvilinear, meaning grid point distances vary with latitude. Standard
   Euclidean lookup would introduce spatial bias.

3. 0.02 degree threshold is a "Sanity Check" - prevents the model from
   predicting sea lice on farms that appear inland due to low-resolution
   coastline mapping.

Reference: https://thredds.met.no/thredds/catalog/fou-hi/norkyst800m/catalog.html

The output constructs the Forcing Function u(t) for the Liquid-KAN architecture.
Variables like temperature will directly modulate the time-constants tau(T)
of the hidden states.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    logger.warning("xarray not installed. Run: pip install xarray netcdf4")

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not installed. Run: pip install scipy")


@dataclass
class NorKyst800Config:
    """
    Configuration for NorKyst-800 data fetching.

    The configuration reflects the protocol requirements:
    - depth_mixed_layer: 3m for mixed layer (biologically relevant)
    - max_dist_threshold: 0.02 degrees (~2km) for sanity checking
    """

    # THREDDS OPeNDAP base URL
    base_url: str = "https://thredds.met.no/thredds/dodsC"

    # Dataset paths on THREDDS server
    # 1-hour aggregated data (most comprehensive)
    hourly_aggregate: str = "sea/norkyst800m/1h/aggregate_be"

    # Variable names (NorKyst-800 uses u_eastward/v_northward for geo-referenced)
    var_temperature: str = "temperature"
    var_salinity: str = "salinity"
    var_u_velocity: str = "u_eastward"
    var_v_velocity: str = "v_northward"

    # Alternative variable names (different files use different conventions)
    var_temperature_alt: Tuple[str, ...] = ("sea_water_temperature", "temp", "votemper")
    var_salinity_alt: Tuple[str, ...] = ("sea_water_salinity", "salt", "vosaline")
    var_u_velocity_alt: Tuple[str, ...] = ("u", "eastward_sea_water_velocity", "u_velocity")
    var_v_velocity_alt: Tuple[str, ...] = ("v", "northward_sea_water_velocity", "v_velocity")

    # Depth selection (use 3m for mixed layer as per protocol)
    depth_surface: float = 0.0
    depth_mixed_layer: float = 3.0
    depth_index: int = 1  # Index 1 typically = 3m in NorKyst-800

    # Spatial mapping parameters
    max_dist_threshold: float = 0.02  # Max distance in degrees (~2km)
    n_neighbors: int = 9  # Number of neighbors to search for land-mask fallback

    # Clustering parameters for optimized extraction
    n_clusters: int = 13  # ~Norwegian Production Areas
    buffer_deg: float = 0.15  # Buffer around clusters in degrees

    # Output paths
    cache_dir: str = "data/cache/norkyst800"
    output_dir: str = "data/raw/norkyst800"


@dataclass
class SpatialMapping:
    """
    Result of spatial mapping from farm to ocean grid.

    Contains the grid indices and distance error for verification.
    """
    locality_id: int
    latitude: float
    longitude: float
    grid_y: int
    grid_x: int
    dist_error_deg: float
    is_valid: bool  # False if dist > threshold (likely on land)


class NorKyst800Client:
    """
    Client for NorKyst-800 data retrieval via OPeNDAP.

    This client implements the Phase 1.2 Environmental Forcing extraction
    with the following key features:

    1. Server-side subsetting via OPeNDAP to avoid memory issues
    2. KD-Tree spatial mapping for curvilinear grid
    3. Clustered bounding-box extraction for efficiency
    4. Distance threshold validation for data quality
    """

    def __init__(self, config: Optional[NorKyst800Config] = None):
        """Initialize the NorKyst-800 client."""
        if not HAS_XARRAY:
            raise ImportError(
                "xarray package required. Install with: pip install xarray netcdf4"
            )
        if not HAS_SCIPY:
            raise ImportError(
                "scipy package required. Install with: pip install scipy"
            )

        self.config = config or NorKyst800Config()

        # Create directories
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Cache for discovered variable names
        self._var_names: Dict[str, str] = {}

        # Cache for discovered dimension names
        self._dim_names: Dict[str, str] = {}

        # Cache for KD-Tree (built once per session)
        self._kdtree: Optional[cKDTree] = None
        self._grid_lat: Optional[np.ndarray] = None
        self._grid_lon: Optional[np.ndarray] = None

    def _find_variable(
        self, ds: xr.Dataset, primary: str, alternatives: Tuple[str, ...]
    ) -> Optional[str]:
        """Find available variable name from primary and alternatives."""
        if primary in ds.data_vars:
            return primary
        for alt in alternatives:
            if alt in ds.data_vars:
                return alt
        return None

    def _discover_variables(self, ds: xr.Dataset) -> Dict[str, str]:
        """Discover actual variable names in the dataset."""
        var_map = {}

        temp_var = self._find_variable(
            ds, self.config.var_temperature, self.config.var_temperature_alt
        )
        if temp_var:
            var_map["temperature"] = temp_var

        sal_var = self._find_variable(
            ds, self.config.var_salinity, self.config.var_salinity_alt
        )
        if sal_var:
            var_map["salinity"] = sal_var

        u_var = self._find_variable(
            ds, self.config.var_u_velocity, self.config.var_u_velocity_alt
        )
        if u_var:
            var_map["u"] = u_var

        v_var = self._find_variable(
            ds, self.config.var_v_velocity, self.config.var_v_velocity_alt
        )
        if v_var:
            var_map["v"] = v_var

        return var_map

    def _discover_dimensions(self, ds: xr.Dataset) -> Dict[str, str]:
        """
        Discover actual dimension names in the dataset.

        NorKyst-800 uses different naming conventions in different files:
        - Y/X (common)
        - eta_rho/xi_rho (ROMS convention)
        - y/x (lowercase)
        - latitude/longitude (rare for grid dims)
        """
        dim_map = {}
        dims = list(ds.dims.keys())

        # Y dimension (rows)
        y_candidates = ["Y", "y", "eta_rho", "eta", "j", "nj"]
        for candidate in y_candidates:
            if candidate in dims:
                dim_map["Y"] = candidate
                break

        # X dimension (columns)
        x_candidates = ["X", "x", "xi_rho", "xi", "i", "ni"]
        for candidate in x_candidates:
            if candidate in dims:
                dim_map["X"] = candidate
                break

        # Depth dimension
        depth_candidates = ["depth", "z", "s_rho", "level", "deptht"]
        for candidate in depth_candidates:
            if candidate in dims:
                dim_map["depth"] = candidate
                break

        # Time dimension
        time_candidates = ["time", "time_counter", "ocean_time"]
        for candidate in time_candidates:
            if candidate in dims:
                dim_map["time"] = candidate
                break

        logger.info(f"Discovered dimensions: {dim_map}")
        return dim_map

    # =========================================================================
    # KD-Tree Spatial Mapper (Section 2.1 of Protocol)
    # =========================================================================

    def build_spatial_mapping(
        self,
        farms_df: pd.DataFrame,
        ds: Optional[xr.Dataset] = None,
    ) -> Tuple[pd.DataFrame, List[SpatialMapping]]:
        """
        Build KD-Tree spatial mapping from farm locations to ocean grid.

        This implements the cKDTree solution from the protocol:
        - Uses k-dimensional tree for mapping each farm to nearest ocean pixel
        - Applies 0.02 degree threshold to filter farms that appear inland
        - Returns mapping with distance errors for verification

        Args:
            farms_df: DataFrame with locality_id, latitude, longitude
            ds: Optional pre-opened dataset (will open if not provided)

        Returns:
            Tuple of:
            - farms_df with added grid_y, grid_x, dist_error_deg columns
            - List of SpatialMapping objects for verification
        """
        logger.info("Building KD-Tree spatial mapping...")

        # Open dataset if not provided
        close_ds = False
        if ds is None:
            url = f"{self.config.base_url}/{self.config.hourly_aggregate}"
            logger.info(f"Opening dataset: {url}")
            ds = xr.open_dataset(url)
            close_ds = True

        try:
            # Get 2D lat/lon arrays
            lat_2d = ds["lat"].values
            lon_2d = ds["lon"].values
            self._grid_lat = lat_2d
            self._grid_lon = lon_2d

            logger.info(f"Grid shape: {lat_2d.shape}")
            logger.info(f"Grid extent: lat [{lat_2d.min():.2f}, {lat_2d.max():.2f}], "
                       f"lon [{lon_2d.min():.2f}, {lon_2d.max():.2f}]")

            # Build KD-Tree from flattened grid points
            grid_points = np.column_stack([lat_2d.ravel(), lon_2d.ravel()])
            self._kdtree = cKDTree(grid_points)

            # Query for each farm
            farm_lats = farms_df["latitude"].values
            farm_lons = farms_df["longitude"].values
            farm_ids = farms_df["locality_id"].values

            farm_points = np.column_stack([farm_lats, farm_lons])
            distances, flat_indices = self._kdtree.query(
                farm_points, k=self.config.n_neighbors
            )

            # Convert flat indices to 2D grid indices (primary match)
            grid_shape = lat_2d.shape
            y_indices = flat_indices[:, 0] // grid_shape[1]
            x_indices = flat_indices[:, 0] % grid_shape[1]

            # Build results
            mappings = []
            result_df = farms_df.copy()
            result_df["grid_y"] = y_indices
            result_df["grid_x"] = x_indices
            result_df["dist_error_deg"] = distances[:, 0]

            valid_count = 0
            invalid_count = 0

            for i in range(len(farm_ids)):
                dist = distances[i, 0]
                is_valid = dist < self.config.max_dist_threshold

                mapping = SpatialMapping(
                    locality_id=farm_ids[i],
                    latitude=farm_lats[i],
                    longitude=farm_lons[i],
                    grid_y=int(y_indices[i]),
                    grid_x=int(x_indices[i]),
                    dist_error_deg=float(dist),
                    is_valid=is_valid,
                )
                mappings.append(mapping)

                if is_valid:
                    valid_count += 1
                else:
                    invalid_count += 1

            # Add validity flag
            result_df["is_ocean_valid"] = result_df["dist_error_deg"] < self.config.max_dist_threshold

            # Log statistics
            avg_dist = distances[:, 0].mean()
            max_dist = distances[:, 0].max()
            logger.info(f"Spatial mapping complete:")
            logger.info(f"  Valid farms (dist < {self.config.max_dist_threshold} deg): {valid_count}")
            logger.info(f"  Invalid farms (likely inland): {invalid_count}")
            logger.info(f"  Average distance error: {avg_dist:.4f} deg")
            logger.info(f"  Maximum distance error: {max_dist:.4f} deg")

            return result_df, mappings

        finally:
            if close_ds:
                ds.close()

    # =========================================================================
    # Temporal Feature Tensor Extraction (Section 2.2 of Protocol)
    # =========================================================================

    def extract_hydrography(
        self,
        farms_df: pd.DataFrame,
        start_date: str,
        end_date: str,
        output_netcdf: bool = True,
    ) -> xr.Dataset:
        """
        Extract hydrography data (the Forcing Function u(t)) for farm locations.

        This constructs the temporal feature tensor that will modulate the
        time-constants tau(T) in the Liquid-KAN architecture.

        Variables extracted:
        - temperature: Water temperature at 3m depth
        - salinity: Sea water salinity at 3m depth
        - u, v: Current velocity components
        - current_speed: Derived from u, v

        Args:
            farms_df: DataFrame with locality_id, latitude, longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_netcdf: If True, save to hydrography_{year}.nc

        Returns:
            xarray Dataset with dimensions (time, farm_id) and environmental vars
        """
        logger.info("=" * 60)
        logger.info("PHASE 1.2: ENVIRONMENTAL FORCING EXTRACTION")
        logger.info("=" * 60)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Farms: {len(farms_df)}")

        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Open aggregated dataset
        url = f"{self.config.base_url}/{self.config.hourly_aggregate}"
        logger.info(f"Connecting to: {url}")

        try:
            ds = xr.open_dataset(url)
            logger.info("Connected successfully")
            logger.info(f"Available variables: {list(ds.data_vars)[:10]}...")
            logger.info(f"Available dimensions: {list(ds.dims.keys())}")

            # Discover variable and dimension names
            self._var_names = self._discover_variables(ds)
            self._dim_names = self._discover_dimensions(ds)
            logger.info(f"Variable mapping: {self._var_names}")

            # Build spatial mapping
            farms_df, mappings = self.build_spatial_mapping(farms_df, ds)

            # Filter to valid ocean points only
            valid_farms = farms_df[farms_df["is_ocean_valid"]].copy()
            if len(valid_farms) < len(farms_df):
                logger.warning(
                    f"Filtered {len(farms_df) - len(valid_farms)} farms with "
                    f"distance > {self.config.max_dist_threshold} deg"
                )

            # Select time range (daily sampling)
            ds_time = ds.sel(time=slice(start_dt, end_dt))
            all_times = pd.to_datetime(ds_time.time.values)

            # Get unique dates
            unique_dates = sorted(set(t.date() for t in all_times))
            logger.info(f"Time steps: {len(unique_dates)} days")

            if len(unique_dates) == 0:
                raise ValueError(f"No data in time range {start_date} to {end_date}")

            # Extract using optimized cluster method
            records = self._extract_clustered(
                ds_time, valid_farms, unique_dates, all_times
            )

            ds.close()

        except Exception as e:
            logger.error(f"Failed to extract hydrography: {e}")
            raise

        # Convert to xarray Dataset
        result_df = pd.DataFrame(records)
        result_ds = self._to_xarray_dataset(result_df, valid_farms)

        # Save to NetCDF if requested
        if output_netcdf:
            year = start_dt.year
            output_path = Path(self.config.output_dir) / f"hydrography_{year}.nc"

            # Remove existing file to avoid permission errors
            if output_path.exists():
                try:
                    output_path.unlink()
                    logger.info(f"Removed existing file: {output_path}")
                except OSError as e:
                    logger.warning(f"Could not remove existing file: {e}")

            try:
                # Use engine='scipy' as fallback for Windows path issues
                result_ds.to_netcdf(output_path, mode='w', engine='scipy')
                logger.info(f"Saved NetCDF to: {output_path}")
            except (PermissionError, OSError) as e:
                logger.warning(f"Could not save NetCDF (path may have special chars): {e}")
                # Save as pickle as fallback
                pkl_path = output_path.with_suffix('.pkl')
                import pickle
                with open(pkl_path, 'wb') as f:
                    pickle.dump(result_ds.to_dict(), f)
                logger.info(f"Saved pickle fallback to: {pkl_path}")

        # Also save CSV for easy inspection
        csv_path = Path(self.config.output_dir) / "environmental_data.csv"
        result_df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV to: {csv_path}")

        logger.info("=" * 60)
        logger.info("ENVIRONMENTAL FORCING EXTRACTION COMPLETE")
        logger.info("=" * 60)

        return result_ds

    def _extract_clustered(
        self,
        ds: xr.Dataset,
        farms_df: pd.DataFrame,
        unique_dates: List,
        all_times: pd.DatetimeIndex,
    ) -> List[Dict]:
        """
        Extract data using clustered bounding-box method.

        Clusters farms into ~13 geographic regions (Production Areas) and
        downloads spatial chunks per day instead of point queries.
        Reduces requests from N_farms x N_days to N_clusters x N_days.
        """
        records = []

        # Cluster farms into geographic regions
        clusters = self._cluster_farms(farms_df)
        logger.info(f"Clustered into {len(clusters)} regions")

        n_days = len(unique_dates)
        depth_idx = self.config.depth_index

        # Get discovered dimension names (with fallbacks)
        y_dim = self._dim_names.get("Y", "Y")
        x_dim = self._dim_names.get("X", "X")
        depth_dim = self._dim_names.get("depth", "depth")
        time_dim = self._dim_names.get("time", "time")

        for cluster_id, cluster_farms in clusters.items():
            cluster_farm_ids = cluster_farms["locality_id"].values
            cluster_y = cluster_farms["grid_y"].values
            cluster_x = cluster_farms["grid_x"].values
            n_farms = len(cluster_farms)

            logger.info(f"  Cluster {cluster_id}: {n_farms} farms")

            # Get bounding box for grid indices
            y_min, y_max = cluster_y.min() - 5, cluster_y.max() + 5
            x_min, x_max = cluster_x.min() - 5, cluster_x.max() + 5

            y_slice = slice(max(0, y_min), y_max)
            x_slice = slice(max(0, x_min), x_max)

            # Process each day
            for day_idx, date in enumerate(unique_dates):
                if day_idx % 50 == 0:
                    logger.info(f"    Day {day_idx + 1}/{n_days}")

                # Find time index for this date
                try:
                    time_mask = [t.date() == date for t in all_times]
                    if not any(time_mask):
                        continue
                    time_idx = time_mask.index(True)
                except (ValueError, IndexError):
                    continue

                # Load spatial chunk for this day using discovered dimension names
                try:
                    isel_kwargs = {time_dim: time_idx}

                    # Only add depth if present
                    if depth_dim in ds.dims:
                        isel_kwargs[depth_dim] = depth_idx

                    # Add spatial dimensions
                    isel_kwargs[y_dim] = y_slice
                    isel_kwargs[x_dim] = x_slice

                    chunk = ds.isel(**isel_kwargs).load()
                except Exception as e:
                    logger.debug(f"Failed to load chunk for {date}: {e}")
                    continue

                # Extract values for all farms in this cluster
                y_offset = y_slice.start
                x_offset = x_slice.start

                for f_idx in range(n_farms):
                    local_y = cluster_y[f_idx] - y_offset
                    local_x = cluster_x[f_idx] - x_offset

                    record = {
                        "locality_id": cluster_farm_ids[f_idx],
                        "date": pd.Timestamp(date),
                    }

                    # Extract each variable
                    for var_key, var_name in self._var_names.items():
                        try:
                            if var_name in chunk:
                                val = float(chunk[var_name].values[local_y, local_x])
                                if not np.isnan(val):
                                    record[var_key] = val
                                else:
                                    record[var_key] = np.nan
                            else:
                                record[var_key] = np.nan
                        except (IndexError, KeyError):
                            record[var_key] = np.nan

                    records.append(record)

        # Compute derived variables
        for record in records:
            u = record.get("u", np.nan)
            v = record.get("v", np.nan)
            if not np.isnan(u) and not np.isnan(v):
                record["current_speed"] = np.sqrt(u**2 + v**2)
            else:
                record["current_speed"] = np.nan

        logger.info(f"Extracted {len(records)} total records")

        # Log data quality summary
        temp_valid = sum(1 for r in records if not np.isnan(r.get("temperature", np.nan)))
        logger.info(f"Temperature coverage: {100*temp_valid/len(records):.1f}%")

        return records

    def _cluster_farms(self, farms_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        Cluster farms into geographic regions based on latitude bands.

        Corresponds roughly to Norwegian Production Areas (PO1-PO13).
        """
        n_clusters = self.config.n_clusters
        lat_min = farms_df["latitude"].min()
        lat_max = farms_df["latitude"].max()

        lat_bins = np.linspace(lat_min - 0.01, lat_max + 0.01, n_clusters + 1)

        farms_df = farms_df.copy()
        farms_df["cluster"] = pd.cut(
            farms_df["latitude"],
            bins=lat_bins,
            labels=range(n_clusters),
            include_lowest=True,
        ).astype(int)

        clusters = {}
        for cluster_id in range(n_clusters):
            cluster_farms = farms_df[farms_df["cluster"] == cluster_id].copy()
            if len(cluster_farms) > 0:
                clusters[cluster_id] = cluster_farms

        return clusters

    def _to_xarray_dataset(
        self, df: pd.DataFrame, farms_df: pd.DataFrame
    ) -> xr.Dataset:
        """
        Convert DataFrame to xarray Dataset with proper structure.

        Output structure matches protocol: (time, farm_id) dimensions
        with temperature, salinity, u, v, current_speed variables.
        """
        # Pivot to wide format
        farm_ids = farms_df["locality_id"].unique()
        dates = df["date"].unique()

        # Initialize arrays
        n_farms = len(farm_ids)
        n_times = len(dates)

        vars_to_extract = ["temperature", "salinity", "u", "v", "current_speed"]
        data_arrays = {var: np.full((n_times, n_farms), np.nan) for var in vars_to_extract}

        # Create lookup indices
        farm_to_idx = {fid: idx for idx, fid in enumerate(farm_ids)}
        date_to_idx = {d: idx for idx, d in enumerate(dates)}

        # Fill arrays
        for _, row in df.iterrows():
            fid = row["locality_id"]
            date = row["date"]

            if fid in farm_to_idx and date in date_to_idx:
                f_idx = farm_to_idx[fid]
                t_idx = date_to_idx[date]

                for var in vars_to_extract:
                    if var in row and not pd.isna(row[var]):
                        data_arrays[var][t_idx, f_idx] = row[var]

        # Create xarray Dataset
        ds = xr.Dataset(
            {
                var: (["time", "farm_id"], data_arrays[var])
                for var in vars_to_extract
            },
            coords={
                "time": dates,
                "farm_id": farm_ids,
            },
        )

        # Add farm metadata as coordinates
        farm_lats = np.array([
            farms_df[farms_df["locality_id"] == fid]["latitude"].values[0]
            for fid in farm_ids
        ])
        farm_lons = np.array([
            farms_df[farms_df["locality_id"] == fid]["longitude"].values[0]
            for fid in farm_ids
        ])

        ds = ds.assign_coords({
            "latitude": ("farm_id", farm_lats),
            "longitude": ("farm_id", farm_lons),
        })

        # Add attributes
        ds.attrs["title"] = "NorKyst-800 Hydrography for Sea Lice Modeling"
        ds.attrs["source"] = "MET Norway THREDDS (thredds.met.no)"
        ds.attrs["depth_m"] = self.config.depth_mixed_layer

        ds["temperature"].attrs = {"units": "degC", "long_name": "Sea water temperature at 3m"}
        ds["salinity"].attrs = {"units": "PSU", "long_name": "Sea water salinity at 3m"}
        ds["u"].attrs = {"units": "m/s", "long_name": "Eastward sea water velocity"}
        ds["v"].attrs = {"units": "m/s", "long_name": "Northward sea water velocity"}
        ds["current_speed"].attrs = {"units": "m/s", "long_name": "Current speed magnitude"}

        return ds


# =========================================================================
# Convenience Function
# =========================================================================

def fetch_environmental_data(
    farms_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    output_dir: str = "data/raw/norkyst800",
) -> xr.Dataset:
    """
    Convenience function to fetch NorKyst-800 environmental data.

    Args:
        farms_df: DataFrame with locality_id, latitude, longitude
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory

    Returns:
        xarray Dataset with environmental variables
    """
    config = NorKyst800Config(output_dir=output_dir)
    client = NorKyst800Client(config)

    return client.extract_hydrography(
        farms_df=farms_df,
        start_date=start_date,
        end_date=end_date,
    )
