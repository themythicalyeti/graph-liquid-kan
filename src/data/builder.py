"""
Phase 2.2: Temporal Alignment Engine (GLKANDatasetBuilder)

Transforms raw geospatial artifacts from Phase 1 into aligned tensors
for the Graph-Liquid-KAN architecture.

Critical Constraint:
The "Liquid" time-constant tau(T) depends on the *continuity* of the input
signal. Discontinuities or misalignment between ocean forcing u(t) and
biological response Y(t) will cause the ODE solver to diverge.

Key Operations:
1. Define master timeline (daily resolution)
2. Process ocean data: resample hourly->daily, normalize
3. Process lice data: map (year,week)->date, create sparse tensor + mask
4. Align treatment events as intervention features
5. Output: X (features), Y (targets), mask (observation indicator)
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from loguru import logger

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


@dataclass
class BuilderConfig:
    """Configuration for dataset builder."""
    # Feature selection
    environmental_features: List[str] = field(default_factory=lambda: [
        "temperature", "salinity", "current_speed", "current_sin", "current_cos"
    ])
    treatment_features: List[str] = field(default_factory=lambda: [
        "is_mechanical", "is_medicinal", "is_cleaner_fish"
    ])

    # Target variables (lice counts)
    target_columns: List[str] = field(default_factory=lambda: [
        "avg_adult_female_lice", "avg_mobile_lice", "avg_stationary_lice"
    ])

    # AUTOREGRESSIVE FEATURES - CRITICAL FOR PREDICTIVE POWER
    # Past lice counts are the best predictor of future lice counts
    use_autoregressive: bool = True  # Enable lagged lice features
    ar_lags: List[int] = field(default_factory=lambda: [7, 14])  # Lag in days (1 week, 2 weeks)

    # Temporal resolution
    resample_freq: str = "1D"  # Daily

    # Normalization (computed from training data)
    normalize_features: bool = True

    # NaN handling
    fill_nan_value: float = 0.0  # For normalized features


class GLKANDatasetBuilder:
    """
    Dataset builder for Graph-Liquid-KAN architecture.

    Bridges the gap between:
    - Discrete biological observations (weekly lice counts)
    - Continuous physical forcing (hourly/daily ocean data)

    The output tensor structure:
    - X: (T, N, F) - Environmental + treatment features
    - Y: (T, N, 3) - Lice counts (adult_female, mobile, stationary)
    - mask: (T, N) - Boolean mask for valid observations
    - times: (T,) - Datetime index
    - site_ids: (N,) - Farm locality IDs
    """

    def __init__(
        self,
        nodes_path: Union[str, Path],
        lice_path: Union[str, Path],
        treatments_path: Union[str, Path],
        hydro_path: Optional[Union[str, Path]] = None,
        config: Optional[BuilderConfig] = None,
    ):
        """
        Initialize the dataset builder.

        Args:
            nodes_path: Path to graph_nodes_metadata.csv
            lice_path: Path to lice_reports.csv
            treatments_path: Path to treatments.csv
            hydro_path: Path to hydrography_*.nc (optional, can set later)
            config: Builder configuration
        """
        if not HAS_XARRAY:
            raise ImportError("xarray required. Install with: pip install xarray")

        self.config = config or BuilderConfig()

        # Load node data
        logger.info(f"Loading nodes from: {nodes_path}")
        self.nodes_df = pd.read_csv(nodes_path)
        self.farm_ids = self.nodes_df["locality_id"].values
        self.n_farms = len(self.farm_ids)

        # Create ID -> index mapping
        self.id_to_idx = {int(fid): i for i, fid in enumerate(self.farm_ids)}

        logger.info(f"Loaded {self.n_farms} farm nodes")

        # Load lice data
        logger.info(f"Loading lice reports from: {lice_path}")
        self.lice_df = pd.read_csv(lice_path)
        logger.info(f"Loaded {len(self.lice_df)} lice records")

        # Load treatment data
        logger.info(f"Loading treatments from: {treatments_path}")
        self.treatments_df = pd.read_csv(treatments_path)
        logger.info(f"Loaded {len(self.treatments_df)} treatment events")

        # Load hydrography if provided
        self.hydro_ds = None
        if hydro_path is not None:
            self.load_hydrography(hydro_path)

        # Normalization statistics (computed from training data)
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None

    def load_hydrography(self, hydro_path: Union[str, Path]) -> None:
        """Load hydrography NetCDF file."""
        logger.info(f"Loading hydrography from: {hydro_path}")
        # Use scipy engine to avoid Windows path issues with special characters
        self.hydro_ds = xr.open_dataset(hydro_path, engine='scipy')
        logger.info(f"Hydrography dimensions: {dict(self.hydro_ds.sizes)}")

    def _week_to_date(self, year: int, week: int) -> Optional[datetime]:
        """
        Convert (year, week) to date (Monday of that week).

        This is the observation date for weekly lice reports.
        """
        try:
            # ISO week date format: year-Www-d where d=1 is Monday
            date_str = f"{year}-W{week:02d}-1"
            return datetime.strptime(date_str, "%G-W%V-%u")
        except ValueError:
            # Handle edge cases (week 53, etc.)
            try:
                # Fallback: use first day of year + week offset
                jan1 = datetime(year, 1, 1)
                return jan1 + timedelta(weeks=week - 1)
            except:
                return None

    def _process_environmental_features(
        self,
        start_date: datetime,
        end_date: datetime,
        dates: pd.DatetimeIndex,
    ) -> np.ndarray:
        """
        Process environmental features from hydrography data.

        Returns:
            X_env: (T, N, F_env) environmental feature tensor
        """
        T = len(dates)
        N = self.n_farms

        if self.hydro_ds is None:
            logger.warning("No hydrography data loaded, using zeros for environmental features")
            return np.zeros((T, N, 5), dtype=np.float32)

        logger.info("Processing environmental features...")

        # Select time range and resample to daily
        hydro_time = self.hydro_ds.sel(time=slice(start_date, end_date))

        # Check if we need to resample
        time_delta = pd.to_datetime(hydro_time.time.values[1]) - pd.to_datetime(hydro_time.time.values[0])
        if time_delta < pd.Timedelta("1D"):
            logger.info(f"Resampling from {time_delta} to daily...")
            hydro_daily = hydro_time.resample(time="1D").mean()
        else:
            hydro_daily = hydro_time

        # Extract variables
        # Note: farm_id dimension should match our farm ordering
        hydro_farm_ids = hydro_daily.farm_id.values

        # Create mapping from hydro farm order to our farm order
        hydro_to_our = {}
        for h_idx, h_fid in enumerate(hydro_farm_ids):
            if int(h_fid) in self.id_to_idx:
                hydro_to_our[h_idx] = self.id_to_idx[int(h_fid)]

        # Extract raw values
        temp = hydro_daily["temperature"].values  # (T_hydro, N_hydro)
        salt = hydro_daily["salinity"].values

        # Current speed and direction
        if "current_speed" in hydro_daily:
            current_speed = hydro_daily["current_speed"].values
        elif "u" in hydro_daily and "v" in hydro_daily:
            u = hydro_daily["u"].values
            v = hydro_daily["v"].values
            current_speed = np.sqrt(u**2 + v**2)
        else:
            current_speed = np.zeros_like(temp)

        # Current direction (sin/cos encoding to avoid discontinuity at 360°)
        if "u" in hydro_daily and "v" in hydro_daily:
            u = hydro_daily["u"].values
            v = hydro_daily["v"].values
            current_angle = np.arctan2(v, u)
            current_sin = np.sin(current_angle)
            current_cos = np.cos(current_angle)
        else:
            current_sin = np.zeros_like(temp)
            current_cos = np.zeros_like(temp)

        # Align time indices
        hydro_dates = pd.to_datetime(hydro_daily.time.values)
        date_to_tidx = {d.date(): i for i, d in enumerate(dates)}
        hydro_date_to_tidx = {d.date(): i for i, d in enumerate(hydro_dates)}

        # Initialize output tensor
        X_env = np.zeros((T, N, 5), dtype=np.float32)

        # Fill tensor
        for our_date in dates:
            our_tidx = date_to_tidx[our_date.date()]

            if our_date.date() in hydro_date_to_tidx:
                hydro_tidx = hydro_date_to_tidx[our_date.date()]

                for h_idx, our_idx in hydro_to_our.items():
                    X_env[our_tidx, our_idx, 0] = temp[hydro_tidx, h_idx]
                    X_env[our_tidx, our_idx, 1] = salt[hydro_tidx, h_idx]
                    X_env[our_tidx, our_idx, 2] = current_speed[hydro_tidx, h_idx]
                    X_env[our_tidx, our_idx, 3] = current_sin[hydro_tidx, h_idx]
                    X_env[our_tidx, our_idx, 4] = current_cos[hydro_tidx, h_idx]

        # Handle NaNs
        nan_count = np.isnan(X_env).sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in environmental features")
            X_env = np.nan_to_num(X_env, nan=0.0)

        logger.info(f"Environmental features shape: {X_env.shape}")
        return X_env

    def _process_treatment_features(
        self,
        start_date: datetime,
        end_date: datetime,
        dates: pd.DatetimeIndex,
    ) -> np.ndarray:
        """
        Process treatment events into binary feature tensor.

        Treatments create discontinuities in lice dynamics that the model
        must handle. We encode as [is_mechanical, is_medicinal, is_cleaner_fish].

        Returns:
            X_treat: (T, N, 3) treatment feature tensor
        """
        T = len(dates)
        N = self.n_farms

        X_treat = np.zeros((T, N, 3), dtype=np.float32)

        # Parse treatment dates
        if "treatment_date" in self.treatments_df.columns:
            self.treatments_df["treatment_date"] = pd.to_datetime(
                self.treatments_df["treatment_date"], errors="coerce"
            )

        date_to_tidx = {d.date(): i for i, d in enumerate(dates)}

        treatment_count = 0
        for _, row in self.treatments_df.iterrows():
            fid = row.get("locality_id")
            if fid not in self.id_to_idx:
                continue

            f_idx = self.id_to_idx[fid]

            # Get treatment date
            treat_date = row.get("treatment_date")
            if pd.isna(treat_date):
                # Try to construct from year/week
                year = row.get("year")
                week = row.get("week")
                if year and week:
                    treat_date = self._week_to_date(int(year), int(week))

            if treat_date is None:
                continue

            if isinstance(treat_date, pd.Timestamp):
                treat_date = treat_date.to_pydatetime()

            if treat_date.date() not in date_to_tidx:
                continue

            t_idx = date_to_tidx[treat_date.date()]

            # Set treatment flags
            if row.get("is_mechanical", 0):
                X_treat[t_idx, f_idx, 0] = 1.0
            if row.get("is_medicinal", 0):
                X_treat[t_idx, f_idx, 1] = 1.0
            if row.get("is_cleaner_fish", 0):
                X_treat[t_idx, f_idx, 2] = 1.0

            treatment_count += 1

        logger.info(f"Processed {treatment_count} treatment events")
        logger.info(f"Treatment features shape: {X_treat.shape}")
        return X_treat

    def _process_lice_labels(
        self,
        start_date: datetime,
        end_date: datetime,
        dates: pd.DatetimeIndex,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process lice counts into sparse label tensor with observation mask.

        The mask indicates which (time, farm) pairs have ground-truth observations.
        The ODE will learn to interpolate dynamics between these truth points.

        Returns:
            Y: (T, N, 3) lice count tensor
            mask: (T, N) boolean observation mask
        """
        T = len(dates)
        N = self.n_farms

        Y = np.zeros((T, N, 3), dtype=np.float32)
        mask = np.zeros((T, N), dtype=bool)

        date_to_tidx = {d.date(): i for i, d in enumerate(dates)}

        obs_count = 0
        mnar_count = 0

        for _, row in self.lice_df.iterrows():
            fid = row.get("locality_id")
            if fid not in self.id_to_idx:
                continue

            f_idx = self.id_to_idx[fid]

            # Get observation date from year/week
            year = row.get("year")
            week = row.get("week")

            if pd.isna(year) or pd.isna(week):
                continue

            obs_date = self._week_to_date(int(year), int(week))
            if obs_date is None:
                continue

            if obs_date.date() not in date_to_tidx:
                continue

            t_idx = date_to_tidx[obs_date.date()]

            # Check MNAR flag
            is_mnar = row.get("is_mnar", False)
            if is_mnar:
                mnar_count += 1
                continue  # Skip MNAR observations (they are systematically biased)

            # Extract lice counts
            adult_female = row.get("avg_adult_female_lice")
            mobile = row.get("avg_mobile_lice")
            stationary = row.get("avg_stationary_lice")

            # Only count as observation if we have the primary target
            if pd.notna(adult_female):
                Y[t_idx, f_idx, 0] = float(adult_female)
                Y[t_idx, f_idx, 1] = float(mobile) if pd.notna(mobile) else 0.0
                Y[t_idx, f_idx, 2] = float(stationary) if pd.notna(stationary) else 0.0
                mask[t_idx, f_idx] = True
                obs_count += 1

        logger.info(f"Processed {obs_count} valid lice observations")
        logger.info(f"Skipped {mnar_count} MNAR records")
        logger.info(f"Observation coverage: {100 * mask.sum() / mask.size:.2f}%")
        logger.info(f"Labels shape: {Y.shape}, Mask shape: {mask.shape}")

        return Y, mask

    def _process_autoregressive_features(
        self,
        Y: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Create autoregressive features from lagged lice counts.

        CRITICAL: Past lice counts are the strongest predictor of future counts.
        This adds lagged versions of Y as input features.

        For each lag L in ar_lags:
        - X_ar[t] = Y[t-L] (shifted by L days)
        - Also includes a binary "has_observation" flag for each lag

        Args:
            Y: (T, N, 3) lice count tensor
            mask: (T, N) observation mask

        Returns:
            X_ar: (T, N, n_ar_features) autoregressive feature tensor
                  where n_ar_features = len(ar_lags) * (3 lice types + 1 mask flag)
        """
        if not self.config.use_autoregressive:
            return np.zeros((Y.shape[0], Y.shape[1], 0), dtype=np.float32)

        T, N, _ = Y.shape
        lags = self.config.ar_lags
        n_features_per_lag = 4  # 3 lice counts + 1 observation flag

        X_ar = np.zeros((T, N, len(lags) * n_features_per_lag), dtype=np.float32)

        logger.info(f"Creating autoregressive features with lags: {lags}")

        for lag_idx, lag in enumerate(lags):
            feat_offset = lag_idx * n_features_per_lag

            # Shift Y by lag days
            if lag < T:
                # Y values from lag days ago
                X_ar[lag:, :, feat_offset:feat_offset+3] = Y[:-lag, :, :]

                # Observation flag from lag days ago
                X_ar[lag:, :, feat_offset+3] = mask[:-lag].astype(np.float32)

            logger.info(f"  Lag {lag} days: {(X_ar[:, :, feat_offset+3] > 0).sum()} observations available")

        # Forward-fill missing values within each farm
        # This propagates the last known observation forward in time
        logger.info("Forward-filling missing autoregressive values...")
        for n in range(N):
            for lag_idx, lag in enumerate(lags):
                feat_offset = lag_idx * n_features_per_lag
                obs_flag_idx = feat_offset + 3

                last_known = np.zeros(3)
                for t in range(T):
                    if X_ar[t, n, obs_flag_idx] > 0:
                        # Have an observation at this lag
                        last_known = X_ar[t, n, feat_offset:feat_offset+3].copy()
                    else:
                        # No observation - use last known value
                        X_ar[t, n, feat_offset:feat_offset+3] = last_known

        logger.info(f"Autoregressive features shape: {X_ar.shape}")
        return X_ar

    def build_dataset(
        self,
        start_date: str,
        end_date: str,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Build the complete dataset for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_path: Optional path to save the dataset

        Returns:
            Dictionary containing:
            - X: (T, N, F) feature tensor
            - Y: (T, N, 3) target tensor
            - mask: (T, N) observation mask
            - times: (T,) datetime array
            - site_ids: (N,) farm IDs
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: BUILDING DATASET TENSORS")
        logger.info("=" * 60)

        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Create master timeline (daily)
        dates = pd.date_range(start=start_dt, end=end_dt, freq=self.config.resample_freq)
        T = len(dates)
        N = self.n_farms

        logger.info(f"Timeline: {start_date} to {end_date} ({T} days)")
        logger.info(f"Farms: {N}")

        # Process lice labels FIRST (needed for autoregressive features)
        logger.info("\n[1/4] Processing lice labels...")
        Y, mask = self._process_lice_labels(start_dt, end_dt, dates)

        # Process autoregressive features (lagged lice counts)
        logger.info("\n[2/4] Processing autoregressive features (lagged lice)...")
        X_ar = self._process_autoregressive_features(Y, mask)

        # Process environmental features
        logger.info("\n[3/4] Processing environmental features...")
        X_env = self._process_environmental_features(start_dt, end_dt, dates)

        # Process treatment features
        logger.info("\n[4/4] Processing treatment features...")
        X_treat = self._process_treatment_features(start_dt, end_dt, dates)

        # Concatenate features: [autoregressive, environmental, treatment]
        # Autoregressive first because it's the most predictive!
        X = np.concatenate([X_ar, X_env, X_treat], axis=-1)

        logger.info(f"\nFinal feature tensor X: {X.shape}")
        logger.info(f"  Autoregressive features: {X_ar.shape[-1]} (CRITICAL for prediction)")
        logger.info(f"  Environmental features: {X_env.shape[-1]}")
        logger.info(f"  Treatment features: {X_treat.shape[-1]}")

        # Normalize features if requested
        if self.config.normalize_features:
            logger.info("\nNormalizing features...")
            X = self._normalize_features(X, fit=True)

        # Convert to tensors
        dataset = {
            "X": torch.from_numpy(X).float(),
            "Y": torch.from_numpy(Y).float(),
            "mask": torch.from_numpy(mask).bool(),
            "times": dates.to_numpy(),
            "site_ids": torch.from_numpy(self.farm_ids).long(),
        }

        # Add normalization stats if computed
        if self.feature_mean is not None:
            dataset["feature_mean"] = torch.from_numpy(self.feature_mean).float()
            dataset["feature_std"] = torch.from_numpy(self.feature_std).float()

        # Save if path provided
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as npz for compatibility
            np.savez(
                output_path,
                X=X,
                Y=Y,
                mask=mask,
                times=dates.to_numpy(),
                site_ids=self.farm_ids,
                feature_mean=self.feature_mean,
                feature_std=self.feature_std,
            )
            logger.info(f"\nSaved dataset to: {output_path}")

        logger.info("\n" + "=" * 60)
        logger.info("DATASET BUILD COMPLETE")
        logger.info("=" * 60)

        return dataset

    def _normalize_features(
        self,
        X: np.ndarray,
        fit: bool = False,
    ) -> np.ndarray:
        """
        Normalize features using z-score normalization.

        Critical: Calculate mean/std over training period only.

        Args:
            X: (T, N, F) feature tensor
            fit: If True, compute statistics; otherwise use existing

        Returns:
            Normalized feature tensor
        """
        if fit or self.feature_mean is None:
            # Compute statistics over all (time, farm) pairs
            # Reshape to (T*N, F) for easy computation
            X_flat = X.reshape(-1, X.shape[-1])

            # Compute per-feature mean and std, ignoring zeros (missing values)
            self.feature_mean = np.zeros(X.shape[-1])
            self.feature_std = np.ones(X.shape[-1])

            for f in range(X.shape[-1]):
                feat = X_flat[:, f]
                valid = feat != 0  # Simple heuristic for missing
                if valid.sum() > 0:
                    self.feature_mean[f] = feat[valid].mean()
                    self.feature_std[f] = feat[valid].std() + 1e-8

        # Apply normalization
        X_norm = (X - self.feature_mean) / self.feature_std

        # Handle NaNs/Infs
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)
        X_norm = np.clip(X_norm, -10, 10)  # Clip extreme values

        logger.info(f"Feature means: {self.feature_mean}")
        logger.info(f"Feature stds: {self.feature_std}")

        return X_norm
