"""
Conformal Prediction for Sea Lice Forecasting

Provides mathematically rigorous uncertainty quantification:
- Instead of "0.4 lice", output "90% chance between 0.2 and 0.6"
- Distribution-free: No assumptions about error distribution
- Guaranteed coverage: If you say 90%, you get 90% (asymptotically)

Key Methods:
1. EnbPI (Ensemble Batch Prediction Intervals) - for time series
2. Adaptive Conformal Inference (ACI) - adjusts for distribution shift
3. Conformalized Quantile Regression - for asymmetric intervals

Reference:
- Xu & Xie (2021): "Conformal Prediction Interval for Dynamic Time-Series"
- Gibbs & Candès (2021): "Adaptive Conformal Inference Under Distribution Shift"

Critical for farmers: "Should I order a treatment ship?"
- If upper bound > 0.5 (regulatory threshold), yes.
- If upper bound < 0.3, probably safe to wait.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, List, Union
from collections import deque
from dataclasses import dataclass


@dataclass
class PredictionInterval:
    """Structured prediction interval output."""
    point_prediction: torch.Tensor  # Central prediction
    lower_bound: torch.Tensor       # Lower confidence bound
    upper_bound: torch.Tensor       # Upper confidence bound
    coverage_level: float           # Nominal coverage (e.g., 0.90)

    def exceeds_threshold(self, threshold: float) -> torch.Tensor:
        """Check if upper bound exceeds a threshold (e.g., 0.5 for regulatory limit)."""
        return self.upper_bound > threshold

    def interval_width(self) -> torch.Tensor:
        """Width of prediction interval (uncertainty measure)."""
        return self.upper_bound - self.lower_bound

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            'prediction': self.point_prediction,
            'lower': self.lower_bound,
            'upper': self.upper_bound,
            'width': self.interval_width(),
        }


class EnbPI(nn.Module):
    """
    Ensemble Batch Prediction Intervals for Time Series.

    EnbPI adapts conformal prediction to time series by:
    1. Using a sliding window of recent residuals for calibration
    2. Updating the quantile adaptively as new data arrives
    3. Handling temporal dependencies through bootstrap aggregation

    The key insight: Use past prediction errors to calibrate future intervals.

    Algorithm:
    1. Collect residuals: r_t = |y_t - ŷ_t| over calibration window
    2. For coverage (1-α), compute q = quantile(residuals, 1-α)
    3. Prediction interval: [ŷ - q, ŷ + q]
    4. After observing y_t, update residual window

    Args:
        coverage: Target coverage level (e.g., 0.90 for 90% intervals)
        window_size: Number of recent residuals to use for calibration
        symmetric: If True, use |y - ŷ|; if False, track upper/lower separately
        adapt_rate: Learning rate for adaptive coverage adjustment (ACI)
    """

    def __init__(
        self,
        coverage: float = 0.90,
        window_size: int = 100,
        symmetric: bool = True,
        adapt_rate: float = 0.01,
    ):
        super().__init__()

        self.coverage = coverage
        self.window_size = window_size
        self.symmetric = symmetric
        self.adapt_rate = adapt_rate

        # Residual storage (circular buffer)
        self.residuals: deque = deque(maxlen=window_size)

        # For asymmetric intervals
        self.residuals_upper: deque = deque(maxlen=window_size)
        self.residuals_lower: deque = deque(maxlen=window_size)

        # Adaptive alpha (for ACI)
        self.alpha = 1.0 - coverage
        self.alpha_t = self.alpha  # Current adaptive alpha

        # Track empirical coverage for diagnostics
        self.coverage_history: List[float] = []
        self.n_predictions = 0
        self.n_covered = 0

    def reset(self):
        """Reset calibration state."""
        self.residuals.clear()
        self.residuals_upper.clear()
        self.residuals_lower.clear()
        self.alpha_t = self.alpha
        self.coverage_history.clear()
        self.n_predictions = 0
        self.n_covered = 0

    def calibrate(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Calibrate conformal predictor using historical predictions.

        This should be called ONCE on a calibration set before making
        predictions. For online learning, use update() instead.

        Args:
            predictions: Model predictions, shape (T, N, D) or (T, N)
            targets: Ground truth values, same shape
            mask: Optional observation mask, shape (T, N)
        """
        # Flatten for simplicity
        if predictions.dim() > 2:
            predictions = predictions.reshape(-1, predictions.shape[-1])
            targets = targets.reshape(-1, targets.shape[-1])
            if mask is not None:
                mask = mask.reshape(-1).unsqueeze(-1).expand_as(predictions)
        elif predictions.dim() == 2:
            if mask is not None:
                mask = mask.reshape(-1)

        # Compute residuals
        if self.symmetric:
            residuals = (targets - predictions).abs()
        else:
            residuals_upper = targets - predictions  # Positive when under-predicted
            residuals_lower = predictions - targets  # Positive when over-predicted

        # Apply mask
        if mask is not None:
            if self.symmetric:
                residuals = residuals[mask.bool()]
            else:
                residuals_upper = residuals_upper[mask.bool()]
                residuals_lower = residuals_lower[mask.bool()]

        # Convert to numpy and store
        if self.symmetric:
            residuals_np = residuals.detach().cpu().numpy().flatten()
            for r in residuals_np[-self.window_size:]:
                self.residuals.append(r)
        else:
            upper_np = residuals_upper.detach().cpu().numpy().flatten()
            lower_np = residuals_lower.detach().cpu().numpy().flatten()
            for u, l in zip(upper_np[-self.window_size:], lower_np[-self.window_size:]):
                self.residuals_upper.append(u)
                self.residuals_lower.append(l)

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        was_covered: Optional[bool] = None,
    ) -> None:
        """
        Online update with new observation.

        Called after ground truth becomes available to update calibration.

        Args:
            prediction: Previous prediction
            target: Observed ground truth
            was_covered: Whether the true value was within the interval
        """
        # Compute new residual
        if self.symmetric:
            residual = (target - prediction).abs().mean().item()
            self.residuals.append(residual)
        else:
            upper = (target - prediction).mean().item()
            lower = (prediction - target).mean().item()
            self.residuals_upper.append(upper)
            self.residuals_lower.append(lower)

        # Adaptive Conformal Inference (ACI) update
        if was_covered is not None:
            self.n_predictions += 1
            if was_covered:
                self.n_covered += 1
                # Covered -> can widen alpha (narrower intervals)
                self.alpha_t = self.alpha_t + self.adapt_rate * (self.alpha - self.alpha_t)
            else:
                # Not covered -> shrink alpha (wider intervals)
                self.alpha_t = self.alpha_t - self.adapt_rate * self.alpha_t

            # Clamp to reasonable range
            self.alpha_t = max(0.01, min(0.5, self.alpha_t))

            # Track coverage
            self.coverage_history.append(self.n_covered / self.n_predictions)

    def get_quantile(self, alpha: Optional[float] = None) -> float:
        """
        Get the calibration quantile for prediction intervals.

        Args:
            alpha: Override alpha (default: use adaptive alpha_t)

        Returns:
            Quantile value for interval construction
        """
        if alpha is None:
            alpha = self.alpha_t

        if len(self.residuals) == 0:
            # Not calibrated yet - return conservative estimate
            return 1.0

        residuals_array = np.array(list(self.residuals))

        # Finite-sample correction: use (n+1)*(1-alpha)/n quantile
        n = len(residuals_array)
        adjusted_level = min((n + 1) * (1 - alpha) / n, 1.0)

        return float(np.quantile(residuals_array, adjusted_level))

    def get_asymmetric_quantiles(
        self,
        alpha: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Get separate upper and lower quantiles for asymmetric intervals."""
        if alpha is None:
            alpha = self.alpha_t

        if len(self.residuals_upper) == 0:
            return 1.0, 1.0

        upper_array = np.array(list(self.residuals_upper))
        lower_array = np.array(list(self.residuals_lower))

        n = len(upper_array)
        adjusted_level = min((n + 1) * (1 - alpha / 2) / n, 1.0)

        q_upper = float(np.quantile(upper_array, adjusted_level))
        q_lower = float(np.quantile(lower_array, adjusted_level))

        return q_upper, q_lower

    def predict_interval(
        self,
        point_prediction: torch.Tensor,
    ) -> PredictionInterval:
        """
        Construct prediction interval around point prediction.

        Args:
            point_prediction: Model's point prediction, any shape

        Returns:
            PredictionInterval with bounds and metadata
        """
        if self.symmetric:
            q = self.get_quantile()
            lower = point_prediction - q
            upper = point_prediction + q
        else:
            q_upper, q_lower = self.get_asymmetric_quantiles()
            lower = point_prediction - q_lower
            upper = point_prediction + q_upper

        # Clamp to non-negative for lice counts
        lower = torch.clamp(lower, min=0.0)
        upper = torch.clamp(upper, min=0.0)

        return PredictionInterval(
            point_prediction=point_prediction,
            lower_bound=lower,
            upper_bound=upper,
            coverage_level=1.0 - self.alpha_t,
        )

    def get_diagnostics(self) -> Dict[str, float]:
        """Get calibration diagnostics."""
        empirical_coverage = self.n_covered / max(self.n_predictions, 1)

        return {
            'target_coverage': self.coverage,
            'empirical_coverage': empirical_coverage,
            'adaptive_alpha': self.alpha_t,
            'n_predictions': self.n_predictions,
            'n_residuals': len(self.residuals),
            'current_quantile': self.get_quantile(),
        }


class ConformalSeaLicePredictor(nn.Module):
    """
    Sea Lice Predictor with Conformal Uncertainty Quantification.

    Wraps a base predictor with EnbPI conformal prediction to provide
    mathematically rigorous prediction intervals.

    Key output for farmers:
    - "There is a 90% chance lice will be between 0.2 and 0.6"
    - "There is a 95% chance lice will NOT exceed 0.5" (regulatory threshold)

    Usage:
        model = SeaLicePredictor(...)
        conformal = ConformalSeaLicePredictor(model, coverage=0.90)

        # Calibrate on held-out data
        conformal.calibrate(val_predictions, val_targets, val_mask)

        # Make predictions with intervals
        result = conformal.predict_with_uncertainty(x, edge_index)
        print(f"Prediction: {result.point_prediction}")
        print(f"90% CI: [{result.lower_bound}, {result.upper_bound}]")
        print(f"Risk of exceeding 0.5: {result.exceeds_threshold(0.5).float().mean()}")

    Args:
        base_model: Underlying SeaLicePredictor or SeaLiceGLKAN
        coverage: Target coverage level (0.90 = 90% intervals)
        calibration_window: Number of residuals to use for calibration
        use_adaptive: Whether to use Adaptive Conformal Inference
        per_farm: Whether to calibrate separately for each farm
    """

    def __init__(
        self,
        base_model: nn.Module,
        coverage: float = 0.90,
        calibration_window: int = 100,
        use_adaptive: bool = True,
        per_farm: bool = False,
    ):
        super().__init__()

        self.base_model = base_model
        self.coverage = coverage
        self.per_farm = per_farm

        # Global conformal predictor
        self.conformal = EnbPI(
            coverage=coverage,
            window_size=calibration_window,
            symmetric=True,
            adapt_rate=0.01 if use_adaptive else 0.0,
        )

        # Per-farm conformal predictors (optional)
        self.farm_conformal: Optional[Dict[int, EnbPI]] = None
        if per_farm:
            self.farm_conformal = {}

        # Track which output dimensions to compute intervals for
        # Default: all output dimensions (adult_female, mobile, attached)
        self.interval_dims = None  # None = all

    def calibrate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        edge_index: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Calibrate conformal predictor on validation data.

        IMPORTANT: Must be called before predict_with_uncertainty()

        Args:
            x: Input features, shape (T, N, F)
            y: Ground truth targets, shape (T, N, D)
            edge_index: Graph edges
            mask: Observation mask, shape (T, N)
            **kwargs: Additional args for base model

        Returns:
            Calibration diagnostics
        """
        # Get base model predictions
        self.base_model.eval()
        with torch.no_grad():
            if hasattr(self.base_model, 'predict'):
                predictions = self.base_model.predict(x, edge_index, **kwargs)
            elif hasattr(self.base_model, 'network'):
                predictions, _ = self.base_model.network(x, edge_index, **kwargs)
            else:
                predictions, _ = self.base_model(x, edge_index, **kwargs)

        # Calibrate global predictor
        self.conformal.calibrate(predictions, y, mask)

        # Per-farm calibration (if enabled)
        if self.per_farm:
            T, N = y.shape[:2]
            for farm_idx in range(N):
                if farm_idx not in self.farm_conformal:
                    self.farm_conformal[farm_idx] = EnbPI(
                        coverage=self.coverage,
                        window_size=max(20, self.conformal.window_size // N),
                    )

                farm_pred = predictions[:, farm_idx]
                farm_target = y[:, farm_idx]
                farm_mask = mask[:, farm_idx] if mask is not None else None

                self.farm_conformal[farm_idx].calibrate(
                    farm_pred.unsqueeze(1),
                    farm_target.unsqueeze(1),
                    farm_mask.unsqueeze(1) if farm_mask is not None else None,
                )

        return self.conformal.get_diagnostics()

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs,
    ) -> PredictionInterval:
        """
        Make predictions with conformal uncertainty intervals.

        Args:
            x: Input features, shape (T, N, F)
            edge_index: Graph edges
            **kwargs: Additional args for base model

        Returns:
            PredictionInterval with point prediction and bounds
        """
        # Get base model predictions
        self.base_model.eval()
        with torch.no_grad():
            if hasattr(self.base_model, 'predict'):
                predictions = self.base_model.predict(x, edge_index, **kwargs)
            elif hasattr(self.base_model, 'network'):
                predictions, _ = self.base_model.network(x, edge_index, **kwargs)
            else:
                predictions, _ = self.base_model(x, edge_index, **kwargs)

        # Construct intervals
        if self.per_farm and self.farm_conformal:
            # Per-farm intervals
            T, N = predictions.shape[:2]
            D = predictions.shape[2] if predictions.dim() > 2 else 1

            lower = torch.zeros_like(predictions)
            upper = torch.zeros_like(predictions)

            for farm_idx in range(N):
                if farm_idx in self.farm_conformal:
                    farm_interval = self.farm_conformal[farm_idx].predict_interval(
                        predictions[:, farm_idx]
                    )
                    lower[:, farm_idx] = farm_interval.lower_bound
                    upper[:, farm_idx] = farm_interval.upper_bound
                else:
                    # Fall back to global
                    global_interval = self.conformal.predict_interval(
                        predictions[:, farm_idx]
                    )
                    lower[:, farm_idx] = global_interval.lower_bound
                    upper[:, farm_idx] = global_interval.upper_bound

            return PredictionInterval(
                point_prediction=predictions,
                lower_bound=lower,
                upper_bound=upper,
                coverage_level=self.coverage,
            )
        else:
            # Global intervals
            return self.conformal.predict_interval(predictions)

    def update_calibration(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        farm_idx: Optional[int] = None,
    ) -> None:
        """
        Online update after observing ground truth.

        Call this when new observations become available to keep
        calibration current.

        Args:
            prediction: Previous prediction
            target: Observed ground truth
            farm_idx: Optional farm index for per-farm update
        """
        # Check coverage
        interval = self.conformal.predict_interval(prediction)
        was_covered = (target >= interval.lower_bound).all() and \
                      (target <= interval.upper_bound).all()

        # Update global
        self.conformal.update(prediction, target, was_covered.item())

        # Update per-farm if applicable
        if self.per_farm and farm_idx is not None and farm_idx in self.farm_conformal:
            farm_interval = self.farm_conformal[farm_idx].predict_interval(prediction)
            farm_covered = (target >= farm_interval.lower_bound).all() and \
                          (target <= farm_interval.upper_bound).all()
            self.farm_conformal[farm_idx].update(prediction, target, farm_covered.item())

    def get_risk_assessment(
        self,
        interval: PredictionInterval,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Assess risk relative to regulatory threshold.

        For Norwegian aquaculture, 0.5 adult females per fish is a critical threshold.

        Args:
            interval: Prediction interval from predict_with_uncertainty
            threshold: Regulatory threshold (default: 0.5)

        Returns:
            Dict with risk metrics:
            - exceeds_threshold: Boolean mask where upper bound > threshold
            - margin_to_threshold: Distance from upper bound to threshold
            - probability_above: Approximate probability of exceeding (from interval)
        """
        exceeds = interval.exceeds_threshold(threshold)
        margin = threshold - interval.upper_bound  # Positive = safe, negative = at risk

        # Approximate probability from interval position
        # If threshold is above upper bound: ~0%
        # If threshold is below lower bound: ~100%
        # Otherwise: interpolate based on position in interval
        width = interval.interval_width().clamp(min=1e-6)
        position = (threshold - interval.lower_bound) / width
        position = position.clamp(0, 1)

        # probability_above ≈ 1 - position (very rough approximation)
        prob_above = 1 - position

        return {
            'exceeds_threshold': exceeds,
            'margin_to_threshold': margin,
            'probability_above': prob_above,
            'threshold': threshold,
        }

    def forward(self, *args, **kwargs):
        """Forward pass returns prediction with uncertainty."""
        return self.predict_with_uncertainty(*args, **kwargs)


class MultiCoveragePredictor(nn.Module):
    """
    Prediction intervals at multiple coverage levels.

    Useful for visualization and decision-making:
    - 50% interval: "Most likely range"
    - 80% interval: "Reasonable range"
    - 95% interval: "Almost certainly within this range"

    Args:
        base_model: Underlying predictor
        coverage_levels: List of coverage levels (e.g., [0.50, 0.80, 0.95])
        calibration_window: Window size for EnbPI
    """

    def __init__(
        self,
        base_model: nn.Module,
        coverage_levels: List[float] = [0.50, 0.80, 0.90, 0.95],
        calibration_window: int = 100,
    ):
        super().__init__()

        self.base_model = base_model
        self.coverage_levels = sorted(coverage_levels)

        # Create conformal predictor for each level
        self.conformal_predictors = nn.ModuleDict({
            f'c{int(c*100)}': EnbPI(
                coverage=c,
                window_size=calibration_window,
            )
            for c in coverage_levels
        })

    def calibrate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        edge_index: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Dict]:
        """Calibrate all coverage levels."""
        self.base_model.eval()
        with torch.no_grad():
            if hasattr(self.base_model, 'predict'):
                predictions = self.base_model.predict(x, edge_index, **kwargs)
            elif hasattr(self.base_model, 'network'):
                predictions, _ = self.base_model.network(x, edge_index, **kwargs)
            else:
                predictions, _ = self.base_model(x, edge_index, **kwargs)

        diagnostics = {}
        for name, conformal in self.conformal_predictors.items():
            conformal.calibrate(predictions, y, mask)
            diagnostics[name] = conformal.get_diagnostics()

        return diagnostics

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs,
    ) -> Dict[str, PredictionInterval]:
        """
        Get prediction intervals at all coverage levels.

        Returns:
            Dict mapping coverage level to PredictionInterval
        """
        self.base_model.eval()
        with torch.no_grad():
            if hasattr(self.base_model, 'predict'):
                predictions = self.base_model.predict(x, edge_index, **kwargs)
            elif hasattr(self.base_model, 'network'):
                predictions, _ = self.base_model.network(x, edge_index, **kwargs)
            else:
                predictions, _ = self.base_model(x, edge_index, **kwargs)

        intervals = {}
        for name, conformal in self.conformal_predictors.items():
            intervals[name] = conformal.predict_interval(predictions)

        return intervals

    def get_uncertainty_summary(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Get comprehensive uncertainty summary for visualization.

        Returns dict with:
        - prediction: Point prediction
        - bands: Dict of (lower, upper) for each coverage level
        """
        intervals = self.predict_with_uncertainty(x, edge_index, **kwargs)

        # Use highest coverage prediction as point estimate
        highest_coverage = f'c{int(max(self.coverage_levels)*100)}'
        prediction = intervals[highest_coverage].point_prediction

        bands = {}
        for name, interval in intervals.items():
            coverage = interval.coverage_level
            bands[f'{int(coverage*100)}%'] = {
                'lower': interval.lower_bound,
                'upper': interval.upper_bound,
                'width': interval.interval_width(),
            }

        return {
            'prediction': prediction,
            'bands': bands,
        }


def evaluate_coverage(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    lower_bounds: torch.Tensor,
    upper_bounds: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Evaluate empirical coverage of prediction intervals.

    Args:
        predictions: Point predictions, shape (T, N, D)
        targets: Ground truth, same shape
        lower_bounds: Lower interval bounds
        upper_bounds: Upper interval bounds
        mask: Observation mask, shape (T, N)

    Returns:
        Dict with coverage metrics
    """
    # Check coverage
    covered = (targets >= lower_bounds) & (targets <= upper_bounds)

    if mask is not None:
        if mask.dim() < covered.dim():
            mask = mask.unsqueeze(-1).expand_as(covered)
        covered = covered & mask
        n_valid = mask.float().sum()
    else:
        n_valid = covered.numel()

    # Overall coverage
    empirical_coverage = covered.float().sum() / max(n_valid, 1)

    # Interval widths
    widths = upper_bounds - lower_bounds
    if mask is not None:
        mean_width = (widths * mask.float()).sum() / max(n_valid, 1)
    else:
        mean_width = widths.mean()

    # Sharpness (narrower is better, conditional on coverage)

    return {
        'empirical_coverage': empirical_coverage.item(),
        'mean_interval_width': mean_width.item(),
        'n_valid': int(n_valid.item()),
    }
