"""
Early Warning Operational Dashboard

Target Audience: Farm Managers
Goal: Actionable decision support ("Do I need to treat next week?")

Features:
1. Conformal Prediction "Cone of Uncertainty" - 14-day forecast with 90% CI
2. Traffic Light Risk Gauge - Current week risk assessment
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

# Import model components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.conformal import ConformalSeaLicePredictor, PredictionInterval


@dataclass
class FarmForecast:
    """Forecast result for a single farm."""
    farm_id: int
    farm_name: str
    dates: List[str]
    point_predictions: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    risk_level: str  # 'green', 'yellow', 'red'
    threshold_crossing_day: Optional[int]


class EarlyWarningDashboard:
    """
    Operational dashboard for farm managers.

    Provides:
    - 14-day forecasts with confidence intervals
    - Traffic light risk assessment
    - Threshold crossing alerts

    Args:
        model: SeaLicePredictor model (will be wrapped with conformal)
        coverage: Confidence interval coverage (default: 0.90)
        threshold: Regulatory threshold for adult female lice (default: 0.5)
        warning_threshold: Early warning threshold (default: 0.3)
    """

    REGULATORY_THRESHOLD = 0.5  # Norwegian regulatory limit
    WARNING_THRESHOLD = 0.3     # Early warning level

    def __init__(
        self,
        model,
        coverage: float = 0.90,
        threshold: float = 0.5,
        warning_threshold: float = 0.3,
    ):
        self.model = model
        self.coverage = coverage
        self.threshold = threshold
        self.warning_threshold = warning_threshold

        # Colors for visualization
        self.colors = {
            'green': '#2ECC71',
            'yellow': '#F1C40F',
            'red': '#E74C3C',
            'prediction': '#3498DB',
            'ci': '#85C1E9',
            'threshold': '#E74C3C',
            'warning': '#F39C12',
        }

        # Wrap model with conformal prediction
        from src.models.conformal import ConformalSeaLicePredictor
        self.conformal_model = ConformalSeaLicePredictor(
            base_model=model,
            coverage=coverage,
            calibration_window=100,
            use_adaptive=True,
        )

        self.is_calibrated = False

    def calibrate(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        edge_index: torch.Tensor,
        mask: torch.Tensor,
        feature_indices: Optional[Dict] = None,
    ) -> Dict:
        """
        Calibrate conformal predictor on historical data.

        Args:
            X: Historical input features (T, N, F)
            Y: Historical targets (T, N, 3)
            edge_index: Graph edges
            mask: Observation mask
            feature_indices: Feature name to index mapping

        Returns:
            Calibration diagnostics
        """
        # Get predictions on calibration set
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'network'):
                predictions, _ = self.model.network(
                    X, edge_index, feature_indices=feature_indices
                )
            else:
                predictions, _ = self.model(
                    X, edge_index, feature_indices=feature_indices
                )

        # Calibrate conformal predictor
        self.conformal_model.conformal.calibrate(predictions, Y, mask)
        self.is_calibrated = True

        return self.conformal_model.conformal.get_diagnostics()

    def forecast(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        farm_idx: int,
        feature_indices: Optional[Dict] = None,
        horizon: int = 14,
        start_date: Optional[str] = None,
    ) -> FarmForecast:
        """
        Generate 14-day forecast for a specific farm.

        Args:
            X: Input features (T, N, F)
            edge_index: Graph edges
            farm_idx: Index of farm to forecast
            feature_indices: Feature name to index mapping
            horizon: Forecast horizon in days (default: 14)
            start_date: Start date string (YYYY-MM-DD)

        Returns:
            FarmForecast with predictions and risk assessment
        """
        if not self.is_calibrated:
            raise RuntimeError("Must call calibrate() before forecast()")

        # Get prediction with uncertainty
        interval = self.conformal_model.predict_with_uncertainty(
            X, edge_index, feature_indices=feature_indices
        )

        # Extract farm-specific predictions (adult female lice, index 0)
        T = min(horizon, interval.point_prediction.shape[0])
        point_pred = interval.point_prediction[:T, farm_idx, 0].cpu().numpy()
        lower = interval.lower_bound[:T, farm_idx, 0].cpu().numpy()
        upper = interval.upper_bound[:T, farm_idx, 0].cpu().numpy()

        # Generate date labels
        if start_date:
            from datetime import datetime, timedelta
            start = datetime.strptime(start_date, "%Y-%m-%d")
            dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(T)]
        else:
            dates = [f"Day {i+1}" for i in range(T)]

        # Assess risk level based on upper bound
        max_upper = upper.max()
        if max_upper >= self.threshold:
            risk_level = 'red'
        elif max_upper >= self.warning_threshold:
            risk_level = 'yellow'
        else:
            risk_level = 'green'

        # Find threshold crossing day
        crossing_days = np.where(upper >= self.threshold)[0]
        threshold_crossing_day = crossing_days[0] if len(crossing_days) > 0 else None

        return FarmForecast(
            farm_id=farm_idx,
            farm_name=f"Farm {farm_idx}",
            dates=dates,
            point_predictions=point_pred,
            lower_bounds=lower,
            upper_bounds=upper,
            risk_level=risk_level,
            threshold_crossing_day=threshold_crossing_day,
        )

    def plot_forecast_cone(
        self,
        forecast: FarmForecast,
        ax: Optional[plt.Axes] = None,
        show_legend: bool = True,
    ) -> plt.Figure:
        """
        Plot conformal prediction "cone of uncertainty".

        Args:
            forecast: FarmForecast object from forecast()
            ax: Optional matplotlib axes
            show_legend: Whether to show legend

        Returns:
            matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.figure

        days = np.arange(len(forecast.dates))

        # Plot confidence interval as shaded area
        ax.fill_between(
            days,
            forecast.lower_bounds,
            forecast.upper_bounds,
            color=self.colors['ci'],
            alpha=0.4,
            label=f'{int(self.coverage*100)}% Confidence Interval'
        )

        # Plot point prediction
        ax.plot(
            days,
            forecast.point_predictions,
            color=self.colors['prediction'],
            linewidth=2,
            marker='o',
            markersize=6,
            label='Point Prediction'
        )

        # Draw regulatory threshold line
        ax.axhline(
            y=self.threshold,
            color=self.colors['threshold'],
            linewidth=2,
            linestyle='--',
            label=f'Regulatory Limit ({self.threshold} lice/fish)'
        )

        # Draw warning threshold
        ax.axhline(
            y=self.warning_threshold,
            color=self.colors['warning'],
            linewidth=1.5,
            linestyle=':',
            label=f'Warning Level ({self.warning_threshold})'
        )

        # Shade risk zone
        ax.fill_between(
            days,
            self.threshold,
            max(forecast.upper_bounds.max() * 1.1, self.threshold * 1.5),
            color=self.colors['red'],
            alpha=0.1,
        )

        # Mark threshold crossing point
        if forecast.threshold_crossing_day is not None:
            ax.axvline(
                x=forecast.threshold_crossing_day,
                color=self.colors['red'],
                linewidth=2,
                linestyle='-.',
                alpha=0.7
            )
            ax.annotate(
                'RISK ALERT',
                xy=(forecast.threshold_crossing_day, self.threshold),
                xytext=(forecast.threshold_crossing_day + 0.5, self.threshold * 1.3),
                fontsize=10,
                fontweight='bold',
                color=self.colors['red'],
                arrowprops=dict(arrowstyle='->', color=self.colors['red'])
            )

        # Styling
        ax.set_xlabel('Days', fontsize=12)
        ax.set_ylabel('Adult Female Lice per Fish', fontsize=12)
        ax.set_title(
            f'{forecast.farm_name} - 14-Day Forecast\n'
            f'Risk Level: {forecast.risk_level.upper()}',
            fontsize=14,
            fontweight='bold'
        )

        ax.set_xticks(days[::2] if len(days) > 10 else days)
        ax.set_xticklabels(
            [forecast.dates[i] for i in range(0, len(days), 2 if len(days) > 10 else 1)],
            rotation=45,
            ha='right'
        )

        ax.set_ylim(0, max(forecast.upper_bounds.max() * 1.2, self.threshold * 1.5))
        ax.grid(True, alpha=0.3)

        if show_legend:
            ax.legend(loc='upper left', framealpha=0.9)

        plt.tight_layout()
        return fig

    def plot_traffic_light_gauge(
        self,
        forecast: FarmForecast,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """
        Plot traffic light risk gauge for current week.

        Args:
            forecast: FarmForecast object
            ax: Optional matplotlib axes

        Returns:
            matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 8))
        else:
            fig = ax.figure

        # Calculate current risk value (max upper bound in first week)
        week_upper = forecast.upper_bounds[:min(7, len(forecast.upper_bounds))]
        risk_value = week_upper.max()

        # Draw traffic light housing
        housing = mpatches.FancyBboxPatch(
            (0.2, 0.1), 0.6, 0.8,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor='#2C3E50',
            edgecolor='#1A252F',
            linewidth=3
        )
        ax.add_patch(housing)

        # Light positions (y-coords: red=top, yellow=middle, green=bottom)
        light_positions = {
            'red': 0.7,
            'yellow': 0.5,
            'green': 0.3,
        }

        # Determine which light is active
        if risk_value >= self.threshold:
            active = 'red'
        elif risk_value >= self.warning_threshold:
            active = 'yellow'
        else:
            active = 'green'

        # Draw lights
        for color, y_pos in light_positions.items():
            if color == active:
                brightness = 1.0
                glow_color = self.colors[color]
            else:
                brightness = 0.2
                glow_color = None

            # Draw glow effect for active light
            if glow_color:
                glow = plt.Circle(
                    (0.5, y_pos), 0.15,
                    color=glow_color,
                    alpha=0.3
                )
                ax.add_patch(glow)

            # Draw light bulb
            light_color = self.colors[color] if brightness == 1.0 else '#34495E'
            light = plt.Circle(
                (0.5, y_pos), 0.1,
                color=light_color,
                ec='#1A252F',
                linewidth=2
            )
            ax.add_patch(light)

        # Add risk value annotation
        ax.text(
            0.5, 0.05,
            f'Upper Bound: {risk_value:.2f}',
            ha='center',
            va='center',
            fontsize=10,
            fontweight='bold',
            color='white'
        )

        # Add threshold labels on the side
        ax.text(1.0, 0.7, f'>{self.threshold}', ha='left', va='center', fontsize=9, color=self.colors['red'])
        ax.text(1.0, 0.5, f'{self.warning_threshold}-{self.threshold}', ha='left', va='center', fontsize=9, color=self.colors['yellow'])
        ax.text(1.0, 0.3, f'<{self.warning_threshold}', ha='left', va='center', fontsize=9, color=self.colors['green'])

        # Status text
        status_text = {
            'green': 'SAFE',
            'yellow': 'WARNING',
            'red': 'HIGH RISK'
        }
        ax.text(
            0.5, 0.95,
            status_text[active],
            ha='center',
            va='top',
            fontsize=14,
            fontweight='bold',
            color=self.colors[active]
        )

        ax.set_xlim(0, 1.3)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(forecast.farm_name, fontsize=12, fontweight='bold')

        return fig

    def create_dashboard(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        farm_indices: List[int],
        feature_indices: Optional[Dict] = None,
        start_date: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create complete early warning dashboard for multiple farms.

        Args:
            X: Input features (T, N, F)
            edge_index: Graph edges
            farm_indices: List of farm indices to display
            feature_indices: Feature name to index mapping
            start_date: Start date for x-axis labels
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure
        """
        n_farms = len(farm_indices)

        # Create figure with subplots: forecast cone + traffic light for each farm
        fig = plt.figure(figsize=(16, 5 * n_farms))

        for i, farm_idx in enumerate(farm_indices):
            # Generate forecast
            forecast = self.forecast(
                X, edge_index, farm_idx,
                feature_indices=feature_indices,
                start_date=start_date,
            )

            # Calculate vertical position (top to bottom)
            row_height = 0.85 / n_farms
            bottom = 0.92 - (i + 1) * row_height

            # Forecast cone (left, larger - 75% width)
            ax_cone = fig.add_axes([0.08, bottom, 0.65, row_height * 0.85])
            self.plot_forecast_cone(forecast, ax=ax_cone)

            # Traffic light gauge (right, smaller - 15% width)
            ax_gauge = fig.add_axes([0.78, bottom, 0.15, row_height * 0.85])
            self.plot_traffic_light_gauge(forecast, ax=ax_gauge)

        fig.suptitle(
            'Early Warning Dashboard - Sea Lice Outbreak Risk Assessment',
            fontsize=16,
            fontweight='bold',
            y=0.98
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Dashboard saved to {save_path}")

        return fig

    def plot_multi_farm_risk_matrix(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        farm_indices: List[int],
        feature_indices: Optional[Dict] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create risk matrix showing all farms at a glance.

        Args:
            X: Input features
            edge_index: Graph edges
            farm_indices: List of farm indices
            feature_indices: Feature mapping
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        n_farms = len(farm_indices)
        n_cols = min(5, n_farms)
        n_rows = (n_farms + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)

        for i, farm_idx in enumerate(farm_indices):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]

            forecast = self.forecast(
                X, edge_index, farm_idx,
                feature_indices=feature_indices,
            )
            self.plot_traffic_light_gauge(forecast, ax=ax)

        # Hide unused subplots
        for i in range(n_farms, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')

        fig.suptitle(
            'Farm Risk Overview',
            fontsize=14,
            fontweight='bold'
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Risk matrix saved to {save_path}")

        return fig


def demo_early_warning():
    """Demo function to test the early warning dashboard."""
    print("Early Warning Dashboard Demo")
    print("=" * 50)

    # This would be run with actual model and data
    print("To use this dashboard:")
    print("1. Load your trained SeaLicePredictor model")
    print("2. Create dashboard: dashboard = EarlyWarningDashboard(model)")
    print("3. Calibrate: dashboard.calibrate(X_calib, Y_calib, edge_index, mask)")
    print("4. Create visualization: dashboard.create_dashboard(X_test, edge_index, [0,1,2])")


if __name__ == "__main__":
    demo_early_warning()
