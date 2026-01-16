"""
Counterfactual "What-If" Simulator

Target Audience: Planners
Goal: Optimizing treatment schedules by comparing different scenarios.

Features:
1. Treatment Scenario Comparison - Bath, Medicine, Cleaner Fish effects
2. Intervention timing analysis
3. Cost-benefit visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from copy import deepcopy

# Import model components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TreatmentScenario:
    """Definition of a treatment scenario."""
    name: str
    description: str
    treatment_type: str  # 'bath', 'medicine', 'cleaner_fish', 'combined'
    intervention_day: int
    duration: int = 1
    color: str = '#3498DB'

    # Treatment indices in the input tensor (based on model documentation)
    # Index 12: Bath treatment
    # Index 13: Medicine treatment
    # Index 14: Cleaner fish

    def get_treatment_indices(self) -> List[int]:
        """Get feature indices for this treatment type."""
        treatment_map = {
            'bath': [12],
            'medicine': [13],
            'cleaner_fish': [14],
            'combined': [12, 13],
            'all': [12, 13, 14],
        }
        return treatment_map.get(self.treatment_type, [12])


@dataclass
class ScenarioResult:
    """Results from running a treatment scenario."""
    scenario: TreatmentScenario
    predictions: np.ndarray  # (T, N, 3) or (T,) for single farm
    adult_female: np.ndarray  # (T,) adult female prediction
    peak_lice: float
    days_above_threshold: int
    threshold_crossing_day: Optional[int]


class WhatIfSimulator:
    """
    Counterfactual "What-If" Simulator for treatment planning.

    Allows users to:
    1. Compare baseline (no treatment) vs treatment scenarios
    2. Test different treatment types at different times
    3. Visualize intervention effectiveness

    Args:
        model: SeaLicePredictor or SeaLiceGLKAN model
        threshold: Regulatory threshold (default: 0.5 lice/fish)
    """

    REGULATORY_THRESHOLD = 0.5

    # Predefined scenarios
    SCENARIOS = {
        'baseline': TreatmentScenario(
            name='Baseline',
            description='No treatment applied',
            treatment_type='none',
            intervention_day=0,
            color='#2C3E50'
        ),
        'bath_early': TreatmentScenario(
            name='Early Bath Treatment',
            description='Bath treatment at day 5',
            treatment_type='bath',
            intervention_day=5,
            color='#E74C3C'
        ),
        'bath_late': TreatmentScenario(
            name='Late Bath Treatment',
            description='Bath treatment at day 10',
            treatment_type='bath',
            intervention_day=10,
            color='#9B59B6'
        ),
        'medicine': TreatmentScenario(
            name='Medicine Treatment',
            description='Oral medicine at day 5',
            treatment_type='medicine',
            intervention_day=5,
            color='#3498DB'
        ),
        'cleaner_fish': TreatmentScenario(
            name='Cleaner Fish',
            description='Add cleaner fish at day 5',
            treatment_type='cleaner_fish',
            intervention_day=5,
            duration=10,  # Cleaner fish stay longer
            color='#2ECC71'
        ),
        'combined': TreatmentScenario(
            name='Combined Treatment',
            description='Bath + Medicine at day 5',
            treatment_type='combined',
            intervention_day=5,
            color='#F1C40F'
        ),
    }

    def __init__(
        self,
        model,
        threshold: float = 0.5,
    ):
        self.model = model
        self.threshold = threshold

        # Get the GLKAN network
        if hasattr(model, 'network'):
            self.glkan = model.network
        else:
            self.glkan = model

        # Treatment feature indices
        self.treatment_indices = {
            'bath': 12,
            'medicine': 13,
            'cleaner_fish': 14,
        }

    def apply_treatment(
        self,
        X: torch.Tensor,
        scenario: TreatmentScenario,
    ) -> torch.Tensor:
        """
        Apply a treatment scenario to input features.

        Args:
            X: Input features (T, N, F)
            scenario: Treatment scenario to apply

        Returns:
            Modified input tensor with treatment flags set
        """
        X_modified = X.clone()

        if scenario.treatment_type == 'none':
            return X_modified

        T = X.shape[0]
        start_day = scenario.intervention_day
        end_day = min(start_day + scenario.duration, T)

        # Get treatment indices
        indices = scenario.get_treatment_indices()

        # Apply treatment flags
        for idx in indices:
            if idx < X.shape[-1]:
                X_modified[start_day:end_day, :, idx] = 1.0

        return X_modified

    def run_scenario(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        scenario: TreatmentScenario,
        farm_idx: int = 0,
        feature_indices: Optional[Dict] = None,
    ) -> ScenarioResult:
        """
        Run a single treatment scenario.

        Args:
            X: Input features (T, N, F)
            edge_index: Graph edges
            scenario: Treatment scenario
            farm_idx: Farm index to analyze
            feature_indices: Feature mapping

        Returns:
            ScenarioResult with predictions and metrics
        """
        # Apply treatment to input
        X_treated = self.apply_treatment(X, scenario)

        # Run model
        self.glkan.eval()
        with torch.no_grad():
            predictions, _ = self.glkan(
                X_treated, edge_index,
                feature_indices=feature_indices,
            )

        # Extract results
        predictions_np = predictions.cpu().numpy()
        adult_female = predictions_np[:, farm_idx, 0]

        # Compute metrics
        peak_lice = adult_female.max()
        days_above = (adult_female > self.threshold).sum()

        crossing_days = np.where(adult_female > self.threshold)[0]
        first_crossing = crossing_days[0] if len(crossing_days) > 0 else None

        return ScenarioResult(
            scenario=scenario,
            predictions=predictions_np,
            adult_female=adult_female,
            peak_lice=peak_lice,
            days_above_threshold=days_above,
            threshold_crossing_day=first_crossing,
        )

    def compare_scenarios(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        scenario_names: List[str],
        farm_idx: int = 0,
        feature_indices: Optional[Dict] = None,
    ) -> Dict[str, ScenarioResult]:
        """
        Compare multiple treatment scenarios.

        Args:
            X: Input features
            edge_index: Graph edges
            scenario_names: List of scenario names to compare
            farm_idx: Farm index to analyze
            feature_indices: Feature mapping

        Returns:
            Dict mapping scenario names to results
        """
        results = {}

        for name in scenario_names:
            if name not in self.SCENARIOS:
                raise ValueError(f"Unknown scenario: {name}")

            scenario = self.SCENARIOS[name]
            result = self.run_scenario(
                X, edge_index, scenario, farm_idx, feature_indices
            )
            results[name] = result

        return results

    def create_custom_scenario(
        self,
        name: str,
        treatment_type: str,
        intervention_day: int,
        duration: int = 1,
        color: str = '#3498DB',
    ) -> TreatmentScenario:
        """
        Create a custom treatment scenario.

        Args:
            name: Scenario name
            treatment_type: 'bath', 'medicine', 'cleaner_fish', 'combined'
            intervention_day: Day to start treatment
            duration: Treatment duration in days
            color: Plot color

        Returns:
            TreatmentScenario object
        """
        return TreatmentScenario(
            name=name,
            description=f'{treatment_type.replace("_", " ").title()} at day {intervention_day}',
            treatment_type=treatment_type,
            intervention_day=intervention_day,
            duration=duration,
            color=color,
        )

    def plot_scenario_comparison(
        self,
        results: Dict[str, ScenarioResult],
        ax: Optional[plt.Axes] = None,
        show_intervention: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot comparison of multiple scenarios.

        Args:
            results: Dict of scenario results
            ax: Optional matplotlib axes
            show_intervention: Whether to show intervention markers
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 8))
        else:
            fig = ax.figure

        # Plot each scenario
        for name, result in results.items():
            days = np.arange(len(result.adult_female))

            ax.plot(
                days,
                result.adult_female,
                color=result.scenario.color,
                linewidth=2.5 if name == 'baseline' else 2,
                linestyle='-' if name == 'baseline' else '--',
                label=f'{result.scenario.name} (peak: {result.peak_lice:.2f})',
                marker='o' if len(days) < 20 else None,
                markersize=4,
            )

            # Mark intervention point
            if show_intervention and result.scenario.intervention_day > 0:
                ax.axvline(
                    x=result.scenario.intervention_day,
                    color=result.scenario.color,
                    linestyle=':',
                    alpha=0.5,
                )
                ax.annotate(
                    result.scenario.name.split()[0],
                    xy=(result.scenario.intervention_day, result.adult_female.max()),
                    xytext=(result.scenario.intervention_day + 0.5, result.adult_female.max() * 1.05),
                    fontsize=8,
                    color=result.scenario.color,
                    alpha=0.7,
                )

        # Draw threshold line
        ax.axhline(
            y=self.threshold,
            color='red',
            linewidth=2,
            linestyle='--',
            label=f'Regulatory Limit ({self.threshold})'
        )

        # Shade danger zone
        ax.axhspan(
            self.threshold,
            ax.get_ylim()[1] if ax.get_ylim()[1] > self.threshold else self.threshold * 1.5,
            alpha=0.1,
            color='red'
        )

        # Styling
        ax.set_xlabel('Days', fontsize=12)
        ax.set_ylabel('Adult Female Lice per Fish', fontsize=12)
        ax.set_title(
            'Treatment Scenario Comparison\n'
            'How different interventions affect lice populations',
            fontsize=14,
            fontweight='bold'
        )

        ax.set_xlim(0, len(list(results.values())[0].adult_female) - 1)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', framealpha=0.9)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Scenario comparison saved to {save_path}")

        return fig

    def plot_intervention_timing_analysis(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        treatment_type: str = 'bath',
        farm_idx: int = 0,
        feature_indices: Optional[Dict] = None,
        intervention_days: Optional[List[int]] = None,
        ax: Optional[plt.Axes] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Analyze optimal intervention timing.

        Shows how treatment effectiveness varies with timing.

        Args:
            X: Input features
            edge_index: Graph edges
            treatment_type: Type of treatment
            farm_idx: Farm to analyze
            feature_indices: Feature mapping
            intervention_days: List of days to test
            ax: Optional axes
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        if intervention_days is None:
            T = X.shape[0]
            intervention_days = list(range(1, min(T - 5, 15)))

        # Create color gradient
        cmap = plt.cm.viridis
        colors = [cmap(i / len(intervention_days)) for i in range(len(intervention_days))]

        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            ax_lines, ax_metrics = axes
        else:
            fig = ax.figure
            ax_lines = ax
            ax_metrics = None

        # Run baseline
        baseline = self.run_scenario(
            X, edge_index, self.SCENARIOS['baseline'], farm_idx, feature_indices
        )

        # Test each intervention day
        results = []
        for i, day in enumerate(intervention_days):
            scenario = self.create_custom_scenario(
                name=f'Day {day}',
                treatment_type=treatment_type,
                intervention_day=day,
                color=mcolors.to_hex(colors[i]),
            )
            result = self.run_scenario(
                X, edge_index, scenario, farm_idx, feature_indices
            )
            results.append(result)

            # Plot trajectory
            days = np.arange(len(result.adult_female))
            ax_lines.plot(
                days,
                result.adult_female,
                color=colors[i],
                linewidth=1.5,
                alpha=0.7,
                label=f'Day {day}' if i % 3 == 0 else None,
            )

        # Plot baseline
        ax_lines.plot(
            days,
            baseline.adult_female,
            color='black',
            linewidth=2.5,
            linestyle='--',
            label='No Treatment'
        )

        # Threshold
        ax_lines.axhline(y=self.threshold, color='red', linestyle='--', linewidth=2)

        ax_lines.set_xlabel('Days', fontsize=12)
        ax_lines.set_ylabel('Adult Female Lice', fontsize=12)
        ax_lines.set_title(
            f'{treatment_type.replace("_", " ").title()} Treatment: Timing Analysis',
            fontsize=14,
            fontweight='bold'
        )
        ax_lines.legend(loc='upper right')
        ax_lines.grid(True, alpha=0.3)

        # Metrics plot
        if ax_metrics is not None:
            peak_values = [r.peak_lice for r in results]
            days_above = [r.days_above_threshold for r in results]

            ax_metrics.bar(
                intervention_days,
                peak_values,
                color=colors,
                edgecolor='white',
                alpha=0.8
            )
            ax_metrics.axhline(
                y=baseline.peak_lice,
                color='black',
                linestyle='--',
                linewidth=2,
                label='No Treatment'
            )
            ax_metrics.axhline(
                y=self.threshold,
                color='red',
                linestyle=':',
                linewidth=2,
                label='Threshold'
            )

            ax_metrics.set_xlabel('Intervention Day', fontsize=12)
            ax_metrics.set_ylabel('Peak Lice Count', fontsize=12)
            ax_metrics.set_title(
                'Peak Lice by Intervention Day',
                fontsize=14,
                fontweight='bold'
            )
            ax_metrics.legend()
            ax_metrics.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Timing analysis saved to {save_path}")

        return fig

    def plot_cost_benefit_analysis(
        self,
        results: Dict[str, ScenarioResult],
        treatment_costs: Optional[Dict[str, float]] = None,
        penalty_per_day: float = 10000,  # Cost per day above threshold
        ax: Optional[plt.Axes] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot cost-benefit analysis of different treatments.

        Args:
            results: Scenario results
            treatment_costs: Dict of treatment costs (NOK)
            penalty_per_day: Regulatory penalty per day above threshold
            ax: Optional axes
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        if treatment_costs is None:
            # Default treatment costs (illustrative, in NOK)
            treatment_costs = {
                'baseline': 0,
                'bath_early': 50000,
                'bath_late': 50000,
                'medicine': 75000,
                'cleaner_fish': 100000,
                'combined': 120000,
            }

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 7))
        else:
            fig = ax.figure

        scenarios = list(results.keys())
        treatment_cost_values = []
        penalty_cost_values = []
        total_costs = []

        for name in scenarios:
            result = results[name]
            t_cost = treatment_costs.get(name, 0)
            p_cost = result.days_above_threshold * penalty_per_day

            treatment_cost_values.append(t_cost / 1000)  # Convert to thousands
            penalty_cost_values.append(p_cost / 1000)
            total_costs.append((t_cost + p_cost) / 1000)

        x = np.arange(len(scenarios))
        width = 0.35

        # Stacked bar chart
        bars1 = ax.bar(x, treatment_cost_values, width, label='Treatment Cost', color='#3498DB')
        bars2 = ax.bar(x, penalty_cost_values, width, bottom=treatment_cost_values,
                       label='Regulatory Penalty', color='#E74C3C')

        # Add total cost labels
        for i, (total, scenario) in enumerate(zip(total_costs, scenarios)):
            ax.annotate(
                f'{total:.0f}k',
                xy=(i, total),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                fontweight='bold'
            )

        # Styling
        ax.set_ylabel('Cost (1000 NOK)', fontsize=12)
        ax.set_xlabel('Treatment Scenario', fontsize=12)
        ax.set_title(
            'Cost-Benefit Analysis\n'
            f'Treatment costs vs regulatory penalties ({penalty_per_day/1000:.0f}k NOK/day above threshold)',
            fontsize=14,
            fontweight='bold'
        )

        ax.set_xticks(x)
        ax.set_xticklabels([results[s].scenario.name for s in scenarios], rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Cost-benefit analysis saved to {save_path}")

        return fig

    def create_what_if_report(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        farm_idx: int = 0,
        feature_indices: Optional[Dict] = None,
        save_dir: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create comprehensive What-If analysis report.

        Args:
            X: Input features
            edge_index: Graph edges
            farm_idx: Farm to analyze
            feature_indices: Feature mapping
            save_dir: Directory to save outputs

        Returns:
            matplotlib Figure
        """
        # Run all predefined scenarios
        scenarios = ['baseline', 'bath_early', 'medicine', 'cleaner_fish']
        results = self.compare_scenarios(
            X, edge_index, scenarios, farm_idx, feature_indices
        )

        # Create figure
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(2, 2, figure=fig)

        # Scenario comparison (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_scenario_comparison(results, ax=ax1)

        # Timing analysis (top-right)
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_intervention_timing_analysis(
            X, edge_index, 'bath', farm_idx, feature_indices, ax=ax2
        )

        # Cost-benefit (bottom-left)
        ax3 = fig.add_subplot(gs[1, 0])
        self.plot_cost_benefit_analysis(results, ax=ax3)

        # Summary table (bottom-right)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        # Create summary table
        table_data = []
        headers = ['Scenario', 'Peak Lice', 'Days Above', 'First Crossing']

        for name, result in results.items():
            crossing = f'Day {result.threshold_crossing_day}' if result.threshold_crossing_day else 'Never'
            table_data.append([
                result.scenario.name,
                f'{result.peak_lice:.3f}',
                str(result.days_above_threshold),
                crossing
            ])

        table = ax4.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            colColours=['#3498DB'] * len(headers),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        ax4.set_title('Scenario Summary', fontsize=14, fontweight='bold', pad=20)

        fig.suptitle(
            f'What-If Treatment Simulator - Farm {farm_idx}',
            fontsize=16,
            fontweight='bold',
            y=1.02
        )

        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            # Save individual plots
            fig_comp, ax_comp = plt.subplots(figsize=(14, 8))
            self.plot_scenario_comparison(results, ax=ax_comp,
                                         save_path=save_path / 'scenario_comparison.png')
            plt.close(fig_comp)

            fig_timing = self.plot_intervention_timing_analysis(
                X, edge_index, 'bath', farm_idx, feature_indices,
                save_path=save_path / 'timing_analysis.png'
            )
            plt.close(fig_timing)

            fig_cost, ax_cost = plt.subplots(figsize=(12, 7))
            self.plot_cost_benefit_analysis(results, ax=ax_cost,
                                           save_path=save_path / 'cost_benefit.png')
            plt.close(fig_cost)

            # Save combined report
            fig.savefig(
                save_path / 'whatif_report.png',
                dpi=150, bbox_inches='tight', facecolor='white'
            )
            print(f"What-If report saved to {save_dir}")

        return fig


def demo_what_if():
    """Demo function for What-If simulator."""
    print("Counterfactual What-If Simulator Demo")
    print("=" * 50)
    print("This tool helps planners optimize treatment schedules:")
    print()
    print("1. Scenario Comparison:")
    print("   - Baseline (no treatment)")
    print("   - Bath treatment (early/late)")
    print("   - Medicine treatment")
    print("   - Cleaner fish deployment")
    print("   - Combined treatments")
    print()
    print("2. Timing Analysis:")
    print("   - Find optimal intervention day")
    print("   - Earlier intervention = lower peak")
    print()
    print("3. Cost-Benefit:")
    print("   - Treatment cost vs regulatory penalties")
    print("   - Find most cost-effective strategy")
    print()
    print("Usage:")
    print("  sim = WhatIfSimulator(model)")
    print("  results = sim.compare_scenarios(X, edge_index, ['baseline', 'bath_early', 'medicine'])")
    print("  sim.plot_scenario_comparison(results, save_path='comparison.png')")


if __name__ == "__main__":
    demo_what_if()
