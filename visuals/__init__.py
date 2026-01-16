"""
Sea Lice Model Visualization Suite

Four categories of visualizations:
1. Early Warning Dashboard - Operational decision support for farm managers
2. Interpretability Suite - Biological curve validation for scientists
3. Spatial Network - Infection dynamics for regulators
4. What-If Simulator - Treatment planning for planners

Usage:
    from visuals import run_all_visualizations
    run_all_visualizations(model, data_path='data/processed')
"""

from .early_warning import EarlyWarningDashboard
from .interpretability import InterpretabilitySuite
from .spatial_network import SpatialNetworkVisualizer
from .counterfactual import WhatIfSimulator

__all__ = [
    'EarlyWarningDashboard',
    'InterpretabilitySuite',
    'SpatialNetworkVisualizer',
    'WhatIfSimulator',
]
