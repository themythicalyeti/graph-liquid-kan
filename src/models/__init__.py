"""
Phase 3: Graph-Liquid-KAN Architecture

The unified framework combining:
- Graphon Topology (continuous graph limits)
- Liquid Time-Constant Dynamics (input-dependent tau)
- Kolmogorov-Arnold Functional Approximation (learnable basis functions)

Modules:
- layers: FastKAN, GraphonAggregator (atomic components)
- cells: LiquidKANCell (the core CfC engine)
- network: GLKANNetwork (full temporal unroller)
- biology: Sea lice-specific biological modules
- sea_lice_network: Complete domain-specific GLKAN
"""

from .layers import FastKAN, GraphonAggregator, GraphKAN
from .cells import LiquidKANCell, GraphLiquidKANCell
from .network import GLKANNetwork, GLKANPredictor
from .biology import (
    # Virtual Source functions (feature engineering)
    compute_fecundity,
    compute_egg_production_rate,
    # KAN-based biological modules
    BelehradekKAN,
    SalinityMortalityKAN,
    LarvalTransportModule,
    KHopGraphConv,
    SeaLiceDynamicsCell,
    PopulationMassConservation,
    create_sea_lice_biology_modules,
)
from .sea_lice_network import SeaLiceGLKAN, SeaLicePINNLoss, SeaLicePredictor
from .conformal import (
    PredictionInterval,
    EnbPI,
    ConformalSeaLicePredictor,
    MultiCoveragePredictor,
    evaluate_coverage,
)

__all__ = [
    # Core layers
    "FastKAN",
    "GraphonAggregator",
    "GraphKAN",
    # Core cells
    "LiquidKANCell",
    "GraphLiquidKANCell",
    # Generic network
    "GLKANNetwork",
    "GLKANPredictor",
    # Virtual Source functions (feature engineering)
    "compute_fecundity",
    "compute_egg_production_rate",
    # Sea lice biology modules
    "BelehradekKAN",
    "SalinityMortalityKAN",
    "LarvalTransportModule",
    "KHopGraphConv",
    "SeaLiceDynamicsCell",
    "PopulationMassConservation",
    "create_sea_lice_biology_modules",
    # Sea lice network
    "SeaLiceGLKAN",
    "SeaLicePINNLoss",
    "SeaLicePredictor",
    # Conformal Prediction (Uncertainty Quantification)
    "PredictionInterval",
    "EnbPI",
    "ConformalSeaLicePredictor",
    "MultiCoveragePredictor",
    "evaluate_coverage",
]
