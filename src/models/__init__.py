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
"""

from .layers import FastKAN, GraphonAggregator
from .cells import LiquidKANCell
from .network import GLKANNetwork

__all__ = [
    "FastKAN",
    "GraphonAggregator",
    "LiquidKANCell",
    "GLKANNetwork",
]
