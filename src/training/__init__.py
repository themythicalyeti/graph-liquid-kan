"""
Phase 4: Optimization & Training Module

Contains:
- losses.py: Physics-Informed Loss functions
- trainer.py: Training Executive loop
- audit.py: Scientific Validation tests
"""

from .losses import PhysicsInformedLoss, GLKANLoss
from .trainer import GLKANTrainer, TrainingConfig

__all__ = [
    "PhysicsInformedLoss",
    "GLKANLoss",
    "GLKANTrainer",
    "TrainingConfig",
]
