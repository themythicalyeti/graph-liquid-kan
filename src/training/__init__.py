"""
Phase 4: Optimization & Training Module

Contains:
- losses.py: Physics-Informed Loss functions
- trainer.py: Training Executive loop
- pretrain.py: Biological module pre-training
- audit.py: Scientific Validation tests
"""

from .losses import PhysicsInformedLoss, GLKANLoss
from .trainer import GLKANTrainer, TrainingConfig
from .pretrain import BiologyPretrainer, BiologyPretrainConfig

__all__ = [
    "PhysicsInformedLoss",
    "GLKANLoss",
    "GLKANTrainer",
    "TrainingConfig",
    "BiologyPretrainer",
    "BiologyPretrainConfig",
]
