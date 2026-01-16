"""
Pre-training Module for Biological KAN Components

This module pre-trains BelehradekKAN and SalinityMortalityKAN on theoretical
biological curves before full model training. This ensures the model starts
with biologically plausible responses rather than random initialization.

Theoretical Curves:
- Belehradek: D(T) = a * (T - alpha)^b where a=0.05, alpha=0, b=1.5
- Salinity: S(sal) = 1 / (1 + exp(-k * (sal - threshold))) where k=0.3, threshold=25

After pre-training, the biological modules should closely match these curves.
During full training, a reduced learning rate (0.5x) allows controlled drift
to adapt to real data while preserving the biological foundation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from loguru import logger


def _get_biology_modules():
    """Lazy import to avoid circular dependency."""
    try:
        # When used as part of installed package
        from ..models.biology import BelehradekKAN, SalinityMortalityKAN
    except ImportError:
        # When src is added to path directly
        from models.biology import BelehradekKAN, SalinityMortalityKAN
    return BelehradekKAN, SalinityMortalityKAN


@dataclass
class BiologyPretrainConfig:
    """Configuration for biological module pre-training."""
    # Training parameters
    n_epochs: int = 200
    learning_rate: float = 1e-2
    batch_size: int = 64

    # Data generation
    n_samples: int = 1000
    temp_range: Tuple[float, float] = (0.0, 25.0)
    salinity_range: Tuple[float, float] = (0.0, 40.0)

    # Theoretical curve parameters
    # Belehradek: D(T) = a * (T - alpha)^b
    belehradek_a: float = 0.05
    belehradek_alpha: float = 0.0  # Biological zero
    belehradek_b: float = 1.5

    # Salinity: S(sal) = 1 / (1 + exp(-k * (sal - threshold)))
    salinity_k: float = 0.3
    salinity_threshold: float = 25.0

    # Checkpoint
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "pretrained_biology.pt"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Module config (must match main model)
    n_bases: int = 8


class BiologyPretrainer:
    """
    Pre-trainer for biological KAN modules.

    Trains BelehradekKAN and SalinityMortalityKAN to match theoretical
    biological curves before being integrated into the full model.

    Usage:
        pretrainer = BiologyPretrainer()
        pretrained = pretrainer.pretrain_all()  # Returns dict of state_dicts

        # Load into model
        model.load_pretrained_biology(pretrained)
    """

    def __init__(self, config: Optional[BiologyPretrainConfig] = None):
        self.config = config or BiologyPretrainConfig()
        self.device = torch.device(self.config.device)

        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _belehradek_theoretical(self, temperature: torch.Tensor) -> torch.Tensor:
        """
        Theoretical Belehradek curve: D(T) = a * (T - alpha)^b

        Args:
            temperature: Temperature in Celsius

        Returns:
            Development rate
        """
        a = self.config.belehradek_a
        alpha = self.config.belehradek_alpha
        b = self.config.belehradek_b

        # Effective temperature (above biological zero)
        t_eff = F.relu(temperature - alpha)

        # Belehradek equation
        dev_rate = a * torch.pow(t_eff + 1e-6, b)

        return dev_rate

    def _salinity_theoretical(self, salinity: torch.Tensor) -> torch.Tensor:
        """
        Theoretical salinity survival curve: S(sal) = 1 / (1 + exp(-k * (sal - threshold)))

        Args:
            salinity: Salinity in PSU

        Returns:
            Survival factor in [0, 1]
        """
        k = self.config.salinity_k
        threshold = self.config.salinity_threshold

        # Sigmoid survival curve
        survival = torch.sigmoid(k * (salinity - threshold))

        return survival

    def _generate_belehradek_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic temperature-development rate pairs."""
        temp_min, temp_max = self.config.temp_range

        # Generate temperatures with some noise
        temperatures = torch.linspace(temp_min, temp_max, self.config.n_samples)
        temperatures = temperatures + torch.randn_like(temperatures) * 0.5
        temperatures = temperatures.clamp(temp_min, temp_max)

        # Compute theoretical development rates
        dev_rates = self._belehradek_theoretical(temperatures)

        return temperatures, dev_rates

    def _generate_salinity_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic salinity-survival pairs."""
        sal_min, sal_max = self.config.salinity_range

        # Generate salinities with some noise
        salinities = torch.linspace(sal_min, sal_max, self.config.n_samples)
        salinities = salinities + torch.randn_like(salinities) * 1.0
        salinities = salinities.clamp(sal_min, sal_max)

        # Compute theoretical survival factors
        survivals = self._salinity_theoretical(salinities)

        return salinities, survivals

    def pretrain_belehradek(
        self,
        verbose: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Pre-train BelehradekKAN on theoretical curve.

        Args:
            verbose: Whether to log progress

        Returns:
            Tuple of (state_dict, final_r2)
        """
        if verbose:
            logger.info("Pre-training BelehradekKAN...")

        # Create module
        BelehradekKAN, _ = _get_biology_modules()
        module = BelehradekKAN(n_bases=self.config.n_bases).to(self.device)

        # Generate data
        temperatures, dev_rates = self._generate_belehradek_data()
        temperatures = temperatures.to(self.device)
        dev_rates = dev_rates.to(self.device)

        # Create dataloader
        dataset = TensorDataset(temperatures, dev_rates)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        # Optimizer
        optimizer = torch.optim.AdamW(module.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.n_epochs
        )

        # Training loop
        best_loss = float('inf')
        best_state = None

        for epoch in range(self.config.n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for temp_batch, target_batch in dataloader:
                optimizer.zero_grad()

                # Forward pass
                pred = module(temp_batch)

                # MSE loss
                loss = F.mse_loss(pred, target_batch)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            epoch_loss /= n_batches

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.cpu().clone() for k, v in module.state_dict().items()}

            if verbose and (epoch + 1) % 50 == 0:
                logger.info(f"  Epoch {epoch + 1}/{self.config.n_epochs}: Loss = {epoch_loss:.6f}")

        # Compute R^2 on full range
        module.load_state_dict(best_state)
        module.to(self.device)

        with torch.no_grad():
            test_temps = torch.linspace(
                self.config.temp_range[0],
                self.config.temp_range[1],
                200
            ).to(self.device)

            pred = module(test_temps)
            target = self._belehradek_theoretical(test_temps)

            ss_res = ((pred - target) ** 2).sum()
            ss_tot = ((target - target.mean()) ** 2).sum()
            r2 = 1 - (ss_res / ss_tot)
            r2 = r2.item()

        if verbose:
            logger.info(f"  BelehradekKAN pre-training complete: R^2 = {r2:.4f}")

        return best_state, r2

    def pretrain_salinity(
        self,
        verbose: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Pre-train SalinityMortalityKAN on theoretical curve.

        Args:
            verbose: Whether to log progress

        Returns:
            Tuple of (state_dict, final_r2)
        """
        if verbose:
            logger.info("Pre-training SalinityMortalityKAN...")

        # Create module
        _, SalinityMortalityKAN = _get_biology_modules()
        module = SalinityMortalityKAN(n_bases=self.config.n_bases).to(self.device)

        # Generate data
        salinities, survivals = self._generate_salinity_data()
        salinities = salinities.to(self.device)
        survivals = survivals.to(self.device)

        # Create dataloader
        dataset = TensorDataset(salinities, survivals)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        # Optimizer
        optimizer = torch.optim.AdamW(module.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.n_epochs
        )

        # Training loop
        best_loss = float('inf')
        best_state = None

        for epoch in range(self.config.n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for sal_batch, target_batch in dataloader:
                optimizer.zero_grad()

                # Forward pass
                pred = module(sal_batch)

                # MSE loss
                loss = F.mse_loss(pred, target_batch)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            epoch_loss /= n_batches

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.cpu().clone() for k, v in module.state_dict().items()}

            if verbose and (epoch + 1) % 50 == 0:
                logger.info(f"  Epoch {epoch + 1}/{self.config.n_epochs}: Loss = {epoch_loss:.6f}")

        # Compute R^2 on full range
        module.load_state_dict(best_state)
        module.to(self.device)

        with torch.no_grad():
            test_sals = torch.linspace(
                self.config.salinity_range[0],
                self.config.salinity_range[1],
                200
            ).to(self.device)

            pred = module(test_sals)
            target = self._salinity_theoretical(test_sals)

            ss_res = ((pred - target) ** 2).sum()
            ss_tot = ((target - target.mean()) ** 2).sum()
            r2 = 1 - (ss_res / ss_tot)
            r2 = r2.item()

        if verbose:
            logger.info(f"  SalinityMortalityKAN pre-training complete: R^2 = {r2:.4f}")

        return best_state, r2

    def pretrain_all(
        self,
        verbose: bool = True,
        save: bool = True,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Pre-train all biological modules.

        Args:
            verbose: Whether to log progress
            save: Whether to save checkpoint

        Returns:
            Dict with 'belehradek' and 'salinity_survival' state_dicts
        """
        logger.info("=" * 60)
        logger.info("PRE-TRAINING BIOLOGICAL MODULES")
        logger.info("=" * 60)
        logger.info(f"Theoretical curves:")
        logger.info(f"  Belehradek: D(T) = {self.config.belehradek_a} * (T - {self.config.belehradek_alpha})^{self.config.belehradek_b}")
        logger.info(f"  Salinity: S(sal) = 1 / (1 + exp(-{self.config.salinity_k} * (sal - {self.config.salinity_threshold})))")
        logger.info("")

        # Pre-train each module
        belehradek_state, belehradek_r2 = self.pretrain_belehradek(verbose=verbose)
        salinity_state, salinity_r2 = self.pretrain_salinity(verbose=verbose)

        # Package results
        pretrained = {
            'belehradek': belehradek_state,
            'salinity_survival': salinity_state,
            'config': {
                'belehradek_a': self.config.belehradek_a,
                'belehradek_alpha': self.config.belehradek_alpha,
                'belehradek_b': self.config.belehradek_b,
                'salinity_k': self.config.salinity_k,
                'salinity_threshold': self.config.salinity_threshold,
                'n_bases': self.config.n_bases,
            },
            'metrics': {
                'belehradek_r2': belehradek_r2,
                'salinity_r2': salinity_r2,
            }
        }

        # Save checkpoint
        if save:
            checkpoint_path = Path(self.config.checkpoint_dir) / self.config.checkpoint_name
            torch.save(pretrained, checkpoint_path)
            logger.info(f"\nSaved pre-trained weights to: {checkpoint_path}")

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("PRE-TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"BelehradekKAN R^2:        {belehradek_r2:.4f} {'[PASS]' if belehradek_r2 > 0.95 else '[WARN]'}")
        logger.info(f"SalinityMortalityKAN R^2: {salinity_r2:.4f} {'[PASS]' if salinity_r2 > 0.95 else '[WARN]'}")

        if belehradek_r2 > 0.95 and salinity_r2 > 0.95:
            logger.info("\nAll biological modules successfully pre-trained!")
        else:
            logger.warning("\nSome modules have R^2 < 0.95. Consider increasing epochs or adjusting parameters.")

        return pretrained

    @staticmethod
    def load_pretrained(
        checkpoint_path: str,
        device: str = "cpu",
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Load pre-trained weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load tensors to

        Returns:
            Dict with pretrained state_dicts
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        logger.info(f"Loaded pre-trained biology from: {checkpoint_path}")

        if 'metrics' in checkpoint:
            logger.info(f"  BelehradekKAN R^2: {checkpoint['metrics']['belehradek_r2']:.4f}")
            logger.info(f"  SalinityKAN R^2: {checkpoint['metrics']['salinity_r2']:.4f}")

        return checkpoint


def get_biology_param_names() -> list:
    """
    Return parameter name patterns for biological modules.

    Used by trainer to identify which parameters should get reduced learning rate.
    """
    return [
        'belehradek.',
        'salinity_survival.',
        'dynamics_cell.temperature_development.',
        'dynamics_cell.salinity_survival.',
    ]
