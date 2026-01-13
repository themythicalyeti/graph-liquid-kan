"""
Phase 4.4: Scientific Validation "Auditor"

Low loss is not enough. We must validate that the model has learned
the PHYSICS of the system, not just curve fitting.

Validation Tests:
1. Counterfactual Test: Temperature +5°C should increase growth rate
2. Long-Horizon Rollout: 90-day prediction should remain bounded
3. Graphon Generalization: 2x nodes should give similar predictions

These tests ensure the model is scientifically valid, not just statistically fit.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class AuditConfig:
    """Configuration for scientific validation."""
    # Counterfactual test
    temperature_delta: float = 5.0  # Temperature increase in °C
    temperature_feature_idx: int = 0  # Index of temperature in feature tensor

    # Long-horizon test
    rollout_days: int = 90  # 3x typical training window

    # Graphon test
    graphon_tolerance: float = 0.10  # 10% deviation tolerance


class ScientificAuditor:
    """
    Scientific Validation Suite for Graph-Liquid-KAN.

    Ensures the model has learned physically meaningful dynamics,
    not just statistical correlations.

    Tests:
    1. Counterfactual: Does temperature affect growth as expected?
    2. Long-Horizon: Is the system stable over extended rollouts?
    3. Graphon: Does the model generalize to different graph sizes?
    """

    def __init__(self, model: nn.Module, config: Optional[AuditConfig] = None):
        self.model = model
        self.config = config or AuditConfig()
        self.device = next(model.parameters()).device

    def run_all_tests(
        self,
        sample_batch: Dict[str, torch.Tensor],
    ) -> Dict[str, bool]:
        """
        Run all scientific validation tests.

        Args:
            sample_batch: A sample batch for testing

        Returns:
            Dictionary of test results (True = passed)
        """
        logger.info("=" * 60)
        logger.info("SCIENTIFIC VALIDATION AUDIT")
        logger.info("=" * 60)

        results = {}

        # Test 1: Counterfactual
        results['counterfactual'] = self.test_counterfactual(sample_batch)

        # Test 2: Long-Horizon
        results['long_horizon'] = self.test_long_horizon_stability(sample_batch)

        # Test 3: Graphon
        results['graphon'] = self.test_graphon_generalization(sample_batch)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("AUDIT SUMMARY")
        logger.info("=" * 60)

        all_passed = True
        for test_name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            symbol = "[x]" if passed else "[ ]"
            logger.info(f"  {symbol} {test_name}: {status}")
            if not passed:
                all_passed = False

        if all_passed:
            logger.info("\nAll scientific validation tests PASSED")
        else:
            logger.warning("\nSome tests FAILED - review model physics")

        return results

    @torch.no_grad()
    def test_counterfactual(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> bool:
        """
        Counterfactual Test: Temperature effect on lice growth.

        Protocol:
        1. Run model with original data
        2. Artificially increase temperature by +5°C
        3. Run model again
        4. Check: Does lice growth rate increase?

        Pass Criteria:
        - If growth increases with temperature -> biological law learned
        - If growth decreases or stays flat -> model failed

        Args:
            batch: Sample batch of data

        Returns:
            True if test passed
        """
        logger.info("\n[TEST 1] Counterfactual: Temperature Effect")
        logger.info("-" * 40)

        self.model.eval()
        batch = self._to_device(batch)

        # Original prediction
        output_original = self.model(batch)
        pred_original = output_original['predictions']

        # Compute growth rate (mean daily change)
        if pred_original.shape[1] > 1:
            growth_original = (pred_original[:, 1:] - pred_original[:, :-1]).mean()
        else:
            growth_original = torch.tensor(0.0)

        # Modify temperature
        x_modified = batch['x'].clone()
        temp_idx = self.config.temperature_feature_idx
        x_modified[..., temp_idx] += self.config.temperature_delta

        batch_modified = {**batch, 'x': x_modified}

        # Counterfactual prediction
        output_modified = self.model(batch_modified)
        pred_modified = output_modified['predictions']

        # Compute counterfactual growth rate
        if pred_modified.shape[1] > 1:
            growth_modified = (pred_modified[:, 1:] - pred_modified[:, :-1]).mean()
        else:
            growth_modified = torch.tensor(0.0)

        # Log results
        logger.info(f"  Original growth rate: {growth_original.item():.6f}")
        logger.info(f"  +{self.config.temperature_delta}°C growth rate: {growth_modified.item():.6f}")
        logger.info(f"  Difference: {(growth_modified - growth_original).item():.6f}")

        # Check: Higher temperature should increase growth
        # (Sea lice reproduce faster in warmer water)
        passed = growth_modified > growth_original

        if passed:
            logger.info("  [PASS] Temperature increases lice growth (biologically correct)")
        else:
            logger.warning("  [FAIL] Temperature does not increase growth (check model physics)")

        return passed

    @torch.no_grad()
    def test_long_horizon_stability(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> bool:
        """
        Long-Horizon Rollout Test: System stability.

        Protocol:
        1. Feed model initial conditions (first time step)
        2. Run for 90 days (3x training window) using only environmental data
        3. Check: Does prediction explode, crash, or remain bounded?

        Pass Criteria:
        - Trajectory remains bounded (Input-to-State Stability)
        - No explosion to infinity
        - No crash to zero

        Args:
            batch: Sample batch of data

        Returns:
            True if test passed
        """
        logger.info("\n[TEST 2] Long-Horizon Stability")
        logger.info("-" * 40)

        self.model.eval()
        batch = self._to_device(batch)

        # Get dimensions
        x = batch['x']  # (B, T, N, F)
        B, T, N, F = x.shape

        # Extend time series by repeating environmental features
        n_rollout = self.config.rollout_days
        n_repeats = (n_rollout // T) + 2

        # Create extended input (cycle environmental data)
        x_extended = x.repeat(1, n_repeats, 1, 1)[:, :n_rollout]

        # Create extended batch
        batch_extended = {
            'x': x_extended,
            'edge_index': batch['edge_index'],
            'time_points': torch.linspace(0, 1, n_rollout, device=self.device),
        }

        # Add other required fields
        if 'degree' in batch:
            batch_extended['degree'] = batch['degree']

        # Run extended rollout
        try:
            # Use the network directly for extended rollout
            if hasattr(self.model, 'network'):
                pred_extended, _ = self.model.network(
                    x=x_extended,
                    edge_index=batch['edge_index'],
                )
            else:
                pred_extended, _ = self.model(
                    x=x_extended,
                    edge_index=batch['edge_index'],
                )
        except Exception as e:
            logger.error(f"  Rollout failed: {e}")
            return False

        # Analyze stability
        pred_mean = pred_extended.mean(dim=(0, 2, 3))  # Mean over batch, nodes, outputs
        pred_std = pred_extended.std(dim=(0, 2, 3))
        pred_min = pred_extended.min().item()
        pred_max = pred_extended.max().item()

        logger.info(f"  Rollout days: {n_rollout}")
        logger.info(f"  Prediction range: [{pred_min:.4f}, {pred_max:.4f}]")
        logger.info(f"  Final mean: {pred_mean[-1].item():.4f}")
        logger.info(f"  Final std: {pred_std[-1].item():.4f}")

        # Check for explosion (values > 100x initial)
        initial_scale = pred_extended[:, :T].abs().mean().item() + 1e-6
        final_scale = pred_extended[:, -T:].abs().mean().item()
        explosion_ratio = final_scale / initial_scale

        logger.info(f"  Scale ratio (final/initial): {explosion_ratio:.2f}x")

        # Check for NaN/Inf
        has_nan = torch.isnan(pred_extended).any().item()
        has_inf = torch.isinf(pred_extended).any().item()

        if has_nan or has_inf:
            logger.error("  [FAIL] NaN or Inf detected in rollout")
            return False

        # Stability criteria
        is_bounded = explosion_ratio < 100 and pred_max < 1000
        not_collapsed = pred_std[-1].item() > 1e-6  # Still has variance

        passed = is_bounded and not_collapsed

        if passed:
            logger.info("  [PASS] Long-horizon rollout is stable")
        else:
            if not is_bounded:
                logger.warning("  [FAIL] Predictions exploded")
            if not not_collapsed:
                logger.warning("  [FAIL] Predictions collapsed to constant")

        return passed

    @torch.no_grad()
    def test_graphon_generalization(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> bool:
        """
        Graphon Generalization Test: Scale invariance.

        Protocol:
        1. Run model with original graph (N nodes)
        2. Create synthetic graph with 2N nodes (duplicate structure)
        3. Run model on 2N graph
        4. Check: Is average prediction roughly the same?

        Pass Criteria:
        - Deviation < 10% indicates proper 1/N normalization
        - If deviation is high, Graphon normalization is broken

        Args:
            batch: Sample batch of data

        Returns:
            True if test passed
        """
        logger.info("\n[TEST 3] Graphon Generalization (Scale Invariance)")
        logger.info("-" * 40)

        self.model.eval()
        batch = self._to_device(batch)

        # Original prediction
        x = batch['x']  # (B, T, N, F)
        edge_index = batch['edge_index']

        # Run original
        if hasattr(self.model, 'network'):
            pred_original, _ = self.model.network(
                x=x,
                edge_index=edge_index,
            )
        else:
            output_original = self.model(batch)
            pred_original = output_original['predictions']

        mean_original = pred_original.abs().mean().item()

        # Create 2N graph by duplicating
        B, T, N, F = x.shape
        x_2n = x.repeat(1, 1, 2, 1)  # (B, T, 2N, F)

        # Duplicate edge index (offset second copy by N)
        edge_index_2n = torch.cat([
            edge_index,
            edge_index + N,
        ], dim=1)

        # Reset cache if model has aggregator
        if hasattr(self.model, 'reset_cache'):
            self.model.reset_cache()
        if hasattr(self.model, 'network') and hasattr(self.model.network, 'reset_cache'):
            self.model.network.reset_cache()

        # Run on 2N graph
        try:
            if hasattr(self.model, 'network'):
                pred_2n, _ = self.model.network(
                    x=x_2n,
                    edge_index=edge_index_2n,
                )
            else:
                batch_2n = {**batch, 'x': x_2n, 'edge_index': edge_index_2n}
                # Need to expand other tensors too
                if 'y' in batch:
                    batch_2n['y'] = batch['y'].repeat(1, 1, 2, 1)
                if 'mask' in batch:
                    batch_2n['mask'] = batch['mask'].repeat(1, 1, 2)
                output_2n = self.model(batch_2n)
                pred_2n = output_2n['predictions']
        except Exception as e:
            logger.error(f"  2N graph failed: {e}")
            return False

        mean_2n = pred_2n.abs().mean().item()

        # Calculate deviation
        if mean_original > 1e-8:
            deviation = abs(mean_2n - mean_original) / mean_original
        else:
            deviation = abs(mean_2n - mean_original)

        logger.info(f"  Original (N={N}) mean prediction: {mean_original:.6f}")
        logger.info(f"  Doubled (N={2*N}) mean prediction: {mean_2n:.6f}")
        logger.info(f"  Relative deviation: {100*deviation:.2f}%")

        passed = deviation < self.config.graphon_tolerance

        if passed:
            logger.info(f"  [PASS] Scale invariance within {100*self.config.graphon_tolerance:.0f}% tolerance")
        else:
            logger.warning(f"  [FAIL] Deviation {100*deviation:.1f}% exceeds {100*self.config.graphon_tolerance:.0f}% tolerance")
            logger.warning("  Check Graphon normalization (1/N) in aggregator")

        return passed

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to model device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }


def run_full_audit(
    model: nn.Module,
    dataloader,
    config: Optional[AuditConfig] = None,
) -> Dict[str, bool]:
    """
    Convenience function to run full scientific audit.

    Args:
        model: Trained model
        dataloader: Data loader (uses first batch)
        config: Audit configuration

    Returns:
        Dictionary of test results
    """
    auditor = ScientificAuditor(model, config)

    # Get a sample batch
    batch = next(iter(dataloader))

    return auditor.run_all_tests(batch)
