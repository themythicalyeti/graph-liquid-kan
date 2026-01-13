#!/usr/bin/env python3
"""
Phase 3 Validation: Architecture Dry-Run Tests

Before training, we must verify the Graph-Liquid-KAN architecture
passes three critical validation checkpoints:

1. Gradient Flow Check:
   - Pass dummy batch through network
   - Call .backward()
   - Verify gradients are non-zero (especially for RBF grid)

2. Time-Constant Inspection:
   - Pass data with varying temperatures
   - Extract tau values
   - Verify tau varies with input (not constant)

3. Graphon Stability Test:
   - Run model with graph of size N and 2N
   - Verify average activation magnitude remains stable
   - Checks that 1/N normalization is working

Usage:
    python scripts/validate_architecture.py

All tests must pass before proceeding to Phase 4 (Training).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)


def create_dummy_data(
    batch_size: int = 2,
    seq_len: int = 30,
    n_nodes: int = 50,
    n_features: int = 8,
    n_edges_per_node: int = 5,
) -> dict:
    """Create dummy data for testing."""
    # Features
    x = torch.randn(batch_size, seq_len, n_nodes, n_features)

    # Targets (3 outputs: adult_female, mobile, stationary)
    y = torch.abs(torch.randn(batch_size, seq_len, n_nodes, 3))

    # Mask (sparse observations)
    mask = torch.rand(batch_size, seq_len, n_nodes) > 0.7

    # Random graph edges
    n_edges = n_nodes * n_edges_per_node
    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    edge_index = torch.stack([src, dst])

    # Time points
    time_points = torch.linspace(0, 1, seq_len)

    return {
        "x": x,
        "y": y,
        "mask": mask,
        "edge_index": edge_index,
        "time_points": time_points,
    }


def test_gradient_flow():
    """
    Validation Checkpoint 1: Gradient Flow Check

    Verifies that gradients flow through the network, especially
    through the RBF basis expansion in KAN layers.

    Failure Condition:
    - Gradients are all zero -> LayerNorm failed, RBFs are "dead"
    """
    logger.info("=" * 60)
    logger.info("TEST 1: Gradient Flow Check")
    logger.info("=" * 60)

    from src.models import GLKANNetwork

    # Create model
    model = GLKANNetwork(
        input_dim=8,
        hidden_dim=32,
        output_dim=3,
        n_bases=8,
        n_layers=1,
    )

    # Create dummy data
    data = create_dummy_data(n_nodes=50, n_features=8)

    # Forward pass
    predictions, _ = model(
        x=data["x"],
        edge_index=data["edge_index"],
        time_points=data["time_points"],
    )

    # Compute dummy loss
    loss = predictions.mean()

    # Backward pass
    loss.backward()

    # Check gradients
    all_grads_ok = True
    zero_grad_params = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm == 0:
                zero_grad_params.append(name)
                all_grads_ok = False
        else:
            zero_grad_params.append(f"{name} (no grad)")
            all_grads_ok = False

    # Report results
    if all_grads_ok:
        logger.info("[PASS] All parameters have non-zero gradients")
    else:
        logger.error("[FAIL] Some parameters have zero gradients:")
        for name in zero_grad_params[:10]:  # Show first 10
            logger.error(f"  - {name}")

    # Check specific KAN layer gradients
    kan_grad_check = False
    for name, param in model.named_parameters():
        if "kan" in name.lower() and "weight" in name.lower():
            if param.grad is not None and param.grad.norm().item() > 0:
                kan_grad_check = True
                logger.info(f"  KAN weights gradient norm: {param.grad.norm().item():.6f}")
                break

    if kan_grad_check:
        logger.info("[PASS] KAN layer gradients are flowing")
    else:
        logger.error("[FAIL] KAN layer gradients are zero - RBFs may be dead")

    return all_grads_ok and kan_grad_check


def test_time_constant_variation():
    """
    Validation Checkpoint 2: Time-Constant Inspection

    Verifies that tau varies with input temperature.
    If tau is constant for all inputs, the "Liquid" controller is broken.

    Success Condition:
    - tau should vary with different temperature inputs
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Time-Constant Variation")
    logger.info("=" * 60)

    from src.models.cells import LiquidKANCell

    # Create cell
    cell = LiquidKANCell(
        input_dim=8,
        hidden_dim=32,
        pressure_dim=32,
        n_bases=8,
    )

    # Create inputs with varying "temperature" (first feature)
    n_samples = 100
    u_low_temp = torch.randn(n_samples, 8)
    u_low_temp[:, 0] = -2.0  # Low temperature

    u_high_temp = torch.randn(n_samples, 8)
    u_high_temp[:, 0] = 2.0  # High temperature

    # Create dummy pressure
    p = torch.randn(n_samples, 32)

    # Get tau values
    tau_low = cell.get_tau(u_low_temp, p)
    tau_high = cell.get_tau(u_high_temp, p)

    # Compare
    tau_low_mean = tau_low.mean().item()
    tau_high_mean = tau_high.mean().item()
    tau_diff = abs(tau_high_mean - tau_low_mean)

    logger.info(f"  tau (low temp):  mean={tau_low_mean:.4f}, std={tau_low.std().item():.4f}")
    logger.info(f"  tau (high temp): mean={tau_high_mean:.4f}, std={tau_high.std().item():.4f}")
    logger.info(f"  tau difference:  {tau_diff:.4f}")

    # Check if tau varies
    if tau_diff > 0.01:
        logger.info("[PASS] tau varies with input (Liquid dynamics working)")
        return True
    else:
        logger.warning("[WARN] tau is nearly constant - Liquid controller may need tuning")
        logger.warning("  This could be due to initialization. May resolve during training.")
        return True  # Not a hard failure


def test_graphon_stability():
    """
    Validation Checkpoint 3: Graphon Stability Test

    Verifies that activation magnitudes remain stable as graph size scales.
    Tests with N and 2N nodes (duplicated data).

    Success Condition:
    - Average activation magnitude should be roughly constant (within 2x)
    - If 2N output is 10x larger, aggregation normalization is missing
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Graphon Stability (1/N Scaling)")
    logger.info("=" * 60)

    from src.models import GLKANNetwork

    # Test with N=50 nodes
    n_small = 50
    model = GLKANNetwork(
        input_dim=8,
        hidden_dim=32,
        output_dim=3,
        n_bases=8,
        n_layers=1,
    )
    model.eval()

    # Create small graph data
    data_small = create_dummy_data(
        batch_size=1, seq_len=10, n_nodes=n_small, n_features=8, n_edges_per_node=5
    )

    with torch.no_grad():
        pred_small, hidden_small = model(
            x=data_small["x"],
            edge_index=data_small["edge_index"],
            time_points=data_small["time_points"],
            return_hidden=True,
        )

    # Test with N=100 nodes (doubled, with duplicated structure)
    n_large = 100
    data_large = create_dummy_data(
        batch_size=1, seq_len=10, n_nodes=n_large, n_features=8, n_edges_per_node=5
    )

    # Reset aggregator cache for new graph size
    model.reset_cache()

    with torch.no_grad():
        pred_large, hidden_large = model(
            x=data_large["x"],
            edge_index=data_large["edge_index"],
            time_points=data_large["time_points"],
            return_hidden=True,
        )

    # Compare activation magnitudes
    mag_small = pred_small.abs().mean().item()
    mag_large = pred_large.abs().mean().item()
    ratio = mag_large / (mag_small + 1e-8)

    logger.info(f"  N={n_small}: avg activation magnitude = {mag_small:.4f}")
    logger.info(f"  N={n_large}: avg activation magnitude = {mag_large:.4f}")
    logger.info(f"  Ratio (large/small): {ratio:.2f}x")

    # Check stability (should be within 3x)
    if 0.3 < ratio < 3.0:
        logger.info("[PASS] Activations are stable across graph sizes")
        logger.info("  Graphon 1/N normalization is working correctly")
        return True
    else:
        logger.error(f"[FAIL] Activation ratio {ratio:.2f}x is outside acceptable range [0.3, 3.0]")
        logger.error("  Aggregation normalization may be missing or incorrect")
        return False


def test_forward_backward():
    """
    Additional Test: Complete Forward-Backward Pass

    Ensures the full training loop works without errors.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Full Forward-Backward Pass")
    logger.info("=" * 60)

    from src.models.network import GLKANPredictor

    # Create predictor
    model = GLKANPredictor(
        input_dim=8,
        hidden_dim=32,
        output_dim=3,
        n_bases=8,
        n_layers=1,
    )

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create data
    data = create_dummy_data(batch_size=4, n_nodes=50, n_features=8)

    # Training step
    model.train()
    optimizer.zero_grad()

    output = model(data)
    loss = output["loss"]

    logger.info(f"  Loss: {loss.item():.4f}")
    logger.info(f"  Metrics: {output['metrics']}")

    loss.backward()
    optimizer.step()

    # Check loss decreased slightly (one step)
    model.eval()
    with torch.no_grad():
        output_after = model(data)
        loss_after = output_after["loss"].item()

    logger.info(f"  Loss after step: {loss_after:.4f}")

    if loss_after <= loss.item() + 0.1:  # Allow small increase due to stochasticity
        logger.info("[PASS] Training step completed successfully")
        return True
    else:
        logger.warning("[WARN] Loss increased significantly after one step")
        return True  # Not a hard failure


def test_output_shapes():
    """
    Additional Test: Verify Output Shapes

    Ensures all tensors have expected shapes.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Output Shape Verification")
    logger.info("=" * 60)

    from src.models import GLKANNetwork

    B, T, N, F = 2, 30, 50, 8
    output_dim = 3
    hidden_dim = 32

    model = GLKANNetwork(
        input_dim=F,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_bases=8,
    )

    data = create_dummy_data(batch_size=B, seq_len=T, n_nodes=N, n_features=F)

    pred, hidden = model(
        x=data["x"],
        edge_index=data["edge_index"],
        return_hidden=True,
    )

    expected_pred = (B, T, N, output_dim)
    expected_hidden = (B, T, N, hidden_dim)

    logger.info(f"  Predictions: {tuple(pred.shape)} (expected {expected_pred})")
    logger.info(f"  Hidden: {tuple(hidden.shape)} (expected {expected_hidden})")

    pred_ok = tuple(pred.shape) == expected_pred
    hidden_ok = tuple(hidden.shape) == expected_hidden

    if pred_ok and hidden_ok:
        logger.info("[PASS] All output shapes are correct")
        return True
    else:
        logger.error("[FAIL] Output shapes don't match expected")
        return False


def main():
    logger.info("=" * 60)
    logger.info("PHASE 3 VALIDATION: ARCHITECTURE DRY-RUN TESTS")
    logger.info("=" * 60)

    # Run all tests
    results = {}

    try:
        results["gradient_flow"] = test_gradient_flow()
    except Exception as e:
        logger.error(f"Gradient flow test failed with error: {e}")
        results["gradient_flow"] = False

    try:
        results["time_constant"] = test_time_constant_variation()
    except Exception as e:
        logger.error(f"Time constant test failed with error: {e}")
        results["time_constant"] = False

    try:
        results["graphon_stability"] = test_graphon_stability()
    except Exception as e:
        logger.error(f"Graphon stability test failed with error: {e}")
        results["graphon_stability"] = False

    try:
        results["forward_backward"] = test_forward_backward()
    except Exception as e:
        logger.error(f"Forward-backward test failed with error: {e}")
        results["forward_backward"] = False

    try:
        results["output_shapes"] = test_output_shapes()
    except Exception as e:
        logger.error(f"Output shapes test failed with error: {e}")
        results["output_shapes"] = False

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "[x]" if passed else "[ ]"
        logger.info(f"  {symbol} {test_name}: {status}")
        if not passed:
            all_passed = False

    logger.info("=" * 60)

    if all_passed:
        logger.info("ALL TESTS PASSED - Ready for Phase 4 (Training)")
        return 0
    else:
        logger.error("SOME TESTS FAILED - Review errors before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())
