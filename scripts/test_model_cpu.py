#!/usr/bin/env python3
"""
Diagnostic script to test model on CPU with smaller data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="DEBUG")

def main():
    logger.info("=" * 60)
    logger.info("MODEL DIAGNOSTIC TEST")
    logger.info("=" * 60)

    # Step 1: Create small synthetic data
    logger.info("\n1. Creating synthetic test data...")

    B = 2   # batch size
    T = 10  # time steps (small)
    N = 100 # nodes (small)
    F = 8   # features

    x = torch.randn(B, T, N, F)
    y = torch.abs(torch.randn(B, T, N, 3))
    mask = torch.ones(B, T, N, dtype=torch.bool)

    # Create small graph (ring topology)
    src = torch.arange(N)
    dst = (torch.arange(N) + 1) % N
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src])
    ])

    logger.info(f"   x: {x.shape}")
    logger.info(f"   y: {y.shape}")
    logger.info(f"   edge_index: {edge_index.shape}")

    # Step 2: Test model components
    logger.info("\n2. Testing FastKAN layer...")
    from src.models.layers import FastKAN

    kan = FastKAN(in_features=F, out_features=64, n_bases=8)
    x_flat = x.view(-1, F)
    try:
        out = kan(x_flat)
        logger.info(f"   FastKAN output: {out.shape} - OK")
    except Exception as e:
        logger.error(f"   FastKAN failed: {e}")
        return

    logger.info("\n3. Testing GraphonAggregator...")
    from src.models.layers import GraphonAggregator

    agg = GraphonAggregator()
    x_node = torch.randn(N, 64)
    try:
        out = agg(x_node, edge_index)
        logger.info(f"   GraphonAggregator output: {out.shape} - OK")
    except Exception as e:
        logger.error(f"   GraphonAggregator failed: {e}")
        return

    logger.info("\n4. Testing LiquidKANCell...")
    from src.models.cells import LiquidKANCell

    cell = LiquidKANCell(input_dim=64, hidden_dim=64, n_bases=8)
    h = torch.randn(N, 64)
    u = torch.randn(N, 64)
    p = torch.randn(N, 64)
    dt = torch.tensor(0.1)
    try:
        h_new = cell(h, u, p, dt)
        logger.info(f"   LiquidKANCell output: {h_new.shape} - OK")
    except Exception as e:
        logger.error(f"   LiquidKANCell failed: {e}")
        return

    logger.info("\n5. Testing GraphLiquidKANCell...")
    from src.models.cells import GraphLiquidKANCell

    glk_cell = GraphLiquidKANCell(input_dim=64, hidden_dim=64, n_bases=8)
    h = torch.randn(N, 64)
    u = torch.randn(N, 64)
    try:
        h_new = glk_cell(h, u, edge_index, dt)
        logger.info(f"   GraphLiquidKANCell output: {h_new.shape} - OK")
    except Exception as e:
        logger.error(f"   GraphLiquidKANCell failed: {e}")
        return

    logger.info("\n6. Testing GLKANNetwork...")
    from src.models.network import GLKANNetwork

    net = GLKANNetwork(
        input_dim=F,
        hidden_dim=64,
        output_dim=3,
        n_bases=8,
        n_layers=1,
    )
    try:
        pred, hidden = net(x, edge_index)
        logger.info(f"   GLKANNetwork output: {pred.shape} - OK")
    except Exception as e:
        logger.error(f"   GLKANNetwork failed: {e}")
        return

    logger.info("\n7. Testing GLKANPredictor...")
    from src.models.network import GLKANPredictor

    model = GLKANPredictor(
        input_dim=F,
        hidden_dim=64,
        output_dim=3,
        n_bases=8,
        n_layers=1,
    )

    batch = {
        'x': x,
        'y': y,
        'mask': mask,
        'edge_index': edge_index,
    }

    try:
        output = model(batch)
        logger.info(f"   GLKANPredictor output: {output['predictions'].shape} - OK")
    except Exception as e:
        logger.error(f"   GLKANPredictor failed: {e}")
        return

    logger.info("\n8. Testing loss computation...")
    from src.training.losses import GLKANLoss

    criterion = GLKANLoss()
    try:
        loss, metrics = criterion(output, batch)
        logger.info(f"   Loss: {loss.item():.4f} - OK")
    except Exception as e:
        logger.error(f"   Loss computation failed: {e}")
        return

    logger.info("\n9. Testing backward pass...")
    try:
        loss.backward()
        logger.info(f"   Backward pass - OK")
    except Exception as e:
        logger.error(f"   Backward pass failed: {e}")
        return

    logger.info("\n10. Testing overfit loop (100 steps)...")
    model = GLKANPredictor(
        input_dim=F,
        hidden_dim=64,
        output_dim=3,
        n_bases=8,
        n_layers=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    initial_loss = None
    for step in range(100):
        optimizer.zero_grad()
        output = model(batch)
        loss, _ = criterion(output, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        if step % 20 == 0:
            logger.info(f"   Step {step}: Loss = {loss.item():.6f}")

    final_loss = loss.item()
    logger.info(f"\n   Initial loss: {initial_loss:.4f}")
    logger.info(f"   Final loss: {final_loss:.4f}")
    logger.info(f"   Reduction: {100*(1 - final_loss/initial_loss):.1f}%")

    if final_loss < initial_loss * 0.5:
        logger.info("\n[PASS] Model can learn on synthetic data")
    else:
        logger.warning("\n[WARN] Model did not reduce loss significantly")

    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC TEST COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
