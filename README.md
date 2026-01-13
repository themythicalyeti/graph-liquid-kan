# Graph-Liquid-KAN: Sea Lice Outbreak Prediction

A novel deep learning architecture combining **Graph Neural Networks**, **Kolmogorov-Arnold Networks (KAN)**, and **Liquid Time-Constant Networks** for predicting sea lice outbreaks in Norwegian salmon farms.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GRAPH-LIQUID-KAN                          │
├─────────────────────────────────────────────────────────────┤
│  Input: Environmental Features (T, N, F)                     │
│         - Temperature, Salinity, Currents                    │
│         - Treatment indicators                               │
├─────────────────────────────────────────────────────────────┤
│  FastKAN Encoder                                             │
│  └─ Gaussian RBF basis functions                             │
│  └─ Learnable non-linearities                                │
├─────────────────────────────────────────────────────────────┤
│  GraphonAggregator (per timestep)                            │
│  └─ 1/N normalized message passing                           │
│  └─ Scale-invariant (N→∞ convergence)                        │
├─────────────────────────────────────────────────────────────┤
│  LiquidKANCell (CfC dynamics)                                │
│  └─ τ(T) = adaptive time constant                            │
│  └─ h(t+dt) = decay·h + (1-decay)·x_eq·gate                  │
├─────────────────────────────────────────────────────────────┤
│  Output: Lice Counts (T, N, 3)                               │
│          - Adult female, Mobile, Stationary                  │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

- **FastKAN Layers**: Gaussian RBF basis functions replacing traditional MLPs
- **Graphon-compliant Aggregation**: 1/N normalization for scale invariance
- **Liquid Time-Constants**: Continuous-time dynamics with adaptive τ
- **Physics-Informed Loss**: L_data + λ_bio·L_bio + λ_stability·L_stability

## Target Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Recall | ≥90% | Catch 9/10 outbreaks |
| Precision | ≥80% | 8/10 predictions correct |
| F1 Score | ≥0.85 | Balance P/R |

## Scientific Validation

Three tests ensure physically meaningful predictions:

1. **Counterfactual**: Temperature +5°C should increase lice growth
2. **Long-Horizon**: 90-day rollout should remain stable
3. **Graphon**: N vs 2N nodes should have <10% deviation

## Project Structure

```
v5/
├── src/
│   ├── data/           # Dataset builders and loaders
│   ├── models/         # GLKAN architecture
│   ├── training/       # Training loop, losses, audit
│   └── ingestion/      # Data fetching clients
├── scripts/
│   ├── run_phase1.py   # Data acquisition
│   ├── run_phase2.py   # Tensor construction
│   ├── train_glkan.py  # Full training
│   └── run_audit.py    # Scientific validation
├── protocols/          # Development documentation
├── deployment.ipynb    # Google Colab notebook
└── config/             # Configuration files
```

## Quick Start

### Local Training (CPU)

```bash
# Install dependencies
pip install -r requirements.txt

# Quick test (200 nodes, 10 epochs)
python scripts/train_reduced.py

# Run scientific audit
python scripts/run_audit.py
```

### GPU Training (Google Colab)

1. Package data: `python scripts/package_for_colab.py`
2. Upload `colab_package/glkan_data.zip` to Google Drive root
3. Open `deployment.ipynb` in Colab
4. Select GPU runtime (T4/A100)
5. Run all cells

## Data Requirements

- **BarentsWatch API**: Lice counts, farm locations, treatments
- **NorKyst-800**: Ocean temperature, salinity, currents
- Credentials needed: `BARENTSWATCH_CLIENT_ID`, `BARENTSWATCH_CLIENT_SECRET`

## Training Configuration

```python
CONFIG = {
    'hidden_dim': 64,
    'n_bases': 8,        # RBF basis functions
    'n_layers': 2,       # GLKAN layers
    'lr': 1e-4,
    'epochs': 100,
    'lambda_bio': 0.1,   # Physics loss weight
    'patience': 15,      # Early stopping
}
```

## Results

After training on full data (1,777 farms, 73K edges):

| Test | Status |
|------|--------|
| Counterfactual | ✅ PASS |
| Long-Horizon | ✅ PASS |
| Graphon | ✅ PASS |

## References

- **KAN**: Liu et al. "KAN: Kolmogorov-Arnold Networks" (2024)
- **Liquid Networks**: Hasani et al. "Liquid Time-constant Networks" (2021)
- **Graphon Theory**: Lovász "Large Networks and Graph Limits" (2012)

## License

MIT License

## Citation

```bibtex
@software{glkan2024,
  title={Graph-Liquid-KAN: Sea Lice Outbreak Prediction},
  year={2024},
  url={https://github.com/YOUR_USERNAME/graph-liquid-kan}
}
```
