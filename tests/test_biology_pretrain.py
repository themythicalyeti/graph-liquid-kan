"""
Test Script for Biological Module Pre-training

This script tests:
1. BiologyPretrainer generates correct synthetic data
2. Pre-trained curves match theoretical R² > 0.95
3. Weights load correctly into all 4 module instances
4. Optimizer has correct LR for each parameter group
"""

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_theoretical_curves():
    """Test that theoretical curve functions are correct."""
    print("\n" + "=" * 60)
    print("TEST: Theoretical Curves")
    print("=" * 60)

    from training.pretrain import BiologyPretrainer, BiologyPretrainConfig

    config = BiologyPretrainConfig()
    pretrainer = BiologyPretrainer(config)

    # Test Belehradek curve: D(T) = 0.05 * (T - 0)^1.5
    temps = torch.tensor([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    expected_rates = 0.05 * torch.pow(F.relu(temps), 1.5)
    actual_rates = pretrainer._belehradek_theoretical(temps)

    print(f"Temperature: {temps.tolist()}")
    print(f"Expected rates: {expected_rates.tolist()}")
    print(f"Actual rates: {actual_rates.tolist()}")

    assert torch.allclose(actual_rates, expected_rates, atol=1e-5), \
        "Belehradek theoretical curve mismatch"
    print("[PASS] Belehradek theoretical curve correct")

    # Test salinity curve: S(sal) = 1 / (1 + exp(-0.3 * (sal - 25)))
    sals = torch.tensor([0.0, 15.0, 25.0, 30.0, 35.0])
    expected_survival = torch.sigmoid(0.3 * (sals - 25.0))
    actual_survival = pretrainer._salinity_theoretical(sals)

    print(f"\nSalinity: {sals.tolist()}")
    print(f"Expected survival: {expected_survival.tolist()}")
    print(f"Actual survival: {actual_survival.tolist()}")

    assert torch.allclose(actual_survival, expected_survival, atol=1e-5), \
        "Salinity theoretical curve mismatch"
    print("[PASS] Salinity theoretical curve correct")


def test_pretrain_quality():
    """Test that pre-trained modules match theory with R² > 0.95."""
    print("\n" + "=" * 60)
    print("TEST: Pre-training Quality (R² > 0.95)")
    print("=" * 60)

    from training.pretrain import BiologyPretrainer, BiologyPretrainConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        config = BiologyPretrainConfig(
            n_epochs=200,
            checkpoint_dir=tmpdir,
        )
        pretrainer = BiologyPretrainer(config)

        # Pre-train both modules
        pretrained = pretrainer.pretrain_all(verbose=True, save=True)

        # Check R² values
        belehradek_r2 = pretrained['metrics']['belehradek_r2']
        salinity_r2 = pretrained['metrics']['salinity_r2']

        print(f"\nBelehradek R²: {belehradek_r2:.4f}")
        print(f"Salinity R²: {salinity_r2:.4f}")

        assert belehradek_r2 > 0.95, f"Belehradek R² = {belehradek_r2:.4f} < 0.95"
        assert salinity_r2 > 0.95, f"Salinity R² = {salinity_r2:.4f} < 0.95"

        print("[PASS] Both modules have R² > 0.95")


def test_weight_loading():
    """Test that weights load correctly into all 4 module instances."""
    print("\n" + "=" * 60)
    print("TEST: Weight Loading to All 4 Instances")
    print("=" * 60)

    from training.pretrain import BiologyPretrainer, BiologyPretrainConfig
    from models.sea_lice_network import SeaLiceGLKAN

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and pre-train
        config = BiologyPretrainConfig(
            n_epochs=100,  # Fewer epochs for faster test
            checkpoint_dir=tmpdir,
        )
        pretrainer = BiologyPretrainer(config)
        pretrained = pretrainer.pretrain_all(verbose=False, save=False)

        # Create model
        model = SeaLiceGLKAN(
            input_dim=15,
            hidden_dim=32,
            output_dim=3,
            n_bases=8,
        )

        # Load pre-trained weights
        model.load_pretrained_biology(pretrained)

        # Verify all 4 instances have same weights
        bel1_state = model.belehradek.state_dict()
        bel2_state = model.dynamics_cell.temperature_development.state_dict()

        sal1_state = model.salinity_survival.state_dict()
        sal2_state = model.dynamics_cell.salinity_survival.state_dict()

        # Check Belehradek instances match
        for key in bel1_state:
            if key in bel2_state:
                match = torch.allclose(bel1_state[key], bel2_state[key], atol=1e-6)
                print(f"Belehradek '{key}': {'MATCH' if match else 'MISMATCH'}")
                assert match, f"Belehradek weights mismatch for key: {key}"

        # Check salinity instances match
        for key in sal1_state:
            if key in sal2_state:
                match = torch.allclose(sal1_state[key], sal2_state[key], atol=1e-6)
                print(f"Salinity '{key}': {'MATCH' if match else 'MISMATCH'}")
                assert match, f"Salinity weights mismatch for key: {key}"

        print("\n[PASS] All 4 module instances have matching weights")


def test_differential_lr():
    """Test that optimizer has correct LR for each parameter group."""
    print("\n" + "=" * 60)
    print("TEST: Differential Learning Rates")
    print("=" * 60)

    import tempfile
    from training.trainer import GLKANTrainer, TrainingConfig
    from training.pretrain import get_biology_param_names
    from models.sea_lice_network import SeaLicePredictor

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create model
        model = SeaLicePredictor(
            input_dim=15,
            hidden_dim=32,
            output_dim=3,
            n_bases=8,
        )

        # Create dummy data loader
        from torch.utils.data import DataLoader, TensorDataset

        # Dummy batch
        x = torch.randn(4, 10, 5, 15)  # (B, T, N, F)
        y = torch.randn(4, 10, 5, 3)
        mask = torch.ones(4, 10, 5)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])

        def collate_fn(batch):
            return {
                'x': x,
                'y': y,
                'mask': mask,
                'edge_index': edge_index,
            }

        dummy_dataset = TensorDataset(x)
        dummy_loader = DataLoader(dummy_dataset, batch_size=1, collate_fn=collate_fn)

        # Create trainer
        config = TrainingConfig(
            learning_rate=0.01,
            bio_lr_scale=0.5,
            pretrain_biology=True,
            checkpoint_dir=tmpdir,
            n_epochs=1,
        )

        trainer = GLKANTrainer(
            model=model,
            train_loader=dummy_loader,
            config=config,
        )

        # Check parameter groups
        print(f"Number of parameter groups: {len(trainer.optimizer.param_groups)}")

        bio_patterns = get_biology_param_names()
        expected_bio_lr = config.learning_rate * config.bio_lr_scale

        for i, group in enumerate(trainer.optimizer.param_groups):
            name = group.get('name', f'group_{i}')
            lr = group['lr']
            n_params = len(group['params'])
            print(f"  Group '{name}': {n_params} params, LR = {lr}")

            if name == 'biology':
                assert abs(lr - expected_bio_lr) < 1e-8, \
                    f"Biology LR = {lr}, expected {expected_bio_lr}"
            elif name == 'other':
                assert abs(lr - config.learning_rate) < 1e-8, \
                    f"Other LR = {lr}, expected {config.learning_rate}"

        print(f"\n[PASS] Differential LR configured correctly")
        print(f"  Main LR: {config.learning_rate}")
        print(f"  Biology LR: {expected_bio_lr} ({config.bio_lr_scale}x)")


def test_load_save_checkpoint():
    """Test that pre-trained weights can be saved and loaded."""
    print("\n" + "=" * 60)
    print("TEST: Save/Load Checkpoint")
    print("=" * 60)

    from training.pretrain import BiologyPretrainer, BiologyPretrainConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and pre-train
        config = BiologyPretrainConfig(
            n_epochs=50,  # Fewer epochs for faster test
            checkpoint_dir=tmpdir,
            checkpoint_name="test_biology.pt",
        )
        pretrainer = BiologyPretrainer(config)
        original = pretrainer.pretrain_all(verbose=False, save=True)

        # Load from checkpoint
        checkpoint_path = Path(tmpdir) / "test_biology.pt"
        loaded = BiologyPretrainer.load_pretrained(str(checkpoint_path))

        # Verify metrics match
        assert 'metrics' in loaded, "Loaded checkpoint missing metrics"
        assert abs(loaded['metrics']['belehradek_r2'] - original['metrics']['belehradek_r2']) < 1e-6
        assert abs(loaded['metrics']['salinity_r2'] - original['metrics']['salinity_r2']) < 1e-6

        # Verify weights match
        for key in original['belehradek']:
            assert torch.allclose(
                loaded['belehradek'][key],
                original['belehradek'][key],
                atol=1e-6
            ), f"Belehradek weight mismatch: {key}"

        print("[PASS] Checkpoint save/load working correctly")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("BIOLOGY PRE-TRAINING TEST SUITE")
    print("=" * 60)

    try:
        test_theoretical_curves()
        test_pretrain_quality()
        test_weight_loading()
        test_differential_lr()
        test_load_save_checkpoint()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n[FAIL] {e}")
        raise
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
