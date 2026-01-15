"""
TimeGAN for Rare Event (Outbreak) Augmentation

Problem: Barentswatch data is 95% "low lice" (zeros and small numbers).
The rare outbreaks (>0.5 lice) are exactly what we need to predict, but
they're too rare for the model to learn effectively.

Solution: Train a TimeGAN to generate synthetic "nightmare scenarios" -
realistic outbreak trajectories that the model can pre-train on.

Key Innovation: Outbreak-Conditioned Generation
Instead of generating random time series, we condition the GAN to generate
specifically the outbreak patterns we care about.

Two Approaches:
1. OutbreakTimeGAN: Pure neural approach (learns from data)
2. SpatialOutbreakSimulator: Physics-informed approach that respects:
   - Hydrodynamic drivers (temperature → fecundity → eggs)
   - Spatial graph structure (infection spreads via currents)
   - Realistic environmental correlations

Architecture (Yoon et al., 2019 "Time-series GAN"):
1. Embedding Network: Real data -> Latent space
2. Recovery Network: Latent space -> Data space
3. Generator: Noise -> Latent sequences
4. Supervisor: Captures temporal dynamics
5. Discriminator: Real vs Fake classification

Training Phases:
1. Autoencoder: Train embedding + recovery (reconstruction)
2. Supervisor: Train temporal dynamics in latent space
3. Joint: Train generator + discriminator adversarially

Usage:
    # Option 1: Pure TimeGAN
    timegan = OutbreakTimeGAN(feature_dim=16, hidden_dim=64)
    timegan.fit(real_data, outbreak_labels)
    synthetic = timegan.generate_outbreaks(n_samples=10000)

    # Option 2: Spatially-aware simulator (RECOMMENDED)
    simulator = SpatialOutbreakSimulator(n_farms=100)
    simulator.fit(X_real, edge_index, env_data)
    synthetic = simulator.generate_outbreak_scenarios(
        n_scenarios=1000,
        edge_index=edge_index,
        env_data=warm_summer_conditions,  # Use real hydrodynamics
    )

Reference:
- Yoon et al. (2019): "Time-series Generative Adversarial Networks"
- https://github.com/jsyoon0823/TimeGAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class TimeGANConfig:
    """Configuration for TimeGAN training."""
    # Architecture
    feature_dim: int = 16          # Input feature dimension
    hidden_dim: int = 64           # Hidden layer dimension
    latent_dim: int = 32           # Latent space dimension
    n_layers: int = 2              # Number of RNN layers

    # Training
    batch_size: int = 32
    epochs_autoencoder: int = 100  # Phase 1: Autoencoder
    epochs_supervisor: int = 100   # Phase 2: Supervisor
    epochs_joint: int = 200        # Phase 3: Joint training

    # Optimization
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999

    # Outbreak conditioning
    outbreak_threshold: float = 0.5  # Lice count threshold for "outbreak"
    outbreak_weight: float = 3.0     # Weight for outbreak samples in training

    # Generation
    noise_dim: int = 32              # Dimension of input noise


class EmbeddingNetwork(nn.Module):
    """
    Maps real time series data to latent space.

    Real data (T, F) -> Latent representation (T, H)
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_layers: int = 2,
    ):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid(),  # Bounded latent space
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Real data, shape (B, T, F)
        Returns:
            h: Latent representation, shape (B, T, H)
        """
        rnn_out, _ = self.rnn(x)
        h = self.fc(rnn_out)
        return h


class RecoveryNetwork(nn.Module):
    """
    Maps latent representation back to data space.

    Latent (T, H) -> Reconstructed data (T, F)
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        feature_dim: int,
        n_layers: int = 2,
    ):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Latent representation, shape (B, T, H)
        Returns:
            x_tilde: Reconstructed data, shape (B, T, F)
        """
        rnn_out, _ = self.rnn(h)
        x_tilde = self.fc(rnn_out)
        return x_tilde


class Generator(nn.Module):
    """
    Generates latent sequences from random noise.

    Noise (T, Z) -> Fake latent (T, H)

    For outbreak conditioning, we concatenate a condition vector.
    """

    def __init__(
        self,
        noise_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_layers: int = 2,
        condition_dim: int = 1,  # Outbreak intensity
    ):
        super().__init__()

        self.noise_dim = noise_dim
        self.condition_dim = condition_dim

        self.rnn = nn.GRU(
            input_size=noise_dim + condition_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z: Random noise, shape (B, T, Z)
            condition: Outbreak intensity, shape (B, 1) or (B, T, 1)
        Returns:
            h_fake: Generated latent sequence, shape (B, T, H)
        """
        B, T, _ = z.shape

        if condition is not None:
            # Expand condition to match sequence length
            if condition.dim() == 2:
                condition = condition.unsqueeze(1).expand(B, T, -1)
            z = torch.cat([z, condition], dim=-1)
        else:
            # Default: zero condition (normal, not outbreak)
            zeros = torch.zeros(B, T, self.condition_dim, device=z.device)
            z = torch.cat([z, zeros], dim=-1)

        rnn_out, _ = self.rnn(z)
        h_fake = self.fc(rnn_out)
        return h_fake


class Supervisor(nn.Module):
    """
    Captures temporal dynamics in latent space.

    Given h_t, predicts h_{t+1}.
    This enforces that the generator learns realistic temporal patterns.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
    ):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Latent sequence, shape (B, T, H)
        Returns:
            h_supervised: Predicted next-step latent, shape (B, T, H)
        """
        rnn_out, _ = self.rnn(h)
        h_supervised = self.fc(rnn_out)
        return h_supervised


class Discriminator(nn.Module):
    """
    Distinguishes real vs fake latent sequences.

    Latent (T, H) -> Probability of real
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
    ):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Latent sequence, shape (B, T, H)
        Returns:
            y: Probability of real, shape (B, T, 1)
        """
        rnn_out, _ = self.rnn(h)
        y = self.fc(rnn_out)
        return y


class OutbreakTimeGAN(nn.Module):
    """
    TimeGAN with Outbreak Conditioning for Sea Lice Simulation.

    Generates realistic synthetic time series with controllable outbreak intensity.

    Key Features:
    1. Learns temporal dynamics from real Barentswatch data
    2. Can generate "nightmare scenarios" with many outbreaks
    3. Produces realistic environmental correlations (temp, salinity, etc.)
    4. Useful for pre-training forecasters on rare events

    Usage:
        # Initialize
        timegan = OutbreakTimeGAN(feature_dim=16, seq_len=52)

        # Train on historical data (with outbreak labels)
        timegan.fit(X_train, outbreak_labels=y_train > 0.5)

        # Generate nightmare scenarios
        X_synthetic = timegan.generate_outbreaks(
            n_samples=10000,
            outbreak_intensity=0.8,  # High outbreak probability
        )

        # Pre-train your forecaster
        forecaster.pretrain(X_synthetic)
        forecaster.finetune(X_real)

    Args:
        feature_dim: Number of input features
        hidden_dim: Hidden layer dimension
        latent_dim: Latent space dimension
        seq_len: Sequence length (e.g., 52 weeks)
        n_layers: Number of RNN layers
        config: Optional full configuration
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        seq_len: int = 52,
        n_layers: int = 2,
        config: Optional[TimeGANConfig] = None,
    ):
        super().__init__()

        self.config = config or TimeGANConfig(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_layers=n_layers,
        )

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        # Core networks
        self.embedder = EmbeddingNetwork(
            feature_dim, hidden_dim, latent_dim, n_layers
        )
        self.recovery = RecoveryNetwork(
            latent_dim, hidden_dim, feature_dim, n_layers
        )
        self.generator = Generator(
            self.config.noise_dim, hidden_dim, latent_dim, n_layers
        )
        self.supervisor = Supervisor(
            latent_dim, hidden_dim, n_layers
        )
        self.discriminator = Discriminator(
            latent_dim, hidden_dim, n_layers
        )

        # Training state
        self.is_trained = False
        self.training_history: Dict[str, List[float]] = {
            'recon_loss': [],
            'sup_loss': [],
            'g_loss': [],
            'd_loss': [],
        }

    def _reconstruction_loss(
        self,
        x: torch.Tensor,
        x_tilde: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Weighted MSE reconstruction loss."""
        mse = (x - x_tilde) ** 2
        if weights is not None:
            mse = mse * weights.unsqueeze(-1).unsqueeze(-1)
        return mse.mean()

    def _supervised_loss(
        self,
        h: torch.Tensor,
        h_supervised: torch.Tensor,
    ) -> torch.Tensor:
        """Supervisor prediction loss."""
        # Compare h[t+1] with supervisor(h[t])
        h_target = h[:, 1:, :]
        h_pred = h_supervised[:, :-1, :]
        return F.mse_loss(h_pred, h_target)

    def _compute_outbreak_weights(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
        weight: float = 3.0,
    ) -> torch.Tensor:
        """
        Compute sample weights to emphasize outbreaks.

        Samples containing outbreak events get higher weight.
        """
        # Assume first feature is adult female lice count
        lice_counts = x[:, :, 0]  # (B, T)

        # Check if any timestep exceeds threshold
        has_outbreak = (lice_counts > threshold).any(dim=1).float()  # (B,)

        # Weight: 1 for normal, `weight` for outbreak
        weights = 1.0 + (weight - 1.0) * has_outbreak

        return weights

    def fit(
        self,
        X: torch.Tensor,
        outbreak_labels: Optional[torch.Tensor] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train TimeGAN on real data.

        Three-phase training:
        1. Autoencoder: Learn embedding + recovery
        2. Supervisor: Learn temporal dynamics
        3. Joint: Adversarial training

        Args:
            X: Real time series data, shape (N, T, F)
            outbreak_labels: Optional binary outbreak indicators
            device: Training device
            verbose: Show progress bars

        Returns:
            Training history dict
        """
        self.to(device)
        X = X.to(device)

        N, T, F = X.shape
        cfg = self.config

        # Compute outbreak weights
        weights = self._compute_outbreak_weights(
            X, cfg.outbreak_threshold, cfg.outbreak_weight
        )

        # Create data loader
        dataset = torch.utils.data.TensorDataset(X, weights)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=True
        )

        # ===================================================================
        # Phase 1: Autoencoder Training
        # ===================================================================
        if verbose:
            print("Phase 1: Training Autoencoder...")

        opt_ae = torch.optim.Adam(
            list(self.embedder.parameters()) + list(self.recovery.parameters()),
            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2)
        )

        for epoch in tqdm(range(cfg.epochs_autoencoder), disable=not verbose):
            epoch_loss = 0.0
            for batch_x, batch_w in loader:
                opt_ae.zero_grad()

                # Forward
                h = self.embedder(batch_x)
                x_tilde = self.recovery(h)

                # Loss (weighted by outbreak)
                loss = self._reconstruction_loss(batch_x, x_tilde, batch_w)

                loss.backward()
                opt_ae.step()

                epoch_loss += loss.item()

            self.training_history['recon_loss'].append(epoch_loss / len(loader))

        # ===================================================================
        # Phase 2: Supervisor Training
        # ===================================================================
        if verbose:
            print("Phase 2: Training Supervisor...")

        opt_sup = torch.optim.Adam(
            self.supervisor.parameters(),
            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2)
        )

        for epoch in tqdm(range(cfg.epochs_supervisor), disable=not verbose):
            epoch_loss = 0.0
            for batch_x, batch_w in loader:
                opt_sup.zero_grad()

                # Get embeddings (frozen)
                with torch.no_grad():
                    h = self.embedder(batch_x)

                # Supervisor prediction
                h_sup = self.supervisor(h)

                # Loss
                loss = self._supervised_loss(h, h_sup)

                loss.backward()
                opt_sup.step()

                epoch_loss += loss.item()

            self.training_history['sup_loss'].append(epoch_loss / len(loader))

        # ===================================================================
        # Phase 3: Joint Training (GAN)
        # ===================================================================
        if verbose:
            print("Phase 3: Joint Adversarial Training...")

        # Generator includes: generator, supervisor, embedder, recovery
        g_params = (
            list(self.generator.parameters()) +
            list(self.supervisor.parameters())
        )
        opt_g = torch.optim.Adam(g_params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2)
        )

        for epoch in tqdm(range(cfg.epochs_joint), disable=not verbose):
            g_loss_epoch = 0.0
            d_loss_epoch = 0.0

            for batch_x, batch_w in loader:
                B = batch_x.shape[0]

                # Generate noise and random outbreak condition
                z = torch.randn(B, T, cfg.noise_dim, device=device)

                # Sample outbreak intensity from distribution emphasizing high values
                # This is key: we WANT to generate outbreaks
                outbreak_intensity = torch.rand(B, 1, device=device) * 0.5 + 0.5  # [0.5, 1.0]

                # ===============================================================
                # Train Discriminator
                # ===============================================================
                opt_d.zero_grad()

                # Real embeddings
                h_real = self.embedder(batch_x)

                # Fake embeddings
                h_fake = self.generator(z, outbreak_intensity)
                h_fake_sup = self.supervisor(h_fake)

                # Discriminator predictions
                y_real = self.discriminator(h_real)
                y_fake = self.discriminator(h_fake_sup)

                # Loss (WGAN-style with soft labels)
                d_loss_real = F.binary_cross_entropy_with_logits(
                    y_real, torch.ones_like(y_real) * 0.9  # Label smoothing
                )
                d_loss_fake = F.binary_cross_entropy_with_logits(
                    y_fake, torch.zeros_like(y_fake) + 0.1
                )
                d_loss = d_loss_real + d_loss_fake

                d_loss.backward()
                opt_d.step()

                # ===============================================================
                # Train Generator
                # ===============================================================
                opt_g.zero_grad()

                # Generate new fakes
                h_fake = self.generator(z, outbreak_intensity)
                h_fake_sup = self.supervisor(h_fake)

                # Adversarial loss (fool discriminator)
                y_fake = self.discriminator(h_fake_sup)
                g_loss_adv = F.binary_cross_entropy_with_logits(
                    y_fake, torch.ones_like(y_fake)
                )

                # Supervised loss (temporal consistency)
                g_loss_sup = self._supervised_loss(h_fake, h_fake_sup)

                # Feature matching loss
                h_real = self.embedder(batch_x)
                g_loss_match = F.mse_loss(
                    h_fake.mean(dim=0), h_real.mean(dim=0)
                )

                g_loss = g_loss_adv + 10 * g_loss_sup + g_loss_match

                g_loss.backward()
                opt_g.step()

                g_loss_epoch += g_loss.item()
                d_loss_epoch += d_loss.item()

            self.training_history['g_loss'].append(g_loss_epoch / len(loader))
            self.training_history['d_loss'].append(d_loss_epoch / len(loader))

        self.is_trained = True
        if verbose:
            print("Training complete!")

        return self.training_history

    def generate(
        self,
        n_samples: int,
        seq_len: Optional[int] = None,
        outbreak_intensity: float = 0.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ) -> torch.Tensor:
        """
        Generate synthetic time series.

        Args:
            n_samples: Number of sequences to generate
            seq_len: Sequence length (default: training seq_len)
            outbreak_intensity: Outbreak condition [0, 1]
                0.0 = normal (low lice)
                1.0 = severe outbreak
            device: Generation device

        Returns:
            X_synthetic: Generated data, shape (n_samples, T, F)
        """
        if not self.is_trained:
            raise RuntimeError("TimeGAN must be trained before generation. Call fit() first.")

        self.eval()
        self.to(device)

        T = seq_len or self.seq_len

        with torch.no_grad():
            # Generate noise
            z = torch.randn(n_samples, T, self.config.noise_dim, device=device)

            # Outbreak condition
            condition = torch.full(
                (n_samples, 1), outbreak_intensity, device=device
            )

            # Generate latent sequence
            h_fake = self.generator(z, condition)
            h_fake = self.supervisor(h_fake)

            # Decode to data space
            x_fake = self.recovery(h_fake)

        return x_fake

    def generate_outbreaks(
        self,
        n_samples: int,
        seq_len: Optional[int] = None,
        min_intensity: float = 0.5,
        max_intensity: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ) -> torch.Tensor:
        """
        Generate synthetic outbreak scenarios ("nightmare simulations").

        This is the key method for rare event augmentation.
        Generates sequences conditioned on high outbreak intensity.

        Args:
            n_samples: Number of outbreak sequences
            seq_len: Sequence length
            min_intensity: Minimum outbreak intensity
            max_intensity: Maximum outbreak intensity
            device: Generation device

        Returns:
            X_outbreaks: Synthetic outbreak data, shape (n_samples, T, F)
        """
        if not self.is_trained:
            raise RuntimeError("TimeGAN must be trained before generation. Call fit() first.")

        self.eval()
        self.to(device)

        T = seq_len or self.seq_len

        with torch.no_grad():
            # Generate noise
            z = torch.randn(n_samples, T, self.config.noise_dim, device=device)

            # Random outbreak intensities in [min, max] range
            intensities = torch.rand(n_samples, 1, device=device)
            intensities = min_intensity + (max_intensity - min_intensity) * intensities

            # Generate latent sequence
            h_fake = self.generator(z, intensities)
            h_fake = self.supervisor(h_fake)

            # Decode to data space
            x_fake = self.recovery(h_fake)

            # Post-process: ensure non-negative lice counts
            x_fake[:, :, 0] = F.relu(x_fake[:, :, 0])  # Adult females
            if x_fake.shape[-1] > 1:
                x_fake[:, :, 1] = F.relu(x_fake[:, :, 1])  # Mobile
            if x_fake.shape[-1] > 2:
                x_fake[:, :, 2] = F.relu(x_fake[:, :, 2])  # Attached

        return x_fake

    def generate_mixed_dataset(
        self,
        n_normal: int,
        n_outbreaks: int,
        seq_len: Optional[int] = None,
        outbreak_intensity_range: Tuple[float, float] = (0.6, 1.0),
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mixed dataset with controlled outbreak ratio.

        Useful for creating balanced training data where outbreaks
        are no longer rare.

        Args:
            n_normal: Number of normal (low-lice) sequences
            n_outbreaks: Number of outbreak sequences
            seq_len: Sequence length
            outbreak_intensity_range: (min, max) intensity for outbreaks
            device: Generation device

        Returns:
            X: Generated data, shape (n_normal + n_outbreaks, T, F)
            labels: Binary outbreak labels, shape (n_normal + n_outbreaks,)
        """
        # Generate normal sequences
        X_normal = self.generate(
            n_normal, seq_len,
            outbreak_intensity=0.1,  # Low intensity
            device=device
        )

        # Generate outbreak sequences
        X_outbreaks = self.generate_outbreaks(
            n_outbreaks, seq_len,
            min_intensity=outbreak_intensity_range[0],
            max_intensity=outbreak_intensity_range[1],
            device=device
        )

        # Combine
        X = torch.cat([X_normal, X_outbreaks], dim=0)
        labels = torch.cat([
            torch.zeros(n_normal, device=device),
            torch.ones(n_outbreaks, device=device),
        ])

        # Shuffle
        perm = torch.randperm(X.shape[0])
        X = X[perm]
        labels = labels[perm]

        return X, labels


class OutbreakAugmenter:
    """
    Complete pipeline for rare event augmentation.

    Wraps TimeGAN with utilities for:
    1. Training on historical data
    2. Generating nightmare scenarios
    3. Creating augmented datasets for pre-training

    Usage:
        augmenter = OutbreakAugmenter(feature_dim=16, seq_len=52)

        # Train on real data
        augmenter.fit(X_real)

        # Create pre-training dataset
        X_pretrain, y_pretrain = augmenter.create_pretraining_data(
            outbreak_ratio=0.5,  # 50% outbreaks (vs ~5% in real data)
            total_samples=10000,
        )

        # Pre-train forecaster
        forecaster.fit(X_pretrain, y_pretrain, epochs=50)

        # Fine-tune on real data
        forecaster.fit(X_real, y_real, epochs=100)
    """

    def __init__(
        self,
        feature_dim: int,
        seq_len: int = 52,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        outbreak_threshold: float = 0.5,
    ):
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.outbreak_threshold = outbreak_threshold

        config = TimeGANConfig(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            outbreak_threshold=outbreak_threshold,
        )

        self.timegan = OutbreakTimeGAN(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            seq_len=seq_len,
            config=config,
        )

        # Store real data statistics for validation
        self.real_mean: Optional[torch.Tensor] = None
        self.real_std: Optional[torch.Tensor] = None
        self.real_outbreak_rate: float = 0.0

    def fit(
        self,
        X: torch.Tensor,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train TimeGAN on historical data.

        Args:
            X: Real data, shape (N, T, F)
            device: Training device
            verbose: Show progress

        Returns:
            Training history
        """
        # Store statistics
        self.real_mean = X.mean(dim=(0, 1))
        self.real_std = X.std(dim=(0, 1))

        # Compute outbreak rate
        lice = X[:, :, 0]
        self.real_outbreak_rate = (lice > self.outbreak_threshold).any(dim=1).float().mean().item()

        if verbose:
            print(f"Real data outbreak rate: {self.real_outbreak_rate:.1%}")
            print(f"Training TimeGAN to generate more outbreaks...")

        return self.timegan.fit(X, device=device, verbose=verbose)

    def create_pretraining_data(
        self,
        outbreak_ratio: float = 0.5,
        total_samples: int = 10000,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create pre-training dataset with elevated outbreak rate.

        Args:
            outbreak_ratio: Fraction of samples that are outbreaks
            total_samples: Total number of samples to generate
            device: Generation device

        Returns:
            X: Generated data, shape (total_samples, T, F)
            labels: Outbreak labels, shape (total_samples,)
        """
        n_outbreaks = int(total_samples * outbreak_ratio)
        n_normal = total_samples - n_outbreaks

        print(f"Generating {n_normal} normal + {n_outbreaks} outbreak sequences...")

        X, labels = self.timegan.generate_mixed_dataset(
            n_normal=n_normal,
            n_outbreaks=n_outbreaks,
            device=device,
        )

        # Validate quality
        synthetic_outbreak_rate = labels.float().mean().item()
        print(f"Synthetic outbreak rate: {synthetic_outbreak_rate:.1%}")
        print(f"Amplification factor: {synthetic_outbreak_rate / max(self.real_outbreak_rate, 0.01):.1f}x")

        return X, labels

    def generate_nightmare_scenarios(
        self,
        n_scenarios: int = 1000,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ) -> torch.Tensor:
        """
        Generate worst-case outbreak scenarios for stress testing.

        These are high-intensity outbreaks for:
        1. Testing model robustness
        2. Preparing for extreme events
        3. Calibrating conformal prediction on tails

        Args:
            n_scenarios: Number of nightmare scenarios
            device: Generation device

        Returns:
            X_nightmare: Extreme outbreak data, shape (n_scenarios, T, F)
        """
        return self.timegan.generate_outbreaks(
            n_samples=n_scenarios,
            min_intensity=0.8,  # Very high outbreak intensity
            max_intensity=1.0,
            device=device,
        )

    def evaluate_synthetic_quality(
        self,
        X_synthetic: torch.Tensor,
        X_real: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate quality of synthetic data vs real data.

        Metrics:
        - Distribution similarity (per feature)
        - Temporal autocorrelation preservation
        - Outbreak pattern realism
        """
        metrics = {}

        # Feature-wise statistics
        syn_mean = X_synthetic.mean(dim=(0, 1))
        syn_std = X_synthetic.std(dim=(0, 1))
        real_mean = X_real.mean(dim=(0, 1))
        real_std = X_real.std(dim=(0, 1))

        metrics['mean_abs_error'] = (syn_mean - real_mean).abs().mean().item()
        metrics['std_abs_error'] = (syn_std - real_std).abs().mean().item()

        # Autocorrelation (lag-1)
        def autocorr(x):
            x_centered = x - x.mean(dim=1, keepdim=True)
            return (x_centered[:, :-1] * x_centered[:, 1:]).mean() / (x.var() + 1e-8)

        syn_autocorr = autocorr(X_synthetic[:, :, 0])
        real_autocorr = autocorr(X_real[:, :, 0])
        metrics['autocorr_error'] = abs(syn_autocorr.item() - real_autocorr.item())

        # Outbreak rate
        syn_outbreak = (X_synthetic[:, :, 0] > self.outbreak_threshold).any(dim=1).float().mean()
        real_outbreak = (X_real[:, :, 0] > self.outbreak_threshold).any(dim=1).float().mean()
        metrics['syn_outbreak_rate'] = syn_outbreak.item()
        metrics['real_outbreak_rate'] = real_outbreak.item()

        return metrics


# ============================================================================
# SPATIAL OUTBREAK SIMULATOR
# Physics-informed generation that respects hydrodynamics and graph structure
# ============================================================================

class SpatialOutbreakSimulator(nn.Module):
    """
    Spatially-Aware Outbreak Simulator for Sea Lice.

    Unlike the basic TimeGAN, this simulator RESPECTS:
    1. HYDRODYNAMICS: Temperature drives fecundity (E = N × f(T))
    2. SPATIAL GRAPH: Outbreaks spread via currents between neighboring farms
    3. SEASONAL PATTERNS: Uses real environmental data as conditioning
    4. TREATMENT DYNAMICS: Accounts for intervention effects

    Key Insight: Real outbreaks are CORRELATED across farms due to:
    - Shared water temperature (regional warming events)
    - Larval transport via currents (farm A infects farm B)
    - Synchronized treatment decisions

    Architecture:
    1. Environmental Encoder: Learns T → outbreak risk mapping
    2. Spatial Propagation: Graph message passing for contagion
    3. Stochastic Dynamics: Adds realistic noise to deterministic skeleton
    4. Intervention Module: Models treatment effects

    Usage:
        simulator = SpatialOutbreakSimulator(n_farms=100)

        # Train on historical outbreak patterns
        simulator.fit(X_real, edge_index, edge_attr)

        # Generate nightmare: "What if summer was 3°C warmer?"
        env_nightmare = env_data.clone()
        env_nightmare[:, :, 0] += 3.0  # Warmer temperature

        X_synthetic = simulator.generate_outbreak_scenarios(
            n_scenarios=1000,
            env_data=env_nightmare,
            edge_index=edge_index,
            edge_attr=edge_attr,
            seed_farms=[12, 45, 78],  # Start outbreak here
        )

    Args:
        n_farms: Number of farms in the spatial network
        feature_dim: Dimension of input features
        hidden_dim: Hidden layer dimension
        env_dim: Dimension of environmental features (temp, sal, u, v)
    """

    def __init__(
        self,
        n_farms: int,
        feature_dim: int = 3,  # adult_female, mobile, attached
        hidden_dim: int = 64,
        env_dim: int = 5,  # temp, sal, current_u, current_v, speed
        outbreak_threshold: float = 0.5,
    ):
        super().__init__()

        self.n_farms = n_farms
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.env_dim = env_dim
        self.outbreak_threshold = outbreak_threshold

        # ===================================================================
        # Environmental Encoder: T, salinity, currents → outbreak risk
        # This learns the mapping from conditions to growth rate
        # ===================================================================
        self.env_encoder = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Temperature → fecundity multiplier (learned Belehrádek-like)
        self.fecundity_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # Ensure positive fecundity
        )

        # ===================================================================
        # Spatial Propagation: Graph message passing for infection spread
        # ===================================================================
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # distance, current_u, current_v, temp
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Attention for neighbor influence
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # ===================================================================
        # Dynamics Module: State evolution over time
        # ===================================================================
        self.dynamics_net = nn.GRUCell(
            input_size=hidden_dim + env_dim + feature_dim,
            hidden_size=hidden_dim,
        )

        # Output decoder: hidden → lice counts
        self.output_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Softplus(),  # Non-negative counts
        )

        # ===================================================================
        # Learned parameters for outbreak dynamics
        # ===================================================================
        # Base reproduction rate
        self.log_r0 = nn.Parameter(torch.tensor(0.0))

        # Spatial transmission coefficient
        self.log_beta = nn.Parameter(torch.tensor(-1.0))

        # Stochastic noise level
        self.log_sigma = nn.Parameter(torch.tensor(-2.0))

        # Temperature sensitivity
        self.temp_sensitivity = nn.Parameter(torch.tensor(0.1))

        # Training state
        self.is_fitted = False

    @property
    def r0(self):
        return torch.exp(self.log_r0)

    @property
    def beta(self):
        return torch.exp(self.log_beta)

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    def compute_fecundity(self, temperature: torch.Tensor) -> torch.Tensor:
        """
        Compute temperature-dependent fecundity.

        Implements: f(T) ≈ a × (T - T0)^b
        But learned from data instead of fixed.

        Args:
            temperature: Water temperature (°C), shape (N,) or (B, N)

        Returns:
            fecundity: Eggs per female per day, same shape
        """
        # Normalize temperature
        temp_norm = (temperature - 10.0) / 5.0  # Center at 10°C
        temp_norm = temp_norm.unsqueeze(-1) if temp_norm.dim() == 1 else temp_norm.unsqueeze(-1)

        fecundity = self.fecundity_net(temp_norm).squeeze(-1)

        return fecundity

    def spatial_aggregation(
        self,
        h: torch.Tensor,
        lice: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute infection pressure from neighboring farms via graph.

        The pressure from farm j to farm i depends on:
        1. Lice count at farm j (source strength)
        2. Current direction and speed (transport probability)
        3. Distance between farms

        Args:
            h: Hidden states, shape (N, hidden_dim)
            lice: Current lice counts, shape (N, feature_dim)
            edge_index: Graph edges, shape (2, E)
            edge_attr: Edge attributes [distance, u, v, temp], shape (E, 4)

        Returns:
            pressure: Infection pressure at each farm, shape (N, hidden_dim)
        """
        N = h.shape[0]
        device = h.device

        src, dst = edge_index[0], edge_index[1]

        # Encode edge features
        edge_features = self.edge_encoder(edge_attr)  # (E, hidden_dim)

        # Source farm hidden states
        h_src = h[src]  # (E, hidden_dim)

        # Compute attention weights
        edge_and_source = torch.cat([edge_features, h_src], dim=-1)
        attention_logits = self.attention_net(edge_and_source).squeeze(-1)  # (E,)

        # Softmax attention per destination
        attention_max = torch.zeros(N, device=device)
        attention_max.scatter_reduce_(0, dst, attention_logits, reduce='amax', include_self=False)
        attention_exp = torch.exp(attention_logits - attention_max[dst])
        attention_sum = torch.zeros(N, device=device)
        attention_sum.scatter_add_(0, dst, attention_exp)
        attention_weights = attention_exp / (attention_sum[dst] + 1e-8)

        # Source lice as infection source (adult females produce eggs)
        source_lice = lice[src, 0]  # Adult females only

        # Weighted contribution
        weighted = attention_weights * source_lice * self.beta
        weighted = weighted.unsqueeze(-1) * h_src

        # Aggregate to destination
        pressure = torch.zeros(N, self.hidden_dim, device=device)
        pressure.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.hidden_dim), weighted)

        return pressure

    def step(
        self,
        h: torch.Tensor,
        lice: torch.Tensor,
        env: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        add_noise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single timestep of outbreak dynamics.

        dL/dt = r0 × f(T) × L + β × Σ_j P(j→i) × L_j - treatment - death

        Args:
            h: Hidden state, shape (N, hidden_dim)
            lice: Current lice counts, shape (N, feature_dim)
            env: Environmental features, shape (N, env_dim)
            edge_index: Graph edges
            edge_attr: Edge attributes
            add_noise: Whether to add stochastic noise

        Returns:
            h_new: Updated hidden state
            lice_new: Updated lice counts
        """
        N = h.shape[0]
        device = h.device

        # Extract temperature
        temperature = env[:, 0]  # First feature is temperature

        # Compute fecundity
        fecundity = self.compute_fecundity(temperature)  # (N,)

        # Spatial infection pressure
        pressure = self.spatial_aggregation(h, lice, edge_index, edge_attr)

        # Encode environment
        env_encoded = self.env_encoder(env)  # (N, hidden_dim)

        # Combined input for dynamics
        dynamics_input = torch.cat([
            env_encoded + pressure,  # Environment + spatial pressure
            env,                      # Raw environmental features
            lice,                     # Current lice counts
        ], dim=-1)

        # Update hidden state
        h_new = self.dynamics_net(dynamics_input, h)

        # Decode to lice counts
        lice_base = self.output_decoder(h_new)

        # Apply fecundity modulation (temperature effect)
        lice_modulated = lice_base * (1 + self.temp_sensitivity * (fecundity - 1))

        # Add stochastic noise
        if add_noise:
            noise = torch.randn_like(lice_modulated) * self.sigma * (lice_modulated + 0.01)
            lice_new = F.relu(lice_modulated + noise)
        else:
            lice_new = lice_modulated

        return h_new, lice_new

    def forward(
        self,
        env_sequence: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        initial_lice: Optional[torch.Tensor] = None,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """
        Generate outbreak trajectory given environmental conditions.

        Args:
            env_sequence: Environmental features, shape (T, N, env_dim)
            edge_index: Graph edges, shape (2, E)
            edge_attr: Edge attributes, shape (E, 4)
            initial_lice: Starting lice counts, shape (N, feature_dim)
            add_noise: Whether to add stochastic noise

        Returns:
            lice_trajectory: Simulated lice counts, shape (T, N, feature_dim)
        """
        T, N, _ = env_sequence.shape
        device = env_sequence.device

        # Initialize
        h = torch.zeros(N, self.hidden_dim, device=device)
        if initial_lice is None:
            lice = torch.zeros(N, self.feature_dim, device=device)
        else:
            lice = initial_lice.clone()

        # Simulate
        trajectory = []
        for t in range(T):
            env_t = env_sequence[t]
            h, lice = self.step(h, lice, env_t, edge_index, edge_attr, add_noise)
            trajectory.append(lice)

        return torch.stack(trajectory, dim=0)

    def fit(
        self,
        X: torch.Tensor,
        env_data: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        epochs: int = 100,
        lr: float = 1e-3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train simulator on real outbreak data.

        Args:
            X: Real lice trajectories, shape (N_samples, T, N_farms, feature_dim)
                or (T, N_farms, feature_dim) for single trajectory
            env_data: Environmental data, shape matching X
            edge_index: Graph edges
            edge_attr: Edge attributes
            mask: Observation mask
            epochs: Number of training epochs
            lr: Learning rate
            device: Training device
            verbose: Show progress

        Returns:
            Training history
        """
        self.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)

        # Handle single vs batch trajectories
        if X.dim() == 3:
            X = X.unsqueeze(0)
            env_data = env_data.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        X = X.to(device)
        env_data = env_data.to(device)
        if mask is not None:
            mask = mask.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history = {'loss': []}

        for epoch in tqdm(range(epochs), disable=not verbose):
            epoch_loss = 0.0
            n_batches = X.shape[0]

            for i in range(n_batches):
                optimizer.zero_grad()

                # Get trajectory
                x_true = X[i]  # (T, N, F)
                env = env_data[i]  # (T, N, env_dim)
                sample_mask = mask[i] if mask is not None else None

                # Initial condition from data
                initial = x_true[0]

                # Simulate
                x_pred = self.forward(
                    env, edge_index, edge_attr,
                    initial_lice=initial,
                    add_noise=False,  # Deterministic during training
                )

                # Loss
                if sample_mask is not None:
                    diff = (x_pred - x_true) ** 2
                    loss = (diff * sample_mask.unsqueeze(-1)).sum() / (sample_mask.sum() + 1e-8)
                else:
                    loss = F.mse_loss(x_pred, x_true)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            history['loss'].append(epoch_loss / n_batches)

        self.is_fitted = True
        return history

    def generate_outbreak_scenarios(
        self,
        n_scenarios: int,
        env_data: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        seed_farms: Optional[List[int]] = None,
        initial_intensity: float = 0.3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ) -> torch.Tensor:
        """
        Generate synthetic outbreak scenarios.

        IMPORTANT: Uses REAL environmental data as conditioning!
        The hydrodynamics drive the outbreak, not random noise.

        Args:
            n_scenarios: Number of scenarios to generate
            env_data: Real environmental conditions, shape (T, N, env_dim)
            edge_index: Graph edges
            edge_attr: Edge attributes
            seed_farms: Which farms to seed outbreak (None = random)
            initial_intensity: Initial lice level at seed farms
            device: Generation device

        Returns:
            scenarios: Synthetic outbreak trajectories, shape (n_scenarios, T, N, F)
        """
        self.eval()
        self.to(device)

        T, N, _ = env_data.shape
        env_data = env_data.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)

        scenarios = []

        for i in range(n_scenarios):
            # Create initial condition with outbreak seeds
            initial = torch.zeros(N, self.feature_dim, device=device)

            if seed_farms is None:
                # Random seed farms (1-5 farms)
                n_seeds = torch.randint(1, min(6, N), (1,)).item()
                seeds = torch.randperm(N)[:n_seeds].tolist()
            else:
                seeds = seed_farms

            # Initialize seeds with outbreak levels
            for farm_idx in seeds:
                # Random intensity around initial_intensity
                intensity = initial_intensity * (0.5 + torch.rand(1).item())
                initial[farm_idx, 0] = intensity  # Adult females
                initial[farm_idx, 1] = intensity * 0.5  # Mobile
                initial[farm_idx, 2] = intensity * 0.3  # Attached

            # Add environmental variation (nightmare mode)
            env_varied = env_data.clone()
            # Random temperature perturbation (simulate warm year)
            temp_perturbation = torch.randn(1, device=device) * 2.0 + 1.0  # +1°C on average
            env_varied[:, :, 0] = env_varied[:, :, 0] + temp_perturbation

            # Generate trajectory with noise
            with torch.no_grad():
                trajectory = self.forward(
                    env_varied, edge_index, edge_attr,
                    initial_lice=initial,
                    add_noise=True,
                )

            scenarios.append(trajectory)

        return torch.stack(scenarios, dim=0)

    def generate_worst_case(
        self,
        env_data: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        temperature_increase: float = 3.0,
        n_seed_farms: int = 5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ) -> torch.Tensor:
        """
        Generate absolute worst-case scenario.

        "What if temperature was 3°C higher and 5 farms got infected?"

        Args:
            env_data: Base environmental data
            edge_index: Graph structure
            edge_attr: Edge attributes
            temperature_increase: How much warmer (°C)
            n_seed_farms: Number of initial outbreak farms
            device: Device

        Returns:
            worst_case: Single worst-case trajectory, shape (T, N, F)
        """
        self.eval()
        self.to(device)

        T, N, _ = env_data.shape
        env_data = env_data.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)

        # Create nightmare conditions
        env_nightmare = env_data.clone()
        env_nightmare[:, :, 0] += temperature_increase  # Warmer

        # Seed highly connected farms (hubs)
        # Compute node degrees
        degrees = torch.zeros(N, device=device)
        degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], device=device))

        # Select highest degree farms as seeds
        _, top_farms = torch.topk(degrees, n_seed_farms)
        seed_farms = top_farms.tolist()

        # Initial outbreak
        initial = torch.zeros(N, self.feature_dim, device=device)
        for farm in seed_farms:
            initial[farm] = torch.tensor([0.5, 0.3, 0.2], device=device)

        # Generate
        with torch.no_grad():
            worst_case = self.forward(
                env_nightmare, edge_index, edge_attr,
                initial_lice=initial,
                add_noise=True,
            )

        return worst_case


class SpatialAugmenter:
    """
    Complete pipeline for spatially-aware outbreak augmentation.

    Combines:
    1. SpatialOutbreakSimulator for physics-informed generation
    2. Real environmental data as conditioning
    3. Graph structure for realistic spatial spread

    Usage:
        augmenter = SpatialAugmenter()

        # Train on real data
        augmenter.fit(X_real, env_real, edge_index, edge_attr)

        # Generate nightmare scenarios using real summer conditions
        summer_env = env_real[summer_mask]  # Use actual warm periods
        X_synthetic = augmenter.generate_pretraining_data(
            n_scenarios=1000,
            env_data=summer_env,
            outbreak_ratio=0.5,
        )
    """

    def __init__(
        self,
        n_farms: int,
        feature_dim: int = 3,
        hidden_dim: int = 64,
        env_dim: int = 5,
        outbreak_threshold: float = 0.5,
    ):
        self.simulator = SpatialOutbreakSimulator(
            n_farms=n_farms,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            env_dim=env_dim,
            outbreak_threshold=outbreak_threshold,
        )
        self.outbreak_threshold = outbreak_threshold

    def fit(
        self,
        X: torch.Tensor,
        env_data: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        **kwargs,
    ):
        """Train on real data."""
        return self.simulator.fit(X, env_data, edge_index, edge_attr, **kwargs)

    def generate_pretraining_data(
        self,
        n_scenarios: int,
        env_data: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        outbreak_ratio: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate pre-training dataset with controlled outbreak ratio.

        Returns:
            X_synthetic: Synthetic lice data, shape (n_scenarios, T, N, F)
            env_synthetic: Corresponding environmental data
            labels: Binary outbreak labels per scenario
        """
        n_outbreaks = int(n_scenarios * outbreak_ratio)
        n_normal = n_scenarios - n_outbreaks

        # Generate outbreak scenarios (warm conditions, seeded outbreaks)
        X_outbreaks = self.simulator.generate_outbreak_scenarios(
            n_scenarios=n_outbreaks,
            env_data=env_data,
            edge_index=edge_index,
            edge_attr=edge_attr,
            initial_intensity=0.3,
            device=device,
        )

        # Generate normal scenarios (same conditions but no seeds)
        X_normal = []
        for _ in range(n_normal):
            with torch.no_grad():
                traj = self.simulator.forward(
                    env_data.to(device),
                    edge_index.to(device),
                    edge_attr.to(device),
                    initial_lice=torch.zeros(env_data.shape[1], self.simulator.feature_dim, device=device),
                    add_noise=True,
                )
            X_normal.append(traj)
        X_normal = torch.stack(X_normal, dim=0)

        # Combine
        X_synthetic = torch.cat([X_normal, X_outbreaks], dim=0)

        # Labels based on actual outbreak occurrence
        labels = (X_synthetic[:, :, :, 0] > self.outbreak_threshold).any(dim=(1, 2)).float()

        # Shuffle
        perm = torch.randperm(X_synthetic.shape[0])
        X_synthetic = X_synthetic[perm]
        labels = labels[perm]

        return X_synthetic, labels


# ============================================================================
# HYBRID SPATIAL OUTBREAK SIMULATOR
# Uses static edge geometry + dynamic flux computation from currents
# ============================================================================

class HybridSpatialOutbreakSimulator(nn.Module):
    """
    Physics-Informed Outbreak Simulator with HYBRID edge attributes.

    KEY INNOVATION: Separates static vs dynamic edge properties
    - STATIC (precomputed): distance, direction vectors between farms
    - DYNAMIC (per-timestep): flux computed from current node features

    This captures the time-varying nature of larval transport:
    - Summer: Strong currents → fast outbreak spread
    - Winter: Weak currents → slow/no spread
    - Counterfactual: "What if currents were 20% stronger?"

    Physics:
        flux(A→B, t) = current_A(t) · direction(A→B)

        If flux > 0: water flows from A toward B (larvae transport possible)
        If flux < 0: water flows away from B (no transport this direction)

    Architecture:
        1. Static edge features loaded once from graph
        2. Dynamic edge_attr computed each timestep from node currents
        3. GNN-style message passing with flux-weighted attention
        4. Liquid neural network for temporal dynamics

    Usage:
        # Initialize with static graph features
        simulator = HybridSpatialOutbreakSimulator(
            n_farms=1777,
            edge_index=graph['edge_index'],
            edge_distance=graph['edge_distance'],
            edge_direction=graph['edge_direction'],
            feature_indices=feature_indices,
        )

        # Train on real data (X contains node features with currents)
        simulator.fit(Y_lice, X_features, edge_index)

        # Generate scenarios - flux computed dynamically from X
        X_synthetic = simulator.generate_outbreak_scenarios(
            n_scenarios=1000,
            env_data=X_features,  # Contains current_u, current_v
        )

    Args:
        n_farms: Number of farms
        edge_index: (2, E) graph edges
        edge_distance: (E,) static distances in km
        edge_direction: (E, 2) static unit direction vectors
        feature_indices: Dict mapping feature names to indices in X
        feature_dim: Lice output dimension (default 3: AF, mobile, attached)
        hidden_dim: Hidden layer size
        decay_km: Distance decay for larval survival
    """

    def __init__(
        self,
        n_farms: int,
        edge_index: torch.Tensor,
        edge_distance: torch.Tensor,
        edge_direction: torch.Tensor,
        feature_indices: Dict[str, int],
        feature_dim: int = 3,
        hidden_dim: int = 64,
        decay_km: float = 15.0,
        outbreak_threshold: float = 0.5,
    ):
        super().__init__()

        self.n_farms = n_farms
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.decay_km = decay_km
        self.outbreak_threshold = outbreak_threshold
        self.feature_indices = feature_indices

        # Register static graph features as buffers (not parameters)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_distance', edge_distance)
        self.register_buffer('edge_direction', edge_direction)

        # Number of edges
        self.n_edges = edge_index.shape[1]

        # ===================================================================
        # Edge encoder: takes dynamic edge_attr [dist, flux, temp, sal]
        # ===================================================================
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Attention for flux-weighted neighbor influence
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # ===================================================================
        # Environmental encoder
        # ===================================================================
        # Determine env_dim from feature_indices
        env_features = ['temperature', 'salinity', 'current_u', 'current_v', 'current_speed']
        self.env_indices = [feature_indices.get(f, i) for i, f in enumerate(env_features)]
        env_dim = len(self.env_indices)

        self.env_encoder = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Temperature → fecundity (learned Belehradek-like)
        self.fecundity_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

        # ===================================================================
        # Dynamics: GRU cell for temporal evolution
        # ===================================================================
        self.dynamics_net = nn.GRUCell(
            input_size=hidden_dim + env_dim + feature_dim,
            hidden_size=hidden_dim,
        )

        # Output decoder
        self.output_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Softplus(),
        )

        # ===================================================================
        # Learned epidemic parameters
        # ===================================================================
        self.log_beta = nn.Parameter(torch.tensor(-1.0))  # Transmission rate
        self.log_sigma = nn.Parameter(torch.tensor(-2.0))  # Noise level
        self.temp_sensitivity = nn.Parameter(torch.tensor(0.1))

        self.is_fitted = False

    @property
    def beta(self):
        return torch.exp(self.log_beta)

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    def compute_dynamic_edge_attr(
        self,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute edge attributes dynamically from current node features.

        Args:
            node_features: (N, F) node features at current timestep

        Returns:
            edge_attr: (E, 4) with [distance, flux, temp_src, sal_src]
        """
        device = node_features.device
        src = self.edge_index[0]

        # Feature indices
        temp_idx = self.feature_indices.get('temperature', 11)
        sal_idx = self.feature_indices.get('salinity', 12)
        u_idx = self.feature_indices.get('current_u', 13)
        v_idx = self.feature_indices.get('current_v', 14)

        # Get currents at source farms
        current_u = node_features[src, u_idx]  # (E,)
        current_v = node_features[src, v_idx]  # (E,)

        # Compute flux: dot product of current and direction
        flux = (current_u * self.edge_direction[:, 0] +
                current_v * self.edge_direction[:, 1])

        # Positive flux only (water flowing toward destination)
        flux_positive = torch.clamp(flux, min=0)

        # Distance decay
        distance_factor = torch.exp(-self.edge_distance / self.decay_km)
        weighted_flux = flux_positive * distance_factor

        # Environmental at source
        temp_src = node_features[src, temp_idx]
        sal_src = node_features[src, sal_idx]

        # Stack into edge_attr
        edge_attr = torch.stack([
            self.edge_distance,
            weighted_flux,
            temp_src,
            sal_src,
        ], dim=1)

        return edge_attr

    def spatial_aggregation(
        self,
        h: torch.Tensor,
        lice: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute infection pressure from neighbors with flux-weighted attention.

        Args:
            h: Hidden states (N, hidden_dim)
            lice: Current lice counts (N, feature_dim)
            edge_attr: Dynamic edge attributes (E, 4)

        Returns:
            pressure: Infection pressure (N, hidden_dim)
        """
        N = h.shape[0]
        device = h.device

        src, dst = self.edge_index[0], self.edge_index[1]

        # Encode edge features (includes flux!)
        edge_features = self.edge_encoder(edge_attr)  # (E, hidden_dim)

        # Source hidden states
        h_src = h[src]  # (E, hidden_dim)

        # Attention: combine edge features and source state
        edge_and_source = torch.cat([edge_features, h_src], dim=-1)
        attention_logits = self.attention_net(edge_and_source).squeeze(-1)

        # Weight by flux (edge_attr[:, 1] is the weighted flux)
        flux_weight = edge_attr[:, 1]  # (E,)
        attention_logits = attention_logits + torch.log(flux_weight + 1e-8)

        # Softmax per destination
        attention_max = torch.zeros(N, device=device)
        attention_max.scatter_reduce_(0, dst, attention_logits, reduce='amax', include_self=False)
        attention_exp = torch.exp(attention_logits - attention_max[dst])
        attention_sum = torch.zeros(N, device=device)
        attention_sum.scatter_add_(0, dst, attention_exp)
        attention_weights = attention_exp / (attention_sum[dst] + 1e-8)

        # Source lice (adult females)
        source_lice = lice[src, 0]

        # Weighted message
        weighted = attention_weights * source_lice * self.beta
        weighted = weighted.unsqueeze(-1) * h_src

        # Aggregate to destination
        pressure = torch.zeros(N, self.hidden_dim, device=device)
        pressure.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.hidden_dim), weighted)

        return pressure

    def step(
        self,
        h: torch.Tensor,
        lice: torch.Tensor,
        node_features: torch.Tensor,
        add_noise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single timestep with dynamic edge_attr computation.

        Args:
            h: Hidden state (N, hidden_dim)
            lice: Current lice counts (N, feature_dim)
            node_features: Full node features (N, F)
            add_noise: Whether to add stochastic noise

        Returns:
            h_new, lice_new
        """
        N = h.shape[0]

        # Compute dynamic edge attributes from current conditions
        edge_attr = self.compute_dynamic_edge_attr(node_features)

        # Extract environmental features
        env = node_features[:, self.env_indices]

        # Temperature for fecundity
        temp_idx = self.feature_indices.get('temperature', 11)
        temperature = node_features[:, temp_idx]
        temp_norm = (temperature - 10.0) / 5.0
        fecundity = self.fecundity_net(temp_norm.unsqueeze(-1)).squeeze(-1)

        # Spatial infection pressure
        pressure = self.spatial_aggregation(h, lice, edge_attr)

        # Encode environment
        env_encoded = self.env_encoder(env)

        # Dynamics input
        dynamics_input = torch.cat([
            env_encoded + pressure,
            env,
            lice,
        ], dim=-1)

        # Update hidden state
        h_new = self.dynamics_net(dynamics_input, h)

        # Decode to lice counts
        lice_base = self.output_decoder(h_new)

        # Temperature modulation (expand fecundity to match feature_dim)
        fecundity_expanded = fecundity.unsqueeze(-1)  # (N,) -> (N, 1)
        lice_modulated = lice_base * (1 + self.temp_sensitivity * (fecundity_expanded - 1))

        # Stochastic noise
        if add_noise:
            noise = torch.randn_like(lice_modulated) * self.sigma * (lice_modulated + 0.01)
            lice_new = F.relu(lice_modulated + noise)
        else:
            lice_new = lice_modulated

        return h_new, lice_new

    def forward(
        self,
        node_features_sequence: torch.Tensor,
        initial_lice: Optional[torch.Tensor] = None,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """
        Generate outbreak trajectory with dynamic flux computation.

        Args:
            node_features_sequence: (T, N, F) full node features over time
            initial_lice: (N, feature_dim) starting lice counts
            add_noise: Whether to add stochastic noise

        Returns:
            lice_trajectory: (T, N, feature_dim)
        """
        T, N, F = node_features_sequence.shape
        device = node_features_sequence.device

        # Initialize
        h = torch.zeros(N, self.hidden_dim, device=device)
        if initial_lice is None:
            lice = torch.zeros(N, self.feature_dim, device=device)
        else:
            lice = initial_lice.clone()

        trajectory = []
        for t in range(T):
            node_features_t = node_features_sequence[t]
            h, lice = self.step(h, lice, node_features_t, add_noise)
            trajectory.append(lice)

        return torch.stack(trajectory, dim=0)

    def fit(
        self,
        Y: torch.Tensor,
        X: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        epochs: int = 100,
        lr: float = 1e-3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train on real outbreak data.

        Args:
            Y: Target lice counts (T, N, feature_dim) or (B, T, N, feature_dim)
            X: Node features (T, N, F) or (B, T, N, F)
            mask: Observation mask
            epochs: Training epochs
            lr: Learning rate
            device: Device
            verbose: Show progress

        Returns:
            Training history
        """
        self.to(device)

        # Handle batched input
        if Y.dim() == 3:
            Y = Y.unsqueeze(0)
            X = X.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        Y = Y.to(device)
        X = X.to(device)
        if mask is not None:
            mask = mask.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history = {'loss': []}

        for epoch in tqdm(range(epochs), disable=not verbose):
            epoch_loss = 0.0
            n_batches = Y.shape[0]

            for i in range(n_batches):
                optimizer.zero_grad()

                y_true = Y[i]  # (T, N, F)
                x_seq = X[i]   # (T, N, features)
                sample_mask = mask[i] if mask is not None else None

                # Initial condition
                initial = y_true[0]

                # Forward (flux computed dynamically!)
                y_pred = self.forward(x_seq, initial_lice=initial, add_noise=False)

                # Loss
                if sample_mask is not None:
                    diff = (y_pred - y_true) ** 2
                    loss = (diff * sample_mask.unsqueeze(-1)).sum() / (sample_mask.sum() + 1e-8)
                else:
                    loss = F.mse_loss(y_pred, y_true)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            history['loss'].append(epoch_loss / n_batches)

        self.is_fitted = True
        return history

    def generate_outbreak_scenarios(
        self,
        n_scenarios: int,
        X: torch.Tensor,
        seed_farms: Optional[List[int]] = None,
        initial_intensity: float = 0.3,
        temperature_perturbation: float = 0.0,
        current_scale: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ) -> torch.Tensor:
        """
        Generate synthetic outbreak scenarios with controllable conditions.

        KEY: Uses node features X which contains currents, so flux is computed
        dynamically. Can perturb temperature/currents for counterfactual analysis.

        Args:
            n_scenarios: Number of scenarios to generate
            X: Node features (T, N, F) with currents for flux computation
            seed_farms: Which farms to seed (None = random)
            initial_intensity: Starting lice level at seed farms
            temperature_perturbation: Add this to temperature (nightmare mode)
            current_scale: Multiply currents by this (stronger = faster spread)
            device: Device

        Returns:
            scenarios: (n_scenarios, T, N, feature_dim)
        """
        self.eval()
        self.to(device)

        T, N, F = X.shape
        X = X.to(device)

        # Get feature indices for perturbation
        temp_idx = self.feature_indices.get('temperature', 11)
        u_idx = self.feature_indices.get('current_u', 13)
        v_idx = self.feature_indices.get('current_v', 14)

        scenarios = []

        for i in range(n_scenarios):
            # Perturb conditions
            X_perturbed = X.clone()

            # Temperature perturbation (nightmare: warmer = more fecundity)
            if temperature_perturbation != 0:
                X_perturbed[:, :, temp_idx] += temperature_perturbation

            # Current scaling (nightmare: stronger currents = faster spread)
            if current_scale != 1.0:
                X_perturbed[:, :, u_idx] *= current_scale
                X_perturbed[:, :, v_idx] *= current_scale

            # Random temperature variation per scenario
            temp_noise = torch.randn(1, device=device) * 1.0
            X_perturbed[:, :, temp_idx] += temp_noise

            # Initial lice conditions
            initial = torch.zeros(N, self.feature_dim, device=device)

            if seed_farms is None:
                n_seeds = torch.randint(1, min(6, N), (1,)).item()
                seeds = torch.randperm(N)[:n_seeds].tolist()
            else:
                seeds = seed_farms

            for farm_idx in seeds:
                intensity = initial_intensity * (0.5 + torch.rand(1).item())
                initial[farm_idx, 0] = intensity  # Adult females
                initial[farm_idx, 1] = intensity * 0.5  # Mobile
                initial[farm_idx, 2] = intensity * 0.3  # Attached

            # Generate trajectory
            with torch.no_grad():
                trajectory = self.forward(
                    X_perturbed,
                    initial_lice=initial,
                    add_noise=True,
                )

            scenarios.append(trajectory)

        return torch.stack(scenarios, dim=0)

    def generate_nightmare_scenario(
        self,
        X: torch.Tensor,
        temperature_increase: float = 3.0,
        current_boost: float = 1.5,
        n_seed_farms: int = 5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ) -> torch.Tensor:
        """
        Generate absolute worst-case: hot summer + strong currents + hub seeding.

        Args:
            X: Node features for flux computation
            temperature_increase: Degrees warmer than normal
            current_boost: Current multiplier (1.5 = 50% stronger)
            n_seed_farms: Number of initial outbreak farms (highest degree)
            device: Device

        Returns:
            worst_case: (T, N, feature_dim)
        """
        self.eval()
        self.to(device)

        T, N, F = X.shape
        X = X.to(device)

        # Find highest-degree farms (hubs) as seeds
        src = self.edge_index[0].cpu()
        degrees = torch.zeros(N)
        for s in src:
            degrees[s] += 1
        _, top_farms = torch.topk(degrees, n_seed_farms)
        seed_farms = top_farms.tolist()

        # Generate with nightmare conditions
        scenarios = self.generate_outbreak_scenarios(
            n_scenarios=1,
            X=X,
            seed_farms=seed_farms,
            initial_intensity=0.5,  # Higher initial
            temperature_perturbation=temperature_increase,
            current_scale=current_boost,
            device=device,
        )

        return scenarios[0]  # Return single scenario


class HybridOutbreakAugmenter:
    """
    Complete pipeline for physics-informed outbreak augmentation.

    Wraps HybridSpatialOutbreakSimulator with utilities for:
    1. Training on historical data
    2. Generating nightmare scenarios with dynamic flux
    3. Creating balanced pre-training datasets

    Usage:
        # Initialize with graph
        augmenter = HybridOutbreakAugmenter.from_graph(
            graph_path='data/processed/spatial_graph.pt',
            feature_indices=feature_indices,
        )

        # Train on real data
        augmenter.fit(Y_lice, X_features)

        # Generate pre-training data (50% outbreaks)
        X_pretrain, Y_pretrain = augmenter.create_pretraining_data(
            X_real=X_features,
            outbreak_ratio=0.5,
            total_samples=5000,
        )
    """

    def __init__(
        self,
        simulator: HybridSpatialOutbreakSimulator,
        outbreak_threshold: float = 0.5,
    ):
        self.simulator = simulator
        self.outbreak_threshold = outbreak_threshold

    @classmethod
    def from_graph(
        cls,
        graph_path: str,
        feature_indices: Dict[str, int],
        n_farms: Optional[int] = None,
        **kwargs,
    ) -> 'HybridOutbreakAugmenter':
        """
        Create augmenter from saved graph file.

        Args:
            graph_path: Path to spatial_graph.pt with edge_direction, edge_distance
            feature_indices: Dict mapping feature names to indices
            n_farms: Number of farms (inferred from graph if None)
            **kwargs: Additional args for simulator
        """
        graph = torch.load(graph_path, map_location='cpu', weights_only=False)

        # Check for required static features
        if 'edge_direction' not in graph:
            raise ValueError(
                "Graph missing 'edge_direction'. Run add_static_edge_features_to_graph() first."
            )

        n_farms = n_farms or graph['n_nodes']

        simulator = HybridSpatialOutbreakSimulator(
            n_farms=n_farms,
            edge_index=graph['edge_index'],
            edge_distance=graph['edge_distance'],
            edge_direction=graph['edge_direction'],
            feature_indices=feature_indices,
            **kwargs,
        )

        return cls(simulator)

    def fit(
        self,
        Y: torch.Tensor,
        X: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, List[float]]:
        """Train simulator on historical data."""
        return self.simulator.fit(Y, X, mask, **kwargs)

    def create_pretraining_data(
        self,
        X_real: torch.Tensor,
        outbreak_ratio: float = 0.5,
        total_samples: int = 5000,
        temperature_range: Tuple[float, float] = (0.0, 3.0),
        current_range: Tuple[float, float] = (0.8, 1.5),
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create pre-training dataset with controlled outbreak ratio.

        Args:
            X_real: Real node features (T, N, F) for flux computation
            outbreak_ratio: Fraction of samples that are outbreaks
            total_samples: Total samples to generate
            temperature_range: (min, max) temperature perturbation for outbreaks
            current_range: (min, max) current scaling for outbreaks
            device: Device

        Returns:
            X_synthetic: Generated lice data (total_samples, T, N, feature_dim)
            labels: Binary outbreak labels
        """
        n_outbreaks = int(total_samples * outbreak_ratio)
        n_normal = total_samples - n_outbreaks

        print(f"Generating {n_normal} normal + {n_outbreaks} outbreak scenarios...")
        print(f"Using dynamic flux from node features")

        # Generate outbreak scenarios (perturbed conditions)
        outbreak_scenarios = []
        for i in tqdm(range(n_outbreaks), desc="Generating outbreaks"):
            # Random perturbation within range
            temp_perturb = temperature_range[0] + torch.rand(1).item() * (
                temperature_range[1] - temperature_range[0]
            )
            current_scale = current_range[0] + torch.rand(1).item() * (
                current_range[1] - current_range[0]
            )

            with torch.no_grad():
                scenario = self.simulator.generate_outbreak_scenarios(
                    n_scenarios=1,
                    X=X_real,
                    initial_intensity=0.3,
                    temperature_perturbation=temp_perturb,
                    current_scale=current_scale,
                    device=device,
                )
            outbreak_scenarios.append(scenario[0])

        # Generate normal scenarios (no perturbation, no seeding)
        normal_scenarios = []
        for i in tqdm(range(n_normal), desc="Generating normal"):
            with torch.no_grad():
                initial = torch.zeros(
                    X_real.shape[1], self.simulator.feature_dim, device=device
                )
                # Small random initial lice (realistic baseline)
                initial[:, 0] = torch.rand(X_real.shape[1], device=device) * 0.1

                scenario = self.simulator.forward(
                    X_real.to(device),
                    initial_lice=initial,
                    add_noise=True,
                )
            normal_scenarios.append(scenario)

        # Combine
        X_outbreaks = torch.stack(outbreak_scenarios, dim=0)
        X_normal = torch.stack(normal_scenarios, dim=0)
        X_synthetic = torch.cat([X_normal, X_outbreaks], dim=0)

        # Labels based on actual outbreak occurrence
        labels = (X_synthetic[:, :, :, 0] > self.outbreak_threshold).any(dim=(1, 2)).float()

        # Shuffle
        perm = torch.randperm(X_synthetic.shape[0])
        X_synthetic = X_synthetic[perm]
        labels = labels[perm]

        # Stats
        actual_outbreak_rate = labels.mean().item()
        print(f"Actual outbreak rate: {actual_outbreak_rate:.1%}")

        return X_synthetic, labels
