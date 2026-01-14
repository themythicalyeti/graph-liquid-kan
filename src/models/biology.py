"""
Sea Lice-Specific Biological Modules for Graph-Liquid-KAN

This module implements domain-specific components for modeling Lepeophtheirus salmonis
(sea lice) population dynamics on salmon farms, based on the theoretical framework
from the FEATURES_FOR_SEA_LICE.MD document.

Key Biological Components:
1. BelehradekKAN: Temperature-dependent development rate (replaces fixed Belehrádek equation)
2. SalinityMortalityKAN: Salinity-dependent survival factor
3. LarvalTransportModule: Cross-infection via ocean current-driven larval dispersal
4. PopulationDynamicsCell: Combines all factors into a single ODE system

Reference Equations:
- Belehrádek: Development Rate(T) = a * (T - α)^b
- Salinity: Mortality increases below ~25 ppt
- Larval Transport: Diffusion-advection based on current speed/direction

Instead of assuming fixed functional forms, we use KANs to LEARN these relationships
from data, enabling symbolic regression to discover potentially better models.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Union

from .layers import FastKAN


# ============================================================================
# VIRTUAL SOURCE: Temperature-Dependent Egg Production
# ============================================================================

def compute_fecundity(
    temperature: Union[np.ndarray, torch.Tensor],
    model: str = "stien2005",
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute temperature-dependent fecundity (eggs per female per day).

    Sea lice egg production rate increases exponentially with temperature.
    A count of "0.5 lice" in summer produces ~10x more eggs than in winter.

    Models available:
    - "stien2005": Based on Stien et al. 2005, Norwegian data
      Fecundity(T) = a * (T - T0)^b where T0 ~ 0°C (biological zero)

    - "heuch2000": Based on Heuch et al. 2000
      Development time ~ 1 / (T - 10) approximately

    - "exponential": Simple exponential Q10 model
      Fecundity(T) = base * Q10^((T-T_ref)/10)

    Args:
        temperature: Water temperature in °C
        model: Fecundity model to use

    Returns:
        fecundity: Eggs per female per day (same shape as temperature)
    """
    is_torch = isinstance(temperature, torch.Tensor)

    if is_torch:
        clip_fn = torch.clamp
        exp_fn = torch.exp
        pow_fn = torch.pow
    else:
        clip_fn = np.clip
        exp_fn = np.exp
        pow_fn = np.power

    # Ensure temperature is above biological zero
    T_min = 0.0  # Biological zero for L. salmonis
    T_eff = clip_fn(temperature - T_min, 0.1, 30.0)  # Effective temperature

    if model == "stien2005":
        # Stien et al. 2005: Development follows Belehrádek
        # Fecundity ~ (T - T0)^1.5 approximately
        # Scaled to give ~30-50 eggs/day at 10°C
        a = 3.0  # Scale factor
        b = 1.5  # Exponent
        fecundity = a * pow_fn(T_eff, b)

    elif model == "heuch2000":
        # Heuch et al. 2000: Egg string production every ~10 days at 10°C
        # ~150-200 eggs per string = ~15-20 eggs/day
        # Rate increases linearly with temperature above 5°C
        base_rate = 15.0  # eggs/day at 10°C
        T_ref = 10.0
        # Linear scaling with temperature
        fecundity = base_rate * (1.0 + 0.1 * (temperature - T_ref))
        fecundity = clip_fn(fecundity, 0.0, 100.0)

    elif model == "exponential":
        # Q10 model: rate doubles every 10°C increase
        base_rate = 15.0  # eggs/day at 10°C
        T_ref = 10.0
        Q10 = 2.5  # Q10 coefficient (rate multiplier per 10°C)
        fecundity = base_rate * pow_fn(Q10, (temperature - T_ref) / 10.0)

    else:
        raise ValueError(f"Unknown fecundity model: {model}")

    return fecundity


def compute_egg_production_rate(
    adult_female_counts: Union[np.ndarray, torch.Tensor],
    temperature: Union[np.ndarray, torch.Tensor],
    model: str = "stien2005",
) -> Union[np.ndarray, torch.Tensor]:
    """
    Virtual Source Generator: Convert raw lice counts to biological signal.

    This is the CRITICAL feature engineering step:
    E = N_females × fecundity(T)

    Why this matters:
    - 0.5 lice at 15°C produces ~80 eggs/day
    - 0.5 lice at 5°C produces ~10 eggs/day
    - The raw count "0.5" hides this 8x difference in infection pressure

    Args:
        adult_female_counts: Number of adult females per fish
        temperature: Water temperature in °C (same shape)
        model: Fecundity model to use

    Returns:
        egg_production: Eggs released per fish per day
    """
    fecundity = compute_fecundity(temperature, model=model)
    egg_production = adult_female_counts * fecundity
    return egg_production


class BelehradekKAN(nn.Module):
    """
    KAN-based Temperature-Dependent Development Rate.

    The classical Belehrádek equation for sea lice development is:
        D(T) = a * (T - α)^b

    where:
        - D(T) is development rate at temperature T
        - a, b are species-specific parameters
        - α is the biological zero (temperature below which no development occurs)

    Instead of fitting these parameters, we use a KAN to LEARN the function D(T).
    This allows:
        1. Discovery of non-Belehrádek functional forms
        2. Automatic adaptation to regional variations
        3. Symbolic extraction of the learned relationship

    The output is interpreted as the inverse time constant for development:
        τ_dev(T) = 1 / D(T)

    High temperature -> High D(T) -> Low τ_dev -> Fast development

    Args:
        n_bases: Number of RBF basis functions (resolution of learned curve)
        temp_range: Expected temperature range for input normalization (°C)
        output_scale: Scale factor for development rate output
    """

    def __init__(
        self,
        n_bases: int = 12,  # Higher resolution for smooth biological curves
        temp_range: Tuple[float, float] = (0.0, 25.0),
        output_scale: float = 1.0,
    ):
        super().__init__()

        self.temp_range = temp_range
        self.output_scale = output_scale

        # Temperature normalization parameters
        temp_min, temp_max = temp_range
        self.register_buffer('temp_mean', torch.tensor((temp_max + temp_min) / 2.0))
        self.register_buffer('temp_std', torch.tensor((temp_max - temp_min) / 2.0))

        # KAN layer to learn D(T)
        # Input: normalized temperature (1D)
        # Output: development rate (1D)
        self.kan = FastKAN(
            in_features=1,
            out_features=1,
            n_bases=n_bases,
            grid_range=(-1.5, 1.5),  # Slightly wider to handle extrapolation
        )

        # Learnable biological zero (α parameter)
        # Initialized to 0°C, but can learn from data
        self.biological_zero = nn.Parameter(torch.tensor(0.0))

    def forward(self, temperature: torch.Tensor) -> torch.Tensor:
        """
        Compute development rate from temperature.

        Args:
            temperature: Water temperature in °C, shape (...,) or (..., 1)

        Returns:
            dev_rate: Development rate, same shape as input
        """
        # Ensure proper shape
        squeeze_last = False
        if temperature.dim() == 0 or (temperature.dim() > 0 and temperature.shape[-1] != 1):
            temperature = temperature.unsqueeze(-1)
            squeeze_last = True

        # Apply biological zero (no development below this temperature)
        # Use softplus to ensure smooth gradient
        temp_effective = temperature - self.biological_zero
        temp_effective = F.softplus(temp_effective)  # Max(0, T - α) smoothly

        # Normalize for KAN input
        temp_normalized = (temp_effective - self.temp_mean) / self.temp_std

        # Compute development rate via KAN
        dev_rate = self.kan(temp_normalized)

        # Ensure positive output and apply scale
        dev_rate = F.softplus(dev_rate) * self.output_scale

        if squeeze_last:
            dev_rate = dev_rate.squeeze(-1)

        return dev_rate

    def get_biological_zero(self) -> float:
        """Return learned biological zero temperature."""
        return self.biological_zero.item()

    def symbolic_extraction(
        self,
        temp_points: int = 100,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract learned function for symbolic regression.

        Returns temperature and development rate curves that can be fitted
        to symbolic expressions (e.g., Belehrádek, exponential, polynomial).

        Args:
            temp_points: Number of points to sample
            device: Device for computation

        Returns:
            temps: Temperature values
            rates: Corresponding development rates
        """
        if device is None:
            device = next(self.parameters()).device

        temps = torch.linspace(
            self.temp_range[0],
            self.temp_range[1],
            temp_points,
            device=device,
        )

        with torch.no_grad():
            rates = self.forward(temps)

        return temps, rates


class SalinityMortalityKAN(nn.Module):
    """
    KAN-based Salinity-Dependent Mortality/Survival Factor.

    Sea lice are marine parasites that require saline water:
    - Above ~25 ppt: Normal survival
    - Below ~25 ppt: Increased mortality (osmotic stress)
    - Freshwater (<5 ppt): High mortality

    The survival factor S(salinity) ∈ [0, 1] modulates population:
        dN/dt = ... - mortality_rate * (1 - S(salinity)) * N

    Instead of using a fixed threshold function, we learn S(salinity) via KAN.

    Args:
        n_bases: Number of RBF basis functions
        salinity_range: Expected salinity range (ppt)
        threshold_init: Initial estimate of stress threshold
    """

    def __init__(
        self,
        n_bases: int = 8,
        salinity_range: Tuple[float, float] = (0.0, 40.0),
        threshold_init: float = 25.0,
    ):
        super().__init__()

        self.salinity_range = salinity_range

        # Normalization
        sal_min, sal_max = salinity_range
        self.register_buffer('sal_mean', torch.tensor((sal_max + sal_min) / 2.0))
        self.register_buffer('sal_std', torch.tensor((sal_max - sal_min) / 2.0))

        # KAN to learn survival function
        self.kan = FastKAN(
            in_features=1,
            out_features=1,
            n_bases=n_bases,
            grid_range=(-1.5, 1.5),
        )

        # Learnable threshold (for interpretability)
        self.stress_threshold = nn.Parameter(torch.tensor(threshold_init))

    def forward(self, salinity: torch.Tensor) -> torch.Tensor:
        """
        Compute survival factor from salinity.

        Args:
            salinity: Water salinity in ppt, shape (...)

        Returns:
            survival: Survival factor in [0, 1], same shape
        """
        squeeze_last = False
        if salinity.dim() == 0 or (salinity.dim() > 0 and salinity.shape[-1] != 1):
            salinity = salinity.unsqueeze(-1)
            squeeze_last = True

        # Normalize
        sal_normalized = (salinity - self.sal_mean) / self.sal_std

        # Compute via KAN
        survival_raw = self.kan(sal_normalized)

        # Sigmoid to constrain to [0, 1]
        survival = torch.sigmoid(survival_raw)

        if squeeze_last:
            survival = survival.squeeze(-1)

        return survival

    def symbolic_extraction(
        self,
        sal_points: int = 100,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract learned salinity-survival curve."""
        if device is None:
            device = next(self.parameters()).device

        salinities = torch.linspace(
            self.salinity_range[0],
            self.salinity_range[1],
            sal_points,
            device=device,
        )

        with torch.no_grad():
            survivals = self.forward(salinities)

        return salinities, survivals


class LarvalTransportModule(nn.Module):
    """
    Larval Transport Model for Cross-Farm Infection.

    Sea lice spread between farms primarily via larval stages (nauplii, copepodids)
    that drift with ocean currents. This module computes the "infection pressure"
    that farm i receives from all other farms based on:

    1. Source strength: Number of adult females at farm j (larval production)
    2. Transport probability: P(j -> i) based on current speed, direction, distance
    3. Viability: Temperature-dependent larval survival during transport

    The transport probability is modeled using an attention-like mechanism:
        P(j -> i) = softmax( f(current_j->i, distance_ij, temp_avg) )

    where f is a KAN that learns the relationship between environmental factors
    and transport success.

    Args:
        hidden_dim: Dimension of hidden representations
        n_bases: Number of RBF bases for KAN
        max_distance: Maximum transport distance (km) for normalization
        max_current_speed: Maximum current speed (m/s) for normalization
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        n_bases: int = 8,
        max_distance: float = 50.0,  # km
        max_current_speed: float = 1.0,  # m/s
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_distance = max_distance
        self.max_current_speed = max_current_speed

        # Edge feature encoder: [distance, current_u, current_v, temp] -> hidden
        # Note: current_u, current_v are east-west and north-south components
        self.edge_encoder = FastKAN(
            in_features=4,  # distance, current_u, current_v, temperature
            out_features=hidden_dim,
            n_bases=n_bases,
        )

        # Attention scorer: edge_features + source_state -> attention weight
        self.attention_kan = FastKAN(
            in_features=hidden_dim + hidden_dim,  # edge + source
            out_features=1,
            n_bases=n_bases,
        )

        # Larval production estimator: adult_female_count -> larvae
        self.larvae_production = FastKAN(
            in_features=1,
            out_features=1,
            n_bases=n_bases,
        )

        # Temperature-dependent viability (optional refinement)
        self.viability_kan = FastKAN(
            in_features=2,  # temperature, transport_time
            out_features=1,
            n_bases=n_bases,
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        adult_female_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute infection pressure from larval transport.

        Args:
            h: Node hidden states, shape (N, hidden_dim)
            edge_index: Graph edges, shape (2, E)
            edge_attr: Edge attributes [distance, current_u, current_v, temp], shape (E, 4)
            adult_female_counts: Optional explicit adult female counts, shape (N,)

        Returns:
            pressure: Infection pressure at each node, shape (N, hidden_dim)
        """
        N = h.shape[0]
        E = edge_index.shape[1]
        device = h.device

        src, dst = edge_index[0], edge_index[1]

        # Normalize edge attributes
        edge_attr_norm = edge_attr.clone()
        edge_attr_norm[:, 0] = edge_attr[:, 0] / self.max_distance  # distance
        edge_attr_norm[:, 1:3] = edge_attr[:, 1:3] / self.max_current_speed  # currents
        edge_attr_norm[:, 3] = (edge_attr[:, 3] - 10.0) / 10.0  # temperature (centered at 10°C)

        # Encode edge features
        edge_features = self.edge_encoder(edge_attr_norm)  # (E, hidden_dim)

        # Get source node states
        h_src = h[src]  # (E, hidden_dim)

        # Compute attention scores
        attention_input = torch.cat([edge_features, h_src], dim=-1)  # (E, 2*hidden_dim)
        attention_raw = self.attention_kan(attention_input).squeeze(-1)  # (E,)

        # Softmax attention per destination node
        # Use scatter_softmax-like operation
        attention_max = torch.zeros(N, device=device).scatter_reduce(
            0, dst, attention_raw, reduce='amax', include_self=False
        )
        attention_stable = attention_raw - attention_max[dst]
        attention_exp = torch.exp(attention_stable)
        attention_sum = torch.zeros(N, device=device).scatter_add(0, dst, attention_exp)
        attention_weights = attention_exp / (attention_sum[dst] + 1e-8)

        # Weight source contributions
        if adult_female_counts is not None:
            # Use explicit counts for larval production
            larvae = self.larvae_production(adult_female_counts[src].unsqueeze(-1))
            larvae = F.softplus(larvae).squeeze(-1)  # (E,)
            weighted_contribution = attention_weights * larvae
            weighted_contribution = weighted_contribution.unsqueeze(-1) * h_src
        else:
            # Use hidden state magnitude as proxy for infection level
            weighted_contribution = attention_weights.unsqueeze(-1) * h_src

        # Aggregate to destination nodes
        pressure = torch.zeros(N, self.hidden_dim, device=device)
        pressure.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.hidden_dim), weighted_contribution)

        return pressure


class KHopGraphConv(nn.Module):
    """
    K-Hop Graph Convolution for Multi-Scale Spatial Aggregation.

    Standard message passing only considers 1-hop neighbors. For sea lice,
    infection can spread over longer distances via multiple hops or direct
    long-range connections (larval transport).

    This module computes:
        h_i' = sum_{k=0}^{K} A^k @ h @ W_k

    where A^k represents k-hop reachability and W_k are learnable weights.

    Instead of explicit matrix powers (expensive), we use iterative message
    passing with different learnable transformations at each hop.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        k_hops: Number of hops (default: 3)
        n_bases: Number of RBF bases for KAN layers
        aggregation: How to combine different hops ('sum', 'concat', 'attention')
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k_hops: int = 3,
        n_bases: int = 8,
        aggregation: str = 'sum',
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.k_hops = k_hops
        self.aggregation = aggregation

        # KAN transformation for each hop
        self.hop_kans = nn.ModuleList([
            FastKAN(
                in_features=in_features,
                out_features=out_features,
                n_bases=n_bases,
            )
            for _ in range(k_hops + 1)  # +1 for 0-hop (self)
        ])

        # Learnable hop weights (for weighted sum aggregation)
        self.hop_weights = nn.Parameter(torch.ones(k_hops + 1))

        # Attention mechanism (for attention aggregation)
        if aggregation == 'attention':
            self.hop_attention = FastKAN(
                in_features=out_features,
                out_features=1,
                n_bases=n_bases,
            )

        # Output projection (for concat aggregation)
        if aggregation == 'concat':
            self.output_proj = FastKAN(
                in_features=out_features * (k_hops + 1),
                out_features=out_features,
                n_bases=n_bases,
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        K-hop graph convolution.

        Args:
            x: Node features, shape (N, in_features)
            edge_index: Edge indices, shape (2, E)
            edge_weight: Optional edge weights, shape (E,)

        Returns:
            out: Updated features, shape (N, out_features)
        """
        N = x.shape[0]
        device = x.device

        # Build sparse adjacency with normalization
        src, dst = edge_index[0], edge_index[1]

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=device)

        # Compute degree for normalization
        degree = torch.zeros(N, device=device)
        degree.scatter_add_(0, src, edge_weight)
        degree = degree.clamp(min=1.0)

        # Normalized edge weights
        edge_weight_norm = edge_weight / degree[src]

        # Collect outputs from each hop
        hop_outputs = []

        # 0-hop: self transformation
        h = self.hop_kans[0](x)
        hop_outputs.append(h)

        # k-hop: iterative message passing
        h_current = x
        for k in range(1, self.k_hops + 1):
            # Message passing: A @ h
            h_aggregated = torch.zeros(N, self.in_features, device=device)
            weighted_messages = edge_weight_norm.unsqueeze(-1) * h_current[src]
            h_aggregated.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.in_features), weighted_messages)

            # Transform
            h_transformed = self.hop_kans[k](h_aggregated)
            hop_outputs.append(h_transformed)

            # Update for next hop
            h_current = h_aggregated

        # Aggregate hop outputs
        if self.aggregation == 'sum':
            # Weighted sum
            weights = F.softmax(self.hop_weights, dim=0)
            out = sum(w * h for w, h in zip(weights, hop_outputs))

        elif self.aggregation == 'concat':
            # Concatenate and project
            out = torch.cat(hop_outputs, dim=-1)
            out = self.output_proj(out)

        elif self.aggregation == 'attention':
            # Attention-weighted sum
            stacked = torch.stack(hop_outputs, dim=1)  # (N, K+1, out_features)
            attention_scores = self.hop_attention(stacked).squeeze(-1)  # (N, K+1)
            attention_weights = F.softmax(attention_scores, dim=1)
            out = (attention_weights.unsqueeze(-1) * stacked).sum(dim=1)

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return out


class SeaLiceDynamicsCell(nn.Module):
    """
    Sea Lice-Specific Population Dynamics Cell.

    This cell models the temporal evolution of sea lice populations incorporating:
    1. Temperature-dependent development (BelehradekKAN)
    2. Salinity-dependent survival (SalinityMortalityKAN)
    3. Infection pressure from neighbors (via graph aggregation)
    4. Treatment effects (when present)

    The ODE system (conceptually):
        dN_nauplii/dt = reproduction - development_to_copepodid - mortality
        dN_copepodid/dt = development_from_nauplii - attachment - mortality
        dN_adult/dt = attachment * success_rate - natural_death - treatment

    Instead of explicit stage modeling, we learn a compressed hidden state
    that captures the essential population dynamics.

    Args:
        env_dim: Dimension of environmental input (temp, sal, currents)
        hidden_dim: Hidden state dimension
        n_bases: Number of RBF bases
        tau_min: Minimum time constant
        tau_max: Maximum time constant
    """

    def __init__(
        self,
        env_dim: int,
        hidden_dim: int,
        n_bases: int = 8,
        tau_min: float = 1.0,
        tau_max: float = 10.0,
    ):
        super().__init__()

        self.env_dim = env_dim
        self.hidden_dim = hidden_dim
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Biological modules
        self.temperature_development = BelehradekKAN(
            n_bases=n_bases,
            temp_range=(0.0, 25.0),
        )

        self.salinity_survival = SalinityMortalityKAN(
            n_bases=n_bases,
            salinity_range=(0.0, 40.0),
        )

        # Context encoder: combines env features with biological factors
        # Input: [temp, salinity, current_u, current_v, dev_rate, survival, ...]
        context_dim = env_dim + 2  # +2 for dev_rate and survival factors

        # KAN for time constant (liquid dynamics)
        self.tau_kan = FastKAN(
            in_features=context_dim + hidden_dim,  # context + pressure
            out_features=hidden_dim,
            n_bases=n_bases,
        )

        # KAN for equilibrium state
        self.equilibrium_kan = FastKAN(
            in_features=context_dim + hidden_dim,
            out_features=hidden_dim,
            n_bases=n_bases,
        )

        # KAN for treatment effect (optional modulation)
        self.treatment_kan = FastKAN(
            in_features=context_dim + 1,  # +1 for treatment indicator
            out_features=hidden_dim,
            n_bases=n_bases,
        )

    def forward(
        self,
        h: torch.Tensor,
        env: torch.Tensor,
        pressure: torch.Tensor,
        dt: torch.Tensor,
        treatment: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Single time step update.

        Args:
            h: Current hidden state, shape (N, hidden_dim)
            env: Environmental features [temp, salinity, ...], shape (N, env_dim)
            pressure: Infection pressure from neighbors, shape (N, hidden_dim)
            dt: Time step
            treatment: Optional treatment indicator, shape (N,) or (N, 1)

        Returns:
            h_new: Updated hidden state
            aux: Auxiliary outputs (tau, dev_rate, survival) for inspection
        """
        N = h.shape[0]
        device = h.device

        # Extract temperature and salinity (assumed to be first 2 features)
        temperature = env[:, 0]  # (N,)
        salinity = env[:, 1] if env.shape[1] > 1 else torch.full((N,), 34.0, device=device)

        # Compute biological factors
        dev_rate = self.temperature_development(temperature)  # (N,)
        survival = self.salinity_survival(salinity)  # (N,)

        # Build context vector
        context = torch.cat([
            env,
            dev_rate.unsqueeze(-1),
            survival.unsqueeze(-1),
        ], dim=-1)  # (N, context_dim)

        # Full context with pressure
        full_context = torch.cat([context, pressure], dim=-1)

        # Compute liquid time constant (modulated by biology)
        tau_raw = self.tau_kan(full_context)
        tau = F.softplus(tau_raw) + self.tau_min
        tau = torch.clamp(tau, min=self.tau_min, max=self.tau_max)

        # Modulate tau by development rate
        # High dev_rate -> low effective tau -> faster dynamics
        tau_effective = tau / (dev_rate.unsqueeze(-1) + 0.1)
        tau_effective = torch.clamp(tau_effective, min=self.tau_min, max=self.tau_max)

        # Compute equilibrium state
        x_equilibrium = self.equilibrium_kan(full_context)

        # Modulate by survival
        x_equilibrium = x_equilibrium * survival.unsqueeze(-1)

        # Apply treatment effect if present
        if treatment is not None:
            if treatment.dim() == 1:
                treatment = treatment.unsqueeze(-1)
            treatment_context = torch.cat([context, treatment], dim=-1)
            treatment_effect = torch.sigmoid(self.treatment_kan(treatment_context))
            x_equilibrium = x_equilibrium * (1 - treatment * treatment_effect)

        # CfC update
        if isinstance(dt, (int, float)):
            dt = torch.tensor(dt, device=device, dtype=h.dtype)
        while dt.dim() < tau_effective.dim():
            dt = dt.unsqueeze(-1)

        decay = torch.exp(-dt / tau_effective)
        h_new = decay * h + (1 - decay) * x_equilibrium

        # Auxiliary outputs for inspection
        aux = {
            'tau': tau_effective,
            'dev_rate': dev_rate,
            'survival': survival,
            'equilibrium': x_equilibrium,
        }

        return h_new, aux


class PopulationMassConservation(nn.Module):
    """
    Physics-Informed Constraint: Mass Conservation for Population Dynamics.

    In a closed system, the total population should only change due to:
    1. Birth (reproduction)
    2. Death (natural mortality, treatment)
    3. Migration (larval transport in/out)

    This module computes the PDE residual for mass conservation:
        dN/dt = births - deaths + migration_in - migration_out

    The residual is used as a loss term to enforce physical consistency.

    This is NOT meant to be a hard constraint but a soft regularizer that
    guides the model toward physically plausible solutions.
    """

    def __init__(self, hidden_dim: int, n_bases: int = 8):
        super().__init__()

        # Learn birth/death rates from hidden state
        self.birth_rate_kan = FastKAN(
            in_features=hidden_dim,
            out_features=1,
            n_bases=n_bases,
        )

        self.death_rate_kan = FastKAN(
            in_features=hidden_dim,
            out_features=1,
            n_bases=n_bases,
        )

    def forward(
        self,
        h_prev: torch.Tensor,
        h_curr: torch.Tensor,
        pressure_in: torch.Tensor,
        pressure_out: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """
        Compute mass conservation residual.

        Args:
            h_prev: Previous hidden state, shape (N, hidden_dim)
            h_curr: Current hidden state, shape (N, hidden_dim)
            pressure_in: Incoming infection pressure, shape (N,)
            pressure_out: Outgoing infection pressure, shape (N,)
            dt: Time step

        Returns:
            residual: Mass conservation violation, scalar
        """
        # Compute population proxy (hidden state magnitude)
        pop_prev = h_prev.norm(dim=-1)  # (N,)
        pop_curr = h_curr.norm(dim=-1)  # (N,)

        # Observed change
        dpop_dt_observed = (pop_curr - pop_prev) / dt  # (N,)

        # Expected change from birth/death
        birth_rate = F.softplus(self.birth_rate_kan(h_prev)).squeeze(-1)  # (N,)
        death_rate = F.softplus(self.death_rate_kan(h_prev)).squeeze(-1)  # (N,)

        births = birth_rate * pop_prev
        deaths = death_rate * pop_prev
        migration = pressure_in - pressure_out

        dpop_dt_expected = births - deaths + migration

        # Residual: difference between observed and expected
        residual = (dpop_dt_observed - dpop_dt_expected) ** 2

        return residual.mean()


# ============================================================================
# Factory function for complete sea lice model
# ============================================================================

def create_sea_lice_biology_modules(
    hidden_dim: int = 64,
    n_bases: int = 8,
    k_hops: int = 3,
) -> Dict[str, nn.Module]:
    """
    Create all sea lice-specific biology modules.

    Returns a dictionary of modules that can be integrated into the main GLKAN.
    """
    return {
        'temperature_development': BelehradekKAN(n_bases=n_bases),
        'salinity_survival': SalinityMortalityKAN(n_bases=n_bases),
        'larval_transport': LarvalTransportModule(hidden_dim=hidden_dim, n_bases=n_bases),
        'khop_conv': KHopGraphConv(
            in_features=hidden_dim,
            out_features=hidden_dim,
            k_hops=k_hops,
            n_bases=n_bases,
        ),
        'dynamics_cell': SeaLiceDynamicsCell(
            env_dim=5,  # temp, sal, current_u, current_v, treatment
            hidden_dim=hidden_dim,
            n_bases=n_bases,
        ),
        'mass_conservation': PopulationMassConservation(
            hidden_dim=hidden_dim,
            n_bases=n_bases,
        ),
    }
