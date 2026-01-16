"""
Spatial Network & Infection Dynamics Visualization

Target Audience: Regulators & Area Managers
Goal: Understanding cross-infection between farms via ocean currents.

Features:
1. Infection Pressure Network Map - Farm network with dynamic flux edges
2. Current-driven transport visualization
3. Ambient infection pressure coloring
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import warnings

# Import model components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class SpatialNetworkVisualizer:
    """
    Spatial network visualization for infection dynamics.

    Visualizes:
    1. Farm network with geographic positions
    2. Dynamic infection pressure edges (based on ocean currents)
    3. Ambient infection pressure at each farm

    Args:
        model: SeaLicePredictor or SeaLiceGLKAN model
    """

    def __init__(self, model):
        self.model = model

        # Get the GLKAN network
        if hasattr(model, 'network'):
            self.glkan = model.network
        else:
            self.glkan = model

        # Color scheme
        self.colors = {
            'low_pressure': '#2ECC71',     # Green - safe
            'med_pressure': '#F1C40F',      # Yellow - warning
            'high_pressure': '#E74C3C',     # Red - danger
            'edge_weak': '#BDC3C7',         # Light gray
            'edge_strong': '#8E44AD',       # Purple
            'current': '#3498DB',           # Blue for currents
            'background': '#F8F9FA',
        }

    def load_farm_positions(
        self,
        graph_data: Dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load farm positions from graph data.

        Args:
            graph_data: Graph data dict with 'pos' or 'node_features'

        Returns:
            (x_coords, y_coords) arrays
        """
        # Determine number of nodes needed
        n_nodes = graph_data.get('num_nodes', 200)

        if 'pos' in graph_data:
            pos = graph_data['pos']
            if isinstance(pos, torch.Tensor):
                pos = pos.cpu().numpy()
            x, y = pos[:, 0], pos[:, 1]
        elif 'node_features' in graph_data:
            # Try to extract lat/lon from node features
            features = graph_data['node_features']
            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy()
            # Assume first two columns are coordinates
            x, y = features[:, 0], features[:, 1]
        else:
            x, y = None, None

        # Generate random positions if needed or extend existing
        if x is None or len(x) < n_nodes:
            np.random.seed(42)
            x_random = np.random.uniform(5, 15, n_nodes)  # Approximate Norwegian coast
            y_random = np.random.uniform(58, 71, n_nodes)
            if x is not None:
                # Extend existing positions
                x = np.concatenate([x, x_random[len(x):]])
                y = np.concatenate([y, y_random[len(y):]])
            else:
                x, y = x_random, y_random

        return x, y

    def compute_infection_pressure(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        feature_indices: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute infection pressure at each node using model forward pass.

        Args:
            X: Input features (T, N, F)
            edge_index: Graph edges
            edge_attr: Edge attributes [distance, current_u, current_v, temp]
            feature_indices: Feature mapping

        Returns:
            (node_pressure, edge_flux) arrays
        """
        self.glkan.eval()

        with torch.no_grad():
            predictions, auxiliary = self.glkan(
                X, edge_index,
                edge_attr=edge_attr,
                feature_indices=feature_indices,
                return_auxiliary=True,
            )

        # Node-level pressure: use prediction magnitude
        # Adult female lice (index 0) as proxy for infection level
        node_pressure = predictions[:, :, 0].mean(dim=0).cpu().numpy()

        # Edge flux: compute from current vectors and distances
        if edge_attr is not None:
            edge_flux = self._compute_edge_flux(edge_attr)
        else:
            # Default uniform flux
            edge_flux = np.ones(edge_index.shape[1]) * 0.5

        return node_pressure, edge_flux

    def _compute_edge_flux(self, edge_attr: torch.Tensor) -> np.ndarray:
        """
        Compute directed flux along edges based on current vectors.

        Flux(A→B) = current_A · direction(A→B) * decay(distance)
        """
        edge_attr_np = edge_attr.cpu().numpy()

        distance = edge_attr_np[:, 0]
        current_u = edge_attr_np[:, 1]  # Eastward
        current_v = edge_attr_np[:, 2]  # Northward

        # Current magnitude
        current_mag = np.sqrt(current_u**2 + current_v**2)

        # Distance decay (15km half-life for larvae)
        decay = np.exp(-distance / 15.0)

        # Flux = current * decay
        flux = current_mag * decay

        # Normalize to [0, 1]
        flux = flux / (flux.max() + 1e-8)

        return flux

    def plot_network_map(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        edge_index: torch.Tensor,
        node_pressure: np.ndarray,
        edge_flux: np.ndarray,
        farm_names: Optional[List[str]] = None,
        ax: Optional[plt.Axes] = None,
        show_labels: bool = False,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot static network map with infection pressure coloring.

        Args:
            x_coords: Farm x coordinates (longitude)
            y_coords: Farm y coordinates (latitude)
            edge_index: Graph edges (2, E)
            node_pressure: Infection pressure per node
            edge_flux: Flux strength per edge
            farm_names: Optional farm name labels
            ax: Optional matplotlib axes
            show_labels: Whether to show farm labels
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 10))
        else:
            fig = ax.figure

        edge_index_np = edge_index.cpu().numpy()
        N = len(x_coords)
        E = edge_index.shape[1]

        # Normalize pressure to [0, 1]
        pressure_norm = (node_pressure - node_pressure.min()) / (node_pressure.max() - node_pressure.min() + 1e-8)

        # Create colormap for nodes
        node_cmap = mcolors.LinearSegmentedColormap.from_list(
            'pressure',
            [self.colors['low_pressure'], self.colors['med_pressure'], self.colors['high_pressure']]
        )

        # Draw edges first (so nodes are on top)
        segments = []
        edge_colors = []
        edge_widths = []

        for i in range(E):
            src, dst = edge_index_np[0, i], edge_index_np[1, i]
            segments.append([(x_coords[src], y_coords[src]), (x_coords[dst], y_coords[dst])])

            # Color by flux strength
            flux = edge_flux[i]
            alpha = max(0.1, flux)
            edge_colors.append((0.5, 0.3, 0.8, alpha))  # Purple with varying alpha
            edge_widths.append(0.5 + flux * 3)

        line_collection = LineCollection(
            segments,
            colors=edge_colors,
            linewidths=edge_widths,
            zorder=1
        )
        ax.add_collection(line_collection)

        # Draw nodes
        scatter = ax.scatter(
            x_coords, y_coords,
            c=pressure_norm,
            cmap=node_cmap,
            s=100 + pressure_norm * 200,  # Size by pressure
            edgecolors='white',
            linewidths=1.5,
            zorder=2
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Infection Pressure', fontsize=10)

        # Add labels if requested
        if show_labels and N <= 30:
            for i in range(N):
                label = farm_names[i] if farm_names else f'{i}'
                ax.annotate(
                    label,
                    (x_coords[i], y_coords[i]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7
                )

        # Styling
        ax.set_xlabel('Longitude (°E)', fontsize=12)
        ax.set_ylabel('Latitude (°N)', fontsize=12)
        ax.set_title(
            'Infection Pressure Network Map\n'
            'Node color/size = infection level, Edge opacity = transport flux',
            fontsize=14,
            fontweight='bold'
        )

        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Add legend
        legend_elements = [
            mpatches.Patch(color=self.colors['low_pressure'], label='Low Risk'),
            mpatches.Patch(color=self.colors['med_pressure'], label='Medium Risk'),
            mpatches.Patch(color=self.colors['high_pressure'], label='High Risk'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Network map saved to {save_path}")

        return fig

    def plot_current_flow(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        current_u: np.ndarray,
        current_v: np.ndarray,
        ax: Optional[plt.Axes] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot ocean current flow vectors.

        Args:
            x_coords: Farm x coordinates
            y_coords: Farm y coordinates
            current_u: Eastward current velocity per node
            current_v: Northward current velocity per node
            ax: Optional matplotlib axes
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 9))
        else:
            fig = ax.figure

        # Current magnitude for coloring
        magnitude = np.sqrt(current_u**2 + current_v**2)

        # Quiver plot
        quiver = ax.quiver(
            x_coords, y_coords,
            current_u, current_v,
            magnitude,
            cmap='coolwarm',
            scale=5,
            width=0.005,
            alpha=0.8
        )

        # Add nodes
        ax.scatter(
            x_coords, y_coords,
            s=50,
            c='black',
            alpha=0.5,
            zorder=3
        )

        # Colorbar
        cbar = plt.colorbar(quiver, ax=ax, shrink=0.8)
        cbar.set_label('Current Speed (m/s)', fontsize=10)

        # Styling
        ax.set_xlabel('Longitude (°E)', fontsize=12)
        ax.set_ylabel('Latitude (°N)', fontsize=12)
        ax.set_title(
            'Ocean Current Flow Field\n'
            'Arrows show current direction and speed',
            fontsize=14,
            fontweight='bold'
        )

        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Current flow saved to {save_path}")

        return fig

    def create_animated_infection(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        edge_index: torch.Tensor,
        X: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        feature_indices: Optional[Dict] = None,
        n_frames: int = 20,
        save_path: Optional[str] = None,
        fps: int = 2,
    ) -> FuncAnimation:
        """
        Create animated visualization of infection spread over time.

        Shows how infection pressure changes and spreads via currents.

        Args:
            x_coords: Farm x coordinates
            y_coords: Farm y coordinates
            edge_index: Graph edges
            X: Input features (T, N, F)
            edge_attr: Edge attributes
            feature_indices: Feature mapping
            n_frames: Number of animation frames
            save_path: Path to save animation (gif or mp4)
            fps: Frames per second

        Returns:
            matplotlib FuncAnimation
        """
        # Get predictions for each time step
        self.glkan.eval()
        T = min(n_frames, X.shape[0])

        with torch.no_grad():
            predictions, _ = self.glkan(
                X[:T], edge_index,
                edge_attr=edge_attr,
                feature_indices=feature_indices,
            )

        # Extract pressure over time
        pressure_over_time = predictions[:, :, 0].cpu().numpy()

        # Compute edge flux (static for simplicity)
        if edge_attr is not None:
            edge_flux = self._compute_edge_flux(edge_attr)
        else:
            edge_flux = np.ones(edge_index.shape[1]) * 0.5

        # Set up figure
        fig, ax = plt.subplots(figsize=(12, 9))

        # Pressure limits for consistent coloring
        vmin = pressure_over_time.min()
        vmax = pressure_over_time.max()

        # Create colormap
        node_cmap = mcolors.LinearSegmentedColormap.from_list(
            'pressure',
            [self.colors['low_pressure'], self.colors['med_pressure'], self.colors['high_pressure']]
        )

        # Initial plot
        edge_index_np = edge_index.cpu().numpy()
        N = len(x_coords)
        E = edge_index.shape[1]

        # Draw edges (static)
        segments = []
        for i in range(E):
            src, dst = edge_index_np[0, i], edge_index_np[1, i]
            segments.append([(x_coords[src], y_coords[src]), (x_coords[dst], y_coords[dst])])

        edge_colors = [(0.5, 0.3, 0.8, max(0.1, flux)) for flux in edge_flux]
        line_collection = LineCollection(segments, colors=edge_colors, linewidths=1, zorder=1)
        ax.add_collection(line_collection)

        # Initial scatter (will be updated)
        scatter = ax.scatter(
            x_coords, y_coords,
            c=pressure_over_time[0],
            cmap=node_cmap,
            vmin=vmin,
            vmax=vmax,
            s=150,
            edgecolors='white',
            linewidths=1.5,
            zorder=2
        )

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Infection Pressure', fontsize=10)

        # Title
        title = ax.set_title(f'Time Step: 0/{T}', fontsize=14, fontweight='bold')

        ax.set_xlabel('Longitude (°E)', fontsize=12)
        ax.set_ylabel('Latitude (°N)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        def animate(frame):
            # Update node colors
            pressure = pressure_over_time[frame]
            scatter.set_array(pressure)

            # Update sizes based on pressure
            pressure_norm = (pressure - vmin) / (vmax - vmin + 1e-8)
            scatter.set_sizes(100 + pressure_norm * 200)

            # Update title
            title.set_text(f'Infection Spread Animation\nTime Step: {frame+1}/{T}')

            return scatter, title

        anim = FuncAnimation(
            fig, animate,
            frames=T,
            interval=1000 // fps,
            blit=False
        )

        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=fps)
            else:
                anim.save(save_path, writer='ffmpeg', fps=fps)
            print(f"Animation saved to {save_path}")

        plt.tight_layout()
        return anim

    def create_spatial_report(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        graph_data: Dict,
        edge_attr: Optional[torch.Tensor] = None,
        feature_indices: Optional[Dict] = None,
        save_dir: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create comprehensive spatial analysis report.

        Args:
            X: Input features
            edge_index: Graph edges
            graph_data: Graph data with positions
            edge_attr: Edge attributes
            feature_indices: Feature mapping
            save_dir: Directory to save outputs

        Returns:
            matplotlib Figure
        """
        # Get number of nodes from data
        N = X.shape[1]

        # Load positions and subset to match data
        x_coords_full, y_coords_full = self.load_farm_positions(graph_data)
        x_coords = x_coords_full[:N]
        y_coords = y_coords_full[:N]

        # Filter edges to only include valid node indices
        edge_index_np = edge_index.cpu().numpy()
        valid_mask = (edge_index_np[0] < N) & (edge_index_np[1] < N)
        edge_index_filtered = edge_index[:, valid_mask]

        # Filter edge_attr if provided
        if edge_attr is not None:
            edge_attr_filtered = edge_attr[valid_mask]
        else:
            edge_attr_filtered = None

        # Compute infection pressure
        node_pressure, edge_flux = self.compute_infection_pressure(
            X, edge_index_filtered, edge_attr_filtered, feature_indices
        )

        # Extract currents if available
        if feature_indices and 'current_u' in feature_indices and 'current_v' in feature_indices:
            current_u = X[:, :, feature_indices['current_u']].mean(dim=0).cpu().numpy()
            current_v = X[:, :, feature_indices['current_v']].mean(dim=0).cpu().numpy()
        else:
            current_u = np.random.randn(N) * 0.1
            current_v = np.random.randn(N) * 0.1

        # Create figure
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 2, figure=fig)

        # Network map (top-left, larger)
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_network_map(
            x_coords, y_coords, edge_index_filtered,
            node_pressure, edge_flux, ax=ax1
        )

        # Current flow (bottom-left)
        ax2 = fig.add_subplot(gs[1, 0])
        self.plot_current_flow(
            x_coords, y_coords,
            current_u, current_v,
            ax=ax2
        )

        # Pressure distribution (bottom-right)
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(node_pressure, bins=30, color=self.colors['med_pressure'], edgecolor='white', alpha=0.8)
        ax3.axvline(
            np.percentile(node_pressure, 90),
            color=self.colors['high_pressure'],
            linestyle='--',
            linewidth=2,
            label='90th percentile'
        )
        ax3.set_xlabel('Infection Pressure', fontsize=12)
        ax3.set_ylabel('Number of Farms', fontsize=12)
        ax3.set_title('Infection Pressure Distribution', fontsize=14, fontweight='bold')
        ax3.legend()

        fig.suptitle(
            'Spatial Infection Dynamics Report',
            fontsize=16,
            fontweight='bold',
            y=1.02
        )

        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            # Save individual plots
            fig_net, ax_net = plt.subplots(figsize=(14, 10))
            self.plot_network_map(
                x_coords, y_coords, edge_index_filtered,
                node_pressure, edge_flux, ax=ax_net,
                save_path=save_path / 'network_map.png'
            )
            plt.close(fig_net)

            fig_curr, ax_curr = plt.subplots(figsize=(12, 9))
            self.plot_current_flow(
                x_coords, y_coords,
                current_u, current_v,
                ax=ax_curr,
                save_path=save_path / 'current_flow.png'
            )
            plt.close(fig_curr)

            # Save combined report
            fig.savefig(
                save_path / 'spatial_report.png',
                dpi=150, bbox_inches='tight', facecolor='white'
            )
            print(f"Spatial report saved to {save_dir}")

        return fig


def demo_spatial_network():
    """Demo function for spatial network visualization."""
    print("Spatial Network & Infection Dynamics Demo")
    print("=" * 50)
    print("This module visualizes cross-infection between farms:")
    print()
    print("1. Network Map:")
    print("   - Nodes = farms, colored by infection pressure")
    print("   - Edges = transport pathways, opacity = flux strength")
    print()
    print("2. Current Flow:")
    print("   - Arrow direction = current direction")
    print("   - Arrow color = current speed")
    print()
    print("3. Animation:")
    print("   - Shows infection spread over time")
    print("   - Reveals how outbreaks propagate via currents")
    print()
    print("Usage:")
    print("  viz = SpatialNetworkVisualizer(model)")
    print("  viz.create_spatial_report(X, edge_index, graph_data, save_dir='visuals/output')")


if __name__ == "__main__":
    demo_spatial_network()
