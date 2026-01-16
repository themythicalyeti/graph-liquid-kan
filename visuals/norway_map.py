"""
Norwegian Coast Visualization for Sea Lice Network

Plots farms on actual Norwegian coastline using real BarentsWatch coordinates.
Uses NorKyst-800 grid alignment for current visualization.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import warnings

# Try to import cartopy for proper map projection (optional)
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    warnings.warn("Cartopy not installed. Using simple lat/lon plot.")


class NorwayMapVisualizer:
    """
    Visualize sea lice network on Norwegian coast.

    Uses real farm coordinates from BarentsWatch and
    can overlay NorKyst-800 current data.
    """

    # Norwegian coast bounding box
    NORWAY_BOUNDS = {
        'lon_min': 4.0,
        'lon_max': 32.0,
        'lat_min': 57.5,
        'lat_max': 71.5,
    }

    # Production area approximate centers (for labeling)
    PRODUCTION_AREAS = {
        1: {'name': 'Svenskegrensen-Jæren', 'lat': 58.5, 'lon': 6.0},
        2: {'name': 'Ryfylke', 'lat': 59.2, 'lon': 6.0},
        3: {'name': 'Karmøy-Sotra', 'lat': 60.0, 'lon': 5.2},
        4: {'name': 'Nordhordland-Stadt', 'lat': 61.0, 'lon': 5.0},
        5: {'name': 'Stadt-Hustadvika', 'lat': 62.3, 'lon': 5.5},
        6: {'name': 'Nordmøre-Sør-Trøndelag', 'lat': 63.5, 'lon': 9.0},
        7: {'name': 'Nord-Trøndelag', 'lat': 64.5, 'lon': 10.5},
        8: {'name': 'Helgeland-Salten', 'lat': 66.0, 'lon': 12.5},
        9: {'name': 'Vestfjorden-Vesterålen', 'lat': 68.0, 'lon': 14.5},
        10: {'name': 'Andøya-Senja', 'lat': 69.0, 'lon': 17.0},
        11: {'name': 'Kvaløya-Loppa', 'lat': 70.0, 'lon': 20.0},
        12: {'name': 'Vest-Finnmark', 'lat': 70.5, 'lon': 24.0},
        13: {'name': 'Øst-Finnmark', 'lat': 70.2, 'lon': 28.0},
    }

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.farm_coords = None
        self.site_ids = None
        self._load_farm_coordinates()

    def _load_farm_coordinates(self):
        """Load farm coordinates from metadata."""
        metadata_path = self.data_dir / "raw/barentswatch/graph_nodes_metadata.csv"
        tensors_path = self.data_dir / "processed/tensors.npz"

        if not metadata_path.exists():
            warnings.warn(f"Metadata not found at {metadata_path}")
            return

        # Load metadata
        df = pd.read_csv(metadata_path)

        # Load site_ids to ensure correct ordering
        if tensors_path.exists():
            data = np.load(tensors_path, allow_pickle=True)
            self.site_ids = data['site_ids']

            # Reorder coordinates to match site_ids
            df_indexed = df.set_index('locality_id')

            lats = []
            lons = []
            names = []

            for sid in self.site_ids:
                if sid in df_indexed.index:
                    row = df_indexed.loc[sid]
                    lats.append(row['latitude'])
                    lons.append(row['longitude'])
                    names.append(row['name'])
                else:
                    lats.append(np.nan)
                    lons.append(np.nan)
                    names.append(f"Unknown-{sid}")

            self.farm_coords = {
                'latitude': np.array(lats),
                'longitude': np.array(lons),
                'names': names,
            }
        else:
            # Use metadata order
            self.farm_coords = {
                'latitude': df['latitude'].values,
                'longitude': df['longitude'].values,
                'names': df['name'].tolist(),
            }
            self.site_ids = df['locality_id'].values

        print(f"Loaded coordinates for {len(self.farm_coords['latitude'])} farms")

    def get_coordinates(self, n_farms: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get farm coordinates, optionally subset."""
        if self.farm_coords is None:
            raise ValueError("Farm coordinates not loaded")

        lats = self.farm_coords['latitude']
        lons = self.farm_coords['longitude']

        if n_farms is not None:
            lats = lats[:n_farms]
            lons = lons[:n_farms]

        return lons, lats  # Return as x, y (lon, lat)

    def plot_norway_basemap(
        self,
        ax: plt.Axes,
        use_cartopy: bool = True,
    ):
        """Add Norway coastline to axes."""
        if HAS_CARTOPY and use_cartopy:
            # Add land, ocean, coastline
            ax.add_feature(cfeature.LAND, facecolor='#F5F5F5', zorder=0)
            ax.add_feature(cfeature.OCEAN, facecolor='#E6F3FF', zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, zorder=1)

            # Set extent
            ax.set_extent([
                self.NORWAY_BOUNDS['lon_min'],
                self.NORWAY_BOUNDS['lon_max'],
                self.NORWAY_BOUNDS['lat_min'],
                self.NORWAY_BOUNDS['lat_max'],
            ], crs=ccrs.PlateCarree())
        else:
            # Simple approximation without cartopy
            ax.set_xlim(self.NORWAY_BOUNDS['lon_min'], self.NORWAY_BOUNDS['lon_max'])
            ax.set_ylim(self.NORWAY_BOUNDS['lat_min'], self.NORWAY_BOUNDS['lat_max'])
            ax.set_facecolor('#E6F3FF')  # Ocean blue
            ax.grid(True, alpha=0.3)

    def plot_farm_network(
        self,
        infection_pressure: Optional[np.ndarray] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_flux: Optional[np.ndarray] = None,
        n_farms: Optional[int] = None,
        ax: Optional[plt.Axes] = None,
        show_edges: bool = False,
        show_names: bool = False,
        title: str = "Norwegian Salmon Farm Network",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot farms on Norwegian coast with infection pressure.

        Args:
            infection_pressure: Infection level per farm (for coloring)
            edge_index: Graph edges (2, E)
            edge_flux: Flux strength per edge
            n_farms: Number of farms to plot (default: all)
            ax: Matplotlib axes
            show_edges: Whether to draw edges
            show_names: Whether to show farm names
            title: Plot title
            save_path: Path to save figure
        """
        # Get coordinates
        lons, lats = self.get_coordinates(n_farms)
        n = len(lons)

        # Create figure
        if ax is None:
            if HAS_CARTOPY:
                fig = plt.figure(figsize=(12, 16))
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            else:
                fig, ax = plt.subplots(figsize=(12, 16))
        else:
            fig = ax.figure

        # Add basemap
        self.plot_norway_basemap(ax, use_cartopy=HAS_CARTOPY)

        # Default infection pressure
        if infection_pressure is None:
            infection_pressure = np.ones(n) * 0.5
        elif len(infection_pressure) > n:
            infection_pressure = infection_pressure[:n]

        # Normalize for coloring
        pressure_norm = (infection_pressure - infection_pressure.min()) / \
                       (infection_pressure.max() - infection_pressure.min() + 1e-8)

        # Color map: green (low) -> yellow -> red (high)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'infection',
            ['#2ECC71', '#F1C40F', '#E74C3C']
        )

        # Draw edges if requested
        if show_edges and edge_index is not None:
            edge_idx = edge_index.cpu().numpy() if isinstance(edge_index, torch.Tensor) else edge_index

            # Filter to valid indices
            valid = (edge_idx[0] < n) & (edge_idx[1] < n)
            edge_idx = edge_idx[:, valid]

            if edge_flux is not None:
                flux = edge_flux[valid] if len(edge_flux) > valid.sum() else edge_flux[:valid.sum()]
            else:
                flux = np.ones(edge_idx.shape[1]) * 0.3

            # Draw edges with varying opacity
            segments = []
            colors = []
            for i in range(edge_idx.shape[1]):
                src, dst = edge_idx[0, i], edge_idx[1, i]
                if not (np.isnan(lons[src]) or np.isnan(lons[dst])):
                    segments.append([(lons[src], lats[src]), (lons[dst], lats[dst])])
                    alpha = min(0.5, flux[i] if i < len(flux) else 0.1)
                    colors.append((0.5, 0.3, 0.8, alpha))

            if segments:
                lc = LineCollection(
                    segments,
                    colors=colors,
                    linewidths=0.5,
                    transform=ccrs.PlateCarree() if HAS_CARTOPY else ax.transData,
                    zorder=2
                )
                ax.add_collection(lc)

        # Plot farms
        transform = ccrs.PlateCarree() if HAS_CARTOPY else None
        scatter_kwargs = {'transform': transform} if transform else {}

        scatter = ax.scatter(
            lons, lats,
            c=pressure_norm,
            cmap=cmap,
            s=30 + pressure_norm * 70,
            edgecolors='white',
            linewidths=0.5,
            alpha=0.8,
            zorder=3,
            **scatter_kwargs
        )

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Infection Pressure', fontsize=10)

        # Add production area labels
        for pa_id, pa_info in self.PRODUCTION_AREAS.items():
            if pa_id <= 8:  # Only label southern areas to avoid clutter
                ax.text(
                    pa_info['lon'], pa_info['lat'],
                    f"PA{pa_id}",
                    fontsize=8,
                    alpha=0.6,
                    ha='center',
                    transform=transform if transform else ax.transData,
                    zorder=4
                )

        # Title and labels
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Longitude (°E)', fontsize=10)
        ax.set_ylabel('Latitude (°N)', fontsize=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Map saved to {save_path}")

        return fig

    def plot_current_field(
        self,
        current_u: np.ndarray,
        current_v: np.ndarray,
        n_farms: Optional[int] = None,
        ax: Optional[plt.Axes] = None,
        title: str = "Ocean Current Field (NorKyst-800)",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot ocean current vectors at farm locations.
        """
        lons, lats = self.get_coordinates(n_farms)
        n = len(lons)

        if len(current_u) > n:
            current_u = current_u[:n]
            current_v = current_v[:n]

        # Create figure
        if ax is None:
            if HAS_CARTOPY:
                fig = plt.figure(figsize=(12, 16))
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            else:
                fig, ax = plt.subplots(figsize=(12, 16))
        else:
            fig = ax.figure

        self.plot_norway_basemap(ax, use_cartopy=HAS_CARTOPY)

        # Current magnitude
        magnitude = np.sqrt(current_u**2 + current_v**2)

        # Plot current vectors
        transform = ccrs.PlateCarree() if HAS_CARTOPY else None
        quiver_kwargs = {'transform': transform} if transform else {}

        q = ax.quiver(
            lons, lats,
            current_u, current_v,
            magnitude,
            cmap='coolwarm',
            scale=3,
            width=0.003,
            alpha=0.7,
            zorder=3,
            **quiver_kwargs
        )

        # Colorbar
        cbar = plt.colorbar(q, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Current Speed (m/s)', fontsize=10)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude (°E)', fontsize=10)
        ax.set_ylabel('Latitude (°N)', fontsize=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Current field saved to {save_path}")

        return fig

    def plot_regional_zoom(
        self,
        region: str = 'trøndelag',
        infection_pressure: Optional[np.ndarray] = None,
        n_farms: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot zoomed view of a specific region.

        Regions: 'rogaland', 'hordaland', 'trøndelag', 'nordland', 'troms', 'finnmark'
        """
        regions = {
            'rogaland': {'lon': (5.0, 7.5), 'lat': (58.5, 60.0)},
            'hordaland': {'lon': (4.5, 7.0), 'lat': (59.5, 61.5)},
            'trøndelag': {'lon': (8.0, 12.0), 'lat': (63.0, 65.5)},
            'nordland': {'lon': (11.0, 16.0), 'lat': (65.5, 68.5)},
            'troms': {'lon': (16.0, 21.0), 'lat': (68.5, 70.5)},
            'finnmark': {'lon': (23.0, 31.0), 'lat': (69.5, 71.5)},
        }

        if region not in regions:
            raise ValueError(f"Unknown region: {region}. Choose from {list(regions.keys())}")

        bounds = regions[region]

        # Get coordinates
        lons, lats = self.get_coordinates(n_farms)

        # Filter to region
        in_region = (
            (lons >= bounds['lon'][0]) & (lons <= bounds['lon'][1]) &
            (lats >= bounds['lat'][0]) & (lats <= bounds['lat'][1])
        )

        lons_region = lons[in_region]
        lats_region = lats[in_region]

        if infection_pressure is not None:
            if len(infection_pressure) > len(in_region):
                infection_pressure = infection_pressure[:len(in_region)]
            pressure_region = infection_pressure[in_region]
        else:
            pressure_region = np.ones(len(lons_region)) * 0.5

        # Create figure
        if HAS_CARTOPY:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.set_extent([bounds['lon'][0], bounds['lon'][1],
                          bounds['lat'][0], bounds['lat'][1]],
                         crs=ccrs.PlateCarree())
        else:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xlim(bounds['lon'])
            ax.set_ylim(bounds['lat'])

        self.plot_norway_basemap(ax, use_cartopy=HAS_CARTOPY)

        # Normalize pressure
        if len(pressure_region) > 0:
            p_norm = (pressure_region - pressure_region.min()) / \
                     (pressure_region.max() - pressure_region.min() + 1e-8)
        else:
            p_norm = np.array([])

        cmap = mcolors.LinearSegmentedColormap.from_list(
            'infection', ['#2ECC71', '#F1C40F', '#E74C3C']
        )

        transform = ccrs.PlateCarree() if HAS_CARTOPY else None
        scatter_kwargs = {'transform': transform} if transform else {}

        scatter = ax.scatter(
            lons_region, lats_region,
            c=p_norm,
            cmap=cmap,
            s=100,
            edgecolors='white',
            linewidths=1,
            alpha=0.9,
            zorder=3,
            **scatter_kwargs
        )

        # Add farm names for small regions
        if len(lons_region) <= 30:
            names = [self.farm_coords['names'][i] for i, is_in in enumerate(in_region) if is_in]
            for i, (lon, lat, name) in enumerate(zip(lons_region, lats_region, names)):
                ax.annotate(
                    name[:15],
                    (lon, lat),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=7,
                    alpha=0.7,
                    **({'transform': transform} if transform else {})
                )

        plt.colorbar(scatter, ax=ax, shrink=0.6, label='Infection Pressure')
        ax.set_title(f'{region.title()} Region - {len(lons_region)} Farms',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Regional map saved to {save_path}")

        return fig


def create_norway_visualization(
    model,
    X: torch.Tensor,
    edge_index: torch.Tensor,
    feature_indices: Optional[Dict] = None,
    output_dir: str = "visuals/output",
):
    """
    Create complete Norwegian coast visualization.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize visualizer
    viz = NorwayMapVisualizer()

    n_farms = X.shape[1]

    # Get model predictions for infection pressure
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'network'):
            preds, _ = model.network(X, edge_index, feature_indices=feature_indices)
        else:
            preds, _ = model(X, edge_index, feature_indices=feature_indices)

    # Use mean adult female prediction as infection pressure
    infection_pressure = preds[:, :, 0].mean(dim=0).cpu().numpy()

    # Extract currents
    if feature_indices and 'current_u' in feature_indices:
        current_u = X[:, :, feature_indices['current_u']].mean(dim=0).cpu().numpy()
        current_v = X[:, :, feature_indices['current_v']].mean(dim=0).cpu().numpy()
    else:
        current_u = np.zeros(n_farms)
        current_v = np.zeros(n_farms)

    # 1. Full Norway map
    print("Creating Norway-wide map...")
    viz.plot_farm_network(
        infection_pressure=infection_pressure,
        edge_index=edge_index,
        n_farms=n_farms,
        show_edges=False,  # Too many edges for full view
        title="Sea Lice Infection Pressure - Norwegian Coast",
        save_path=str(output_path / "norway_infection_map.png"),
    )
    plt.close()

    # 2. Current field
    print("Creating current field map...")
    viz.plot_current_field(
        current_u=current_u,
        current_v=current_v,
        n_farms=n_farms,
        title="Ocean Currents at Farm Locations (NorKyst-800)",
        save_path=str(output_path / "norway_currents.png"),
    )
    plt.close()

    # 3. Regional zooms
    for region in ['trøndelag', 'nordland', 'hordaland']:
        print(f"Creating {region} regional map...")
        try:
            viz.plot_regional_zoom(
                region=region,
                infection_pressure=infection_pressure,
                n_farms=n_farms,
                save_path=str(output_path / f"region_{region}.png"),
            )
            plt.close()
        except Exception as e:
            print(f"  Warning: Could not create {region} map: {e}")

    print(f"\nNorway visualizations saved to {output_path}")


if __name__ == "__main__":
    # Demo
    print("Norway Map Visualizer")
    print("=" * 50)

    viz = NorwayMapVisualizer()

    # Simple test plot
    fig = viz.plot_farm_network(
        title="Norwegian Salmon Farms",
        save_path="visuals/output/norway_test.png"
    )
    plt.close()

    print("Test complete!")
