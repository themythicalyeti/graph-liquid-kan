"""
Phase 1: Data Acquisition Architecture

Modules:
- barentswatch: Biological data (sites, lice counts, treatments)
- norkyst800: Environmental forcing (temperature, salinity, currents)
"""

from .barentswatch import BarentsWatchClient, Site, LiceReport, TreatmentEvent
from .norkyst800 import NorKyst800Client, NorKyst800Config

__all__ = [
    "BarentsWatchClient",
    "Site",
    "LiceReport",
    "TreatmentEvent",
    "NorKyst800Client",
    "NorKyst800Config",
]
