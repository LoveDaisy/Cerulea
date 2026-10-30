"""Physical constants and default numerical settings.

All quantities are SI unless the unit is given in the name.  Wavelengths are
carried in nanometres in the public API (that is what every spectral table
uses) and converted to metres only where the physics requires it.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Earth and atmosphere
# ---------------------------------------------------------------------------
R_EARTH = 6.371e6  # mean Earth radius [m]
G0 = 9.80665  # standard gravity at sea level [m s^-2]
R_SPECIFIC_AIR = 287.05287  # specific gas constant of dry air [J kg^-1 K^-1]
M_AIR = 28.9644e-3  # molar mass of dry air [kg mol^-1]
P0 = 101325.0  # standard sea-level pressure [Pa]
T0 = 288.15  # standard sea-level temperature [K]

#: atmospheric top used by the model [m]
Z_TOA = 120.0e3

# ---------------------------------------------------------------------------
# Molecules
# ---------------------------------------------------------------------------
K_B = 1.380649e-23  # Boltzmann constant [J K^-1]
N_A = 6.02214076e23  # Avogadro constant [mol^-1]
RYDBERG = 10973731.6  # [m^-1]

#: depolarisation factor of air (Bodhaine 1999)
DEPOLARIZATION = 0.0279
#: number density of standard dry air at 15 degC / 1013.25 hPa [m^-3]
N_STD_AIR = 2.546899e25

#: 1 Dobson Unit expressed as a column density [molecules m^-2]
DU = 2.687e20

# ---------------------------------------------------------------------------
# Sun
# ---------------------------------------------------------------------------
SOLAR_CONSTANT = 1361.0  # [W m^-2]

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
#: default spectral grid: 1 nm from 380 to 780 nm
WL_DEFAULT = np.arange(380.0, 781.0, 1.0)
#: a cheaper grid for large sweeps
WL_COARSE = np.arange(380.0, 781.0, 5.0)
