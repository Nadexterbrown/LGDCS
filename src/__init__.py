"""
LGDCS - Lagrangian Gas Dynamics with Coupled Flame Speed
=========================================================

A 1D Lagrangian hydrodynamics solver for compressible flows
with flame-piston coupling.

Main Classes:
-------------
LagrangianSolver : The core solver for 1D Lagrangian gas dynamics
EOS : Unified Equation of State with auto-detection (Cantera default, ideal gas fallback)

EOS Usage:
----------
The EOS class automatically detects Cantera availability:
- If Cantera is available: Uses Cantera for accurate thermodynamics
- If Cantera is unavailable: Falls back to ideal gas calculations

>>> from src import EOS, LagrangianSolver
>>>
>>> # Recommended: Use gas_config for combustion simulations
>>> eos = EOS(gas_config={'T': 300, 'P': 101325, 'Fuel': 'CH4',
...                        'Oxidizer': 'O2:1, N2:3.76', 'Phi': 1.0,
...                        'mech': 'gri30.yaml'})
>>>
>>> # Simple: Auto-detect with air (or ideal gas fallback)
>>> eos = EOS()
>>>
>>> # Check backend
>>> print(f"Using Cantera: {eos.use_cantera}")
"""

# Core solver components
from .lagrangian import (
    LagrangianSolver,
    EOS,
    GasState,
    ExactRiemannSolver,
    ConservationDiagnostics,
    ArtificialViscosityType,
    LimiterType,
    PistonVelocityModel,
    compute_exact_riemann,
    compute_L1_error,
    compute_L2_error,
    compute_Linf_error,
)

# Flame speed providers
from .flame import (
    FlameSpeedProvider,
    ConstantFlameSpeed,
    TabularFlameSpeed,
    CSVTabulatedFlameSpeed,
    CanteraFlameSpeed,
    ActivationEnergyFlameSpeed,
    generate_flame_speed_table,
)

__all__ = [
    # Core solver
    'LagrangianSolver',
    # EOS
    'EOS',
    # State and diagnostics
    'GasState',
    'ExactRiemannSolver',
    'ConservationDiagnostics',
    # Enums
    'ArtificialViscosityType',
    'LimiterType',
    'PistonVelocityModel',
    # Flame speed providers
    'FlameSpeedProvider',
    'ConstantFlameSpeed',
    'TabularFlameSpeed',
    'CSVTabulatedFlameSpeed',
    'CanteraFlameSpeed',
    'ActivationEnergyFlameSpeed',
    'generate_flame_speed_table',
    # Utility functions
    'compute_exact_riemann',
    'compute_L1_error',
    'compute_L2_error',
    'compute_Linf_error',
]
