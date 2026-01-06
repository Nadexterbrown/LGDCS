"""
Flame-Elongation Piston Simulation with Coupled Flame Speed
=============================================================

This script simulates flame-driven piston dynamics where the piston velocity
is coupled to the local flame speed through the relationship:

    u_piston = (sigma - 1) * density_ratio * S_L

where:
    sigma        : Flame stretch/elongation parameter (can be time-dependent)
    density_ratio: Ratio of unburned to burned gas density (rho_u / rho_b)
    S_L          : Laminar flame speed (function of T, P, phi)

The coupling uses sub-iteration to achieve self-consistency between:
    - Thermodynamic state at piston face -> flame speed
    - Flame speed -> piston velocity
    - Piston velocity -> thermodynamic evolution

Features:
---------
1. Iterative flame-piston coupling with sub-iteration
2. Multiple flame speed models (constant, power-law, tabulated, Cantera)
3. Time-dependent sigma parameter for flame elongation
4. Comprehensive diagnostics for coupling convergence

Usage:
------
1. Edit the CONFIGURATION section below
2. Run: python flame_elongation_piston.py

Author: Generated for research purposes
Date: 2024
"""

import os
import sys
from pathlib import Path

# Add ConsL parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# CLTORC path (used for importing problem_setup via importlib)
CLTORC_PATH = Path(__file__).parent.parent.parent.parent / 'CLTORC'

import numpy as np
from typing import Callable, Optional, Dict, List, Union, TYPE_CHECKING
from dataclasses import dataclass
import warnings

if TYPE_CHECKING:
    from src.lagrangian import EOS

# =============================================================================
# USER CONFIGURATION - Edit these parameters to configure the simulation
# =============================================================================

# --- Domain Parameters ---
DOMAIN_LENGTH = 20          # Domain length [m]
N_CELLS = 400                # Number of computational cells

# --- Initial Conditions (via CLTORC input_params) ---
# Set USE_INPUT_PARAMS = True to use Cantera-based initial conditions
# Set USE_INPUT_PARAMS = False to use simple ideal gas with values below
USE_INPUT_PARAMS = True

# input_params configuration (used when USE_INPUT_PARAMS = True)
INPUT_PARAMS_CONFIG = {
    'T': 503,                    # Temperature [K]
    'P': 10e5,                   # Pressure [Pa]
    'Phi': 1.0,                  # Equivalence ratio
    'Fuel': 'H2',                # Fuel species
    'Oxidizer': 'O2',            # Oxidizer species (use 'O2:1, N2:3.76' for air)
    'mech': '../../chemical_mechanism/LiDryer.yaml',  # Cantera mechanism file (relative to script)
    'temperature_unit': 'K',
    'pressure_unit': 'Pa',
    'phi_convention': 'scale_fuel',
    'use_dry_air': False,
}

# Simple ideal gas fallback (used when USE_INPUT_PARAMS = False)
INITIAL_TEMPERATURE = 300.0  # Initial gas temperature [K]
INITIAL_PRESSURE = 101325.0  # Initial gas pressure [Pa] (1 atm)
GAMMA = 1.4                  # Ratio of specific heats
R_GAS = 287.0                # Specific gas constant [J/(kg·K)]

# --- Flame Speed Configuration ---
# Options: 'constant', 'tabulated', 'tabulated_csv', 'cantera'
# Note: Power-law mode has been removed. Use tabulated data or Cantera for T,P-dependent flame speeds.
FLAME_SPEED_MODE = 'tabulated_csv'

# For 'constant' mode:
CONSTANT_FLAME_SPEED = 0.4   # Constant flame speed [m/s]
CONSTANT_EXPANSION_RATIO = 7.0  # Expansion ratio rho_u/rho_b

# For 'cantera' mode: Uses fuel, oxidizer, mechanism, and phi from INPUT_PARAMS_CONFIG

# For 'tabulated_csv' mode: Uses pre-computed flame properties from CSV file
TABULATED_CSV_PATH = '../cantera_data/output/flame_properties.csv'  # Relative to script dir

# --- Equivalence Ratio (fallback when USE_INPUT_PARAMS=False) ---
EQUIVALENCE_RATIO = 1.0      # Fuel-air equivalence ratio phi

# --- Piston Velocity Model ---
# u_piston = (sigma - 1) * density_ratio * S_L
#
# density_ratio: rho_unburned / rho_burned (fixed or from flame speed provider)
# sigma: Flame elongation/stretch factor

# Density ratio configuration
USE_PROVIDER_DENSITY = True  # If True, use density from flame speed provider
                             # If False, use fixed DENSITY_RATIO value
DENSITY_RATIO = 7.0          # Fixed density ratio (used if USE_PROVIDER_DENSITY=False)

# Sigma configuration
# Options: 'constant', 'linear', 'power_law', 'tuned_power_law', 'custom'
SIGMA_MODE = 'tuned_power_law'

# For 'constant' sigma:
SIGMA_VALUE = 1.0            # Constant sigma value

# For 'linear' sigma: sigma(t) = sigma_0 + sigma_rate * t
SIGMA_INITIAL = 1.0          # Initial sigma value
SIGMA_RATE = 10000.0         # Rate of sigma increase [1/s]
SIGMA_MAX = 20.0             # Maximum sigma value

# For 'power_law' sigma: sigma(t) = sigma_0 * (1 + k*t)^n
# This models accelerating flame stretch where elongation grows nonlinearly
SIGMA_POWER_LAW_INITIAL = 1.0    # Initial sigma value (sigma_0)
SIGMA_POWER_LAW_K = 100          # Rate coefficient k [1/s]
SIGMA_POWER_LAW_N = 2            # Power law exponent n (0.5 = sqrt growth, 1.0 = linear, 2.0 = quadratic)
SIGMA_POWER_LAW_MAX = 20.0       # Maximum sigma value (cap)

# For 'tuned_power_law' sigma: sigma(t) = sigma_0 * (1 + k*t)^n
# where k is AUTO-COMPUTED so that sigma(t_target) = sigma_target for ANY exponent n
# This allows fair comparison of different exponents on the same time scale
SIGMA_TUNED_INITIAL = 1.0        # Initial sigma value (sigma_0)
SIGMA_TUNED_TARGET = 20.0        # Target sigma value at t_target
SIGMA_TUNED_T_TARGET = 0.01      # Time at which sigma should equal sigma_target [s]
SIGMA_TUNED_N = 2.0              # Power law exponent n (shape parameter)
SIGMA_TUNED_MAX = 50.0           # Maximum sigma value (cap, can be > target for extrapolation)

# --- Coupling Parameters ---
COUPLING_MAX_ITERATIONS = 100 # Maximum sub-iterations per timestep
COUPLING_TOLERANCE = 1e-3    # Convergence tolerance for piston velocity (1% relative change)
COUPLING_RELAXATION = 0.7    # Under-relaxation factor (0.5-1.0)

# --- Simulation Parameters ---
T_FINAL = 0.1               # Final simulation time [s]
CFL = 0.4                    # CFL number for timestep stability
PISTON_SIDE = 'left'         # Which side has the piston: 'left' or 'right'

# --- Termination Conditions ---
# Simulation terminates when ANY enabled condition is met
ENABLE_VELOCITY_TERMINATION = True  # Enable termination based on piston velocity
TERMINATION_VELOCITY = 1500.0         # Terminate when |u_piston| >= this value [m/s]
TERMINATION_VELOCITY_MODE = 'above'  # 'above': terminate when u >= threshold
                                     # 'below': terminate when u <= threshold

# --- Boundary Conditions ---
# Right boundary: 'wall' (reflecting) or 'open' (non-reflecting)
RIGHT_BC = 'open'

# --- Output Settings ---
OUTPUT_DIR = 'tabulated_flame_elongation_results'
SAVE_INTERVAL = 10           # Save solution every N steps
PRINT_INTERVAL = 50          # Print progress every N steps
SAVE_PLOTS = True            # Generate and save plots
PLOT_XT_DIAGRAMS = True      # Generate x-t surface plots (pressure, density, velocity, energy)

# --- Flame Profile Saving Configuration ---
# Save flame profiles (T, u, HRR) from each FreeFlame simulation during simulation
SAVE_FLAME_PROFILES = True               # Enable/disable profile saving
FLAME_PROFILES_SUBDIR = 'flame_profiles' # Subdirectory within OUTPUT_DIR

# =============================================================================
# END OF USER CONFIGURATION
# =============================================================================


# =============================================================================
# SIGMA (ELONGATION FACTOR) FUNCTIONS
# =============================================================================

@dataclass
class SigmaConfig:
    """Configuration for sigma (flame elongation) parameter."""
    mode: str = 'constant'   # 'constant', 'linear', 'power_law', 'tuned_power_law', 'custom'
    value: float = 1.0       # For constant mode
    initial: float = 1.0     # For linear mode
    rate: float = 0.0        # For linear mode [1/s]
    max_value: float = 10.0  # For linear mode (cap)
    # Power law parameters: sigma(t) = sigma_0 * (1 + k*t)^n
    power_law_initial: float = 1.0   # sigma_0
    power_law_k: float = 100.0       # Rate coefficient k [1/s]
    power_law_n: float = 0.5         # Exponent n
    power_law_max: float = 20.0      # Maximum sigma (cap)
    # Tuned power law parameters: k is auto-computed so sigma(t_target) = sigma_target
    tuned_initial: float = 1.0       # sigma_0
    tuned_target: float = 20.0       # Target sigma at t_target
    tuned_t_target: float = 0.01     # Time to reach target [s]
    tuned_n: float = 2.0             # Exponent n (shape)
    tuned_max: float = 50.0          # Maximum sigma (cap)
    custom_func: Optional[Callable[[float], float]] = None  # For custom mode


def create_sigma_function(config: SigmaConfig) -> Callable[[float], float]:
    """
    Create a sigma function from configuration.

    Parameters:
    -----------
    config : SigmaConfig
        Configuration for sigma

    Returns:
    --------
    sigma_func : callable
        Function that takes time and returns sigma value
    """
    mode = config.mode.lower()

    if mode == 'constant':
        sigma_val = config.value
        def sigma_func(t: float) -> float:
            return sigma_val
        return sigma_func

    elif mode == 'linear':
        sigma_0 = config.initial
        rate = config.rate
        sigma_max = config.max_value
        def sigma_func(t: float) -> float:
            return min(sigma_0 + rate * t, sigma_max)
        return sigma_func

    elif mode == 'power_law':
        # Power law: sigma(t) = sigma_0 * (1 + k*t)^n
        sigma_0 = config.power_law_initial
        k = config.power_law_k
        n = config.power_law_n
        sigma_max = config.power_law_max
        def sigma_func(t: float) -> float:
            sigma = sigma_0 * (1.0 + k * t) ** n
            return min(sigma, sigma_max)
        return sigma_func

    elif mode == 'tuned_power_law':
        # Tuned power law: sigma(t) = sigma_0 * (1 + k*t)^n
        # where k is computed so that sigma(t_target) = sigma_target
        #
        # Derivation:
        #   sigma_target = sigma_0 * (1 + k * t_target)^n
        #   (sigma_target / sigma_0)^(1/n) = 1 + k * t_target
        #   k = [(sigma_target / sigma_0)^(1/n) - 1] / t_target
        #
        sigma_0 = config.tuned_initial
        sigma_target = config.tuned_target
        t_target = config.tuned_t_target
        n = config.tuned_n
        sigma_max = config.tuned_max

        # Compute k so that sigma(t_target) = sigma_target
        ratio = sigma_target / sigma_0
        k = (ratio ** (1.0 / n) - 1.0) / t_target

        # Print the computed k for user reference
        print(f"\n[Tuned Power Law] sigma(t) = {sigma_0} * (1 + {k:.4f}*t)^{n}")
        print(f"  Computed k = {k:.4f} so that sigma({t_target}s) = {sigma_target}")
        print(f"  At t=0: sigma = {sigma_0:.2f}")
        print(f"  At t={t_target}s: sigma = {sigma_target:.2f}")
        print(f"  Max sigma cap: {sigma_max:.2f}\n")

        def sigma_func(t: float) -> float:
            sigma = sigma_0 * (1.0 + k * t) ** n
            return min(sigma, sigma_max)
        return sigma_func

    elif mode == 'custom':
        if config.custom_func is None:
            raise ValueError("custom_func must be provided for 'custom' mode")
        return config.custom_func

    else:
        raise ValueError(f"Unknown sigma mode: {mode}")


# =============================================================================
# CUSTOM PISTON VELOCITY MODEL
# =============================================================================

class FlameElongationPistonVelocity:
    """
    Piston velocity model based on flame elongation:

        u_piston = (sigma - 1) * density_ratio * S_L

    This class provides a custom piston velocity function compatible with
    the LagrangianSolver's flame coupling mechanism.

    Parameters:
    -----------
    sigma_func : callable
        Function returning sigma at time t: sigma(t) -> float
    density_ratio : float or None
        Fixed density ratio (rho_u/rho_b). If None, uses provider's value.
    use_provider_density : bool
        If True, get density ratio from flame speed provider.
    """

    def __init__(self,
                 sigma_func: Callable[[float], float],
                 density_ratio: Optional[float] = None,
                 use_provider_density: bool = True):
        self.sigma_func = sigma_func
        self.fixed_density_ratio = density_ratio
        self.use_provider_density = use_provider_density

        # For diagnostics
        self.last_sigma = 1.0
        self.last_density_ratio = 1.0
        self.last_flame_speed = 0.0
        self.last_piston_velocity = 0.0

    def __call__(self, S_L: float, T: float, P: float, rho: float,
                 t: float = 0.0, rho_burned: Optional[float] = None) -> float:
        """
        Compute piston velocity.

        Parameters:
        -----------
        S_L : float
            Laminar flame speed [m/s]
        T : float
            Temperature at piston face [K]
        P : float
            Pressure at piston face [Pa]
        rho : float
            Unburned gas density at piston face [kg/m³]
        t : float
            Current simulation time [s]
        rho_burned : float, optional
            Burned gas density [kg/m³] (from flame speed provider)

        Returns:
        --------
        u_piston : float
            Piston velocity [m/s]
        """
        # Get sigma at current time
        sigma = self.sigma_func(t)
        self.last_sigma = sigma

        # Get density ratio
        if self.use_provider_density and rho_burned is not None:
            density_ratio = rho / rho_burned
        elif self.fixed_density_ratio is not None:
            density_ratio = self.fixed_density_ratio
        else:
            # Fallback: estimate from adiabatic flame
            # T_burned ~ T + 1800K for stoichiometric hydrocarbon
            T_burned = T + 1800.0
            density_ratio = T_burned / T  # Approximate for constant pressure
        self.last_density_ratio = density_ratio

        # Store flame speed for diagnostics
        self.last_flame_speed = S_L

        # Compute piston velocity: u = (sigma - 1) * density_ratio * S_L
        u_piston = (sigma - 1.0) * density_ratio * S_L
        self.last_piston_velocity = u_piston

        return u_piston


def create_flame_elongation_velocity_wrapper(
    sigma_func: Callable[[float], float],
    density_ratio: Optional[float] = None,
    use_provider_density: bool = True
) -> Callable[[float, float, float, float], float]:
    """
    Create a wrapper function for use with LagrangianSolver's custom_piston_velocity_func.

    The solver calls: custom_piston_velocity_func(S_L, T, P, rho) -> u_piston

    This wrapper adds time-dependency by capturing the solver's current_time.

    Note: The solver will need to be modified to pass burned density to the
    custom function, or we use the fixed density ratio approach.
    """
    model = FlameElongationPistonVelocity(
        sigma_func=sigma_func,
        density_ratio=density_ratio,
        use_provider_density=use_provider_density
    )

    # Store reference for diagnostics access
    create_flame_elongation_velocity_wrapper._model = model

    def velocity_func(S_L: float, T: float, P: float, rho: float) -> float:
        # Note: time is not available in the standard callback signature
        # For time-dependent sigma, the solver needs modification
        # For now, use t=0 (constant sigma) or modify solver to pass time
        return model(S_L, T, P, rho, t=0.0, rho_burned=None)

    return velocity_func


# =============================================================================
# FLAME SPEED PROVIDER SETUP
# =============================================================================

def create_flame_speed_provider(mode: str, **kwargs):
    """
    Create a flame speed provider based on mode.

    Parameters:
    -----------
    mode : str
        Provider mode: 'constant', 'tabulated', 'tabulated_csv', 'cantera'
    **kwargs : dict
        Mode-specific parameters

    Returns:
    --------
    provider : FlameSpeedProvider
        The configured flame speed provider
    """
    from src.flame import (
        ConstantFlameSpeed,
        TabularFlameSpeed,
        CSVTabulatedFlameSpeed,
        CanteraFlameSpeed
    )

    mode = mode.lower()

    if mode == 'constant':
        return ConstantFlameSpeed(
            S_L=kwargs.get('S_L', 0.4),
            expansion_ratio=kwargs.get('expansion_ratio', 7.0)
        )

    elif mode == 'tabulated':
        if 'table_file' in kwargs:
            return TabularFlameSpeed.from_file(kwargs['table_file'])
        else:
            raise ValueError("table_file must be provided for 'tabulated' mode")

    elif mode == 'tabulated_csv':
        if 'csv_path' in kwargs:
            return CSVTabulatedFlameSpeed(
                csv_path=kwargs['csv_path'],
                extrapolate=kwargs.get('extrapolate', False),
                verbose=kwargs.get('verbose', True)
            )
        else:
            raise ValueError("csv_path must be provided for 'tabulated_csv' mode")

    elif mode == 'cantera':
        return CanteraFlameSpeed(
            mechanism=kwargs.get('mechanism', 'gri30.yaml'),
            fuel=kwargs.get('fuel', 'CH4'),
            oxidizer=kwargs.get('oxidizer', 'O2:1, N2:3.76'),
            save_flame_profiles=kwargs.get('save_flame_profiles', False),
            flame_profiles_dir=kwargs.get('flame_profiles_dir', None)
        )

    else:
        raise ValueError(f"Unknown flame speed mode: {mode}")


# =============================================================================
# PROBLEM SETUP
# =============================================================================

def setup_flame_elongation_problem(
    domain_length: float = 1.0,
    n_cells: int = 200,
    initial_temperature: float = 300.0,
    initial_pressure: float = 101325.0,
    gamma: float = 1.4,
    R_gas: float = 287.0,
    eos: Optional['EOS'] = None,
    gas_config: Optional[Dict] = None,
    flame_speed_provider = None,
    equivalence_ratio: float = 1.0,
    sigma_func: Callable[[float], float] = None,
    density_ratio: Optional[float] = None,
    use_provider_density: bool = True,
    coupling_max_iterations: int = 10,
    coupling_tolerance: float = 1e-4,
    coupling_relaxation: float = 0.7,
    piston_side: str = 'left',
    right_bc: str = 'open'
) -> Dict:
    """
    Set up flame-elongation piston problem.

    EOS Selection Priority:
    1. Explicit eos parameter if provided
    2. EOS with gas_config (INPUT_PARAMS_CONFIG style) - uses Cantera
    3. EOS with air at STP (fallback with warning)
    4. EOS with ideal gas mode if Cantera not available (fallback with warning)

    Returns:
    --------
    setup : dict
        Dictionary containing all setup parameters and objects
    """
    from src.lagrangian import EOS, PistonVelocityModel

    # Create grid
    x = np.linspace(0, domain_length, n_cells)
    dx = x[1] - x[0]

    # Initial conditions (uniform quiescent gas)
    rho_initial = initial_pressure / (R_gas * initial_temperature)
    rho = rho_initial * np.ones(n_cells)
    u = np.zeros(n_cells)
    p = initial_pressure * np.ones(n_cells)

    # =================================================================
    # EOS Selection Logic: Use unified EOS class with auto-detection
    # =================================================================
    if eos is not None:
        # Priority 1: Explicit EOS provided
        _eos = eos
    elif gas_config is not None:
        # Priority 2: Create EOS from gas_config (uses Cantera if available)
        _eos = EOS(gas_config=gas_config, gamma=gamma, R_gas=R_gas)
        # Update initial conditions from EOS if Cantera is being used
        if _eos.use_cantera and _eos.gas is not None:
            initial_temperature = _eos.gas.T
            initial_pressure = _eos.gas.P
            rho_initial = _eos.gas.density
            gamma = _eos.gamma
            R_gas = _eos.R_gas
            rho = rho_initial * np.ones(n_cells)
            p = initial_pressure * np.ones(n_cells)
    else:
        # Priority 3: Default EOS (auto-detect Cantera with air, or ideal gas)
        _eos = EOS(gamma=gamma, R_gas=R_gas)

    # Create sigma function if not provided
    if sigma_func is None:
        sigma_func = lambda t: 1.0

    # Create custom piston velocity function
    # This implements: u_piston = (sigma - 1) * density_ratio * S_L
    piston_model = FlameElongationPistonVelocity(
        sigma_func=sigma_func,
        density_ratio=density_ratio,
        use_provider_density=use_provider_density
    )

    # Wrapper for solver's extended signature: f(S_L, T, P, rho, rho_b, t)
    def custom_velocity_func(S_L: float, T: float, P: float, rho: float,
                             rho_b: float = None, t: float = 0.0) -> float:
        return piston_model(S_L, T, P, rho, t=t, rho_burned=rho_b)

    # Set boundary conditions
    if piston_side == 'left':
        bc_left = 'piston'
        bc_right = right_bc.lower()
    else:
        bc_left = right_bc.lower()
        bc_right = 'piston'

    return {
        'x': x,
        'dx': dx,
        'n_cells': n_cells,
        'domain_length': domain_length,
        'rho': rho,
        'u': u,
        'p': p,
        'T_initial': initial_temperature,
        'P_initial': initial_pressure,
        'rho_initial': rho_initial,
        'eos': _eos,
        'gamma': gamma,
        'R_gas': R_gas,
        'bc_left': bc_left,
        'bc_right': bc_right,
        'piston_side': piston_side,
        'flame_speed_provider': flame_speed_provider,
        'equivalence_ratio': equivalence_ratio,
        'custom_velocity_func': custom_velocity_func,
        'piston_model': piston_model,
        'sigma_func': sigma_func,
        'coupling_max_iterations': coupling_max_iterations,
        'coupling_tolerance': coupling_tolerance,
        'coupling_relaxation': coupling_relaxation,
    }


# =============================================================================
# SIMULATION RUNNER
# =============================================================================

def _run_with_termination(
    solver,
    t_final: float,
    save_interval: int,
    print_interval: int,
    termination_velocity: float,
    termination_velocity_mode: str,
    flame_provider = None
) -> Dict:
    """
    Run simulation with early termination based on piston velocity.

    This is a manual time-stepping loop that mirrors solver.solve() but
    adds termination checking after each step.
    """
    # Initialize history storage
    history = {
        't': [],
        'x': [],
        'x_centers': [],
        'rho': [],
        'u': [],
        'p': [],
        'E': [],
        'tau': [],
        'T': [],  # Temperature from EOS
        'solver_piston_velocity': [],  # Track actual solver piston velocity
        'solver_flame_speed': [],       # Track actual solver flame speed
    }

    # Save initial state
    def save_state(t):
        history['t'].append(t)
        history['x'].append(solver.x.copy())
        history['x_centers'].append(solver.x_centers.copy())
        history['rho'].append(solver.state.rho.copy())
        history['u'].append(solver.state.u.copy())
        history['p'].append(solver.state.p.copy())
        history['E'].append(solver.state.E.copy())
        history['tau'].append(solver.state.tau.copy())
        # Calculate and save temperature from EOS (correct for mixture)
        T = solver.eos.temperature(solver.state.rho, solver.state.p)
        if T is not None:
            history['T'].append(T.copy())
        else:
            # Fallback: should not happen if EOS is properly configured
            history['T'].append(np.zeros_like(solver.state.rho))
        # Save actual solver values
        history['solver_piston_velocity'].append(
            getattr(solver, '_current_coupled_piston_velocity', 0.0)
        )
        history['solver_flame_speed'].append(
            getattr(solver, '_current_flame_speed', 0.0)
        )

    # Compute initial flame speed before saving initial state
    # Get initial conditions at piston face
    if solver.piston_side == 'left':
        rho_init = solver.state.rho[0]
        p_init = solver.state.p[0]
    else:
        rho_init = solver.state.rho[-1]
        p_init = solver.state.p[-1]
    T_init = solver.eos.temperature(np.array([rho_init]), np.array([p_init]))[0]
    phi = getattr(solver, '_equivalence_ratio', 1.0)

    # Get initial flame speed from provider
    if flame_provider is not None:
        S_L_init = flame_provider.get_flame_speed(T_init, p_init, phi)
        rho_b_init = flame_provider.get_burned_density(T_init, p_init, phi)
        solver._current_flame_speed = S_L_init
        # Compute initial piston velocity
        if hasattr(solver, '_custom_piston_velocity_func') and solver._custom_piston_velocity_func is not None:
            u_piston_init = solver._custom_piston_velocity_func(S_L_init, T_init, p_init, rho_init, rho_b_init, 0.0)
        else:
            u_piston_init = S_L_init  # Fallback
        solver._current_coupled_piston_velocity = u_piston_init

    save_state(0.0)

    t = 0.0
    step = 0
    terminated_early = False
    termination_reason = None

    print(f"\nRunning simulation with velocity termination check...")

    while t < t_final:
        # Compute timestep
        dt = solver._compute_time_step()
        if t + dt > t_final:
            dt = t_final - t

        # Take a step
        solver.step(dt)
        t += dt
        step += 1

        # Get current piston velocity
        u_piston = solver._current_coupled_piston_velocity

        # Check termination condition
        if termination_velocity_mode == 'above':
            if u_piston >= termination_velocity:
                terminated_early = True
                termination_reason = f"u_piston ({u_piston:.2f} m/s) >= threshold ({termination_velocity:.2f} m/s)"
        elif termination_velocity_mode == 'below':
            if u_piston <= termination_velocity:
                terminated_early = True
                termination_reason = f"u_piston ({u_piston:.2f} m/s) <= threshold ({termination_velocity:.2f} m/s)"

        # Save state at intervals
        if step % save_interval == 0:
            save_state(t)

        # Print progress at intervals
        if step % print_interval == 0:
            # Get coupling iterations from solver (last entry in history)
            iter_history = getattr(solver, '_coupling_iterations_history', [])
            coupling_iters = iter_history[-1] if iter_history else 0
            flame_speed = getattr(solver, '_current_flame_speed', 0.0)
            print(f"  Step {step}: t = {t*1000:.4f} ms, u_piston = {u_piston:.2f} m/s, S_L = {flame_speed:.2f} m/s, iters = {coupling_iters}")

            # Trigger flame profile save at output interval (if provider supports it)
            if flame_provider is not None and hasattr(flame_provider, 'trigger_profile_save'):
                flame_provider.trigger_profile_save(step)

        # Check for early termination
        if terminated_early:
            print(f"\n*** EARLY TERMINATION ***")
            print(f"  Reason: {termination_reason}")
            print(f"  Time: t = {t*1000:.4f} ms (step {step})")
            # Save final state
            if step % save_interval != 0:
                save_state(t)
            break

    if not terminated_early:
        # Save final state if not already saved
        if step % save_interval != 0:
            save_state(t)
        print(f"\nSimulation completed: t = {t*1000:.4f} ms ({step} steps)")

    # Add termination info to history
    history['terminated_early'] = terminated_early
    history['termination_reason'] = termination_reason
    history['final_step'] = step

    return history


def run_flame_elongation_simulation(
    t_final: float,
    setup: Dict,
    cfl: float = 0.5,
    save_interval: int = 10,
    print_interval: int = 100,
    output_dir: Optional[str] = None,
    enable_velocity_termination: bool = False,
    termination_velocity: float = 100.0,
    termination_velocity_mode: str = 'above'
) -> Dict:
    """
    Run flame-elongation piston simulation with coupled flame speed.

    Parameters:
    -----------
    t_final : float
        Final simulation time [s]
    setup : dict
        Setup dictionary from setup_flame_elongation_problem()
    cfl : float
        CFL number for timestep
    save_interval : int
        Save solution every N steps
    print_interval : int
        Print progress every N steps
    output_dir : str, optional
        Directory to save outputs
    enable_velocity_termination : bool
        Enable early termination based on piston velocity (default False)
    termination_velocity : float
        Velocity threshold for termination [m/s] (default 100.0)
    termination_velocity_mode : str
        'above': terminate when u_piston >= threshold
        'below': terminate when u_piston <= threshold

    Returns:
    --------
    results : dict
        Simulation results including history and final state
    """
    from src.lagrangian import (
        LagrangianSolver, ArtificialViscosityType, LimiterType, PistonVelocityModel
    )

    print("=" * 70)
    print("FLAME-ELONGATION PISTON SIMULATION")
    print("=" * 70)
    print(f"Domain: [0, {setup['domain_length']}] m")
    print(f"Cells: {setup['n_cells']}")
    print(f"Initial T: {setup['T_initial']:.1f} K")
    print(f"Initial P: {setup['P_initial']/1e5:.2f} bar")
    print(f"Initial rho: {setup['rho_initial']:.4f} kg/m³")
    print(f"Final time: {t_final*1000:.2f} ms")
    print(f"BC Left: {setup['bc_left']}")
    print(f"BC Right: {setup['bc_right']}")
    print(f"Piston side: {setup['piston_side']}")
    print(f"Flame coupling: ENABLED")
    print(f"  Max iterations: {setup['coupling_max_iterations']}")
    print(f"  Tolerance: {setup['coupling_tolerance']}")
    print(f"  Relaxation: {setup['coupling_relaxation']}")

    # Test flame speed provider
    provider = setup['flame_speed_provider']

    # Show Cantera flame speed parameters if using Cantera
    if hasattr(provider, 'mechanism'):
        print(f"\nCantera Flame Speed Configuration:")
        print(f"  Mechanism: {provider.mechanism}")
        print(f"  Fuel: {provider.fuel}")
        print(f"  Oxidizer: {provider.oxidizer}")
        print(f"  Equivalence ratio: {setup['equivalence_ratio']:.2f}")

    S_L_test = provider.get_flame_speed(
        setup['T_initial'], setup['P_initial'], setup['equivalence_ratio']
    )
    rho_b_test = provider.get_burned_density(
        setup['T_initial'], setup['P_initial'], setup['equivalence_ratio']
    )
    print(f"\nFlame Speed Provider Test (local state T={setup['T_initial']:.1f} K, P={setup['P_initial']/1e5:.2f} bar):")
    print(f"  S_L at initial conditions: {S_L_test:.4f} m/s")
    print(f"  Burned density: {rho_b_test:.4f} kg/m³")
    print(f"  Expansion ratio: {setup['rho_initial']/rho_b_test:.2f}")

    # Test piston velocity
    sigma_test = setup['sigma_func'](0.0)
    u_piston_test = setup['custom_velocity_func'](
        S_L_test, setup['T_initial'], setup['P_initial'], setup['rho_initial']
    )
    print(f"\nPiston Velocity Model Test:")
    print(f"  sigma(0): {sigma_test:.2f}")
    print(f"  u_piston = ({sigma_test:.2f} - 1) * {setup['rho_initial']/rho_b_test:.2f} * {S_L_test:.4f}")
    print(f"  u_piston at t=0: {u_piston_test:.4f} m/s")

    if enable_velocity_termination:
        print(f"\nTermination Condition: ENABLED")
        print(f"  Mode: {termination_velocity_mode}")
        print(f"  Threshold: {termination_velocity:.2f} m/s")
    print("=" * 70)

    # Create solver with flame coupling
    # Note: piston_velocity callback is required by solver even with flame coupling
    # We provide a dummy that returns 0 - it will be overridden by flame coupling
    solver = LagrangianSolver(
        eos=setup['eos'],
        cfl=cfl,
        bc_left=setup['bc_left'],
        bc_right=setup['bc_right'],
        av_type=ArtificialViscosityType.NOH_QH,
        av_Cq=1.0,
        av_Cl=0.3,
        av_Ch=0.1,
        limiter=LimiterType.VANLEER,
        use_predictor_corrector=True,
        piston_velocity=lambda t: 0.0,  # Dummy - overridden by flame coupling
        piston_side=setup['piston_side'],
        # Flame coupling parameters
        enable_flame_coupling=True,
        flame_speed_provider=setup['flame_speed_provider'],
        piston_velocity_model=PistonVelocityModel.CUSTOM,
        equivalence_ratio=setup['equivalence_ratio'],
        coupling_max_iterations=setup['coupling_max_iterations'],
        coupling_tolerance=setup['coupling_tolerance'],
        coupling_relaxation=setup['coupling_relaxation'],
        custom_piston_velocity_func=setup['custom_velocity_func'],
    )

    # Initialize solver
    solver.initialize(
        setup['x'],
        setup['rho'],
        setup['u'],
        setup['p']
    )

    # Run simulation with optional early termination
    if enable_velocity_termination:
        # Manual time-stepping loop with termination check
        history = _run_with_termination(
            solver=solver,
            t_final=t_final,
            save_interval=save_interval,
            print_interval=print_interval,
            termination_velocity=termination_velocity,
            termination_velocity_mode=termination_velocity_mode,
            flame_provider=setup.get('flame_speed_provider')
        )
    else:
        # Standard solve
        history = solver.solve(
            t_final=t_final,
            save_interval=save_interval,
            print_interval=print_interval
        )

    # Ensure temperature is in history (calculate from EOS if not present)
    if 'T' not in history or len(history['T']) == 0:
        history['T'] = []
        for i in range(len(history['t'])):
            rho = history['rho'][i]
            p = history['p'][i]
            T = solver.eos.temperature(rho, p)
            if T is not None:
                history['T'].append(T.copy() if hasattr(T, 'copy') else np.array(T))
            else:
                # Fallback using ideal gas with setup R_gas
                R_gas = setup.get('R_gas', 287.0)
                history['T'].append(p / (rho * R_gas))

    # Get coupling diagnostics
    coupling_diag = solver.get_coupling_diagnostics()
    print(f"\nCoupling Diagnostics:")
    print(f"  Average iterations: {coupling_diag['avg_iterations']:.2f}")
    print(f"  Convergence rate: {coupling_diag['convergence_rate']*100:.1f}%")
    print(f"  Final flame speed: {solver._current_flame_speed:.4f} m/s")
    print(f"  Final piston velocity: {solver._current_coupled_piston_velocity:.4f} m/s")

    # Compute flame speed, sigma, and piston velocity history at saved times
    times = np.array(history['t'])
    sigma_func = setup['sigma_func']
    flame_provider = setup['flame_speed_provider']
    phi = setup['equivalence_ratio']

    # Use actual solver values if available (from _run_with_termination)
    if 'solver_piston_velocity' in history and len(history['solver_piston_velocity']) > 0:
        # Use actual solver-computed values
        piston_velocity_history = list(history['solver_piston_velocity'])
        flame_speed_history = list(history['solver_flame_speed'])

        # Still compute sigma and density ratio from state
        sigma_history = []
        density_ratio_history = []
        for i, t in enumerate(times):
            sigma_history.append(sigma_func(t))
            # Compute density ratio from state
            rho_snapshot = history['rho'][i]
            p_snapshot = history['p'][i]
            if setup['piston_side'] == 'left':
                rho_piston = rho_snapshot[0]
                P_piston = p_snapshot[0]
            else:
                rho_piston = rho_snapshot[-1]
                P_piston = p_snapshot[-1]
            R_gas = setup.get('R_gas', 287.0)
            T_piston = P_piston / (rho_piston * R_gas)
            rho_b = flame_provider.get_burned_density(T_piston, P_piston, phi)
            density_ratio = rho_piston / rho_b if rho_b > 1e-10 else 7.0
            density_ratio_history.append(density_ratio)
    else:
        # Fallback: recompute from history (less accurate)
        flame_speed_history = []
        sigma_history = []
        piston_velocity_history = []
        density_ratio_history = []

        for i, t in enumerate(times):
            rho_snapshot = history['rho'][i]
            p_snapshot = history['p'][i]

            if setup['piston_side'] == 'left':
                rho_piston = rho_snapshot[0]
                P_piston = p_snapshot[0]
            else:
                rho_piston = rho_snapshot[-1]
                P_piston = p_snapshot[-1]

            R_gas = setup.get('R_gas', 287.0)
            T_piston = P_piston / (rho_piston * R_gas)

            S_L = flame_provider.get_flame_speed(T_piston, P_piston, phi)
            rho_b = flame_provider.get_burned_density(T_piston, P_piston, phi)

            sigma = sigma_func(t)
            density_ratio = rho_piston / rho_b if rho_b > 1e-10 else 7.0
            u_piston = (sigma - 1.0) * density_ratio * S_L

            flame_speed_history.append(S_L)
            sigma_history.append(sigma)
            piston_velocity_history.append(u_piston)
            density_ratio_history.append(density_ratio)

    # Package results
    results = {
        'solver': solver,
        'history': history,
        'setup': setup,
        't_final': t_final,
        'actual_t_final': times[-1] if len(times) > 0 else 0.0,
        'diagnostics': solver.get_diagnostics(),
        'coupling_diagnostics': coupling_diag,
        'piston_model': setup['piston_model'],
        # Flame-elongation specific outputs
        'times': times,
        'flame_speed_history': np.array(flame_speed_history),
        'sigma_history': np.array(sigma_history),
        'piston_velocity_history': np.array(piston_velocity_history),
        'density_ratio_history': np.array(density_ratio_history),
        # Termination info
        'terminated_early': history.get('terminated_early', False),
        'termination_reason': history.get('termination_reason', None),
    }

    # Save outputs if requested
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save final state
        np.savez(
            output_path / 'final_state.npz',
            x=solver.x_centers,
            rho=solver.state.rho,
            u=solver.state.u,
            p=solver.state.p,
            E=solver.state.E,
            tau=solver.state.tau,
            m=solver.m_centers
        )

        # Save flame speed vs elongation history
        np.savez(
            output_path / 'flame_elongation_history.npz',
            times=times,
            flame_speed=np.array(flame_speed_history),
            sigma=np.array(sigma_history),
            piston_velocity=np.array(piston_velocity_history),
            density_ratio=np.array(density_ratio_history)
        )

        print(f"\nResults saved to: {output_path}")

    return results


# =============================================================================
# PLOTTING
# =============================================================================

def plot_flame_elongation_results(results: Dict, save_path: Optional[str] = None):
    """Create visualization of flame-elongation simulation results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    solver = results['solver']
    history = results['history']
    setup = results['setup']
    coupling_diag = results['coupling_diagnostics']

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))

    # Density
    ax = axes[0, 0]
    ax.plot(solver.x_centers, solver.state.rho, 'b-', lw=2)
    ax.axhline(setup['rho_initial'], color='k', ls='--', alpha=0.5, label='Initial')
    ax.set_xlabel('Position x [m]')
    ax.set_ylabel('Density [kg/m³]')
    ax.set_title('Density Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Velocity
    ax = axes[0, 1]
    ax.plot(solver.x_centers, solver.state.u, 'r-', lw=2)
    ax.set_xlabel('Position x [m]')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_title('Velocity Profile')
    ax.grid(True, alpha=0.3)

    # Pressure
    ax = axes[1, 0]
    ax.plot(solver.x_centers, solver.state.p / 1e5, 'g-', lw=2)
    ax.axhline(setup['P_initial']/1e5, color='k', ls='--', alpha=0.5, label='Initial')
    ax.set_xlabel('Position x [m]')
    ax.set_ylabel('Pressure [bar]')
    ax.set_title('Pressure Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Temperature
    ax = axes[1, 1]
    T = setup['eos'].temperature(solver.state.rho, solver.state.p)
    if T is not None:
        ax.plot(solver.x_centers, T, 'm-', lw=2)
        ax.axhline(setup['T_initial'], color='k', ls='--', alpha=0.5, label='Initial')
        ax.set_ylabel('Temperature [K]')
        ax.legend()
    else:
        ax.plot(solver.x_centers, solver.state.e, 'c-', lw=2)
        ax.set_ylabel('Internal Energy [J/kg]')
    ax.set_xlabel('Position x [m]')
    ax.set_title('Temperature/Energy Profile')
    ax.grid(True, alpha=0.3)

    # Entropy
    ax = axes[2, 0]
    eos = setup['eos']
    if hasattr(eos, 'entropy'):
        s = eos.entropy(solver.state.rho, solver.state.p)
        # Compute initial entropy for reference
        s_initial = eos.entropy(
            np.array([setup['rho_initial']]),
            np.array([setup['P_initial']])
        )[0]
        ax.plot(solver.x_centers, s, 'c-', lw=2)
        ax.axhline(s_initial, color='k', ls='--', alpha=0.5, label='Initial')
        ax.set_ylabel('Entropy [J/(kg·K)]')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Entropy not available', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Position x [m]')
    ax.set_title('Entropy Profile')
    ax.grid(True, alpha=0.3)

    # Piston position over time
    ax = axes[2, 1]
    times = np.array(history['t'])
    piston_positions = [h[0] for h in history['x']]
    ax.plot(times * 1000, piston_positions, 'b-', lw=2)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Piston Position [m]')
    ax.set_title('Piston Position vs Time')
    ax.grid(True, alpha=0.3)

    # Coupling iterations
    ax = axes[3, 0]
    iterations = coupling_diag['iterations']
    if iterations:
        ax.plot(range(len(iterations)), iterations, 'g-', lw=1, marker='.', markersize=2)
        ax.axhline(coupling_diag['avg_iterations'], color='r', ls='--',
                   label=f'Average: {coupling_diag["avg_iterations"]:.2f}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Coupling Iterations')
        ax.set_title(f'Coupling Convergence ({coupling_diag["convergence_rate"]*100:.1f}% converged)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Internal Energy profile
    ax = axes[3, 1]
    ax.plot(solver.x_centers, solver.state.e / 1e3, 'orange', lw=2)
    e_initial = setup['P_initial'] / ((setup.get('gamma', 1.4) - 1.0) * setup['rho_initial'])
    ax.axhline(e_initial / 1e3, color='k', ls='--', alpha=0.5, label='Initial')
    ax.set_xlabel('Position x [m]')
    ax.set_ylabel('Internal Energy [kJ/kg]')
    ax.set_title('Internal Energy Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Flame-Elongation Piston Simulation at t = {results['t_final']*1000:.2f} ms\n"
                 f"u_piston = (sigma - 1) * (rho_u/rho_b) * S_L",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.savefig('flame_elongation_results.png', dpi=150, bbox_inches='tight')
        print("Figure saved to: flame_elongation_results.png")

    plt.close()


def plot_flame_speed_vs_elongation(results: Dict, save_path: Optional[str] = None):
    """
    Plot flame speed vs elongation (sigma) and related quantities.

    Creates a figure with:
    - Flame speed vs sigma
    - Flame speed vs time
    - Sigma vs time
    - Piston velocity vs time
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    times = results['times']
    S_L = results['flame_speed_history']
    sigma = results['sigma_history']
    u_piston = results['piston_velocity_history']
    density_ratio = results['density_ratio_history']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Flame speed vs sigma
    ax = axes[0, 0]
    sc = ax.scatter(sigma, S_L, c=times*1000, cmap='viridis', s=20, alpha=0.8)
    ax.set_xlabel('Elongation Factor (sigma)')
    ax.set_ylabel('Flame Speed S_L [m/s]')
    ax.set_title('Flame Speed vs Elongation')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Time [ms]')
    ax.grid(True, alpha=0.3)

    # Flame speed vs time
    ax = axes[0, 1]
    ax.plot(times * 1000, S_L, 'b-', lw=2, marker='o', markersize=3)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Flame Speed S_L [m/s]')
    ax.set_title('Flame Speed vs Time')
    ax.grid(True, alpha=0.3)

    # Sigma and density ratio vs time
    ax = axes[1, 0]
    ax.plot(times * 1000, sigma, 'g-', lw=2, label='sigma')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Sigma', color='g')
    ax.tick_params(axis='y', labelcolor='g')
    ax.set_title('Elongation Factor & Density Ratio vs Time')
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(times * 1000, density_ratio, 'r--', lw=2, label='rho_u/rho_b')
    ax2.set_ylabel('Density Ratio', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # Piston velocity vs time
    ax = axes[1, 1]
    ax.plot(times * 1000, u_piston, 'm-', lw=2, marker='o', markersize=3)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Piston Velocity [m/s]')
    ax.set_title('Piston Velocity vs Time')
    ax.grid(True, alpha=0.3)

    # Add formula annotation
    ax.annotate(r'$u_{piston} = (\sigma - 1) \cdot \frac{\rho_u}{\rho_b} \cdot S_L$',
                xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Flame Speed vs Elongation Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Flame speed vs elongation plot saved to: {save_path}")
    else:
        plt.savefig('flame_speed_vs_elongation.png', dpi=150, bbox_inches='tight')
        print("Flame speed vs elongation plot saved to: flame_speed_vs_elongation.png")

    plt.close()


def plot_xt_diagram(
        results: Dict,
        variable: str = 'pressure',
        save_path: Optional[str] = None,
        use_log: bool = False
) -> None:
    """
    Create x-t surface plot of simulation history.

    Parameters:
    -----------
    results : dict
        Results from run_flame_elongation_simulation()
    variable : str
        Variable to plot: 'pressure', 'density', 'velocity', 'energy', 'temperature', 'entropy'
    save_path : str, optional
        Path to save figure
    use_log : bool
        Use logarithmic color scale
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    history = results['history']
    setup = results['setup']
    times = np.array(history['t'])

    # Get R_gas for temperature calculation
    R_gas = setup.get('R_gas', 287.0)

    # Collect data for scatter plot
    x_values = []
    t_values = []
    var_values = []

    for i, t in enumerate(times):
        x_centers = history['x_centers'][i]

        if variable == 'velocity':
            u = history['u'][i]
            # Average to cell centers if interface values
            if len(u) == len(x_centers) + 1:
                vals = 0.5 * (u[:-1] + u[1:])
            else:
                vals = u
        elif variable == 'density':
            vals = history['rho'][i]
        elif variable == 'pressure':
            vals = history['p'][i]
        elif variable == 'energy':
            vals = history['E'][i]
        elif variable == 'temperature':
            # Use stored temperature from history (calculated via EOS during simulation)
            if 'T' in history and len(history['T']) > i:
                vals = history['T'][i]
            else:
                # Fallback: calculate from ideal gas law (may be inaccurate for non-air mixtures)
                rho = np.array(history['rho'][i])
                p = np.array(history['p'][i])
                vals = p / (rho * R_gas)
        elif variable == 'entropy':
            # Use stored entropy from history or compute from EOS
            if 's' in history and len(history['s']) > i:
                vals = history['s'][i]
            else:
                # Compute entropy from EOS
                eos = setup['eos']
                rho = np.array(history['rho'][i])
                p = np.array(history['p'][i])
                if hasattr(eos, 'entropy'):
                    vals = eos.entropy(rho, p)
                else:
                    # Fallback to ideal gas entropy formula
                    gamma = setup.get('gamma', 1.4)
                    p_ref = 101325.0
                    rho_ref = 1.0
                    c_v = R_gas / (gamma - 1.0)
                    vals = c_v * (np.log(p / p_ref) - gamma * np.log(rho / rho_ref))
        else:
            raise ValueError(f"Unknown variable: {variable}")

        for j in range(len(x_centers)):
            x_values.append(x_centers[j])
            t_values.append(t)
            var_values.append(vals[j])

    x_values = np.asarray(x_values)
    t_values = np.asarray(t_values)
    var_values = np.asarray(var_values)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    if use_log and np.all(var_values > 0):
        sc = ax.scatter(x_values, t_values * 1e3, c=var_values, cmap='rainbow',
                        s=2, alpha=0.8, norm=LogNorm())
    else:
        sc = ax.scatter(x_values, t_values * 1e3, c=var_values, cmap='rainbow',
                        s=2, alpha=0.8)

    cbar = plt.colorbar(sc, ax=ax)

    # Set labels based on variable
    labels = {
        'pressure': ('Pressure [Pa]', 'Pressure'),
        'density': ('Density [kg/m³]', 'Density'),
        'velocity': ('Velocity [m/s]', 'Velocity'),
        'energy': ('Internal Energy [J/kg]', 'Internal Energy'),
        'temperature': ('Temperature [K]', 'Temperature'),
        'entropy': ('Entropy [J/(kg·K)]', 'Entropy'),
    }
    ylabel, title = labels.get(variable, (variable, variable))
    cbar.set_label(ylabel)

    ax.set_xlabel('Position x [m]')
    ax.set_ylabel('Time [ms]')
    ax.set_title(f'x-t Diagram: {title}')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"x-t diagram ({variable}) saved to: {save_path}")
    else:
        filename = f'xt_diagram_{variable}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"x-t diagram saved to: {filename}")

    plt.close()


def plot_all_xt_diagrams(results: Dict, output_dir: Optional[Path] = None) -> None:
    """
    Generate all x-t diagrams for pressure, density, velocity, energy, temperature, and entropy.

    Parameters:
    -----------
    results : dict
        Results from run_flame_elongation_simulation()
    output_dir : Path, optional
        Directory to save plots
    """
    variables = ['pressure', 'density', 'velocity', 'energy', 'temperature', 'entropy']

    for var in variables:
        if output_dir:
            save_path = str(output_dir / f'xt_diagram_{var}.png')
        else:
            save_path = None

        try:
            plot_xt_diagram(results, variable=var, save_path=save_path)
        except Exception as e:
            print(f"Warning: Failed to create x-t diagram for {var}: {e}")


def plot_tabulated_flame_speed_surface(flame_provider, save_path: Optional[str] = None) -> None:
    """
    Plot 3D surface of tabulated flame speed data (T vs P vs Su).

    Parameters:
    -----------
    flame_provider : CSVTabulatedFlameSpeed
        The flame speed provider with loaded tabulated data
    save_path : str, optional
        Path to save the figure
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Check if provider has the required attributes
    if not hasattr(flame_provider, 'Su_grid'):
        print("Warning: Flame provider does not have tabulated data for surface plot")
        return

    T_values = flame_provider.T_values
    P_values = flame_provider.P_values
    Su_grid = flame_provider.Su_grid

    # Create meshgrid for plotting
    T_mesh, P_mesh = np.meshgrid(T_values, P_values / 1e5, indexing='ij')  # P in bar

    # Create figure with multiple views
    fig = plt.figure(figsize=(16, 12))

    # 3D Surface plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf = ax1.plot_surface(T_mesh, P_mesh, Su_grid, cmap='viridis',
                            edgecolor='none', alpha=0.9)
    ax1.set_xlabel('Temperature [K]')
    ax1.set_ylabel('Pressure [bar]')
    ax1.set_zlabel('Flame Speed [m/s]')
    ax1.set_title('Laminar Flame Speed Surface')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='Su [m/s]')

    # Contour plot (top view)
    ax2 = fig.add_subplot(2, 2, 2)
    # Mask NaN values for contour plot
    Su_masked = np.ma.masked_invalid(Su_grid)
    levels = np.linspace(np.nanmin(Su_grid), np.nanmax(Su_grid), 20)
    contour = ax2.contourf(T_mesh, P_mesh, Su_masked, levels=levels, cmap='viridis')
    ax2.set_xlabel('Temperature [K]')
    ax2.set_ylabel('Pressure [bar]')
    ax2.set_title('Flame Speed Contours')
    fig.colorbar(contour, ax=ax2, label='Su [m/s]')

    # Flame speed vs Temperature at different pressures
    ax3 = fig.add_subplot(2, 2, 3)
    n_curves = min(5, len(P_values))
    P_indices = np.linspace(0, len(P_values) - 1, n_curves, dtype=int)
    colors = plt.cm.coolwarm(np.linspace(0, 1, n_curves))
    for i, idx in enumerate(P_indices):
        P_bar = P_values[idx] / 1e5
        Su_slice = Su_grid[:, idx]
        valid = ~np.isnan(Su_slice)
        if np.any(valid):
            ax3.plot(T_values[valid], Su_slice[valid], '-o', color=colors[i],
                     label=f'P = {P_bar:.1f} bar', markersize=4)
    ax3.set_xlabel('Temperature [K]')
    ax3.set_ylabel('Flame Speed [m/s]')
    ax3.set_title('Flame Speed vs Temperature')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # Flame speed vs Pressure at different temperatures
    ax4 = fig.add_subplot(2, 2, 4)
    n_curves = min(5, len(T_values))
    T_indices = np.linspace(0, len(T_values) - 1, n_curves, dtype=int)
    colors = plt.cm.coolwarm(np.linspace(0, 1, n_curves))
    for i, idx in enumerate(T_indices):
        T_K = T_values[idx]
        Su_slice = Su_grid[idx, :]
        valid = ~np.isnan(Su_slice)
        if np.any(valid):
            ax4.plot(P_values[valid] / 1e5, Su_slice[valid], '-o', color=colors[i],
                     label=f'T = {T_K:.0f} K', markersize=4)
    ax4.set_xlabel('Pressure [bar]')
    ax4.set_ylabel('Flame Speed [m/s]')
    ax4.set_title('Flame Speed vs Pressure')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    # Get diagnostics for title
    diag = flame_provider.get_diagnostics()
    plt.suptitle(f"Tabulated Flame Speed Data\n"
                 f"T: {diag['T_range'][0]:.0f}-{diag['T_range'][1]:.0f} K, "
                 f"P: {diag['P_range'][0]/1e5:.1f}-{diag['P_range'][1]/1e5:.1f} bar, "
                 f"phi = {diag['phi']:.2f}\n"
                 f"Valid points: {diag['n_valid']}/{diag['n_T']*diag['n_P']}",
                 fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Tabulated flame speed surface plot saved to: {save_path}")
    else:
        plt.savefig('tabulated_flame_speed_surface.png', dpi=150, bbox_inches='tight')
        print("Tabulated flame speed surface plot saved to: tabulated_flame_speed_surface.png")

    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function to run flame-elongation piston simulation."""
    # Get script directory for relative paths
    script_dir = Path(__file__).parent

    # Resolve output directory
    output_dir = script_dir / OUTPUT_DIR if OUTPUT_DIR else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create sigma function
    sigma_config = SigmaConfig(
        mode=SIGMA_MODE,
        value=SIGMA_VALUE,
        initial=SIGMA_INITIAL,
        rate=SIGMA_RATE,
        max_value=SIGMA_MAX,
        power_law_initial=SIGMA_POWER_LAW_INITIAL,
        power_law_k=SIGMA_POWER_LAW_K,
        power_law_n=SIGMA_POWER_LAW_N,
        power_law_max=SIGMA_POWER_LAW_MAX,
        tuned_initial=SIGMA_TUNED_INITIAL,
        tuned_target=SIGMA_TUNED_TARGET,
        tuned_t_target=SIGMA_TUNED_T_TARGET,
        tuned_n=SIGMA_TUNED_N,
        tuned_max=SIGMA_TUNED_MAX
    )
    sigma_func = create_sigma_function(sigma_config)

    # Set up initial conditions
    if USE_INPUT_PARAMS:
        # Temporarily add CLTORC to path to import problem_setup
        sys.path.insert(0, str(CLTORC_PATH))

        # Import CLTORC's problem_setup
        import src.problem_setup as cltorc_problem_setup
        initialize_parameters = cltorc_problem_setup.initialize_parameters
        input_params = cltorc_problem_setup.input_params

        # Remove CLTORC from path and clean up module cache
        sys.path.remove(str(CLTORC_PATH))
        modules_to_remove = [key for key in sys.modules.keys() if key == 'src' or key.startswith('src.')]
        for mod in modules_to_remove:
            del sys.modules[mod]

        # Resolve mechanism path relative to script directory
        mech_path = INPUT_PARAMS_CONFIG.get('mech', 'gri30.yaml')
        if not os.path.isabs(mech_path):
            mech_path = str(script_dir / mech_path)

        # Initialize with config
        initialize_parameters(
            T=INPUT_PARAMS_CONFIG['T'],
            P=INPUT_PARAMS_CONFIG['P'],
            Phi=INPUT_PARAMS_CONFIG['Phi'],
            Fuel=INPUT_PARAMS_CONFIG['Fuel'],
            mech=mech_path,
            temperature_unit=INPUT_PARAMS_CONFIG.get('temperature_unit', 'K'),
            pressure_unit=INPUT_PARAMS_CONFIG.get('pressure_unit', 'Pa'),
            phi_convention=INPUT_PARAMS_CONFIG.get('phi_convention', 'scale_fuel'),
            use_dry_air=INPUT_PARAMS_CONFIG.get('use_dry_air', False),
        )

        # Build gas and extract properties
        gas = input_params.build_gas()
        initial_temperature = gas.T
        initial_pressure = gas.P
        initial_density = gas.density
        gamma = gas.cp / gas.cv
        # Calculate R_gas from Cantera gas properties (R_universal / M)
        R_gas_mixture = 8314.0 / gas.mean_molecular_weight  # J/(kg·K)

        print(f"\nUsing CLTORC input_params:")
        print(f"  T = {gas.T:.2f} K, P = {gas.P/1e5:.2f} bar")
        print(f"  rho = {gas.density:.4f} kg/m³, gamma = {gamma:.4f}")
        print(f"  R_gas = {R_gas_mixture:.2f} J/(kg·K), M = {gas.mean_molecular_weight:.2f} g/mol")
        print(f"  Composition: {input_params.X}")

        # Use Cantera EOS
        use_cantera_eos = True
        cantera_mechanism = mech_path
        cantera_composition = input_params.X
    else:
        # Use simple ideal gas
        initial_temperature = INITIAL_TEMPERATURE
        initial_pressure = INITIAL_PRESSURE
        initial_density = initial_pressure / (R_GAS * initial_temperature)
        gamma = GAMMA
        R_gas_mixture = R_GAS  # Use configured R_GAS for simple ideal gas
        use_cantera_eos = False
        cantera_mechanism = None
        cantera_composition = None

    # Create flame speed provider
    if FLAME_SPEED_MODE == 'constant':
        flame_provider = create_flame_speed_provider(
            'constant',
            S_L=CONSTANT_FLAME_SPEED,
            expansion_ratio=CONSTANT_EXPANSION_RATIO
        )
    elif FLAME_SPEED_MODE == 'tabulated_csv':
        # Resolve CSV path relative to script directory
        csv_path = TABULATED_CSV_PATH
        if not os.path.isabs(csv_path):
            csv_path = str(script_dir / csv_path)

        flame_provider = create_flame_speed_provider(
            'tabulated_csv',
            csv_path=csv_path,
            extrapolate=False,
            verbose=True
        )

        # Plot the tabulated flame speed surface
        if SAVE_PLOTS:
            surface_plot_path = str(output_dir / 'tabulated_flame_speed_surface.png') if output_dir else None
            plot_tabulated_flame_speed_surface(flame_provider, save_path=surface_plot_path)
    elif FLAME_SPEED_MODE == 'cantera':
        # Cantera mode requires INPUT_PARAMS_CONFIG for fuel, oxidizer, and mechanism
        if not use_cantera_eos:
            raise ValueError("FLAME_SPEED_MODE='cantera' requires USE_INPUT_PARAMS=True")

        # Configure flame profile saving directory
        flame_profiles_dir = None
        if SAVE_FLAME_PROFILES:
            flame_profiles_dir = str(output_dir / FLAME_PROFILES_SUBDIR) if output_dir else FLAME_PROFILES_SUBDIR

        flame_provider = create_flame_speed_provider(
            'cantera',
            mechanism=cantera_mechanism,
            fuel=INPUT_PARAMS_CONFIG['Fuel'],
            oxidizer=INPUT_PARAMS_CONFIG.get('Oxidizer', 'O2'),
            save_flame_profiles=SAVE_FLAME_PROFILES,
            flame_profiles_dir=flame_profiles_dir
        )
    else:
        raise ValueError(f"Unknown flame speed mode: {FLAME_SPEED_MODE}")

    # Use equivalence ratio from INPUT_PARAMS_CONFIG when available
    equivalence_ratio = INPUT_PARAMS_CONFIG['Phi'] if use_cantera_eos else EQUIVALENCE_RATIO

    # Prepare gas_config for EOS creation (uses INPUT_PARAMS_CONFIG when available)
    gas_config_for_eos = None
    if use_cantera_eos:
        # Pass the resolved mechanism path and composition to setup
        gas_config_for_eos = {
            'mech': cantera_mechanism,
            'T': initial_temperature,
            'P': initial_pressure,
            'Fuel': INPUT_PARAMS_CONFIG.get('Fuel'),
            'Oxidizer': INPUT_PARAMS_CONFIG.get('Oxidizer', 'O2:1, N2:3.76'),
            'Phi': INPUT_PARAMS_CONFIG.get('Phi', 1.0),
            'composition': cantera_composition,  # Pre-built composition string
        }

    # Set up problem (EOS is created internally using gas_config or defaults)
    setup = setup_flame_elongation_problem(
        domain_length=DOMAIN_LENGTH,
        n_cells=N_CELLS,
        initial_temperature=initial_temperature,
        initial_pressure=initial_pressure,
        gamma=gamma,
        R_gas=R_gas_mixture,
        gas_config=gas_config_for_eos,  # Pass gas_config for Cantera EOS
        flame_speed_provider=flame_provider,
        equivalence_ratio=equivalence_ratio,
        sigma_func=sigma_func,
        density_ratio=DENSITY_RATIO if not USE_PROVIDER_DENSITY else None,
        use_provider_density=USE_PROVIDER_DENSITY,
        coupling_max_iterations=COUPLING_MAX_ITERATIONS,
        coupling_tolerance=COUPLING_TOLERANCE,
        coupling_relaxation=COUPLING_RELAXATION,
        piston_side=PISTON_SIDE,
        right_bc=RIGHT_BC
    )

    # Report EOS backend being used
    if setup['eos'].use_cantera:
        print(f"Using Cantera backend with {cantera_mechanism if cantera_mechanism else 'air.yaml'}")
    else:
        print("Using ideal gas backend")

    # Run simulation
    results = run_flame_elongation_simulation(
        t_final=T_FINAL,
        setup=setup,
        cfl=CFL,
        save_interval=SAVE_INTERVAL,
        print_interval=PRINT_INTERVAL,
        output_dir=str(output_dir) if output_dir else None,
        enable_velocity_termination=ENABLE_VELOCITY_TERMINATION,
        termination_velocity=TERMINATION_VELOCITY,
        termination_velocity_mode=TERMINATION_VELOCITY_MODE
    )

    # Plot results
    if SAVE_PLOTS:
        plot_path = None
        if output_dir:
            plot_path = str(output_dir / 'flame_elongation_results.png')
        plot_flame_elongation_results(results, save_path=plot_path)

        # Plot flame speed vs elongation
        elongation_plot_path = None
        if output_dir:
            elongation_plot_path = str(output_dir / 'flame_speed_vs_elongation.png')
        plot_flame_speed_vs_elongation(results, save_path=elongation_plot_path)

        # Plot x-t diagrams if enabled
        if PLOT_XT_DIAGRAMS:
            print("\nGenerating x-t diagrams...")
            plot_all_xt_diagrams(results, output_dir=output_dir)

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
