"""
Clavin Flame-Elongation Piston Simulation
==========================================

This script simulates flame-driven piston dynamics using the full activation energy
correlation from Tofaili & Clavin (2021):

    S_L = S_L0 * (T_b/T_b0)^2 * (T_u/T_u0)^(3/2) * exp( Ea/(2R) * (1/T_b0 - 1/T_b) )

Key features:
    - Initial flame speed (S_L0) computed using Cantera FreeFlame
    - Burned gas temperature (T_b0) from the same FreeFlame solution
    - Reference unburned temperature (T_u0) stored for prefactor calculation
    - Activation energy (Ea) computed using inert dilution method (burning flux)
    - T_b at current conditions from Cantera equilibrate('HP')
    - Temperature prefactors: (T_b/T_b0)^2 and (T_u/T_u0)^(3/2)

The piston velocity is determined by the flame elongation formula:
    u_piston = (sigma - 1) * density_ratio * S_L

References:
    - Tofaili & Clavin, Combustion and Flame 232 (2021) 111522
    - Law, Combustion Physics (textbook)

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
from typing import Callable, Optional, Dict
from dataclasses import dataclass
import warnings

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# --- Domain Parameters ---
DOMAIN_LENGTH = 1.6           # Domain length [m]
N_CELLS = 400                 # Number of computational cells

# --- Initial Conditions (via CLTORC input_params) ---
INPUT_PARAMS_CONFIG = {
    'T': 503,                    # Temperature [K]
    'P': 10e5,                 # Pressure [Pa]
    'Phi': 1.0,                  # Equivalence ratio
    'Fuel': 'H2',                # Fuel species
    'Oxidizer': 'O2',            # Oxidizer species
    'mech': '../../chemical_mechanism/LiDryer.yaml',  # Cantera mechanism file
    'temperature_unit': 'K',
    'pressure_unit': 'Pa',
    'phi_convention': 'scale_fuel',
    'use_dry_air': False,
}

# --- Activation Energy Flame Speed Configuration ---
# Inert dilution method for activation energy calculation
ACTIVATION_ENERGY_N_DILUTION_POINTS = 3   # Number of dilution points for Arrhenius fit
ACTIVATION_ENERGY_DILUTION_RANGE = (0.0, 0.2)  # Range of additional inert moles
ACTIVATION_ENERGY_INERT_SPECIES = 'N2'    # Inert species for dilution

# --- Sigma (Flame Elongation) Configuration ---
# Options: 'constant', 'linear'
SIGMA_MODE = 'linear'
SIGMA_INITIAL = 1.0          # Initial sigma value
SIGMA_RATE = 1000.0          # Rate of sigma increase [1/s]
SIGMA_MAX = 20.0             # Maximum sigma value

# --- Coupling Parameters ---
COUPLING_MAX_ITERATIONS = 100
COUPLING_TOLERANCE = 5e-3
COUPLING_RELAXATION = 0.7

# --- Simulation Parameters ---
T_FINAL = 0.1                # Final simulation time [s]
CFL = 0.4
PISTON_SIDE = 'left'
RIGHT_BC = 'open'

# --- Termination Conditions ---
ENABLE_VELOCITY_TERMINATION = True
TERMINATION_VELOCITY = 1500.0
TERMINATION_VELOCITY_MODE = 'above'

# --- Output Settings ---
OUTPUT_DIR = 'clavin_flame_results'
SAVE_INTERVAL = 10
PRINT_INTERVAL = 50
SAVE_PLOTS = True
PLOT_XT_DIAGRAMS = True
PLOT_ARRHENIUS = True        # Plot Arrhenius fit used for Ea

# =============================================================================
# END OF USER CONFIGURATION
# =============================================================================


@dataclass
class SigmaConfig:
    """Configuration for sigma (flame elongation) parameter."""
    mode: str = 'constant'
    value: float = 1.0
    initial: float = 1.0
    rate: float = 0.0
    max_value: float = 10.0
    custom_func: Optional[Callable[[float], float]] = None


def create_sigma_function(config: SigmaConfig) -> Callable[[float], float]:
    """Create a sigma function from configuration."""
    mode = config.mode.lower()

    if mode == 'constant':
        sigma_val = config.value
        return lambda t: sigma_val

    elif mode == 'linear':
        sigma_0 = config.initial
        rate = config.rate
        sigma_max = config.max_value
        return lambda t: min(sigma_0 + rate * t, sigma_max)

    elif mode == 'custom':
        if config.custom_func is None:
            raise ValueError("custom_func must be provided for 'custom' mode")
        return config.custom_func

    else:
        raise ValueError(f"Unknown sigma mode: {mode}")


class FlameElongationPistonVelocity:
    """
    Piston velocity model: u_piston = (sigma - 1) * density_ratio * S_L
    """

    def __init__(self,
                 sigma_func: Callable[[float], float],
                 use_provider_density: bool = True):
        self.sigma_func = sigma_func
        self.use_provider_density = use_provider_density
        self.last_sigma = 1.0
        self.last_density_ratio = 1.0
        self.last_flame_speed = 0.0
        self.last_piston_velocity = 0.0

    def __call__(self, S_L: float, T: float, P: float, rho: float,
                 t: float = 0.0, rho_burned: Optional[float] = None) -> float:
        sigma = self.sigma_func(t)
        self.last_sigma = sigma

        if self.use_provider_density and rho_burned is not None:
            density_ratio = rho / rho_burned
        else:
            T_burned = T + 1800.0
            density_ratio = T_burned / T
        self.last_density_ratio = density_ratio
        self.last_flame_speed = S_L

        u_piston = (sigma - 1.0) * density_ratio * S_L
        self.last_piston_velocity = u_piston

        return u_piston


def _run_with_termination(
    solver,
    t_final: float,
    save_interval: int,
    print_interval: int,
    termination_velocity: float,
    termination_velocity_mode: str,
    flame_provider=None,
    piston_side: str = 'left'
) -> Dict:
    """Run simulation with early termination based on piston velocity."""
    history = {
        't': [], 'x': [], 'x_centers': [], 'rho': [], 'u': [], 'p': [],
        'E': [], 'tau': [], 'T': [],
        'solver_piston_velocity': [], 'solver_flame_speed': [],
    }

    def save_state(t):
        history['t'].append(t)
        history['x'].append(solver.x.copy())
        history['x_centers'].append(solver.x_centers.copy())
        history['rho'].append(solver.state.rho.copy())
        history['u'].append(solver.state.u.copy())
        history['p'].append(solver.state.p.copy())
        history['E'].append(solver.state.E.copy())
        history['tau'].append(solver.state.tau.copy())
        T = solver.eos.temperature(solver.state.rho, solver.state.p)
        if T is not None:
            history['T'].append(T.copy())
        else:
            history['T'].append(np.zeros_like(solver.state.rho))
        history['solver_piston_velocity'].append(
            getattr(solver, '_current_coupled_piston_velocity', 0.0))
        history['solver_flame_speed'].append(
            getattr(solver, '_current_flame_speed', 0.0))

    # Compute initial flame speed at t=0 before saving first state
    if flame_provider is not None:
        # Get initial conditions at piston face
        if piston_side == 'left':
            rho_init = solver.state.rho[0]
            p_init = solver.state.p[0]
        else:
            rho_init = solver.state.rho[-1]
            p_init = solver.state.p[-1]
        T_init = solver.eos.temperature(np.array([rho_init]), np.array([p_init]))[0]
        phi = getattr(solver, 'equivalence_ratio', 1.0)
        S_L_init = flame_provider.get_flame_speed(T_init, p_init, phi)
        solver._current_flame_speed = S_L_init
        solver._current_coupled_piston_velocity = 0.0  # At t=0, piston hasn't moved yet

    save_state(0.0)

    t = 0.0
    step = 0
    terminated_early = False
    termination_reason = None

    print(f"\nRunning simulation with velocity termination check...")

    while t < t_final:
        dt = solver._compute_time_step()
        if t + dt > t_final:
            dt = t_final - t

        solver.step(dt)
        t += dt
        step += 1

        u_piston = solver._current_coupled_piston_velocity

        if termination_velocity_mode == 'above':
            if u_piston >= termination_velocity:
                terminated_early = True
                termination_reason = f"u_piston ({u_piston:.2f} m/s) >= threshold ({termination_velocity:.2f} m/s)"
        elif termination_velocity_mode == 'below':
            if u_piston <= termination_velocity:
                terminated_early = True
                termination_reason = f"u_piston ({u_piston:.2f} m/s) <= threshold ({termination_velocity:.2f} m/s)"

        if step % save_interval == 0:
            save_state(t)

        if step % print_interval == 0:
            iter_history = getattr(solver, '_coupling_iterations_history', [])
            coupling_iters = iter_history[-1] if iter_history else 0
            flame_speed = getattr(solver, '_current_flame_speed', 0.0)
            print(f"  Step {step}: t = {t*1000:.4f} ms, u_piston = {u_piston:.2f} m/s, "
                  f"S_L = {flame_speed:.2f} m/s, iters = {coupling_iters}")

        if terminated_early:
            print(f"\n*** EARLY TERMINATION ***")
            print(f"  Reason: {termination_reason}")
            print(f"  Time: t = {t*1000:.4f} ms (step {step})")
            if step % save_interval != 0:
                save_state(t)
            break

    if not terminated_early:
        if step % save_interval != 0:
            save_state(t)
        print(f"\nSimulation completed: t = {t*1000:.4f} ms ({step} steps)")

    history['terminated_early'] = terminated_early
    history['termination_reason'] = termination_reason
    history['final_step'] = step

    return history


def plot_results(results: Dict, output_dir: Path):
    """Create visualization of simulation results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    solver = results['solver']
    history = results['history']
    setup = results['setup']
    coupling_diag = results['coupling_diagnostics']
    flame_provider = results['flame_provider']

    # Get Ea diagnostics
    ea_diag = flame_provider.get_diagnostics()

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

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
    ax.set_xlabel('Position x [m]')
    ax.set_title('Temperature Profile')
    ax.grid(True, alpha=0.3)

    # Piston position over time
    ax = axes[2, 0]
    times = np.array(history['t'])
    piston_positions = [h[0] for h in history['x']]
    ax.plot(times * 1000, piston_positions, 'b-', lw=2)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Piston Position [m]')
    ax.set_title('Piston Position vs Time')
    ax.grid(True, alpha=0.3)

    # Coupling iterations
    ax = axes[2, 1]
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

    plt.suptitle(f"Clavin Flame-Elongation Simulation at t = {results['actual_t_final']*1000:.2f} ms\n"
                 f"Ea = {ea_diag['Ea_kJ_mol']:.2f} kJ/mol, S_L0 = {ea_diag['S_L0']:.4f} m/s, "
                 f"T_b0 = {ea_diag['T_b0']:.0f} K",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / 'clavin_simulation_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    plt.close()


def plot_flame_speed_comparison(results: Dict, output_dir: Path):
    """Compare activation energy flame speed with history."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    times = results['times']
    S_L = results['flame_speed_history']
    u_piston = results['piston_velocity_history']
    sigma = results['sigma_history']
    density_ratio = results['density_ratio_history']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Flame speed vs time
    ax = axes[0, 0]
    ax.plot(times * 1000, S_L, 'b-', lw=2, marker='o', markersize=3)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Flame Speed S_L [m/s]')
    ax.set_title('Flame Speed vs Time (Activation Energy Method)')
    ax.grid(True, alpha=0.3)

    # Piston velocity vs time
    ax = axes[0, 1]
    ax.plot(times * 1000, u_piston, 'm-', lw=2, marker='o', markersize=3)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Piston Velocity [m/s]')
    ax.set_title('Piston Velocity vs Time')
    ax.grid(True, alpha=0.3)

    # Sigma and density ratio
    ax = axes[1, 0]
    ax.plot(times * 1000, sigma, 'g-', lw=2, label='sigma')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Sigma', color='g')
    ax.tick_params(axis='y', labelcolor='g')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(times * 1000, density_ratio, 'r--', lw=2, label='rho_u/rho_b')
    ax2.set_ylabel('Density Ratio', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.set_title('Elongation Factor & Density Ratio vs Time')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # Flame speed vs sigma
    ax = axes[1, 1]
    sc = ax.scatter(sigma, S_L, c=times*1000, cmap='viridis', s=30, alpha=0.8)
    ax.set_xlabel('Elongation Factor (sigma)')
    ax.set_ylabel('Flame Speed S_L [m/s]')
    ax.set_title('Flame Speed vs Elongation')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Time [ms]')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Flame Speed vs Elongation Analysis (Clavin Method)', fontsize=12, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / 'clavin_flame_speed_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Flame speed analysis saved to: {save_path}")
    plt.close()


def plot_xt_diagram(results: Dict, variable: str, output_dir: Path):
    """Create x-t surface plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    history = results['history']
    times = np.array(history['t'])

    x_values = []
    t_values = []
    var_values = []

    for i, t in enumerate(times):
        x_centers = history['x_centers'][i]

        if variable == 'velocity':
            u = history['u'][i]
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
            if 'T' in history and len(history['T']) > i:
                vals = history['T'][i]
            else:
                continue
        else:
            continue

        for j in range(len(x_centers)):
            x_values.append(x_centers[j])
            t_values.append(t)
            var_values.append(vals[j])

    x_values = np.asarray(x_values)
    t_values = np.asarray(t_values)
    var_values = np.asarray(var_values)

    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(x_values, t_values * 1e3, c=var_values, cmap='rainbow', s=2, alpha=0.8)
    cbar = plt.colorbar(sc, ax=ax)

    labels = {
        'pressure': ('Pressure [Pa]', 'Pressure'),
        'density': ('Density [kg/m³]', 'Density'),
        'velocity': ('Velocity [m/s]', 'Velocity'),
        'energy': ('Internal Energy [J/kg]', 'Internal Energy'),
        'temperature': ('Temperature [K]', 'Temperature'),
    }
    ylabel, title = labels.get(variable, (variable, variable))
    cbar.set_label(ylabel)

    ax.set_xlabel('Position x [m]')
    ax.set_ylabel('Time [ms]')
    ax.set_title(f'x-t Diagram: {title} (Clavin Method)')
    plt.tight_layout()

    save_path = output_dir / f'xt_diagram_{variable}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"x-t diagram ({variable}) saved to: {save_path}")
    plt.close()


def main():
    """Main function to run Clavin flame-elongation piston simulation."""
    script_dir = Path(__file__).parent
    output_dir = script_dir / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create sigma function
    sigma_config = SigmaConfig(
        mode=SIGMA_MODE,
        initial=SIGMA_INITIAL,
        rate=SIGMA_RATE,
        max_value=SIGMA_MAX
    )
    sigma_func = create_sigma_function(sigma_config)

    # Initialize from CLTORC - must import BEFORE ConsL's src module
    # Remove any 'src' modules from cache before importing CLTORC's src
    modules_to_remove = [key for key in list(sys.modules.keys()) if key == 'src' or key.startswith('src.')]
    for mod in modules_to_remove:
        del sys.modules[mod]

    sys.path.insert(0, str(CLTORC_PATH))
    import src.problem_setup as cltorc_problem_setup
    initialize_parameters = cltorc_problem_setup.initialize_parameters
    input_params = cltorc_problem_setup.input_params

    # Clean up CLTORC modules and restore path
    sys.path.remove(str(CLTORC_PATH))
    modules_to_remove = [key for key in list(sys.modules.keys()) if key == 'src' or key.startswith('src.')]
    for mod in modules_to_remove:
        del sys.modules[mod]

    # Now import ConsL modules (after CLTORC cleanup)
    from src import (
        LagrangianSolver, CanteraEOS, IdealGasEOS, ActivationEnergyFlameSpeed,
        ArtificialViscosityType, LimiterType, PistonVelocityModel
    )

    # Resolve mechanism path
    mech_path = INPUT_PARAMS_CONFIG.get('mech', 'gri30.yaml')
    if not os.path.isabs(mech_path):
        mech_path = str(script_dir / mech_path)

    # Initialize parameters
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
    R_gas_mixture = 8314.0 / gas.mean_molecular_weight

    print(f"\n{'='*70}")
    print("CLAVIN FLAME-ELONGATION PISTON SIMULATION")
    print(f"{'='*70}")
    print(f"\nInitial Conditions (from CLTORC):")
    print(f"  T = {gas.T:.2f} K, P = {gas.P/1e5:.2f} bar")
    print(f"  rho = {gas.density:.4f} kg/m³, gamma = {gamma:.4f}")
    print(f"  R_gas = {R_gas_mixture:.2f} J/(kg·K), M = {gas.mean_molecular_weight:.2f} g/mol")
    print(f"  Composition: {input_params.X}")

    # Create ActivationEnergyFlameSpeed provider
    print(f"\n{'='*70}")
    print("CREATING ACTIVATION ENERGY FLAME SPEED PROVIDER")
    print(f"{'='*70}")

    flame_provider = ActivationEnergyFlameSpeed(
        mechanism=mech_path,
        fuel=INPUT_PARAMS_CONFIG['Fuel'],
        oxidizer=INPUT_PARAMS_CONFIG.get('Oxidizer', 'O2'),
        T0=initial_temperature,
        P0=initial_pressure,
        phi0=INPUT_PARAMS_CONFIG['Phi'],
        n_dilution_points=ACTIVATION_ENERGY_N_DILUTION_POINTS,
        dilution_range=ACTIVATION_ENERGY_DILUTION_RANGE,
        inert_species=ACTIVATION_ENERGY_INERT_SPECIES,
    )

    # Plot Arrhenius fit
    if PLOT_ARRHENIUS:
        arrhenius_path = str(output_dir / 'arrhenius_fit.png')
        flame_provider.plot_arrhenius(save_path=arrhenius_path)

    # Create grid and initial conditions
    x = np.linspace(0, DOMAIN_LENGTH, N_CELLS)
    rho = initial_density * np.ones(N_CELLS)
    u = np.zeros(N_CELLS)
    p = initial_pressure * np.ones(N_CELLS)

    # Create piston velocity model
    piston_model = FlameElongationPistonVelocity(
        sigma_func=sigma_func,
        use_provider_density=True
    )

    def custom_velocity_func(S_L, T, P, rho, rho_b=None, t=0.0):
        return piston_model(S_L, T, P, rho, t=t, rho_burned=rho_b)

    # Create Cantera EOS
    import cantera as ct

    try:
        eos_gas = ct.Solution(mech_path)
        eos_gas.TPX = initial_temperature, initial_pressure, input_params.X
        eos = CanteraEOS(eos_gas, composition=input_params.X)
        print(f"\nUsing Cantera EOS with {mech_path}")
    except Exception as e:
        warnings.warn(f"Failed to create Cantera EOS: {e}, using ideal gas EOS")
        eos = IdealGasEOS(gamma=gamma, R_gas=R_gas_mixture)

    # Create setup dict
    setup = {
        'x': x,
        'n_cells': N_CELLS,
        'domain_length': DOMAIN_LENGTH,
        'rho': rho,
        'u': u,
        'p': p,
        'T_initial': initial_temperature,
        'P_initial': initial_pressure,
        'rho_initial': initial_density,
        'eos': eos,
        'gamma': gamma,
        'R_gas': R_gas_mixture,
        'bc_left': 'piston',
        'bc_right': RIGHT_BC,
        'piston_side': PISTON_SIDE,
        'equivalence_ratio': INPUT_PARAMS_CONFIG['Phi'],
    }

    # Create solver
    print(f"\n{'='*70}")
    print("RUNNING SIMULATION")
    print(f"{'='*70}")
    print(f"Domain: [0, {DOMAIN_LENGTH}] m, {N_CELLS} cells")
    print(f"Final time: {T_FINAL*1000:.2f} ms")
    print(f"Sigma mode: {SIGMA_MODE} (initial={SIGMA_INITIAL}, rate={SIGMA_RATE}, max={SIGMA_MAX})")

    solver = LagrangianSolver(
        eos=eos,
        cfl=CFL,
        bc_left='piston',
        bc_right=RIGHT_BC,
        av_type=ArtificialViscosityType.NOH_QH,
        av_Cq=1.0,
        av_Cl=0.3,
        av_Ch=0.1,
        limiter=LimiterType.VANLEER,
        use_predictor_corrector=True,
        piston_velocity=lambda t: 0.0,
        piston_side=PISTON_SIDE,
        enable_flame_coupling=True,
        flame_speed_provider=flame_provider,
        piston_velocity_model=PistonVelocityModel.CUSTOM,
        equivalence_ratio=INPUT_PARAMS_CONFIG['Phi'],
        coupling_max_iterations=COUPLING_MAX_ITERATIONS,
        coupling_tolerance=COUPLING_TOLERANCE,
        coupling_relaxation=COUPLING_RELAXATION,
        custom_piston_velocity_func=custom_velocity_func,
    )

    solver.initialize(x, rho, u, p)

    # Run simulation
    history = _run_with_termination(
        solver=solver,
        t_final=T_FINAL,
        save_interval=SAVE_INTERVAL,
        print_interval=PRINT_INTERVAL,
        termination_velocity=TERMINATION_VELOCITY,
        termination_velocity_mode=TERMINATION_VELOCITY_MODE,
        flame_provider=flame_provider,
        piston_side=PISTON_SIDE
    )

    # Ensure temperature in history
    if 'T' not in history or len(history['T']) == 0:
        history['T'] = []
        for i in range(len(history['t'])):
            T = solver.eos.temperature(history['rho'][i], history['p'][i])
            if T is not None:
                history['T'].append(T.copy())
            else:
                history['T'].append(history['p'][i] / (history['rho'][i] * R_gas_mixture))

    # Get coupling diagnostics
    coupling_diag = solver.get_coupling_diagnostics()
    print(f"\nCoupling Diagnostics:")
    print(f"  Average iterations: {coupling_diag['avg_iterations']:.2f}")
    print(f"  Convergence rate: {coupling_diag['convergence_rate']*100:.1f}%")
    print(f"  Final flame speed: {solver._current_flame_speed:.4f} m/s")
    print(f"  Final piston velocity: {solver._current_coupled_piston_velocity:.4f} m/s")

    # Compute histories
    times = np.array(history['t'])
    piston_velocity_history = list(history['solver_piston_velocity'])
    flame_speed_history = list(history['solver_flame_speed'])
    sigma_history = [sigma_func(t) for t in times]
    density_ratio_history = []

    phi = INPUT_PARAMS_CONFIG['Phi']
    for i in range(len(times)):
        rho_snapshot = history['rho'][i]
        p_snapshot = history['p'][i]
        if PISTON_SIDE == 'left':
            rho_piston = rho_snapshot[0]
            P_piston = p_snapshot[0]
        else:
            rho_piston = rho_snapshot[-1]
            P_piston = p_snapshot[-1]
        T_piston = P_piston / (rho_piston * R_gas_mixture)
        rho_b = flame_provider.get_burned_density(T_piston, P_piston, phi)
        density_ratio = rho_piston / rho_b if rho_b > 1e-10 else 7.0
        density_ratio_history.append(density_ratio)

    # Package results
    results = {
        'solver': solver,
        'history': history,
        'setup': setup,
        't_final': T_FINAL,
        'actual_t_final': times[-1] if len(times) > 0 else 0.0,
        'coupling_diagnostics': coupling_diag,
        'flame_provider': flame_provider,
        'piston_model': piston_model,
        'times': times,
        'flame_speed_history': np.array(flame_speed_history),
        'sigma_history': np.array(sigma_history),
        'piston_velocity_history': np.array(piston_velocity_history),
        'density_ratio_history': np.array(density_ratio_history),
        'terminated_early': history.get('terminated_early', False),
        'termination_reason': history.get('termination_reason', None),
    }

    # Save outputs
    np.savez(
        output_dir / 'final_state.npz',
        x=solver.x_centers,
        rho=solver.state.rho,
        u=solver.state.u,
        p=solver.state.p,
        E=solver.state.E,
        tau=solver.state.tau,
        m=solver.m_centers
    )

    ea_diag = flame_provider.get_diagnostics()
    np.savez(
        output_dir / 'activation_energy_data.npz',
        S_L0=ea_diag['S_L0'],
        T_b0=ea_diag['T_b0'],
        T_u0=ea_diag['T_u0'],
        rho_b0=ea_diag['rho_b0'],
        Ea=ea_diag['Ea'],
        Ea_kJ_mol=ea_diag['Ea_kJ_mol'],
        inert_species=ea_diag['inert_species'],
        dilution_amounts=ea_diag['arrhenius_data']['dilution_amounts'],
        T_ad_values=ea_diag['arrhenius_data']['T_ad_values'],
        S_L_values=ea_diag['arrhenius_data']['S_L_values'],
        burning_flux_values=ea_diag['arrhenius_data']['burning_flux_values'],
        R_squared=ea_diag['arrhenius_data']['R_squared'],
    )

    np.savez(
        output_dir / 'flame_elongation_history.npz',
        times=times,
        flame_speed=np.array(flame_speed_history),
        sigma=np.array(sigma_history),
        piston_velocity=np.array(piston_velocity_history),
        density_ratio=np.array(density_ratio_history)
    )

    print(f"\nResults saved to: {output_dir}")

    # Plot results
    if SAVE_PLOTS:
        print("\nGenerating plots...")
        plot_results(results, output_dir)
        plot_flame_speed_comparison(results, output_dir)

        if PLOT_XT_DIAGRAMS:
            for var in ['pressure', 'density', 'velocity', 'energy', 'temperature']:
                try:
                    plot_xt_diagram(results, var, output_dir)
                except Exception as e:
                    print(f"Warning: Failed to create x-t diagram for {var}: {e}")

    print(f"\n{'='*70}")
    print("SIMULATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nActivation Energy Method Summary (Clavin Formulation):")
    print(f"  Ea = {ea_diag['Ea_kJ_mol']:.2f} kJ/mol")
    print(f"  S_L0 = {ea_diag['S_L0']:.4f} m/s")
    print(f"  T_b0 = {ea_diag['T_b0']:.1f} K (adiabatic flame temperature)")
    print(f"  T_u0 = {ea_diag['T_u0']:.1f} K (reference unburned temperature)")
    print(f"  Inert dilution: {ea_diag['inert_species']} (range {ea_diag['dilution_range']})")
    print(f"  Arrhenius R² = {ea_diag['arrhenius_data']['R_squared']:.4f}")

    return results


if __name__ == "__main__":
    main()
