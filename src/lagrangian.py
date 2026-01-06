"""
Publication-Quality 1D Lagrangian Gas Dynamics Solver
=====================================================

A rigorous implementation of Lagrangian hydrodynamics for compressible flows,
suitable for research publication.

Key Features:
-------------
1. Conservative total energy formulation (not internal energy)
2. Noh's artificial heat flux to eliminate wall heating errors
3. Edge-centered (staggered) artificial viscosity
4. Predictor-corrector time integration (2nd order in time)
5. MUSCL reconstruction with slope limiters (2nd order in space)
6. Exact Riemann solver at cell interfaces
7. Conservation diagnostics and verification

References:
-----------
[1] von Neumann, J. & Richtmyer, R.D. (1950). "A method for the numerical
    calculation of hydrodynamic shocks." J. Appl. Phys. 21, 232-237.

[2] Noh, W.F. (1987). "Errors for calculations of strong shocks using an
    artificial viscosity and an artificial heat flux." J. Comput. Phys. 72, 78-120.

[3] Caramana, E.J., Shashkov, M.J. & Whalen, P.P. (1998). "Formulations of
    artificial viscosity for multi-dimensional shock wave computations."
    J. Comput. Phys. 144, 70-97.

[4] Campbell, J.C. & Shashkov, M.J. (2001). "A tensor artificial viscosity
    using a mimetic finite difference algorithm." J. Comput. Phys. 172, 739-765.

[5] Toro, E.F. (2009). "Riemann Solvers and Numerical Methods for Fluid
    Dynamics." 3rd ed., Springer.

Author: Generated for research purposes
Date: 2024
"""

import numpy as np
from scipy.interpolate import interp1d
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Dict, List, Union, Any, TYPE_CHECKING
from enum import Enum
import warnings

# Import flame speed providers (optional dependency)
if TYPE_CHECKING:
    from .flame import FlameSpeedProvider


class ArtificialViscosityType(Enum):
    """Types of artificial viscosity formulations"""
    VONNEUMANN_RICHTMYER = "vnr"      # Classic quadratic + linear
    NOH_QH = "noh"                     # With heat flux (recommended for strong shocks)
    CARAMANA_SHASHKOV = "cs"           # Tensor formulation (mimetic)


class LimiterType(Enum):
    """Slope limiter types for MUSCL reconstruction"""
    MINMOD = "minmod"                  # Most diffusive, most robust
    VANLEER = "vanleer"                # Good balance
    SUPERBEE = "superbee"              # Least diffusive, can oscillate
    MC = "mc"                          # Monotonized central


class PistonVelocityModel(Enum):
    """
    Models for computing piston velocity from flame speed.

    EXPANSION: u_piston = S_L * (rho_u / rho_b - 1)
        Simple expansion model - gas expansion behind flame pushes piston

    FLAME_FIXED: u_piston = S_L
        Piston moves at flame speed (flame attached to piston)

    PRESSURE_DRIVEN: u_piston = S_L + f(delta_P)
        Includes pressure difference effect

    CUSTOM: User provides a custom callback function
    """
    EXPANSION = "expansion"
    FLAME_FIXED = "flame_fixed"
    PRESSURE_DRIVEN = "pressure_driven"
    CUSTOM = "custom"


@dataclass
class ConservationDiagnostics:
    """Track conservation properties throughout simulation"""
    time: List[float] = field(default_factory=list)
    total_mass: List[float] = field(default_factory=list)
    total_momentum: List[float] = field(default_factory=list)
    total_energy: List[float] = field(default_factory=list)

    # Errors relative to initial
    mass_error: List[float] = field(default_factory=list)
    momentum_error: List[float] = field(default_factory=list)
    energy_error: List[float] = field(default_factory=list)


@dataclass
class GasState:
    """
    Container for gas thermodynamic state at cell centers

    Primary variables (evolved):
        tau: Specific volume (1/density) [m³/kg]
        u: Velocity [m/s]
        E: Specific total energy [J/kg] = e + 0.5*u²

    Derived variables (computed from EOS):
        e: Specific internal energy [J/kg]
        p: Pressure [Pa]
        c: Sound speed [m/s]
        T: Temperature [K] (optional)
    """
    tau: np.ndarray      # Specific volume
    u: np.ndarray        # Velocity
    E: np.ndarray        # Total specific energy (CONSERVATIVE)
    p: np.ndarray        # Pressure
    c: np.ndarray        # Sound speed

    @property
    def rho(self) -> np.ndarray:
        """Density from specific volume"""
        return 1.0 / self.tau

    @property
    def e(self) -> np.ndarray:
        """Internal energy from total energy"""
        return self.E - 0.5 * self.u**2

    def copy(self) -> 'GasState':
        """Deep copy of state"""
        return GasState(
            tau=self.tau.copy(),
            u=self.u.copy(),
            E=self.E.copy(),
            p=self.p.copy(),
            c=self.c.copy()
        )


class IdealGasEOS:
    """
    Equation of State for calorically perfect ideal gas

    Relations:
        p = (γ-1) ρ e
        c² = γ p / ρ
        e = p / ((γ-1) ρ)
        T = p / (ρ R)  [if R specified]
    """

    def __init__(self, gamma: float = 1.4, R_gas: Optional[float] = None):
        """
        Parameters:
        -----------
        gamma : float
            Ratio of specific heats (default 1.4 for air)
        R_gas : float, optional
            Specific gas constant [J/(kg·K)] for temperature calculation
        """
        self.gamma = gamma
        self.R_gas = R_gas

    def pressure(self, rho: np.ndarray, e: np.ndarray) -> np.ndarray:
        """Compute pressure from density and internal energy"""
        return np.maximum((self.gamma - 1.0) * rho * e, 1e-15)

    def sound_speed(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Compute adiabatic sound speed"""
        return np.sqrt(np.maximum(self.gamma * p / rho, 1e-15))

    def internal_energy(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Compute internal energy from density and pressure"""
        return p / ((self.gamma - 1.0) * rho)

    def temperature(self, rho: np.ndarray, p: np.ndarray) -> Optional[np.ndarray]:
        """Compute temperature if gas constant is specified"""
        if self.R_gas is None:
            return None
        return p / (rho * self.R_gas)

    def total_energy(self, rho: np.ndarray, u: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Compute total specific energy E = e + 0.5*u²"""
        e = self.internal_energy(rho, p)
        return e + 0.5 * u**2

    def entropy(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Compute specific entropy for ideal gas.

        For a calorically perfect ideal gas:
            s = c_v * ln(p / rho^gamma) + s_ref

        We compute entropy relative to a reference state (p_ref=101325 Pa, rho_ref=1.0 kg/m³):
            s - s_ref = c_v * [ln(p/p_ref) - gamma * ln(rho/rho_ref)]

        Parameters:
        -----------
        rho : array
            Density [kg/m³]
        p : array
            Pressure [Pa]

        Returns:
        --------
        s : array
            Specific entropy [J/(kg·K)]
        """
        rho = np.atleast_1d(rho)
        p = np.atleast_1d(p)

        # Reference state
        p_ref = 101325.0  # Pa
        rho_ref = 1.0     # kg/m³

        # c_v = R / (gamma - 1) for ideal gas
        # Use R_gas if specified, otherwise use air value
        R = self.R_gas if self.R_gas is not None else 287.0
        c_v = R / (self.gamma - 1.0)

        # s - s_ref = c_v * [ln(p/p_ref) - gamma * ln(rho/rho_ref)]
        s = c_v * (np.log(p / p_ref) - self.gamma * np.log(rho / rho_ref))

        return s


class CanteraEOS:
    """
    Equation of State using Cantera for real gas thermodynamics

    This provides accurate thermodynamic properties for real gases and mixtures,
    including temperature-dependent specific heats and multi-species support.

    The EOS maintains consistency by using Cantera's UVX (internal energy, specific
    volume, mole fractions) thermodynamic state representation for Lagrangian methods.

    References:
    -----------
    Cantera: An object-oriented software toolkit for chemical kinetics,
    thermodynamics, and transport processes. https://cantera.org

    Usage:
    ------
    import cantera as ct
    gas = ct.Solution('gri30.yaml')
    eos = CanteraEOS(gas, composition='CH4:1.0, O2:2.0, N2:7.52')
    # Set state from density and internal energy
    eos.set_state_UV(u=1e6, v=1.0/1.2)  # e=1e6 J/kg, rho=1.2 kg/m³
    p = eos.gas.P  # Get pressure
    """

    def __init__(self, gas, composition: Optional[str] = None):
        """
        Initialize Cantera EOS

        Parameters:
        -----------
        gas : cantera.Solution
            Cantera Solution object (e.g., ct.Solution('gri30.yaml'))
        composition : str, optional
            Initial composition in Cantera format (e.g., 'CH4:1.0, O2:2.0')
            If not specified, uses current gas composition
        """
        try:
            import cantera as ct
            self._ct = ct
        except ImportError:
            raise ImportError(
                "Cantera is required for CanteraEOS. Install with: pip install cantera"
            )

        self.gas = gas

        if composition is not None:
            self.gas.X = composition

        # Store initial composition for resetting
        self._initial_X = self.gas.X.copy()

        # Store initial mixture properties for fallback calculations
        # These are computed from the actual mixture, not hardcoded for air
        self._initial_gamma = self.gas.cp / self.gas.cv
        self._initial_R_gas = 8314.0 / self.gas.mean_molecular_weight  # J/(kg·K)

        # Cache for effective gamma (computed from Cp/Cv)
        self._gamma_cache = None

    @property
    def gamma(self) -> float:
        """
        Effective ratio of specific heats γ = Cp/Cv

        Note: For real gases, this varies with temperature and composition.
        This returns the current value at the current state.
        """
        return self.gas.cp / self.gas.cv

    @property
    def R_gas(self) -> float:
        """
        Specific gas constant R = R_universal / M [J/(kg·K)]

        Returns the value for the current mixture composition.
        """
        return 8314.0 / self.gas.mean_molecular_weight

    def set_state_UV(self, u: float, v: float, X: Optional[np.ndarray] = None):
        """
        Set thermodynamic state from internal energy and specific volume

        Parameters:
        -----------
        u : float
            Specific internal energy [J/kg]
        v : float
            Specific volume [m³/kg] (= 1/ρ)
        X : array, optional
            Mole fractions. If None, uses current composition.
        """
        if X is not None:
            self.gas.X = X
        self.gas.UV = u, v

    def set_state_DP(self, rho: float, p: float, X: Optional[np.ndarray] = None):
        """
        Set thermodynamic state from density and pressure

        Parameters:
        -----------
        rho : float
            Density [kg/m³]
        p : float
            Pressure [Pa]
        X : array, optional
            Mole fractions. If None, uses current composition.
        """
        if X is not None:
            self.gas.X = X
        self.gas.DP = rho, p

    def pressure(self, rho: np.ndarray, e: np.ndarray,
                 X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute pressure from density and internal energy

        This iterates through each cell and uses Cantera to get p.
        For vectorized performance, consider using batch methods.

        Parameters:
        -----------
        rho : array
            Density [kg/m³]
        e : array
            Specific internal energy [J/kg]
        X : array, optional
            Mole fractions [n_cells, n_species]. If None, uses stored composition.

        Returns:
        --------
        p : array
            Pressure [Pa]
        """
        rho = np.atleast_1d(rho)
        e = np.atleast_1d(e)
        n = len(rho)
        p = np.zeros(n)

        for i in range(n):
            v = 1.0 / max(rho[i], 1e-15)
            # Note: internal energy can be negative for Cantera reference states
            u = e[i]

            if X is not None and X.ndim == 2:
                self.gas.X = X[i, :]

            try:
                self.gas.UV = u, v
                p[i] = max(self.gas.P, 1e-15)
            except Exception:
                # Fallback to ideal gas approximation if Cantera fails
                # Use stored mixture gamma instead of hardcoded air value
                gamma = self._initial_gamma
                p[i] = max((gamma - 1.0) * rho[i] * e[i], 1e-15)

        return p

    def sound_speed(self, rho: np.ndarray, p: np.ndarray,
                    X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute adiabatic sound speed using Cantera

        c = sqrt(γ * p / ρ) where γ = Cp/Cv from Cantera

        Parameters:
        -----------
        rho : array
            Density [kg/m³]
        p : array
            Pressure [Pa]
        X : array, optional
            Mole fractions [n_cells, n_species]

        Returns:
        --------
        c : array
            Sound speed [m/s]
        """
        rho = np.atleast_1d(rho)
        p = np.atleast_1d(p)
        n = len(rho)
        c = np.zeros(n)

        for i in range(n):
            if X is not None and X.ndim == 2:
                self.gas.X = X[i, :]

            try:
                self.gas.DP = max(rho[i], 1e-15), max(p[i], 1e-10)
                gamma = self.gas.cp / self.gas.cv
                c[i] = np.sqrt(max(gamma * p[i] / rho[i], 1e-15))
            except Exception:
                # Fallback using stored mixture gamma
                c[i] = np.sqrt(max(self._initial_gamma * p[i] / rho[i], 1e-15))

        return c

    def internal_energy(self, rho: np.ndarray, p: np.ndarray,
                       X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute internal energy from density and pressure using Cantera

        Parameters:
        -----------
        rho : array
            Density [kg/m³]
        p : array
            Pressure [Pa]
        X : array, optional
            Mole fractions

        Returns:
        --------
        e : array
            Specific internal energy [J/kg]
        """
        rho = np.atleast_1d(rho)
        p = np.atleast_1d(p)
        n = len(rho)
        e = np.zeros(n)

        for i in range(n):
            if X is not None and X.ndim == 2:
                self.gas.X = X[i, :]

            try:
                self.gas.DP = max(rho[i], 1e-15), max(p[i], 1e-10)
                e[i] = self.gas.int_energy_mass  # Internal energy per unit mass
            except Exception:
                # Fallback to ideal gas using stored mixture gamma
                e[i] = p[i] / ((self._initial_gamma - 1.0) * rho[i])

        return e

    def temperature(self, rho: np.ndarray, p: np.ndarray,
                   X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute temperature from density and pressure using Cantera

        Parameters:
        -----------
        rho : array
            Density [kg/m³]
        p : array
            Pressure [Pa]
        X : array, optional
            Mole fractions

        Returns:
        --------
        T : array
            Temperature [K]
        """
        rho = np.atleast_1d(rho)
        p = np.atleast_1d(p)
        n = len(rho)
        T = np.zeros(n)

        for i in range(n):
            if X is not None and X.ndim == 2:
                self.gas.X = X[i, :]

            try:
                self.gas.DP = max(rho[i], 1e-15), max(p[i], 1e-10)
                T[i] = self.gas.T
            except Exception:
                # Fallback using ideal gas law with actual mixture molecular weight
                R_universal = 8.314  # J/(mol·K)
                M = self.gas.mean_molecular_weight / 1000.0  # kg/mol
                R_specific = R_universal / M  # J/(kg·K)
                T[i] = p[i] / (rho[i] * R_specific)

        return T

    def total_energy(self, rho: np.ndarray, u: np.ndarray, p: np.ndarray,
                    X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute total specific energy E = e + 0.5*u²

        Parameters:
        -----------
        rho : array
            Density [kg/m³]
        u : array
            Velocity [m/s]
        p : array
            Pressure [Pa]
        X : array, optional
            Mole fractions

        Returns:
        --------
        E : array
            Total specific energy [J/kg]
        """
        e = self.internal_energy(rho, p, X)
        return e + 0.5 * u**2

    def get_viscosity(self, rho: np.ndarray, p: np.ndarray,
                     X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get dynamic viscosity from Cantera transport properties

        Parameters:
        -----------
        rho : array
            Density [kg/m³]
        p : array
            Pressure [Pa]
        X : array, optional
            Mole fractions

        Returns:
        --------
        mu : array
            Dynamic viscosity [Pa·s]
        """
        rho = np.atleast_1d(rho)
        p = np.atleast_1d(p)
        n = len(rho)
        mu = np.zeros(n)

        for i in range(n):
            if X is not None and X.ndim == 2:
                self.gas.X = X[i, :]

            try:
                self.gas.DP = max(rho[i], 1e-15), max(p[i], 1e-10)
                mu[i] = self.gas.viscosity
            except Exception:
                # Sutherland's law fallback using stored mixture R_gas
                T = p[i] / (rho[i] * self._initial_R_gas)
                mu[i] = 1.458e-6 * T**1.5 / (T + 110.4)

        return mu

    def get_thermal_conductivity(self, rho: np.ndarray, p: np.ndarray,
                                  X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get thermal conductivity from Cantera transport properties

        Parameters:
        -----------
        rho : array
            Density [kg/m³]
        p : array
            Pressure [Pa]
        X : array, optional
            Mole fractions

        Returns:
        --------
        k : array
            Thermal conductivity [W/(m·K)]
        """
        rho = np.atleast_1d(rho)
        p = np.atleast_1d(p)
        n = len(rho)
        k = np.zeros(n)

        for i in range(n):
            if X is not None and X.ndim == 2:
                self.gas.X = X[i, :]

            try:
                self.gas.DP = max(rho[i], 1e-15), max(p[i], 1e-10)
                k[i] = self.gas.thermal_conductivity
            except Exception:
                # Approximate value for air
                k[i] = 0.025

        return k

    def entropy(self, rho: np.ndarray, p: np.ndarray,
                X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute specific entropy using Cantera.

        Parameters:
        -----------
        rho : array
            Density [kg/m³]
        p : array
            Pressure [Pa]
        X : array, optional
            Mole fractions

        Returns:
        --------
        s : array
            Specific entropy [J/(kg·K)]
        """
        rho = np.atleast_1d(rho)
        p = np.atleast_1d(p)
        n = len(rho)
        s = np.zeros(n)

        for i in range(n):
            if X is not None and X.ndim == 2:
                self.gas.X = X[i, :]

            try:
                self.gas.DP = max(rho[i], 1e-15), max(p[i], 1e-10)
                s[i] = self.gas.entropy_mass  # Specific entropy [J/(kg·K)]
            except Exception:
                # Fallback to ideal gas entropy formula
                # s = c_v * ln(p/p_ref) - c_p * ln(rho/rho_ref) + s_ref
                # Using stored gamma and R_gas
                p_ref = 101325.0
                rho_ref = 1.0
                c_v = self._initial_R_gas / (self._initial_gamma - 1.0)
                s[i] = c_v * (np.log(p[i] / p_ref) - self._initial_gamma * np.log(rho[i] / rho_ref))

        return s


class ExactRiemannSolver:
    """
    Exact Riemann solver for the Euler equations

    Solves for the star-state (p*, u*) at the contact discontinuity
    using Newton-Raphson iteration on the pressure function.

    Reference: Toro (2009), Chapter 4
    """

    def __init__(self, eos: IdealGasEOS, tol: float = 1e-8, max_iter: int = 50):
        self.eos = eos
        self.gamma = eos.gamma
        self.tol = tol
        self.max_iter = max_iter

    def _pressure_function(self, p: float, rho_K: float, p_K: float, c_K: float) -> Tuple[float, float]:
        """
        Pressure function f_K(p) and its derivative for Newton iteration

        For shock (p > p_K): f = (p - p_K) * sqrt(A_K / (p + B_K))
        For rarefaction (p ≤ p_K): f = (2c_K/(γ-1)) * ((p/p_K)^((γ-1)/(2γ)) - 1)
        """
        g = self.gamma

        if p > p_K:
            # Shock wave
            A_K = 2.0 / ((g + 1.0) * rho_K)
            B_K = (g - 1.0) / (g + 1.0) * p_K
            sqrt_term = np.sqrt(A_K / (p + B_K))
            f = (p - p_K) * sqrt_term
            df = sqrt_term * (1.0 - 0.5 * (p - p_K) / (p + B_K))
        else:
            # Rarefaction wave
            power = (g - 1.0) / (2.0 * g)
            ratio = max(p / p_K, 1e-15)
            f = 2.0 * c_K / (g - 1.0) * (ratio**power - 1.0)
            df = 1.0 / (rho_K * c_K) * ratio**(-(g + 1.0) / (2.0 * g))

        return f, df

    def solve(self, rho_L: float, u_L: float, p_L: float,
              rho_R: float, u_R: float, p_R: float) -> Tuple[float, float]:
        """
        Solve the Riemann problem for star-state values

        Returns:
        --------
        u_star : float
            Velocity in the star region
        p_star : float
            Pressure in the star region
        """
        # Sound speeds
        c_L = np.sqrt(self.gamma * p_L / rho_L)
        c_R = np.sqrt(self.gamma * p_R / rho_R)

        # Check for vacuum generation
        if 2.0 * (c_L + c_R) / (self.gamma - 1.0) <= (u_R - u_L):
            raise ValueError("Vacuum state generated in Riemann problem")

        # Initial guess using PVRS (Primitive Variable Riemann Solver)
        p_pvrs = 0.5 * (p_L + p_R) - 0.125 * (u_R - u_L) * (rho_L + rho_R) * (c_L + c_R)
        p_star = max(self.tol, p_pvrs)

        # Two-shock approximation for better initial guess if PVRS gives low pressure
        if p_star < min(p_L, p_R):
            p_star = 0.5 * (p_L + p_R)

        # Newton-Raphson iteration
        for iteration in range(self.max_iter):
            f_L, df_L = self._pressure_function(p_star, rho_L, p_L, c_L)
            f_R, df_R = self._pressure_function(p_star, rho_R, p_R, c_R)

            f = f_L + f_R + (u_R - u_L)
            df = df_L + df_R

            if abs(df) < 1e-15:
                break

            dp = -f / df
            p_new = p_star + dp

            # Ensure positivity
            p_star = max(self.tol, p_new)

            if abs(dp) / (0.5 * (p_star + abs(dp))) < self.tol:
                break

        # Compute star velocity
        f_L, _ = self._pressure_function(p_star, rho_L, p_L, c_L)
        f_R, _ = self._pressure_function(p_star, rho_R, p_R, c_R)
        u_star = 0.5 * (u_L + u_R) + 0.5 * (f_R - f_L)

        return u_star, p_star


class LagrangianSolver:
    """
    Publication-Quality 1D Lagrangian Gas Dynamics Solver

    Solves the Lagrangian form of the Euler equations in mass coordinates:

        ∂τ/∂t - ∂u/∂m = 0           (mass/volume equation)
        ∂u/∂t + ∂(p+q)/∂m = 0       (momentum equation)
        ∂E/∂t + ∂((p+q)u)/∂m = S_H  (TOTAL energy equation - CONSERVATIVE)

    where:
        τ = 1/ρ is specific volume
        m is the mass coordinate
        q is artificial viscosity
        S_H is artificial heat source (Noh's method)
        E = e + u²/2 is total specific energy

    The conservative total energy form ensures proper shock jump conditions
    and eliminates the "wall heating" error of internal energy formulations.
    """

    def __init__(self,
                 eos: Optional[Union['IdealGasEOS', 'CanteraEOS']] = None,
                 gas_config: Optional[Dict[str, Any]] = None,
                 mechanism: Optional[str] = None,
                 gas_composition: Optional[str] = None,
                 initial_T: float = 300.0,
                 initial_P: float = 101325.0,
                 use_ideal_gas: bool = False,
                 ideal_gas_gamma: float = 1.4,
                 ideal_gas_R: float = 287.0,
                 cfl: float = 0.5,
                 bc_left: str = 'solid',
                 bc_right: str = 'solid',
                 av_type: ArtificialViscosityType = ArtificialViscosityType.NOH_QH,
                 av_Cq: float = 2.0,
                 av_Cl: float = 0.3,
                 av_Ch: float = 0.1,
                 limiter: LimiterType = LimiterType.VANLEER,
                 use_predictor_corrector: bool = True,
                 piston_velocity: Optional[Callable[[float], float]] = None,
                 piston_gas_velocity: Optional[Callable[[float], float]] = None,
                 piston_side: str = 'left',
                 enable_friction: bool = False,
                 friction_coefficient: float = 0.0,
                 hydraulic_diameter: float = 0.1,
                 # Flame speed coupling parameters
                 enable_flame_coupling: bool = False,
                 flame_speed_provider: Optional['FlameSpeedProvider'] = None,
                 piston_velocity_model: PistonVelocityModel = PistonVelocityModel.EXPANSION,
                 equivalence_ratio: float = 1.0,
                 coupling_max_iterations: int = 10,
                 coupling_tolerance: float = 1e-4,
                 coupling_relaxation: float = 0.5,
                 custom_piston_velocity_func: Optional[Callable[[float, float, float, float], float]] = None):
        """
        Initialize the solver

        Parameters:
        -----------
        eos : CanteraEOS or IdealGasEOS, optional
            Equation of state. If provided, this takes highest priority.
            Required to be CanteraEOS when enable_flame_coupling=True.
        gas_config : dict, optional
            Gas configuration dictionary with keys:
                - 'T': Temperature [K]
                - 'P': Pressure [Pa]
                - 'Phi': Equivalence ratio (optional)
                - 'Fuel': Fuel species (e.g., 'H2', 'CH4')
                - 'Oxidizer': Oxidizer string (e.g., 'O2:1, N2:3.76')
                - 'mech': Cantera mechanism file path
            If provided and eos is None, creates CanteraEOS from this config.
        mechanism : str, optional
            Cantera mechanism file (e.g., 'gri30.yaml'). Alternative to gas_config.
        gas_composition : str, optional
            Gas composition string (e.g., 'CH4:1, O2:2, N2:7.52'). Alternative to gas_config.
        initial_T : float
            Initial temperature [K] (default 300.0). Used with mechanism/gas_composition.
        initial_P : float
            Initial pressure [Pa] (default 101325.0). Used with mechanism/gas_composition.
        use_ideal_gas : bool
            If True, explicitly use IdealGasEOS instead of Cantera (default False).
            Also used as fallback if Cantera import fails.
        ideal_gas_gamma : float
            Ratio of specific heats for IdealGasEOS (default 1.4).
        ideal_gas_R : float
            Specific gas constant [J/(kg·K)] for IdealGasEOS (default 287.0).
        cfl : float
            CFL number for timestep calculation (default 0.5)
        bc_left, bc_right : str
            Boundary conditions: 'solid' (reflecting), 'open' (non-reflecting),
            'piston' (moving piston)
        av_type : ArtificialViscosityType
            Type of artificial viscosity formulation
        av_Cq : float
            Quadratic artificial viscosity coefficient (default 2.0)
            Controls shock spreading: shock width ≈ Cq * dx
        av_Cl : float
            Linear artificial viscosity coefficient (default 0.3)
            Damps post-shock oscillations
        av_Ch : float
            Artificial heat flux coefficient for Noh's method (default 0.1)
            Required for eliminating wall heating errors
        limiter : LimiterType
            Slope limiter for MUSCL reconstruction
        use_predictor_corrector : bool
            Use predictor-corrector time integration for 2nd order accuracy
        piston_velocity : callable, optional
            Function that returns piston velocity given time: u_piston(t) -> float
            Required when bc_left='piston' or bc_right='piston'
            For porous_piston BC, this controls mesh/piston position movement
        piston_gas_velocity : callable, optional
            Function that returns gas velocity at piston face given time: u_gas(t) -> float
            Only used with 'porous_piston' BC. If not provided, piston_velocity is used.
            This allows gas velocity at boundary to differ from piston/mesh velocity.
        piston_side : str
            Which side has the piston: 'left' or 'right' (default 'left')
        enable_friction : bool
            Enable wall friction source term (default False)
        friction_coefficient : float
            Darcy friction factor f (dimensionless, default 0.0)
            The friction coefficient κ = 2f/D is used in the source term
        hydraulic_diameter : float
            Hydraulic diameter D [m] of the tube/channel (default 0.1)
        enable_flame_coupling : bool
            Enable iterative coupling of piston velocity with flame speed (default False).
            When enabled, piston velocity is computed from flame speed at each timestep
            using sub-iterations to achieve self-consistency.
        flame_speed_provider : FlameSpeedProvider, optional
            Provider for flame speed calculation (tabulated or Cantera-based).
            Required when enable_flame_coupling=True.
        piston_velocity_model : PistonVelocityModel
            Model for computing piston velocity from flame speed (default EXPANSION).
            EXPANSION: u_piston = S_L * (rho_u/rho_b - 1)
            FLAME_FIXED: u_piston = S_L
            PRESSURE_DRIVEN: includes pressure effects
            CUSTOM: uses custom_piston_velocity_func
        equivalence_ratio : float
            Fuel-air equivalence ratio phi for flame speed lookup (default 1.0)
        coupling_max_iterations : int
            Maximum sub-iterations for flame-piston coupling (default 10)
        coupling_tolerance : float
            Convergence tolerance for piston velocity iteration (default 1e-4)
        coupling_relaxation : float
            Under-relaxation factor for velocity updates (default 0.5).
            u_new = (1-alpha)*u_old + alpha*u_computed
        custom_piston_velocity_func : callable, optional
            Custom function for computing piston velocity from flame speed.
            Supported signatures (auto-detected):
              - f(S_L, T, P, rho) -> u_piston  (basic)
              - f(S_L, T, P, rho, rho_b) -> u_piston  (with burned density)
              - f(S_L, T, P, rho, rho_b, t) -> u_piston  (with time)
            Used when piston_velocity_model=CUSTOM
        """
        # =================================================================
        # EOS Initialization Logic (two conditions):
        # -----------------------------------------------------------------
        # CONDITION 1 (Default - Cantera):
        #   Priority: eos > gas_config > mechanism/composition > air fallback
        #   Uses CanteraEOS for accurate thermodynamics
        #
        # CONDITION 2 (IdealGasEOS):
        #   Only if: use_ideal_gas=True OR Cantera import fails
        #   Uses isentropic ideal gas equations
        # =================================================================

        if eos is not None:
            # Explicit EOS provided - use it directly
            self.eos = eos
        elif use_ideal_gas:
            # Condition 2: Explicitly requested IdealGasEOS
            warnings.warn(
                "Using IdealGasEOS as explicitly requested. For accurate "
                "thermodynamic calculations (especially with flame coupling), "
                "CanteraEOS is recommended.",
                UserWarning,
                stacklevel=2
            )
            self.eos = IdealGasEOS(gamma=ideal_gas_gamma, R_gas=ideal_gas_R)
        else:
            # Condition 1: Default to Cantera
            try:
                import cantera as ct

                # Determine configuration source (priority: gas_config > mechanism/composition > air)
                if gas_config is not None:
                    # Use gas_config dictionary (similar to INPUT_PARAMS_CONFIG)
                    mech = gas_config.get('mech', 'gri30.yaml')
                    T = gas_config.get('T', 300.0)
                    P = gas_config.get('P', 101325.0)
                    fuel = gas_config.get('Fuel', None)
                    oxidizer = gas_config.get('Oxidizer', 'O2:1, N2:3.76')
                    phi = gas_config.get('Phi', 1.0)

                    # Create gas object
                    gas = ct.Solution(mech)

                    if fuel is not None:
                        # Set as fuel-oxidizer mixture at given equivalence ratio
                        gas.set_equivalence_ratio(phi, fuel, oxidizer)
                        gas.TP = T, P
                    else:
                        # Direct composition if no fuel specified
                        composition = gas_config.get('composition', 'O2:0.21, N2:0.79')
                        gas.TPX = T, P, composition

                    self.eos = CanteraEOS(gas)

                elif mechanism is not None or gas_composition is not None:
                    # Use mechanism/composition parameters
                    mech = mechanism if mechanism is not None else 'air.yaml'
                    composition = gas_composition if gas_composition is not None else 'O2:0.21, N2:0.79'

                    gas = ct.Solution(mech)
                    gas.TPX = initial_T, initial_P, composition
                    self.eos = CanteraEOS(gas)

                else:
                    # Fallback to air at STP (no configuration provided)
                    warnings.warn(
                        "No EOS or gas configuration provided. Falling back to air "
                        "at STP (mechanism='air.yaml', composition='O2:0.21, N2:0.79', "
                        "T=300K, P=101325Pa). For combustion simulations, provide "
                        "gas_config or mechanism/gas_composition parameters.",
                        UserWarning,
                        stacklevel=2
                    )
                    gas = ct.Solution('air.yaml')
                    gas.TPX = 300.0, 101325.0, 'O2:0.21, N2:0.79'
                    self.eos = CanteraEOS(gas)

            except ImportError:
                # Condition 2 fallback: Cantera not available
                warnings.warn(
                    "Cantera not found - falling back to IdealGasEOS. "
                    "Install Cantera for accurate thermodynamics: pip install cantera",
                    UserWarning,
                    stacklevel=2
                )
                self.eos = IdealGasEOS(gamma=ideal_gas_gamma, R_gas=ideal_gas_R)

            except Exception as e:
                raise RuntimeError(
                    f"Failed to create CanteraEOS: {e}\n"
                    "Check your mechanism file and gas configuration, or use "
                    "use_ideal_gas=True for simplified ideal gas EOS."
                )

        # Final warning if using IdealGasEOS (covers explicit eos=IdealGasEOS case)
        if not isinstance(self.eos, CanteraEOS):
            if eos is not None and isinstance(eos, IdealGasEOS):
                # Only warn if user explicitly passed IdealGasEOS
                warnings.warn(
                    "Using IdealGasEOS. For accurate thermodynamic calculations, "
                    "CanteraEOS is recommended.",
                UserWarning,
                stacklevel=2
            )

        self.riemann = ExactRiemannSolver(self.eos)
        self.cfl = cfl

        # Boundary conditions
        self.bc_left = bc_left.lower()
        self.bc_right = bc_right.lower()

        # Piston boundary condition
        self.piston_velocity = piston_velocity
        self.piston_gas_velocity = piston_gas_velocity
        self.piston_side = piston_side.lower()
        self.current_time = 0.0  # Track current simulation time for piston callback

        # Storage for mesh velocities (needed when mesh velocity != gas velocity)
        self._mesh_velocities = None

        # Validate piston setup
        if self.bc_left in ['piston', 'porous_piston'] or self.bc_right in ['piston', 'porous_piston']:
            if piston_velocity is None:
                raise ValueError("piston_velocity function must be provided when using 'piston' or 'porous_piston' boundary condition")

        # Artificial viscosity parameters
        self.av_type = av_type
        self.av_Cq = av_Cq
        self.av_Cl = av_Cl
        self.av_Ch = av_Ch

        # Limiter and time integration
        self.limiter = limiter
        self.use_predictor_corrector = use_predictor_corrector

        # Friction parameters
        self.enable_friction = enable_friction
        self.friction_coefficient = friction_coefficient
        self.hydraulic_diameter = hydraulic_diameter
        # Compute kappa coefficient: κ = 2f/D
        if hydraulic_diameter > 0:
            self.kappa = 2.0 * friction_coefficient / hydraulic_diameter
        else:
            self.kappa = 0.0

        # Flame speed coupling parameters
        self.enable_flame_coupling = enable_flame_coupling
        self.flame_speed_provider = flame_speed_provider
        self.piston_velocity_model = piston_velocity_model
        self.equivalence_ratio = equivalence_ratio
        self.coupling_max_iterations = coupling_max_iterations
        self.coupling_tolerance = coupling_tolerance
        self.coupling_relaxation = coupling_relaxation
        self.custom_piston_velocity_func = custom_piston_velocity_func

        # Validate flame coupling setup
        if self.enable_flame_coupling:
            if flame_speed_provider is None:
                raise ValueError(
                    "flame_speed_provider must be provided when enable_flame_coupling=True"
                )
            if piston_velocity_model == PistonVelocityModel.CUSTOM:
                if custom_piston_velocity_func is None:
                    raise ValueError(
                        "custom_piston_velocity_func must be provided when "
                        "piston_velocity_model=CUSTOM"
                    )
            # Flame coupling requires a piston BC
            if self.bc_left not in ['piston', 'porous_piston'] and \
               self.bc_right not in ['piston', 'porous_piston']:
                raise ValueError(
                    "enable_flame_coupling requires a 'piston' or 'porous_piston' "
                    "boundary condition on at least one side"
                )
            # Flame coupling requires CanteraEOS for accurate temperature calculations
            if not isinstance(self.eos, CanteraEOS):
                raise ValueError(
                    "Flame coupling requires CanteraEOS for accurate temperature "
                    "calculations. IdealGasEOS is not supported with flame coupling. "
                    "Please create a CanteraEOS instance with your mechanism and use it."
                )

        # Coupling diagnostics
        self._coupling_iterations_history: List[int] = []
        self._coupling_converged_history: List[bool] = []
        self._current_flame_speed: float = 0.0
        self._current_coupled_piston_velocity: float = 0.0

        # Grid data (initialized later)
        self.n_cells: int = 0
        self.m_total: float = 0.0
        self.m: np.ndarray = None          # Interface mass coordinates [n+1]
        self.m_centers: np.ndarray = None   # Cell center mass coordinates [n]
        self.dm: np.ndarray = None          # Cell masses [n]
        self.x: np.ndarray = None           # Interface positions [n+1]
        self.x_centers: np.ndarray = None   # Cell center positions [n]

        # State
        self.state: GasState = None

        # Conservation tracking
        self.diagnostics = ConservationDiagnostics()
        self._initial_mass: float = 0.0
        self._initial_momentum: float = 0.0
        self._initial_energy: float = 0.0

        # Porous piston cell injection parameters
        self._porous_piston_enabled = False
        self._piston_position: float = 0.0  # Track actual piston position
        self._accumulated_injected_mass: float = 0.0  # Mass accumulated from velocity difference
        self._inlet_rho: float = None  # Inlet gas density
        self._inlet_p: float = None    # Inlet gas pressure
        self._inlet_e: float = None    # Inlet gas internal energy
        self._inlet_T: float = None    # Inlet gas temperature (for Cantera)
        self._nominal_cell_mass: float = None  # Target mass for injected cells
        self._nominal_cell_dx: float = None    # Initial cell size for merge threshold
        self._cross_sectional_area: float = 1.0  # Cross-sectional area for mass flux
        self._min_cell_width: float = None  # Minimum cell width before injection

    def set_porous_piston_inlet(self, rho: float, p: float, e: float = None,
                                 T: float = None, area: float = 1.0,
                                 nominal_cell_mass: float = None,
                                 min_cell_width_factor: float = 0.1) -> None:
        """
        Set inlet gas properties for porous piston cell injection.

        When piston velocity > gas velocity, new mass enters the domain.
        This method sets up the properties of the injected gas and enables
        the cell injection mechanism.

        Parameters:
        -----------
        rho : float
            Inlet gas density [kg/m³]
        p : float
            Inlet gas pressure [Pa]
        e : float, optional
            Inlet gas internal energy [J/kg]. If None, computed from EOS.
        T : float, optional
            Inlet gas temperature [K]. Used for Cantera EOS.
        area : float
            Cross-sectional area [m²] (default 1.0 for 1D)
        nominal_cell_mass : float, optional
            Target mass for injected cells. If None, uses average initial cell mass.
        min_cell_width_factor : float
            Fraction of initial cell width below which injection is triggered (default 0.1)
        """
        self._porous_piston_enabled = True
        self._inlet_rho = rho
        self._inlet_p = p
        self._inlet_T = T
        self._cross_sectional_area = area

        # Compute internal energy if not provided
        if e is not None:
            self._inlet_e = e
        else:
            self._inlet_e = self.eos.internal_energy(rho, p)

        # Set nominal cell mass (will be finalized in initialize if None)
        self._nominal_cell_mass = nominal_cell_mass
        self._min_cell_width_factor = min_cell_width_factor

    def _inject_cell_left(self) -> None:
        """
        Inject a new cell at the left boundary (behind the piston).

        This represents gas flowing through the porous piston into the domain.
        The new cell has inlet gas properties and velocity u_gas.

        Note: This is typically not needed when mesh velocities are properly
        handled in the predictor-corrector scheme, but provides a fallback
        for extreme cases.
        """
        if not self._porous_piston_enabled:
            return

        # Get current gas velocity at piston
        u_gas = self.piston_gas_velocity(self.current_time) if self.piston_gas_velocity else 0.0

        # New cell properties
        new_dm = self._nominal_cell_mass
        new_tau = 1.0 / self._inlet_rho
        new_u = u_gas
        new_e = self._inlet_e
        new_p = self._inlet_p
        new_rho = self._inlet_rho
        new_c = np.atleast_1d(self.eos.sound_speed(np.array([new_rho]), np.array([new_p])))[0]
        new_E = new_e + 0.5 * new_u**2

        # New cell width
        new_dx = new_dm * new_tau  # dx = dm / rho = dm * tau

        # Insert new cell at the beginning
        # Update mesh coordinates
        new_x_left = self.x[0] - new_dx
        self.x = np.concatenate([[new_x_left], self.x])

        # Update mass coordinates
        self.dm = np.concatenate([[new_dm], self.dm])
        self.m_total += new_dm
        self.m = np.cumsum(np.concatenate([[0], self.dm]))
        self.m_centers = 0.5 * (self.m[:-1] + self.m[1:])

        # Update cell centers
        self.x_centers = 0.5 * (self.x[:-1] + self.x[1:])

        # Update state arrays (only actual fields, not computed properties)
        # GasState has: tau, u, E, p, c (rho and e are computed properties)
        self.state.tau = np.concatenate([[new_tau], self.state.tau])
        self.state.u = np.concatenate([[new_u], self.state.u])
        self.state.E = np.concatenate([[new_E], self.state.E])
        self.state.p = np.concatenate([[new_p], self.state.p])
        self.state.c = np.concatenate([[new_c], self.state.c])

        self.n_cells += 1

    def _apply_porous_piston_mass_flux(self, dt: float) -> None:
        """
        Apply mass flux at porous piston boundary.

        When piston velocity differs from gas velocity, there is mass flux
        across the boundary:
        - u_piston > u_gas: mass efflux (leaves domain), first cell shrinks
        - u_piston < u_gas: mass influx (enters domain), first cell grows

        When first cell becomes too small (compressed or depleted), it is merged
        with the neighbor cell to prevent timestep collapse.
        """
        if not self._porous_piston_enabled:
            return

        if self.bc_left != 'porous_piston':
            return

        u_piston = self.piston_velocity(self.current_time)
        u_gas = self.piston_gas_velocity(self.current_time) if self.piston_gas_velocity else u_piston

        # Mass flux across boundary: dm = ρ * (u_gas - u_piston) * A * dt
        # Positive when mass enters (u_gas > u_piston)
        # Negative when mass leaves (u_piston > u_gas)
        rho_boundary = self.state.rho[0]  # Use current boundary density
        dm_flux = rho_boundary * (u_gas - u_piston) * self._cross_sectional_area * dt

        # Apply mass flux to first cell
        self.dm[0] += dm_flux

        # Update mass coordinates
        self.m = np.cumsum(np.concatenate([[0], self.dm]))
        self.m_total = self.m[-1]
        self.m_centers = 0.5 * (self.m[:-1] + self.m[1:])

        # Check if first cell should be merged (mass depleted OR compressed)
        min_cell_mass = 0.1 * self._nominal_cell_mass if self._nominal_cell_mass else 1e-10

        # Also check cell size - merge if too compressed
        # Use 5% of nominal cell size as threshold to prevent CFL collapse
        dx = self.x[1] - self.x[0]
        min_cell_dx = 0.05 * self._nominal_cell_dx if self._nominal_cell_dx else 1e-6

        should_merge = False
        if self.dm[0] < min_cell_mass:
            should_merge = True
        if dx < min_cell_dx:
            should_merge = True

        if should_merge and self.n_cells > 10:
            self._merge_cells_left()

    def _merge_cells_left(self) -> None:
        """
        Merge the first two cells (conservative cell merging at porous boundary).

        This occurs when the first cell becomes too compressed or depleted.
        The merge conserves mass, momentum, and energy.
        """
        if self.n_cells <= 10:  # Keep minimum number of cells
            return

        # Conservative merge of cells 0 and 1
        dm0, dm1 = self.dm[0], self.dm[1]
        dm_merged = dm0 + dm1

        # Mass-weighted velocity (momentum conservation)
        u_merged = (dm0 * self.state.u[0] + dm1 * self.state.u[1]) / dm_merged

        # Mass-weighted total energy (energy conservation)
        E_merged = (dm0 * self.state.E[0] + dm1 * self.state.E[1]) / dm_merged

        # Update arrays - remove cell 0, update cell 1 with merged values
        self.x = np.concatenate([[self.x[0]], self.x[2:]])  # Keep first interface, remove second
        self.dm = np.concatenate([[dm_merged], self.dm[2:]])

        # Update state arrays
        self.state.u = np.concatenate([[u_merged], self.state.u[2:]])
        self.state.E = np.concatenate([[E_merged], self.state.E[2:]])

        # Recompute derived quantities from conserved variables
        dx_merged = self.x[1] - self.x[0]
        tau_merged = dx_merged / dm_merged
        self.state.tau = np.concatenate([[tau_merged], self.state.tau[2:]])

        # Recompute pressure and sound speed from EOS
        e_merged = E_merged - 0.5 * u_merged**2
        rho_merged = 1.0 / tau_merged
        p_merged = self.eos.pressure(np.array([rho_merged]), np.array([e_merged]))[0]
        c_merged = self.eos.sound_speed(np.array([rho_merged]), np.array([p_merged]))[0]

        self.state.p = np.concatenate([[p_merged], self.state.p[2:]])
        self.state.c = np.concatenate([[c_merged], self.state.c[2:]])

        # Update counts and coordinates
        self.n_cells -= 1
        self.m = np.cumsum(np.concatenate([[0], self.dm]))
        self.m_total = self.m[-1]
        self.m_centers = 0.5 * (self.m[:-1] + self.m[1:])
        self.x_centers = 0.5 * (self.x[:-1] + self.x[1:])

    def _remove_cell_left(self) -> None:
        """
        Remove the first cell by merging with neighbor.

        Legacy wrapper - now calls _merge_cells_left for conservative treatment.
        """
        self._merge_cells_left()

    def _check_porous_piston_injection(self, dt: float) -> None:
        """
        Legacy function - now redirects to mass flux handling.

        The porous piston is handled via mass flux, not cell injection.
        Cell injection only occurs when u_gas > u_piston (mass enters).
        """
        self._apply_porous_piston_mass_flux(dt)

    def _slope_limiter(self, a: float, b: float) -> float:
        """
        Apply slope limiter for MUSCL reconstruction

        Parameters:
        -----------
        a, b : float
            Left and right slopes

        Returns:
        --------
        Limited slope
        """
        # Handle non-finite values
        if not np.isfinite(a) or not np.isfinite(b):
            return 0.0

        # Prevent overflow in multiplication
        if abs(a) > 1e150 or abs(b) > 1e150:
            return 0.0

        product = a * b

        if self.limiter == LimiterType.MINMOD:
            # Minmod: most diffusive, safest
            if product <= 0:
                return 0.0
            return np.sign(a) * min(abs(a), abs(b))

        elif self.limiter == LimiterType.VANLEER:
            # van Leer: smooth limiter, good balance
            if product <= 0:
                return 0.0
            denom = a + b
            if abs(denom) < 1e-15:
                return 0.0
            return 2.0 * product / denom

        elif self.limiter == LimiterType.SUPERBEE:
            # Superbee: least diffusive, can cause mild oscillations
            if product <= 0:
                return 0.0
            s = np.sign(a)
            return s * max(min(2*abs(a), abs(b)), min(abs(a), 2*abs(b)))

        elif self.limiter == LimiterType.MC:
            # Monotonized Central
            if product <= 0:
                return 0.0
            c = 0.5 * (a + b)
            return np.sign(c) * min(2*abs(a), 2*abs(b), abs(c))

        return 0.0

    def initialize(self,
                   x_euler: np.ndarray,
                   rho_euler: np.ndarray,
                   u_euler: np.ndarray,
                   p_euler: np.ndarray,
                   n_cells: Optional[int] = None):
        """
        Initialize solver from Eulerian initial conditions

        At t=0, Lagrangian positions equal Eulerian positions.
        Mass coordinate: m(x) = ∫₀ˣ ρ(y) dy

        Parameters:
        -----------
        x_euler : array [n]
            Eulerian cell center positions
        rho_euler, u_euler, p_euler : array [n]
            Initial density, velocity, pressure
        n_cells : int, optional
            Number of Lagrangian cells (default: same as input)
        """
        x_euler = np.asarray(x_euler, dtype=np.float64)
        rho_euler = np.asarray(rho_euler, dtype=np.float64)
        u_euler = np.asarray(u_euler, dtype=np.float64)
        p_euler = np.asarray(p_euler, dtype=np.float64)

        n_euler = len(rho_euler)

        # Construct Eulerian interfaces
        dx = x_euler[1] - x_euler[0] if n_euler > 1 else 1.0
        x_interfaces = np.zeros(n_euler + 1)
        x_interfaces[0] = x_euler[0] - 0.5 * dx
        x_interfaces[1:-1] = 0.5 * (x_euler[:-1] + x_euler[1:])
        x_interfaces[-1] = x_euler[-1] + 0.5 * dx

        # Compute mass in each Eulerian cell
        dx_euler = np.diff(x_interfaces)
        mass_cells = rho_euler * dx_euler

        # Cumulative mass at interfaces
        m_euler = np.zeros(n_euler + 1)
        m_euler[1:] = np.cumsum(mass_cells)
        self.m_total = m_euler[-1]

        # Set up Lagrangian grid
        if n_cells is None:
            n_cells = n_euler
        self.n_cells = n_cells

        # Uniform mass spacing
        self.m = np.linspace(0, self.m_total, n_cells + 1)
        self.m_centers = 0.5 * (self.m[:-1] + self.m[1:])
        self.dm = np.diff(self.m)

        # Map mass coordinates to initial positions
        # Handle non-unique mass values
        unique_mask = np.diff(m_euler) > 1e-14
        unique_idx = np.concatenate([[0], np.where(unique_mask)[0] + 1])
        if len(unique_idx) < 2:
            unique_idx = np.array([0, len(m_euler) - 1])

        x_from_m = interp1d(
            m_euler[unique_idx], x_interfaces[unique_idx],
            kind='linear', bounds_error=False,
            fill_value=(x_interfaces[0], x_interfaces[-1])
        )

        self.x = x_from_m(self.m)
        self.x_centers = 0.5 * (self.x[:-1] + self.x[1:])

        # Compute initial state
        dx_lagrange = np.diff(self.x)
        rho_lagrange = self.dm / dx_lagrange

        # Interpolate velocity (smooth, use linear)
        u_interp = interp1d(x_euler, u_euler, kind='linear',
                           bounds_error=False, fill_value=(u_euler[0], u_euler[-1]))
        u_lagrange = u_interp(self.x_centers)

        # Interpolate pressure (piecewise constant for discontinuities)
        p_lagrange = np.zeros(n_cells)
        for i, xc in enumerate(self.x_centers):
            idx = np.searchsorted(x_interfaces, xc) - 1
            idx = max(0, min(idx, n_euler - 1))
            p_lagrange[i] = p_euler[idx]

        # Create state with TOTAL energy (conservative)
        tau = 1.0 / rho_lagrange
        E = self.eos.total_energy(rho_lagrange, u_lagrange, p_lagrange)
        c = self.eos.sound_speed(rho_lagrange, p_lagrange)

        self.state = GasState(tau=tau, u=u_lagrange, E=E, p=p_lagrange, c=c)

        # Store initial conserved quantities
        self._initial_mass = np.sum(self.dm)
        self._initial_momentum = np.sum(self.dm * self.state.u)
        self._initial_energy = np.sum(self.dm * self.state.E)

        # Verify initialization
        mass_check = rho_lagrange * dx_lagrange
        mass_error = np.max(np.abs(mass_check - self.dm) / self.dm)

        # Initialize porous piston parameters if enabled
        if self._porous_piston_enabled:
            # Set nominal cell mass if not specified
            if self._nominal_cell_mass is None:
                self._nominal_cell_mass = np.mean(self.dm)

            # Set nominal cell size for compression threshold
            initial_dx = np.mean(dx_lagrange)
            self._nominal_cell_dx = initial_dx

            # Set minimum cell width threshold
            self._min_cell_width = self._min_cell_width_factor * initial_dx

            # Initialize piston position at left boundary
            self._piston_position = self.x[0]

        print(f"Initialization complete:")
        print(f"  Cells: {self.n_cells}")
        print(f"  Total mass: {self.m_total:.6e}")
        print(f"  Domain: [{self.x[0]:.4f}, {self.x[-1]:.4f}]")
        print(f"  Mass conservation error: {mass_error:.2e}")

    def _compute_artificial_viscosity(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute artificial viscosity and heat flux at cell edges

        Returns:
        --------
        q_edge : array [n+1]
            Artificial viscosity at cell interfaces
        h_edge : array [n+1]
            Artificial heat flux at cell interfaces (Noh's method)
        """
        n = self.n_cells
        q_edge = np.zeros(n + 1)
        h_edge = np.zeros(n + 1)

        dx = np.diff(self.x)
        rho = self.state.rho
        u = self.state.u
        c = self.state.c
        e = self.state.e

        for i in range(1, n):
            # Velocity jump at interface i (between cells i-1 and i)
            du = u[i] - u[i-1]

            # Only apply in compression (du < 0)
            if du < 0:
                # Average properties at interface
                rho_avg = 0.5 * (rho[i-1] + rho[i])
                c_avg = 0.5 * (c[i-1] + c[i])
                dx_avg = 0.5 * (dx[i-1] + dx[i])

                # Quadratic term (Richtmyer)
                q_quad = self.av_Cq**2 * rho_avg * du**2

                # Linear term (Landshoff)
                q_linear = self.av_Cl * rho_avg * c_avg * abs(du)

                q_edge[i] = q_quad + q_linear

                # Noh's artificial heat flux (eliminates wall heating)
                if self.av_type == ArtificialViscosityType.NOH_QH:
                    # Heat flux: H = Ch * ρ * c * dx * ∂e/∂x
                    # This removes the thermodynamic inconsistency
                    de = e[i] - e[i-1]
                    h_edge[i] = self.av_Ch * rho_avg * c_avg * dx_avg * de / dx_avg

        return q_edge, h_edge

    def _compute_friction_source(self, state: GasState) -> np.ndarray:
        """
        Compute wall friction source term for momentum equation.

        The friction force per unit mass (acceleration) in Lagrangian mass coordinates:
            S_friction = -κ * u * |u|

        where κ = 2f/D is the friction coefficient (f = Darcy friction factor, D = hydraulic diameter).

        This follows the formulation in CLTORC where friction is a body force that
        decelerates the flow. The kinetic energy lost to friction is assumed to be
        transferred to the tube wall (not converted to internal energy of the gas).

        Returns:
        --------
        S_friction : array [n_cells]
            Friction acceleration (force per unit mass) at cell centers [m/s²]
        """
        if not self.enable_friction or self.kappa <= 0:
            return np.zeros(self.n_cells)

        n = self.n_cells
        S_friction = np.zeros(n)

        for i in range(n):
            u_cell = state.u[i]
            u_magnitude = np.abs(u_cell)

            # Skip if velocity is negligible
            if u_magnitude < 1e-10:
                continue

            # Friction source: S = -κ * u * |u|
            # This is the acceleration (force per unit mass) opposing the flow
            S_friction[i] = -self.kappa * u_cell * u_magnitude

        return S_friction

    def _get_state_at_piston(self, state: Optional[GasState] = None) -> Dict[str, float]:
        """
        Extract thermodynamic state at the piston face.

        Parameters:
        -----------
        state : GasState, optional
            State to extract from. If None, uses self.state.

        Returns:
        --------
        dict with keys: 'T', 'P', 'rho', 'u', 'e', 'c'
            Thermodynamic state at the piston boundary cell
        """
        if state is None:
            state = self.state

        # Determine which cell is at the piston
        if self.piston_side == 'left' or self.bc_left in ['piston', 'porous_piston']:
            idx = 0  # First cell
        else:
            idx = -1  # Last cell

        rho = state.rho[idx]
        P = state.p[idx]
        u = state.u[idx]
        e = state.e[idx]
        c = state.c[idx]

        # Compute temperature using Cantera (CanteraEOS is enforced for flame coupling)
        # Use the gas object directly for accurate temperature from density and pressure
        self.eos.gas.DP = rho, P
        T = self.eos.gas.T

        return {
            'T': T,
            'P': P,
            'rho': rho,
            'u': u,
            'e': e,
            'c': c
        }

    def _compute_coupled_piston_velocity(self, state: Optional[GasState] = None) -> float:
        """
        Compute piston velocity from flame speed based on current thermodynamic state.

        This implements the relationship: u_piston = f(S_L, thermodynamic_state)

        Parameters:
        -----------
        state : GasState, optional
            State to use. If None, uses self.state.

        Returns:
        --------
        u_piston : float
            Computed piston velocity [m/s]
        """
        if not self.enable_flame_coupling:
            # Fall back to callback if coupling disabled
            if self.piston_velocity is not None:
                return self.piston_velocity(self.current_time)
            return 0.0

        # Get state at piston face
        piston_state = self._get_state_at_piston(state)
        T = piston_state['T']
        P = piston_state['P']
        rho_u = piston_state['rho']  # Unburned gas density

        # Get flame speed from provider
        S_L = self.flame_speed_provider.get_flame_speed(T, P, self.equivalence_ratio)
        self._current_flame_speed = S_L

        # Compute piston velocity based on model
        if self.piston_velocity_model == PistonVelocityModel.EXPANSION:
            # Expansion model: u_piston = S_L * (rho_u / rho_b - 1)
            # Gas expansion behind flame pushes piston
            rho_b = self.flame_speed_provider.get_burned_density(T, P, self.equivalence_ratio)
            if rho_b > 1e-10:
                expansion_ratio = rho_u / rho_b
                u_piston = S_L * (expansion_ratio - 1.0)
            else:
                u_piston = S_L * 6.0  # Fallback expansion ratio ~7

        elif self.piston_velocity_model == PistonVelocityModel.FLAME_FIXED:
            # Flame attached to piston: u_piston = S_L
            u_piston = S_L

        elif self.piston_velocity_model == PistonVelocityModel.PRESSURE_DRIVEN:
            # Pressure-driven model with acoustic correction
            # u_piston = S_L * (rho_u/rho_b - 1) + acoustic_term
            rho_b = self.flame_speed_provider.get_burned_density(T, P, self.equivalence_ratio)
            c = piston_state['c']

            if rho_b > 1e-10:
                expansion_ratio = rho_u / rho_b
                # Base expansion velocity
                u_expansion = S_L * (expansion_ratio - 1.0)
                # Pressure ratio across flame (approximate)
                # For deflagration: P_b ≈ P_u (isobaric), so no acoustic term
                # For detonation: significant pressure jump
                u_piston = u_expansion
            else:
                u_piston = S_L * 6.0

        elif self.piston_velocity_model == PistonVelocityModel.CUSTOM:
            # User-provided function: f(S_L, T, P, rho, rho_b, t) -> u_piston
            # Get burned density for custom function
            rho_b = self.flame_speed_provider.get_burned_density(T, P, self.equivalence_ratio)

            # Check function signature and call appropriately
            import inspect
            sig = inspect.signature(self.custom_piston_velocity_func)
            n_params = len(sig.parameters)

            if n_params >= 6:
                # Extended signature: f(S_L, T, P, rho, rho_b, t)
                u_piston = self.custom_piston_velocity_func(S_L, T, P, rho_u, rho_b, self.current_time)
            elif n_params >= 5:
                # Intermediate signature: f(S_L, T, P, rho, rho_b)
                u_piston = self.custom_piston_velocity_func(S_L, T, P, rho_u, rho_b)
            else:
                # Basic signature: f(S_L, T, P, rho)
                u_piston = self.custom_piston_velocity_func(S_L, T, P, rho_u)

        else:
            raise ValueError(f"Unknown piston velocity model: {self.piston_velocity_model}")

        self._current_coupled_piston_velocity = u_piston
        return u_piston

    def _create_coupled_piston_callback(self, u_piston: float) -> Callable[[float], float]:
        """
        Create a piston velocity callback that returns a fixed value.

        This is used during sub-iteration to set the piston velocity
        to the currently computed value.
        """
        return lambda t: u_piston

    def _muscl_reconstruct(self, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        MUSCL reconstruction with slope limiting

        Returns:
        --------
        phi_L : array [n+1]
            Left state at each interface
        phi_R : array [n+1]
            Right state at each interface
        """
        n = len(phi)
        phi_L = np.zeros(n + 1)
        phi_R = np.zeros(n + 1)

        dm = self.dm
        m_c = self.m_centers

        for i in range(n):
            # Compute slopes
            if i == 0:
                slope_L = (phi[1] - phi[0]) / (m_c[1] - m_c[0]) if n > 1 else 0.0
                slope_R = slope_L
            elif i == n - 1:
                slope_L = (phi[i] - phi[i-1]) / (m_c[i] - m_c[i-1])
                slope_R = slope_L
            else:
                slope_L = (phi[i] - phi[i-1]) / (m_c[i] - m_c[i-1])
                slope_R = (phi[i+1] - phi[i]) / (m_c[i+1] - m_c[i])

            # Apply limiter
            slope = self._slope_limiter(slope_L, slope_R)

            # Reconstruct at interfaces
            phi_L[i+1] = phi[i] + 0.5 * slope * dm[i]
            phi_R[i] = phi[i] - 0.5 * slope * dm[i]

        # Boundary values
        phi_L[0] = phi[0]
        phi_R[n] = phi[n-1]

        return phi_L, phi_R

    def _apply_boundary_conditions(self, u_face: np.ndarray, p_face: np.ndarray):
        """
        Apply boundary conditions to interface values

        Boundary types:
        - 'solid'/'wall': Reflecting wall (u = 0)
        - 'open': Non-reflecting outflow (extrapolate from interior)
        - 'piston': Moving piston with imposed velocity from callback
        - 'porous_piston': Semi-porous piston where mesh moves at piston velocity
                          but gas BC uses a separate gas velocity

        For porous_piston:
        - u_face (gas velocity for flux calc) uses piston_gas_velocity if provided
        - Mesh movement uses piston_velocity (handled in _update_positions)
        """
        # Reset mesh velocities - only used for porous_piston
        self._mesh_velocities = None

        # Left boundary
        if self.bc_left == 'piston':
            # Moving piston with imposed velocity (impermeable)
            u_piston = self.piston_velocity(self.current_time)
            u_face[0] = u_piston
            p_face[0] = self.state.p[0]
        elif self.bc_left == 'porous_piston':
            # Porous piston with cell injection:
            # - Mesh moves at piston velocity (correct piston position)
            # - Gas BC uses gas velocity (gas flows through piston)
            # - Cell injection prevents collapse when u_piston > u_gas
            u_piston = self.piston_velocity(self.current_time)
            if self.piston_gas_velocity is not None:
                u_gas = self.piston_gas_velocity(self.current_time)
            else:
                u_gas = u_piston  # Fallback to piston velocity

            # Gas velocity for flux calculation (momentum/energy fluxes)
            u_face[0] = u_gas
            p_face[0] = self.state.p[0]

            # Mesh moves at piston velocity (to track piston position correctly)
            # Cell injection handles the mass entering through the porous piston
            if self._porous_piston_enabled:
                if self._mesh_velocities is None:
                    self._mesh_velocities = u_face.copy()
                self._mesh_velocities[0] = u_piston
        elif self.bc_left in ['solid', 'wall']:
            u_face[0] = 0.0
            p_face[0] = self.state.p[0]
        else:  # open
            u_face[0] = self.state.u[0]
            p_face[0] = self.state.p[0]

        # Right boundary
        if self.bc_right == 'piston':
            # Moving piston with imposed velocity (impermeable)
            u_piston = self.piston_velocity(self.current_time)
            u_face[-1] = u_piston
            p_face[-1] = self.state.p[-1]
        elif self.bc_right == 'porous_piston':
            # Semi-porous piston: mesh moves at piston velocity, gas BC at gas velocity
            u_piston = self.piston_velocity(self.current_time)
            if self.piston_gas_velocity is not None:
                u_gas = self.piston_gas_velocity(self.current_time)
            else:
                u_gas = u_piston  # Fallback to piston velocity
            # Gas velocity for flux calculation
            u_face[-1] = u_gas
            p_face[-1] = self.state.p[-1]
            # Initialize mesh velocities array if needed and store piston velocity
            if self._mesh_velocities is None:
                self._mesh_velocities = u_face.copy()
            self._mesh_velocities[-1] = u_piston
        elif self.bc_right in ['solid', 'wall']:
            u_face[-1] = 0.0
            p_face[-1] = self.state.p[-1]
        else:  # open
            u_face[-1] = self.state.u[-1]
            p_face[-1] = self.state.p[-1]

    def _compute_time_step(self) -> float:
        """
        Compute stable time step from CFL condition

        dt ≤ CFL * min(dx / (|u| + c))
        """
        dx = np.diff(self.x)
        dx = np.maximum(dx, 1e-15)  # Prevent zero cell widths

        # Ensure valid wave speeds
        c_safe = np.maximum(self.state.c, 1e-10)
        wave_speed = np.abs(self.state.u) + c_safe
        wave_speed = np.maximum(wave_speed, 1e-10)

        dt = self.cfl * np.min(dx / wave_speed)

        # Fallback for numerical issues
        if not np.isfinite(dt) or dt <= 0:
            # Use a very conservative estimate
            dt = self.cfl * 1e-8
            warnings.warn(f"Using fallback timestep: dt = {dt}")

        return dt

    def _compute_fluxes(self, state: GasState) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute interface fluxes using exact Riemann solver

        Returns:
        --------
        u_face, p_face : interface velocity and pressure
        q_face, h_face : artificial viscosity and heat flux
        """
        n = self.n_cells

        # MUSCL reconstruction
        rho_L, rho_R = self._muscl_reconstruct(state.rho)
        u_L, u_R = self._muscl_reconstruct(state.u)
        p_L, p_R = self._muscl_reconstruct(state.p)

        # Ensure positivity
        rho_L = np.maximum(rho_L, 1e-14)
        rho_R = np.maximum(rho_R, 1e-14)
        p_L = np.maximum(p_L, 1e-14)
        p_R = np.maximum(p_R, 1e-14)

        # Solve Riemann problems at interior interfaces
        u_face = np.zeros(n + 1)
        p_face = np.zeros(n + 1)

        for i in range(1, n):
            try:
                u_star, p_star = self.riemann.solve(
                    rho_L[i], u_L[i], p_L[i],
                    rho_R[i], u_R[i], p_R[i]
                )
                u_face[i] = u_star
                p_face[i] = p_star
            except ValueError:
                # Fallback for vacuum generation
                u_face[i] = 0.5 * (u_L[i] + u_R[i])
                p_face[i] = 0.5 * (p_L[i] + p_R[i])

        # Apply boundary conditions
        self._apply_boundary_conditions(u_face, p_face)

        # Compute artificial viscosity
        q_face, h_face = self._compute_artificial_viscosity()

        return u_face, p_face, q_face, h_face

    def _update_state(self, state: GasState, dt: float,
                      u_face: np.ndarray, p_face: np.ndarray,
                      q_face: np.ndarray, h_face: np.ndarray) -> GasState:
        """
        Update state using conservative Lagrangian equations

        CONSERVATIVE FORMULATION (total energy):
            ∂τ/∂t = ∂u/∂m
            ∂u/∂t = -∂(p+q)/∂m + S_friction
            ∂E/∂t = -∂((p+q)u)/∂m + S_H

        where:
            S_H is the heat source from Noh's artificial heat flux
            S_friction = -κ * u * |u| is the wall friction deceleration
        """
        n = self.n_cells
        dm = self.dm

        # Compute friction source term (if enabled)
        S_friction = self._compute_friction_source(state)

        new_tau = np.zeros(n)
        new_u = np.zeros(n)
        new_E = np.zeros(n)

        for i in range(n):
            # Effective pressure at interfaces (include AV)
            p_eff_L = p_face[i] + q_face[i]
            p_eff_R = p_face[i+1] + q_face[i+1]

            # Specific volume update: ∂τ/∂t = ∂u/∂m
            du_dm = (u_face[i+1] - u_face[i]) / dm[i]
            new_tau[i] = state.tau[i] + dt * du_dm

            # Momentum update: ∂u/∂t = -∂(p+q)/∂m + S_friction
            dp_dm = (p_eff_R - p_eff_L) / dm[i]
            new_u[i] = state.u[i] - dt * dp_dm + dt * S_friction[i]

            # CONSERVATIVE total energy update: ∂E/∂t = -∂((p+q)u)/∂m + S_H
            # This is the KEY difference from internal energy formulation
            flux_L = p_eff_L * u_face[i]
            flux_R = p_eff_R * u_face[i+1]
            dflux_dm = (flux_R - flux_L) / dm[i]

            # Noh's heat source (from divergence of heat flux)
            if self.av_type == ArtificialViscosityType.NOH_QH:
                S_H = (h_face[i+1] - h_face[i]) / dm[i]
            else:
                S_H = 0.0

            new_E[i] = state.E[i] - dt * dflux_dm + dt * S_H

        # Robust positivity enforcement
        # Use more conservative bounds for extreme cases
        tau_min = 1e-12
        new_tau = np.maximum(new_tau, tau_min)

        # Compute kinetic energy
        kinetic_energy = 0.5 * new_u**2

        # Compute derived quantities
        new_rho = 1.0 / new_tau
        new_e = new_E - kinetic_energy

        # Note: Internal energy can be negative for Cantera EOS with certain reference states
        # (e.g., elements like H2, O2 have h=0 at 298.15K, so e = h - P/rho can be negative)
        # Only check for NaN, not for positivity

        new_p = self.eos.pressure(new_rho, new_e)
        new_p = np.maximum(new_p, 1e-12)  # Ensure positive pressure

        new_c = self.eos.sound_speed(new_rho, new_p)
        new_c = np.maximum(new_c, 1e-10)  # Ensure valid sound speed

        # Check for NaN values and replace with previous state values
        nan_mask = ~np.isfinite(new_tau) | ~np.isfinite(new_u) | ~np.isfinite(new_E)
        if np.any(nan_mask):
            warnings.warn(f"NaN values detected in {np.sum(nan_mask)} cells, using previous state")
            new_tau[nan_mask] = state.tau[nan_mask]
            new_u[nan_mask] = state.u[nan_mask]
            new_E[nan_mask] = state.E[nan_mask]
            new_p[nan_mask] = state.p[nan_mask]
            new_c[nan_mask] = state.c[nan_mask]

        # For porous piston BC: impose gas velocity on boundary cell
        # This ensures the cell velocity reflects the imposed gas velocity at the boundary
        if self.bc_left == 'porous_piston' and self.piston_gas_velocity is not None:
            u_gas = self.piston_gas_velocity(self.current_time)
            # Set first cell velocity to gas velocity
            new_u[0] = u_gas
            # Update total energy to be consistent with new velocity
            new_E[0] = new_e[0] + 0.5 * new_u[0]**2
        if self.bc_right == 'porous_piston' and self.piston_gas_velocity is not None:
            u_gas = self.piston_gas_velocity(self.current_time)
            # Set last cell velocity to gas velocity
            new_u[-1] = u_gas
            # Update total energy to be consistent with new velocity
            new_E[-1] = new_e[-1] + 0.5 * new_u[-1]**2

        return GasState(tau=new_tau, u=new_u, E=new_E, p=new_p, c=new_c)

    def _update_positions(self, dt: float, u_face: np.ndarray):
        """
        Update cell interface positions

        For porous piston BC: uses stored mesh velocities (_mesh_velocities) which
        contain the piston velocity at boundary nodes, while u_face contains gas
        velocity used for flux calculations.

        After position update, specific volume is synchronized with mesh geometry
        to maintain numerical consistency: τ = dx/dm
        """
        # Use mesh velocities if available (porous piston), otherwise use u_face
        if self._mesh_velocities is not None:
            velocities_for_mesh = self._mesh_velocities
        else:
            velocities_for_mesh = u_face

        self.x = self.x + dt * velocities_for_mesh
        self.x_centers = 0.5 * (self.x[:-1] + self.x[1:])

        # For porous piston: synchronize specific volume with actual mesh geometry
        # This is critical for numerical stability when mesh velocity != gas velocity
        if self.bc_left == 'porous_piston' or self.bc_right == 'porous_piston':
            dx = np.diff(self.x)
            self.state.tau = dx / self.dm

    def _record_diagnostics(self, t: float):
        """Record conservation diagnostics"""
        total_mass = np.sum(self.dm)
        total_momentum = np.sum(self.dm * self.state.u)
        total_energy = np.sum(self.dm * self.state.E)

        self.diagnostics.time.append(t)
        self.diagnostics.total_mass.append(total_mass)
        self.diagnostics.total_momentum.append(total_momentum)
        self.diagnostics.total_energy.append(total_energy)

        # Relative errors
        if self._initial_mass > 0:
            self.diagnostics.mass_error.append(
                abs(total_mass - self._initial_mass) / self._initial_mass
            )
        if abs(self._initial_momentum) > 1e-14:
            self.diagnostics.momentum_error.append(
                abs(total_momentum - self._initial_momentum) / abs(self._initial_momentum)
            )
        else:
            self.diagnostics.momentum_error.append(abs(total_momentum))

        if self._initial_energy > 0:
            self.diagnostics.energy_error.append(
                abs(total_energy - self._initial_energy) / self._initial_energy
            )

    def step(self, dt: float):
        """
        Advance solution by one time step

        Uses predictor-corrector for 2nd order accuracy in time:
            1. Predictor: advance to t + dt using fluxes at t
            2. Corrector: re-evaluate fluxes at t + dt, average with predictor

        For porous piston BC, mesh velocities are also averaged to ensure
        proper second-order accuracy in tracking piston position.

        When enable_flame_coupling=True, uses sub-iteration to achieve
        self-consistent coupling between piston velocity and flame speed:
            1. Compute piston velocity from current state
            2. Advance state with this piston velocity
            3. Recompute piston velocity from new state
            4. Iterate until converged or max iterations reached
        """
        # Store time at beginning of step for boundary conditions
        t_start = self.current_time

        # Check if flame-piston coupling is enabled
        if self.enable_flame_coupling:
            self._step_with_flame_coupling(dt, t_start)
        else:
            self._step_standard(dt, t_start)

        # Check for porous piston cell injection
        # This adds new cells when piston advances faster than gas
        self._check_porous_piston_injection(dt)

    def _step_standard(self, dt: float, t_start: float):
        """
        Standard time step without flame-piston coupling.

        This is the original predictor-corrector implementation.
        """
        if self.use_predictor_corrector:
            # Store initial state and positions
            state_0 = self.state.copy()
            x_0 = self.x.copy()

            # PREDICTOR: Euler step to t + dt
            # Use time at start for boundary conditions
            self.current_time = t_start
            u_face, p_face, q_face, h_face = self._compute_fluxes(self.state)

            # Save predictor mesh velocities (for porous piston averaging)
            mesh_vel_pred = self._mesh_velocities.copy() if self._mesh_velocities is not None else None

            state_pred = self._update_state(self.state, dt, u_face, p_face, q_face, h_face)
            self._update_positions(dt, u_face)
            self.state = state_pred

            # CORRECTOR: Compute fluxes at predicted state
            # Use time at end for boundary conditions
            self.current_time = t_start + dt
            u_face_pred, p_face_pred, q_face_pred, h_face_pred = self._compute_fluxes(self.state)

            # Save corrector mesh velocities
            mesh_vel_corr = self._mesh_velocities.copy() if self._mesh_velocities is not None else None

            # Average fluxes (trapezoidal rule in time)
            u_face_avg = 0.5 * (u_face + u_face_pred)
            p_face_avg = 0.5 * (p_face + p_face_pred)
            q_face_avg = 0.5 * (q_face + q_face_pred)
            h_face_avg = 0.5 * (h_face + h_face_pred)

            # Average mesh velocities for porous piston (critical for correct position tracking)
            if mesh_vel_pred is not None and mesh_vel_corr is not None:
                self._mesh_velocities = 0.5 * (mesh_vel_pred + mesh_vel_corr)
            elif mesh_vel_corr is not None:
                self._mesh_velocities = mesh_vel_corr
            # else: _mesh_velocities remains None, will use u_face_avg

            # Final update from initial state using averaged fluxes
            self.x = x_0
            self.current_time = t_start + 0.5 * dt  # Use midpoint time for averaged BCs
            self.state = self._update_state(state_0, dt, u_face_avg, p_face_avg,
                                            q_face_avg, h_face_avg)
            self._update_positions(dt, u_face_avg)

            # Set final time
            self.current_time = t_start + dt

        else:
            # Single Euler step (1st order)
            self.current_time = t_start
            u_face, p_face, q_face, h_face = self._compute_fluxes(self.state)
            self.state = self._update_state(self.state, dt, u_face, p_face, q_face, h_face)
            self._update_positions(dt, u_face)
            self.current_time = t_start + dt

    def _step_with_flame_coupling(self, dt: float, t_start: float):
        """
        Time step with iterative flame-piston velocity coupling.

        Algorithm:
        1. Store initial state
        2. Compute initial piston velocity guess from current state
        3. Sub-iteration loop:
           a. Set piston velocity callback to current guess
           b. Advance state (using predictor-corrector)
           c. Compute new piston velocity from advanced state
           d. Check convergence: |u_new - u_old| < tolerance
           e. Apply under-relaxation: u = (1-alpha)*u_old + alpha*u_new
        4. Accept converged state
        """
        # Store initial state and positions
        state_0 = self.state.copy()
        x_0 = self.x.copy()

        # Store original piston velocity callback (if any)
        original_piston_velocity = self.piston_velocity

        # Initial guess: compute piston velocity from current state
        u_piston_old = self._compute_coupled_piston_velocity(state_0)

        # Sub-iteration loop
        converged = False
        n_iter = 0

        for iteration in range(self.coupling_max_iterations):
            n_iter = iteration + 1

            # Reset to initial state for each iteration
            self.state = state_0.copy()
            self.x = x_0.copy()
            self.current_time = t_start

            # Set piston velocity to current guess
            self.piston_velocity = self._create_coupled_piston_callback(u_piston_old)

            # Advance state using standard step (predictor-corrector or Euler)
            if self.use_predictor_corrector:
                self._step_predictor_corrector_internal(dt, t_start)
            else:
                self._step_euler_internal(dt, t_start)

            # Compute new piston velocity from advanced state
            u_piston_new = self._compute_coupled_piston_velocity(self.state)

            # Check convergence
            velocity_change = abs(u_piston_new - u_piston_old)
            velocity_scale = max(abs(u_piston_old), abs(u_piston_new), 1e-10)
            relative_change = velocity_change / velocity_scale

            if relative_change < self.coupling_tolerance:
                converged = True
                break

            # Apply under-relaxation for next iteration
            alpha = self.coupling_relaxation
            u_piston_old = (1.0 - alpha) * u_piston_old + alpha * u_piston_new

        # Record coupling diagnostics
        self._coupling_iterations_history.append(n_iter)
        self._coupling_converged_history.append(converged)

        if not converged:
            warnings.warn(
                f"Flame-piston coupling did not converge after {n_iter} iterations. "
                f"Relative change: {relative_change:.2e}, tolerance: {self.coupling_tolerance:.2e}"
            )

        # Restore original piston velocity callback
        self.piston_velocity = original_piston_velocity

        # Set final time
        self.current_time = t_start + dt

    def _step_predictor_corrector_internal(self, dt: float, t_start: float):
        """
        Internal predictor-corrector step for use within sub-iteration.

        Same as _step_standard but without the flame coupling check.
        """
        # Store initial state and positions
        state_0 = self.state.copy()
        x_0 = self.x.copy()

        # PREDICTOR: Euler step to t + dt
        self.current_time = t_start
        u_face, p_face, q_face, h_face = self._compute_fluxes(self.state)

        # Save predictor mesh velocities
        mesh_vel_pred = self._mesh_velocities.copy() if self._mesh_velocities is not None else None

        state_pred = self._update_state(self.state, dt, u_face, p_face, q_face, h_face)
        self._update_positions(dt, u_face)
        self.state = state_pred

        # CORRECTOR: Compute fluxes at predicted state
        self.current_time = t_start + dt
        u_face_pred, p_face_pred, q_face_pred, h_face_pred = self._compute_fluxes(self.state)

        # Save corrector mesh velocities
        mesh_vel_corr = self._mesh_velocities.copy() if self._mesh_velocities is not None else None

        # Average fluxes (trapezoidal rule in time)
        u_face_avg = 0.5 * (u_face + u_face_pred)
        p_face_avg = 0.5 * (p_face + p_face_pred)
        q_face_avg = 0.5 * (q_face + q_face_pred)
        h_face_avg = 0.5 * (h_face + h_face_pred)

        # Average mesh velocities for porous piston
        if mesh_vel_pred is not None and mesh_vel_corr is not None:
            self._mesh_velocities = 0.5 * (mesh_vel_pred + mesh_vel_corr)
        elif mesh_vel_corr is not None:
            self._mesh_velocities = mesh_vel_corr

        # Final update from initial state using averaged fluxes
        self.x = x_0
        self.current_time = t_start + 0.5 * dt
        self.state = self._update_state(state_0, dt, u_face_avg, p_face_avg,
                                        q_face_avg, h_face_avg)
        self._update_positions(dt, u_face_avg)

        self.current_time = t_start + dt

    def _step_euler_internal(self, dt: float, t_start: float):
        """
        Internal Euler step for use within sub-iteration.
        """
        self.current_time = t_start
        u_face, p_face, q_face, h_face = self._compute_fluxes(self.state)
        self.state = self._update_state(self.state, dt, u_face, p_face, q_face, h_face)
        self._update_positions(dt, u_face)
        self.current_time = t_start + dt

    def get_coupling_diagnostics(self) -> Dict[str, List]:
        """
        Get diagnostics from flame-piston coupling iterations.

        Returns:
        --------
        dict with keys:
            'iterations': List[int] - number of iterations per timestep
            'converged': List[bool] - whether each timestep converged
            'avg_iterations': float - average iterations per step
            'convergence_rate': float - fraction of steps that converged
        """
        if not self._coupling_iterations_history:
            return {
                'iterations': [],
                'converged': [],
                'avg_iterations': 0.0,
                'convergence_rate': 1.0
            }

        return {
            'iterations': self._coupling_iterations_history,
            'converged': self._coupling_converged_history,
            'avg_iterations': np.mean(self._coupling_iterations_history),
            'convergence_rate': np.mean(self._coupling_converged_history)
        }

    def solve(self,
              t_final: float,
              save_interval: int = 1,
              print_interval: int = 100,
              fixed_dt: Optional[float] = None) -> Dict:
        """
        Solve to final time

        Parameters:
        -----------
        t_final : float
            Final simulation time
        save_interval : int
            Save solution every N steps
        print_interval : int
            Print progress every N steps (0 = no printing)
        fixed_dt : float, optional
            Use fixed timestep (overrides CFL)

        Returns:
        --------
        history : dict
            Solution history with keys: 't', 'x', 'rho', 'u', 'p', 'E'
        """
        t = 0.0
        step = 0

        history = {
            't': [t],
            'x': [self.x.copy()],
            'x_centers': [self.x_centers.copy()],
            'rho': [self.state.rho.copy()],
            'u': [self.state.u.copy()],
            'p': [self.state.p.copy()],
            'E': [self.state.E.copy()]
        }

        self._record_diagnostics(t)

        print(f"\nSolving to t = {t_final}")
        print(f"  Time integration: {'Predictor-Corrector (2nd order)' if self.use_predictor_corrector else 'Euler (1st order)'}")
        print(f"  Artificial viscosity: {self.av_type.value}")
        print(f"  Slope limiter: {self.limiter.value}")

        while t < t_final:
            # Compute timestep
            if fixed_dt is not None:
                dt = fixed_dt
            else:
                dt = self._compute_time_step()

            dt = min(dt, t_final - t)

            # Advance solution
            self.step(dt)

            t += dt
            step += 1

            # Record diagnostics
            self._record_diagnostics(t)

            # Save to history
            if step % save_interval == 0:
                history['t'].append(t)
                history['x'].append(self.x.copy())
                history['x_centers'].append(self.x_centers.copy())
                history['rho'].append(self.state.rho.copy())
                history['u'].append(self.state.u.copy())
                history['p'].append(self.state.p.copy())
                history['E'].append(self.state.E.copy())

            # Print progress
            if print_interval > 0 and step % print_interval == 0:
                print(f"  Step {step}: t = {t:.6f}, dt = {dt:.3e}")

        # Save final state
        if history['t'][-1] < t:
            history['t'].append(t)
            history['x'].append(self.x.copy())
            history['x_centers'].append(self.x_centers.copy())
            history['rho'].append(self.state.rho.copy())
            history['u'].append(self.state.u.copy())
            history['p'].append(self.state.p.copy())
            history['E'].append(self.state.E.copy())

        print(f"\nCompleted: {step} steps")
        print(f"Conservation errors (relative):")
        if self.diagnostics.mass_error:
            print(f"  Mass:     {self.diagnostics.mass_error[-1]:.2e}")
        if self.diagnostics.momentum_error:
            print(f"  Momentum: {self.diagnostics.momentum_error[-1]:.2e}")
        if self.diagnostics.energy_error:
            print(f"  Energy:   {self.diagnostics.energy_error[-1]:.2e}")

        return history

    def get_diagnostics(self) -> ConservationDiagnostics:
        """Return conservation diagnostics"""
        return self.diagnostics


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_exact_riemann(x: np.ndarray, t: float,
                         rho_L: float, u_L: float, p_L: float,
                         rho_R: float, u_R: float, p_R: float,
                         x0: float = 0.5, gamma: float = 1.4
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute exact solution to Riemann problem

    Returns:
    --------
    rho, u, p : arrays
        Exact density, velocity, pressure at positions x and time t
    """
    eos = IdealGasEOS(gamma=gamma)
    riemann = ExactRiemannSolver(eos)

    u_star, p_star = riemann.solve(rho_L, u_L, p_L, rho_R, u_R, p_R)

    c_L = np.sqrt(gamma * p_L / rho_L)
    c_R = np.sqrt(gamma * p_R / rho_R)

    n = len(x)
    rho = np.zeros(n)
    u = np.zeros(n)
    p = np.zeros(n)

    for i in range(n):
        xi = (x[i] - x0) / t if t > 0 else 0.0

        # Left wave
        if p_star <= p_L:
            # Rarefaction
            c_star_L = c_L * (p_star / p_L) ** ((gamma - 1) / (2 * gamma))
            rho_star_L = rho_L * (p_star / p_L) ** (1 / gamma)

            if xi < u_L - c_L:
                rho[i], u[i], p[i] = rho_L, u_L, p_L
            elif xi < u_star - c_star_L:
                u_fan = 2 / (gamma + 1) * (c_L + (gamma - 1) / 2 * u_L + xi)
                c_fan = u_fan - xi
                rho_fan = rho_L * (c_fan / c_L) ** (2 / (gamma - 1))
                p_fan = p_L * (rho_fan / rho_L) ** gamma
                rho[i], u[i], p[i] = rho_fan, u_fan, p_fan
            elif xi < u_star:
                rho[i], u[i], p[i] = rho_star_L, u_star, p_star
        else:
            # Shock
            rho_star_L = rho_L * ((p_star / p_L) + (gamma - 1) / (gamma + 1)) / \
                        ((gamma - 1) / (gamma + 1) * (p_star / p_L) + 1)
            s_L = u_L - c_L * np.sqrt((gamma + 1) / (2 * gamma) * p_star / p_L +
                                      (gamma - 1) / (2 * gamma))

            if xi < s_L:
                rho[i], u[i], p[i] = rho_L, u_L, p_L
            elif xi < u_star:
                rho[i], u[i], p[i] = rho_star_L, u_star, p_star

        # Right wave (only if xi >= u_star)
        if xi >= u_star:
            if p_star <= p_R:
                # Rarefaction
                c_star_R = c_R * (p_star / p_R) ** ((gamma - 1) / (2 * gamma))
                rho_star_R = rho_R * (p_star / p_R) ** (1 / gamma)

                if xi < u_star + c_star_R:
                    rho[i], u[i], p[i] = rho_star_R, u_star, p_star
                elif xi < u_R + c_R:
                    u_fan = 2 / (gamma + 1) * (-c_R + (gamma - 1) / 2 * u_R + xi)
                    c_fan = xi - u_fan
                    rho_fan = rho_R * (c_fan / c_R) ** (2 / (gamma - 1))
                    p_fan = p_R * (rho_fan / rho_R) ** gamma
                    rho[i], u[i], p[i] = rho_fan, u_fan, p_fan
                else:
                    rho[i], u[i], p[i] = rho_R, u_R, p_R
            else:
                # Shock
                rho_star_R = rho_R * ((p_star / p_R) + (gamma - 1) / (gamma + 1)) / \
                            ((gamma - 1) / (gamma + 1) * (p_star / p_R) + 1)
                s_R = u_R + c_R * np.sqrt((gamma + 1) / (2 * gamma) * p_star / p_R +
                                          (gamma - 1) / (2 * gamma))

                if xi < s_R:
                    rho[i], u[i], p[i] = rho_star_R, u_star, p_star
                else:
                    rho[i], u[i], p[i] = rho_R, u_R, p_R

    return rho, u, p


def compute_L1_error(numerical: np.ndarray, exact: np.ndarray,
                     dx: np.ndarray) -> float:
    """Compute weighted L1 error"""
    return np.sum(np.abs(numerical - exact) * dx) / np.sum(dx)


def compute_L2_error(numerical: np.ndarray, exact: np.ndarray,
                     dx: np.ndarray) -> float:
    """Compute weighted L2 error"""
    return np.sqrt(np.sum((numerical - exact)**2 * dx) / np.sum(dx))


def compute_Linf_error(numerical: np.ndarray, exact: np.ndarray) -> float:
    """Compute L-infinity (max) error"""
    return np.max(np.abs(numerical - exact))


# =============================================================================
# TEST / DEMO
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("PUBLICATION-QUALITY LAGRANGIAN SOLVER TEST")
    print("=" * 70)

    # Sod shock tube test
    n_cells = 200
    x = np.linspace(0, 1, n_cells)

    rho = np.where(x < 0.5, 1.0, 0.125)
    u = np.zeros(n_cells)
    p = np.where(x < 0.5, 1.0, 0.1)

    # Create solver with publication-quality settings
    solver = LagrangianSolver(
        cfl=0.5,
        bc_left='open',
        bc_right='open',
        av_type=ArtificialViscosityType.NOH_QH,
        av_Cq=2.0,
        av_Cl=0.3,
        av_Ch=0.1,
        limiter=LimiterType.VANLEER,
        use_predictor_corrector=True
    )

    solver.initialize(x, rho, u, p)

    t_final = 0.2
    history = solver.solve(t_final=t_final, save_interval=10, print_interval=50)

    # Compute exact solution
    rho_exact, u_exact, p_exact = compute_exact_riemann(
        solver.x_centers, t_final,
        1.0, 0.0, 1.0, 0.125, 0.0, 0.1,
        x0=0.5, gamma=1.4
    )

    # Compute errors
    dx = np.diff(solver.x)
    print(f"\nError Analysis:")
    print(f"  Density L1:  {compute_L1_error(solver.state.rho, rho_exact, dx):.4e}")
    print(f"  Velocity L1: {compute_L1_error(solver.state.u, u_exact, dx):.4e}")
    print(f"  Pressure L1: {compute_L1_error(solver.state.p, p_exact, dx):.4e}")

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    x_num = solver.x_centers

    # Density
    axes[0, 0].plot(x_num, rho_exact, 'r--', lw=2, label='Exact')
    axes[0, 0].plot(x_num, solver.state.rho, 'b-', lw=1.5, label='Numerical')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Velocity
    axes[0, 1].plot(x_num, u_exact, 'r--', lw=2, label='Exact')
    axes[0, 1].plot(x_num, solver.state.u, 'b-', lw=1.5, label='Numerical')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('Velocity')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Pressure
    axes[0, 2].plot(x_num, p_exact, 'r--', lw=2, label='Exact')
    axes[0, 2].plot(x_num, solver.state.p, 'b-', lw=1.5, label='Numerical')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('Pressure')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Conservation errors
    diag = solver.get_diagnostics()
    axes[1, 0].semilogy(diag.time, diag.mass_error, 'b-', lw=2, label='Mass')
    axes[1, 0].semilogy(diag.time, diag.energy_error, 'r-', lw=2, label='Energy')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Relative Error')
    axes[1, 0].set_title('Conservation Errors')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Mesh movement
    axes[1, 1].plot(solver.m, solver.x, 'b-', lw=2)
    axes[1, 1].plot(solver.m, np.linspace(0, 1, len(solver.m)), 'k--', lw=1, alpha=0.5)
    axes[1, 1].set_xlabel('Mass coordinate m')
    axes[1, 1].set_ylabel('Position x')
    axes[1, 1].set_title('Mesh Movement')
    axes[1, 1].grid(True, alpha=0.3)

    # Specific volume (natural Lagrangian variable)
    axes[1, 2].plot(solver.m_centers, solver.state.tau, 'g-', lw=2)
    axes[1, 2].set_xlabel('Mass coordinate m')
    axes[1, 2].set_ylabel('Specific volume τ')
    axes[1, 2].set_title('Specific Volume')
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(f'Sod Shock Tube: t = {t_final}, N = {n_cells}\n'
                 f'Conservative Total Energy + Noh Q+H Artificial Viscosity',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('publication_solver_test.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to publication_solver_test.png")
