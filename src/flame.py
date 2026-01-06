"""
Flame Speed Providers for Coupled Piston-Flame Simulations
===========================================================

Provides interfaces and implementations for computing laminar flame speed
as a function of thermodynamic state (T, P, phi).

Two main implementations:
1. TabularFlameSpeed - Pre-computed lookup table with interpolation
2. CanteraFlameSpeed - Direct calculation using Cantera's flame solver

Author: Generated for research purposes
Date: 2024
"""

import os
import tempfile
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Union
import warnings


class FlameSpeedProvider(ABC):
    """
    Abstract base class for flame speed calculation.

    Provides a unified interface for obtaining laminar flame speed S_L
    as a function of thermodynamic state.
    """

    @abstractmethod
    def get_flame_speed(self, T: float, P: float, phi: float = 1.0) -> float:
        """
        Compute laminar flame speed at given conditions.

        Parameters:
        -----------
        T : float
            Unburned gas temperature [K]
        P : float
            Pressure [Pa]
        phi : float
            Equivalence ratio (default 1.0 for stoichiometric)

        Returns:
        --------
        S_L : float
            Laminar flame speed [m/s]
        """
        pass

    @abstractmethod
    def get_burned_density(self, T: float, P: float, phi: float = 1.0) -> float:
        """
        Compute burned gas density at given unburned conditions.

        Parameters:
        -----------
        T : float
            Unburned gas temperature [K]
        P : float
            Pressure [Pa]
        phi : float
            Equivalence ratio

        Returns:
        --------
        rho_b : float
            Burned gas density [kg/m^3]
        """
        pass


class ConstantFlameSpeed(FlameSpeedProvider):
    """
    Constant flame speed provider for testing and simple cases.

    Useful for verification and cases where flame speed variation is negligible.
    """

    def __init__(self, S_L: float = 0.4, expansion_ratio: float = 7.0):
        """
        Parameters:
        -----------
        S_L : float
            Constant laminar flame speed [m/s] (default 0.4 for CH4-air)
        expansion_ratio : float
            Density ratio rho_u / rho_b (default 7.0 for typical hydrocarbon)
        """
        self.S_L = S_L
        self.expansion_ratio = expansion_ratio

    def get_flame_speed(self, T: float, P: float, phi: float = 1.0) -> float:
        return self.S_L

    def get_burned_density(self, T: float, P: float, phi: float = 1.0) -> float:
        # Approximate unburned density from ideal gas
        R_air = 287.0  # J/(kg·K)
        rho_u = P / (R_air * T)
        return rho_u / self.expansion_ratio


class TabularFlameSpeed(FlameSpeedProvider):
    """
    Pre-tabulated flame speed with trilinear interpolation.

    The table is a 3D array indexed by (T, P, phi).
    Interpolation uses scipy's RegularGridInterpolator for efficiency.
    """

    def __init__(self,
                 T_values: np.ndarray,
                 P_values: np.ndarray,
                 phi_values: np.ndarray,
                 S_L_table: np.ndarray,
                 rho_b_table: Optional[np.ndarray] = None,
                 extrapolate: bool = False):
        """
        Parameters:
        -----------
        T_values : array [n_T]
            Temperature grid points [K]
        P_values : array [n_P]
            Pressure grid points [Pa]
        phi_values : array [n_phi]
            Equivalence ratio grid points
        S_L_table : array [n_T, n_P, n_phi]
            Flame speed values [m/s]
        rho_b_table : array [n_T, n_P, n_phi], optional
            Burned density values [kg/m^3]
        extrapolate : bool
            If True, extrapolate outside table bounds; if False, clamp to edges
        """
        from scipy.interpolate import RegularGridInterpolator

        self.T_values = np.asarray(T_values)
        self.P_values = np.asarray(P_values)
        self.phi_values = np.asarray(phi_values)
        self.S_L_table = np.asarray(S_L_table)
        self.extrapolate = extrapolate

        # Validate table dimensions
        expected_shape = (len(T_values), len(P_values), len(phi_values))
        if S_L_table.shape != expected_shape:
            raise ValueError(f"S_L_table shape {S_L_table.shape} doesn't match "
                             f"grid dimensions {expected_shape}")

        # Create interpolators
        bounds_error = not extrapolate
        fill_value = None if extrapolate else np.nan

        self._S_L_interp = RegularGridInterpolator(
            (T_values, P_values, phi_values),
            S_L_table,
            method='linear',
            bounds_error=bounds_error,
            fill_value=fill_value
        )

        if rho_b_table is not None:
            self.rho_b_table = np.asarray(rho_b_table)
            self._rho_b_interp = RegularGridInterpolator(
                (T_values, P_values, phi_values),
                rho_b_table,
                method='linear',
                bounds_error=bounds_error,
                fill_value=fill_value
            )
        else:
            self.rho_b_table = None
            self._rho_b_interp = None

    def get_flame_speed(self, T: float, P: float, phi: float = 1.0) -> float:
        """Interpolate flame speed from table"""
        # Clamp to table bounds if not extrapolating
        if not self.extrapolate:
            T = np.clip(T, self.T_values[0], self.T_values[-1])
            P = np.clip(P, self.P_values[0], self.P_values[-1])
            phi = np.clip(phi, self.phi_values[0], self.phi_values[-1])

        try:
            S_L = float(self._S_L_interp((T, P, phi)))
            if np.isnan(S_L):
                warnings.warn(f"NaN flame speed at T={T}, P={P}, phi={phi}")
                return 0.0
            return max(S_L, 0.0)
        except Exception as e:
            warnings.warn(f"Interpolation failed: {e}")
            return 0.0

    def get_burned_density(self, T: float, P: float, phi: float = 1.0) -> float:
        """Interpolate burned density from table"""
        if self._rho_b_interp is None:
            # Fallback: estimate from ideal gas with adiabatic temperature rise
            Delta_T = 1800.0  # Approximate for stoichiometric
            T_burned = T + Delta_T
            return P / (287.0 * T_burned)

        if not self.extrapolate:
            T = np.clip(T, self.T_values[0], self.T_values[-1])
            P = np.clip(P, self.P_values[0], self.P_values[-1])
            phi = np.clip(phi, self.phi_values[0], self.phi_values[-1])

        try:
            rho_b = float(self._rho_b_interp((T, P, phi)))
            if np.isnan(rho_b):
                return P / (287.0 * (T + 1800.0))
            return max(rho_b, 1e-6)
        except Exception:
            return P / (287.0 * (T + 1800.0))

    @classmethod
    def from_file(cls, filepath: str) -> 'TabularFlameSpeed':
        """
        Load table from file (NumPy .npz format).

        Expected keys: 'T', 'P', 'phi', 'S_L', optionally 'rho_b'
        """
        data = np.load(filepath)
        return cls(
            T_values=data['T'],
            P_values=data['P'],
            phi_values=data['phi'],
            S_L_table=data['S_L'],
            rho_b_table=data.get('rho_b', None)
        )

    def save(self, filepath: str):
        """Save table to file"""
        save_dict = {
            'T': self.T_values,
            'P': self.P_values,
            'phi': self.phi_values,
            'S_L': self.S_L_table
        }
        if self.rho_b_table is not None:
            save_dict['rho_b'] = self.rho_b_table
        np.savez(filepath, **save_dict)


class CSVTabulatedFlameSpeed(FlameSpeedProvider):
    """
    2D tabulated flame speed from CSV file with bilinear interpolation.

    Reads flame properties from a CSV file (e.g., from cantera_data/output/flame_properties.csv)
    and provides 2D interpolation on (T, P) for flame speed and burned density.

    Expected CSV format:
        T [K],P [Pa],phi,Su [m/s],delta_f [m],T_ad [K],rho_u [kg/m3],rho_b [kg/m3],...

    The data is assumed to be on a regular grid in T and P (at fixed phi).
    """

    def __init__(self,
                 csv_path: str,
                 extrapolate: bool = False,
                 phi_column: str = 'phi',
                 verbose: bool = True):
        """
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file containing flame properties
        extrapolate : bool
            If True, extrapolate outside table bounds; if False, clamp to edges
        phi_column : str
            Name of the equivalence ratio column (default 'phi')
        verbose : bool
            If True, print loading information
        """
        from scipy.interpolate import RegularGridInterpolator
        import pandas as pd
        from pathlib import Path

        self.csv_path = csv_path
        self.extrapolate = extrapolate

        # Load CSV
        if verbose:
            print(f"\nCSVTabulatedFlameSpeed: Loading {csv_path}")

        df = pd.read_csv(csv_path)

        # Clean column names (remove units in brackets)
        df.columns = [col.split('[')[0].strip() for col in df.columns]

        # Filter only converged solutions
        if 'converged' in df.columns:
            n_total = len(df)
            df = df[df['converged'] == True]
            n_converged = len(df)
            if verbose:
                print(f"  Loaded {n_converged}/{n_total} converged solutions")
        else:
            if verbose:
                print(f"  Loaded {len(df)} data points")

        # Extract unique T and P values (sorted)
        self.T_values = np.sort(df['T'].unique())
        self.P_values = np.sort(df['P'].unique())
        self.n_T = len(self.T_values)
        self.n_P = len(self.P_values)

        # Get phi value (assumed constant)
        self.phi = df[phi_column].iloc[0] if phi_column in df.columns else 1.0

        if verbose:
            print(f"  T range: {self.T_values[0]:.1f} - {self.T_values[-1]:.1f} K ({self.n_T} points)")
            print(f"  P range: {self.P_values[0] / 1e5:.2f} - {self.P_values[-1] / 1e5:.2f} bar ({self.n_P} points)")
            print(f"  Phi: {self.phi:.3f}")

        # Create 2D grids for Su and rho_b
        # Grid shape: (n_T, n_P)
        self.Su_grid = np.full((self.n_T, self.n_P), np.nan)
        self.rho_b_grid = np.full((self.n_T, self.n_P), np.nan)
        self.T_ad_grid = np.full((self.n_T, self.n_P), np.nan)

        # Fill grids from dataframe
        for _, row in df.iterrows():
            i_T = np.searchsorted(self.T_values, row['T'])
            i_P = np.searchsorted(self.P_values, row['P'])

            # Handle edge cases from searchsorted
            if i_T >= self.n_T:
                i_T = self.n_T - 1
            if i_P >= self.n_P:
                i_P = self.n_P - 1

            # Verify we found the right indices
            if (abs(self.T_values[i_T] - row['T']) < 1.0 and
                    abs(self.P_values[i_P] - row['P']) < 100.0):
                self.Su_grid[i_T, i_P] = row['Su']
                if 'rho_b' in df.columns:
                    self.rho_b_grid[i_T, i_P] = row['rho_b']
                if 'T_ad' in df.columns:
                    self.T_ad_grid[i_T, i_P] = row['T_ad']

        # Count valid points
        n_valid = np.sum(~np.isnan(self.Su_grid))
        if verbose:
            print(f"  Valid grid points: {n_valid}/{self.n_T * self.n_P}")

        # Create interpolators
        bounds_error = not extrapolate
        fill_value = None if extrapolate else np.nan

        self._Su_interp = RegularGridInterpolator(
            (self.T_values, self.P_values),
            self.Su_grid,
            method='linear',
            bounds_error=bounds_error,
            fill_value=fill_value
        )

        if not np.all(np.isnan(self.rho_b_grid)):
            self._rho_b_interp = RegularGridInterpolator(
                (self.T_values, self.P_values),
                self.rho_b_grid,
                method='linear',
                bounds_error=bounds_error,
                fill_value=fill_value
            )
        else:
            self._rho_b_interp = None

        if not np.all(np.isnan(self.T_ad_grid)):
            self._T_ad_interp = RegularGridInterpolator(
                (self.T_values, self.P_values),
                self.T_ad_grid,
                method='linear',
                bounds_error=bounds_error,
                fill_value=fill_value
            )
        else:
            self._T_ad_interp = None

        if verbose:
            Su_valid = self.Su_grid[~np.isnan(self.Su_grid)]
            print(f"  Su range: {np.min(Su_valid):.4f} - {np.max(Su_valid):.4f} m/s")

    def get_flame_speed(self, T: float, P: float, phi: float = 1.0) -> float:
        """
        Interpolate flame speed from table.

        Parameters:
        -----------
        T : float
            Unburned gas temperature [K]
        P : float
            Pressure [Pa]
        phi : float
            Equivalence ratio (ignored - table is at fixed phi)

        Returns:
        --------
        S_L : float
            Laminar flame speed [m/s]
        """
        # Clamp to table bounds if not extrapolating
        if not self.extrapolate:
            T = np.clip(T, self.T_values[0], self.T_values[-1])
            P = np.clip(P, self.P_values[0], self.P_values[-1])

        try:
            S_L = float(self._Su_interp((T, P)))
            if np.isnan(S_L):
                warnings.warn(f"NaN flame speed at T={T:.1f}K, P={P / 1e5:.2f}bar - using nearest valid")
                return self._get_nearest_valid(T, P, self.Su_grid)
            return max(S_L, 0.0)
        except Exception as e:
            warnings.warn(f"Interpolation failed at T={T:.1f}K, P={P / 1e5:.2f}bar: {e}")
            return self._get_nearest_valid(T, P, self.Su_grid)

    def get_burned_density(self, T: float, P: float, phi: float = 1.0) -> float:
        """
        Interpolate burned density from table.

        Parameters:
        -----------
        T : float
            Unburned gas temperature [K]
        P : float
            Pressure [Pa]
        phi : float
            Equivalence ratio (ignored - table is at fixed phi)

        Returns:
        --------
        rho_b : float
            Burned gas density [kg/m^3]
        """
        if self._rho_b_interp is None:
            # Fallback: estimate from ideal gas with T_ad
            T_ad = self.get_adiabatic_temperature(T, P, phi)
            return P / (287.0 * T_ad)

        if not self.extrapolate:
            T = np.clip(T, self.T_values[0], self.T_values[-1])
            P = np.clip(P, self.P_values[0], self.P_values[-1])

        try:
            rho_b = float(self._rho_b_interp((T, P)))
            if np.isnan(rho_b):
                return self._get_nearest_valid(T, P, self.rho_b_grid)
            return max(rho_b, 1e-6)
        except Exception:
            return self._get_nearest_valid(T, P, self.rho_b_grid)

    def get_adiabatic_temperature(self, T: float, P: float, phi: float = 1.0) -> float:
        """
        Interpolate adiabatic flame temperature from table.

        Parameters:
        -----------
        T : float
            Unburned gas temperature [K]
        P : float
            Pressure [Pa]
        phi : float
            Equivalence ratio (ignored)

        Returns:
        --------
        T_ad : float
            Adiabatic flame temperature [K]
        """
        if self._T_ad_interp is None:
            # Fallback estimate
            return T + 2000.0

        if not self.extrapolate:
            T = np.clip(T, self.T_values[0], self.T_values[-1])
            P = np.clip(P, self.P_values[0], self.P_values[-1])

        try:
            T_ad = float(self._T_ad_interp((T, P)))
            if np.isnan(T_ad):
                return self._get_nearest_valid(T, P, self.T_ad_grid)
            return T_ad
        except Exception:
            return self._get_nearest_valid(T, P, self.T_ad_grid)

    def _get_nearest_valid(self, T: float, P: float, grid: np.ndarray) -> float:
        """Find nearest valid (non-NaN) value in grid."""
        # Find closest indices
        i_T = np.argmin(np.abs(self.T_values - T))
        i_P = np.argmin(np.abs(self.P_values - P))

        # If valid, return it
        if not np.isnan(grid[i_T, i_P]):
            return grid[i_T, i_P]

        # Search expanding neighborhood for valid value
        for radius in range(1, max(self.n_T, self.n_P)):
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    ii = i_T + di
                    jj = i_P + dj
                    if 0 <= ii < self.n_T and 0 <= jj < self.n_P:
                        if not np.isnan(grid[ii, jj]):
                            return grid[ii, jj]

        # Last resort: return median of valid values
        valid = grid[~np.isnan(grid)]
        return np.median(valid) if len(valid) > 0 else 0.0

    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get the T and P bounds of the table."""
        return {
            'T': (self.T_values[0], self.T_values[-1]),
            'P': (self.P_values[0], self.P_values[-1]),
        }

    def get_diagnostics(self) -> Dict:
        """Get diagnostic information about the table."""
        Su_valid = self.Su_grid[~np.isnan(self.Su_grid)]
        return {
            'csv_path': self.csv_path,
            'n_T': self.n_T,
            'n_P': self.n_P,
            'T_range': (self.T_values[0], self.T_values[-1]),
            'P_range': (self.P_values[0], self.P_values[-1]),
            'phi': self.phi,
            'n_valid': len(Su_valid),
            'Su_min': float(np.min(Su_valid)) if len(Su_valid) > 0 else np.nan,
            'Su_max': float(np.max(Su_valid)) if len(Su_valid) > 0 else np.nan,
            'Su_mean': float(np.mean(Su_valid)) if len(Su_valid) > 0 else np.nan,
        }


class CanteraFlameSpeed(FlameSpeedProvider):
    """
    Direct flame speed calculation using Cantera's 1D flame solver.

    This provides accurate flame speed values but is computationally expensive.
    A cache is used to avoid redundant calculations for similar states.
    Supports restart from previous solution for faster convergence.

    Requires Cantera to be installed.
    """

    def __init__(self,
                 mechanism: str = 'gri30.yaml',
                 fuel: str = 'H2',
                 oxidizer: str = 'O2:1',
                 width: float = 1e-8,
                 loglevel: int = 0,
                 cache_size: int = 100,
                 cache_tolerance: float = 0.01,
                 save_flame_profiles: bool = False,
                 flame_profiles_dir: Optional[str] = None):
        """
        Parameters:
        -----------
        mechanism : str
            Cantera mechanism file (e.g., 'gri30.yaml')
        fuel : str
            Fuel species name
        oxidizer : str
            Oxidizer composition string
        width : float
            Initial flame domain width [m] (default 1e-8, Cantera auto-expands)
        loglevel : int
            Cantera solver verbosity (0 = silent)
        cache_size : int
            Maximum number of cached flame speed values
        cache_tolerance : float
            Relative tolerance for cache lookup (e.g., 0.01 = 1%)
        save_flame_profiles : bool
            If True, save flame profiles (T, u, HRR) at output intervals
        flame_profiles_dir : str, optional
            Directory to save flame profiles. If None and save_flame_profiles is True,
            uses 'flame_profiles' in current directory.
        """
        try:
            import cantera as ct
            self._ct = ct
        except ImportError:
            raise ImportError(
                "Cantera is required for CanteraFlameSpeed. "
                "Install with: pip install cantera"
            )

        self.mechanism = mechanism
        self.fuel = fuel
        self.oxidizer = oxidizer
        self.width = width
        self.loglevel = loglevel

        # Initialize gas object
        self.gas = ct.Solution(mechanism)

        # Cache for flame speed values: {(T, P, phi): (S_L, rho_b)}
        self._cache: Dict[Tuple[float, float, float], Tuple[float, float]] = {}
        self._cache_size = cache_size
        self._cache_tolerance = cache_tolerance
        self._cache_order = []  # LRU tracking

        # Restart support using YAML save/restore for robust continuation
        self._last_flame = None
        self._last_successful_T = None
        self._last_successful_P = None
        self._has_saved_solution = False
        # Create restart file path
        if flame_profiles_dir:
            self._restart_file = os.path.join(flame_profiles_dir, 'cantera_flame_restart.yaml')
        else:
            self._restart_file = os.path.join(tempfile.gettempdir(), 'cantera_flame_restart.yaml')

        # Flame profile saving configuration
        self.save_flame_profiles = save_flame_profiles
        self._should_save_profile = False  # Flag to trigger saving at output intervals
        self._save_step_number = 0  # Actual time step number for filename
        if save_flame_profiles:
            if flame_profiles_dir is None:
                flame_profiles_dir = 'flame_profiles'
            from pathlib import Path
            self.flame_profiles_dir = Path(flame_profiles_dir)
            self.flame_profiles_dir.mkdir(parents=True, exist_ok=True)
            print(f"Flame profiles will be saved to: {self.flame_profiles_dir}")
        else:
            self.flame_profiles_dir = None

    def trigger_profile_save(self, step: int = 0):
        """Call this to trigger saving a profile on the next flame solve.

        Parameters:
            step: The actual simulation time step number for the filename
        """
        self._should_save_profile = True
        self._save_step_number = step

    def _find_cached(self, T: float, P: float, phi: float) -> Optional[Tuple[float, float]]:
        """Look for a cached value within tolerance"""
        for (T_c, P_c, phi_c), result in self._cache.items():
            if (abs(T - T_c) / T_c < self._cache_tolerance and
                    abs(P - P_c) / P_c < self._cache_tolerance and
                    abs(phi - phi_c) / phi_c < self._cache_tolerance):
                return result
        return None

    def _add_to_cache(self, T: float, P: float, phi: float, S_L: float, rho_b: float):
        """Add result to cache, evicting oldest if full"""
        key = (T, P, phi)

        if len(self._cache) >= self._cache_size:
            # Remove oldest entry
            oldest_key = self._cache_order.pop(0)
            del self._cache[oldest_key]

        self._cache[key] = (S_L, rho_b)
        self._cache_order.append(key)

    def _save_flame_profile(self, flame, T_u: float, P: float, phi: float) -> str:
        """
        Save flame profile figure with sequenced naming.

        Creates a single figure with Temperature, Velocity, and HRR subplots.

        Parameters:
            flame: Cantera FreeFlame object after solve
            T_u: Unburned gas temperature [K]
            P: Pressure [Pa]
            phi: Equivalence ratio

        Returns:
            Path to saved figure
        """
        if not self.save_flame_profiles or self.flame_profiles_dir is None:
            return None

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Get flame profile data
        x_mm = flame.grid * 1000  # Convert to mm for plotting
        T = flame.T
        u = flame.velocity
        hrr = flame.heat_release_rate
        S_L = flame.velocity[0]

        dpi = 150
        idx = self._save_step_number  # Use actual time step number

        # Create single figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), dpi=dpi, sharex=True)

        # Temperature Profile
        ax1.plot(x_mm, T, 'r-', linewidth=2)
        ax1.set_ylabel('Temperature (K)', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Cantera Free Flame Profiles (T_u={T_u:.1f} K, P={P / 1e5:.2f} bar, S_L={S_L * 100:.2f} cm/s)',
                      fontsize=14, fontweight='bold')

        # Heat Release Rate Profile with fill
        hrr_GW = hrr / 1e9  # Convert to GW/m³
        ax2.plot(x_mm, hrr_GW, 'orange', linewidth=2)
        ax2.fill_between(x_mm, 0, hrr_GW, alpha=0.3, color='orange')
        ax2.set_ylabel('HRR (GW/m³)', fontsize=11)
        ax2.grid(True, alpha=0.3)

        # Velocity Profile
        u_cms = u * 100  # Convert to cm/s
        ax3.plot(x_mm, u_cms, 'b-', linewidth=2)
        ax3.axhline(y=u_cms[0], color='g', linestyle='--', alpha=0.5)
        ax3.text(x_mm[-1] * 0.7, u_cms[0] * 1.05, f'S_L = {u_cms[0]:.1f} cm/s', fontsize=10, color='g')
        ax3.set_ylabel('Velocity (cm/s)', fontsize=11)
        ax3.set_xlabel('Position (mm)', fontsize=12)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        fig_path = self.flame_profiles_dir / f'flame_profile_step_{idx:04d}.png'
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        return str(fig_path)

    def _save_flame_solution(self, flame) -> None:
        """Save the current flame solution to YAML file for restart."""
        if flame is not None:
            try:
                flame.save(self._restart_file, name='solution',
                           description=f'T={self._last_successful_T:.1f}K, P={self._last_successful_P / 1e5:.1f}bar',
                           overwrite=True)
                self._has_saved_solution = True
            except Exception:
                pass  # Silently fail - restart is optional optimization

    def _load_flame_solution(self, flame, T: float, P: float, phi: float) -> bool:
        """Load a previously saved flame solution into a NEW flame object."""
        if not self._has_saved_solution or not os.path.exists(self._restart_file):
            return False
        try:
            flame.restore(self._restart_file, name='solution', loglevel=0)
            flame.inlet.T = T
            flame.inlet.X = self.gas.X
            flame.P = P
            flame.set_max_grid_points(flame.flame, 2000)
            return True
        except Exception:
            # Remove corrupted restart file
            self._has_saved_solution = False
            try:
                if os.path.exists(self._restart_file):
                    os.remove(self._restart_file)
            except Exception:
                pass
            return False

    def _solve_flame(self, T: float, P: float, phi: float) -> Tuple[float, float]:
        """
        Solve 1D freely-propagating flame to get flame speed.

        Uses Cantera's set_equivalence_ratio to properly handle stoichiometry
        for any fuel-oxidizer combination. Supports restart from previous solution
        using YAML save/restore for robustness.

        Parameters:
        -----------
        T : float
            Unburned gas temperature [K]
        P : float
            Pressure [Pa]
        phi : float
            Equivalence ratio

        Returns:
        --------
        S_L : float
            Laminar flame speed [m/s]
        rho_b : float
            Burned gas density [kg/m^3]
        """
        ct = self._ct

        # Set gas state using Cantera's equivalence ratio method
        # This properly handles stoichiometry for any fuel-oxidizer combination
        self.gas.TP = T, P
        self.gas.set_equivalence_ratio(phi, self.fuel, self.oxidizer)

        # Store unburned density
        rho_u = self.gas.density

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Attempt 0: Try fast path - reuse last flame object directly
                if attempt == 0 and self._last_flame is not None:
                    flame = self._last_flame
                    flame.inlet.T = T
                    flame.inlet.X = self.gas.X
                    flame.P = P
                    # Set refinement criteria
                    P_bar = P / 1e5
                    if P_bar > 100:
                        flame.set_refine_criteria(ratio=4, slope=0.01, curve=0.01, prune=0.0)
                    else:
                        flame.set_refine_criteria(ratio=3, slope=0.02, curve=0.02)
                    # Solve with existing grid (fast)
                    flame.solve(loglevel=self.loglevel, refine_grid=True, auto=False)

                # Attempt 1+: Create new flame, restore from YAML if available
                elif attempt > 0 and self._has_saved_solution:
                    flame = ct.FreeFlame(self.gas, width=self.width)
                    if self._load_flame_solution(flame, T, P, phi):
                        P_bar = P / 1e5
                        if P_bar > 100:
                            flame.set_refine_criteria(ratio=4, slope=0.01, curve=0.01, prune=0.0)
                        else:
                            flame.set_refine_criteria(ratio=3, slope=0.02, curve=0.02)
                        flame.solve(loglevel=self.loglevel, refine_grid=True, auto=False)
                    else:
                        raise RuntimeError("YAML restore failed")

                # Fresh solve (no previous solution or all else failed)
                else:
                    flame = ct.FreeFlame(self.gas, width=self.width)
                    flame.set_initial_guess()
                    P_bar = P / 1e5
                    if P_bar > 100:
                        flame.set_refine_criteria(ratio=4, slope=0.01, curve=0.01, prune=0.0)
                    else:
                        flame.set_refine_criteria(ratio=3, slope=0.02, curve=0.02)
                    flame.solve(loglevel=self.loglevel, refine_grid=True, auto=True)

                # Get flame speed (velocity at inlet)
                S_L = flame.velocity[0]

                # Get burned gas density (at outlet)
                rho_b = flame.density[-1]

                # Store successful flame for restart
                self._last_flame = flame
                self._last_successful_T = T
                self._last_successful_P = P
                # Save to YAML for robust restart on next solve
                self._save_flame_solution(flame)

                # Save flame profile only if triggered (at output intervals)
                if self._should_save_profile:
                    self._save_flame_profile(flame, T, P, phi)
                    self._should_save_profile = False  # Reset flag

                return S_L, rho_b

            except Exception as e:
                if attempt < max_retries - 1:
                    # Reset and retry with fresh flame
                    self._last_flame = None
                    continue
                else:
                    # Reset restart state
                    self._last_flame = None
                    self._has_saved_solution = False
                    # Raise error - no fallback to power-law approximations
                    raise RuntimeError(
                        f"Cantera flame solve failed after {max_retries} attempts at "
                        f"T={T:.1f} K, P={P:.0f} Pa, phi={phi:.3f}. "
                        f"Last error: {e}. "
                        f"Consider using pre-tabulated flame speed data (CSVTabulatedFlameSpeed) "
                        f"or adjusting solver parameters."
                    )

    def get_flame_speed(self, T: float, P: float, phi: float = 1.0) -> float:
        """Get flame speed, using cache if available"""
        # Check cache
        cached = self._find_cached(T, P, phi)
        if cached is not None:
            return cached[0]

        # Solve flame
        S_L, rho_b = self._solve_flame(T, P, phi)

        # Cache result
        self._add_to_cache(T, P, phi, S_L, rho_b)

        return S_L

    def get_burned_density(self, T: float, P: float, phi: float = 1.0) -> float:
        """Get burned density, using cache if available"""
        # Check cache
        cached = self._find_cached(T, P, phi)
        if cached is not None:
            return cached[1]

        # Solve flame (will also cache S_L)
        S_L, rho_b = self._solve_flame(T, P, phi)

        # Cache result
        self._add_to_cache(T, P, phi, S_L, rho_b)

        return rho_b

    def clear_cache(self):
        """Clear the flame speed cache"""
        self._cache.clear()
        self._cache_order.clear()


class ActivationEnergyFlameSpeed(FlameSpeedProvider):
    """
    Flame speed provider using activation energy correlation (Clavin formulation).

    Uses the full Clavin equation from Tofaili & Clavin (2021):
        S_L = S_L0 * (T_b/T_b0)^2 * (T_u/T_u0)^(3/2) * exp( Ea/(2R) * (1/T_b0 - 1/T_b) )

    Where:
        S_L0  : Reference flame speed at initial conditions (from Cantera FreeFlame)
        T_b0  : Reference burned gas temperature (from same FreeFlame solution)
        T_u0  : Reference unburned gas temperature [K]
        T_u   : Current unburned gas temperature [K]
        T_b   : Burned gas temperature at current conditions (from equilibrate HP)
        Ea    : Overall activation energy (from inert dilution method)

    The activation energy is computed using the inert dilution method:
        1. Run FreeFlame at multiple N2 dilution levels (constant T_u)
        2. Compute burning flux = rho_molar * S_L for each
        3. Plot ln(burning_flux) vs 1/T_ad
        4. Extract Ea from slope: Ea = -2 * R * slope

    References:
        - Tofaili & Clavin, Combustion and Flame 232 (2021) 111522
        - Law, Combustion Physics (textbook)
    """

    def __init__(self,
                 mechanism: str,
                 fuel: str,
                 oxidizer: str,
                 T0: float,
                 P0: float,
                 phi0: float = 1.0,
                 n_dilution_points: int = 3,
                 dilution_range: Tuple[float, float] = (0.0, 0.2),
                 inert_species: str = 'N2',
                 width: float = 0.03,
                 loglevel: int = 0,
                 save_flame_profiles: bool = False,
                 flame_profiles_dir: Optional[str] = None):
        """
        Parameters:
        -----------
        mechanism : str
            Cantera mechanism file (e.g., 'gri30.yaml', 'LiDryer.yaml')
        fuel : str
            Fuel species name (e.g., 'H2', 'CH4')
        oxidizer : str
            Oxidizer species (e.g., 'O2', 'O2:1, N2:3.76')
        T0 : float
            Reference unburned gas temperature [K]
        P0 : float
            Reference pressure [Pa]
        phi0 : float
            Reference equivalence ratio (default 1.0)
        n_dilution_points : int
            Number of dilution points for activation energy calculation (default 3)
        dilution_range : tuple (min, max)
            Range of additional inert moles to add (default 0.0-0.2)
        inert_species : str
            Inert species for dilution (default 'N2')
        width : float
            Initial flame domain width [m] (default 0.03)
        loglevel : int
            Cantera solver verbosity (0 = silent)
        save_flame_profiles : bool
            If True, save flame profiles (T, u, HRR) from each FreeFlame simulation
            during initialization (default False)
        flame_profiles_dir : str, optional
            Directory to save flame profiles. If None and save_flame_profiles is True,
            uses 'flame_profiles' in current directory.
        """
        try:
            import cantera as ct
            self._ct = ct
        except ImportError:
            raise ImportError(
                "Cantera is required for ActivationEnergyFlameSpeed. "
                "Install with: pip install cantera"
            )

        # Flame profile saving configuration
        self.save_flame_profiles = save_flame_profiles
        if save_flame_profiles:
            if flame_profiles_dir is None:
                flame_profiles_dir = 'flame_profiles'
            from pathlib import Path
            self.flame_profiles_dir = Path(flame_profiles_dir)
            self.flame_profiles_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Flame profiles will be saved to: {self.flame_profiles_dir}")
        else:
            self.flame_profiles_dir = None
        self._profile_counter = 0  # Counter for sequenced file naming

        self.mechanism = mechanism
        self.fuel = fuel
        self.oxidizer = oxidizer
        self.T0 = T0
        self.P0 = P0
        self.phi0 = phi0
        self.width = width
        self.loglevel = loglevel
        self.n_dilution_points = n_dilution_points
        self.dilution_range = dilution_range
        self.inert_species = inert_species

        # Store T_u0 for flame speed equation (reference unburned temperature)
        self.T_u0 = T0

        # Initialize gas object
        self.gas = ct.Solution(mechanism)

        # Compute reference values from Cantera FreeFlame
        print(f"\nActivationEnergyFlameSpeed: Computing reference values...")
        print(f"  Mechanism: {mechanism}")
        print(f"  Fuel: {fuel}, Oxidizer: {oxidizer}")
        print(f"  Reference conditions: T_u0={T0:.1f} K, P0={P0 / 1e5:.2f} bar, phi={phi0:.2f}")

        # Get S_L0, T_b0, rho_b0 from FreeFlame at reference conditions
        self.S_L0, self.T_b0, self.rho_b0 = self._compute_initial_flame_properties()
        print(f"  S_L0 = {self.S_L0:.4f} m/s")
        print(f"  T_b0 = {self.T_b0:.1f} K (adiabatic flame temperature)")
        print(f"  rho_b0 = {self.rho_b0:.4f} kg/m³")

        # Compute activation energy using Law's method
        self.Ea, self.arrhenius_data = self._compute_activation_energy()
        print(f"  Ea = {self.Ea / 1000:.2f} kJ/mol (from Law's method)")
        print(f"  R² = {self.arrhenius_data['R_squared']:.4f}")

        # Universal gas constant
        self.R = 8.314  # J/(mol·K)

        # Cache for burned temperature calculations
        self._T_b_cache: Dict[Tuple[float, float, float], float] = {}
        self._rho_b_cache: Dict[Tuple[float, float, float], float] = {}

    def _compute_initial_flame_properties(self) -> Tuple[float, float, float]:
        """
        Compute S_L0, T_b0, and rho_b0 using Cantera FreeFlame at initial conditions.

        Returns:
            S_L0: Initial flame speed [m/s]
            T_b0: Initial burned gas temperature [K] (from flame.T[-1])
            rho_b0: Initial burned gas density [kg/m³] (from flame.density[-1])
        """
        ct = self._ct

        # Set gas state
        self.gas.TP = self.T0, self.P0
        self.gas.set_equivalence_ratio(self.phi0, self.fuel, self.oxidizer)

        try:
            # Create and solve flame
            flame = ct.FreeFlame(self.gas, width=self.width)
            flame.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
            flame.solve(loglevel=self.loglevel, auto=True)

            S_L0 = flame.velocity[0]
            T_b0 = flame.T[-1]
            rho_b0 = flame.density[-1]

            # Save flame profile if enabled
            self._save_flame_profile(
                flame,
                label='reference',
                T_u=self.T0,
                P=self.P0,
                description=f"Reference flame at T_u0={self.T0:.1f}K, P0={self.P0 / 1e5:.2f}bar, phi={self.phi0:.2f}"
            )

            return S_L0, T_b0, rho_b0

        except Exception as e:
            raise RuntimeError(f"Failed to compute initial flame properties: {e}")

    def _compute_activation_energy(self) -> Tuple[float, Dict]:
        """
        Compute overall activation energy using the inert dilution method.

        Method (following Clavin's approach):
            1. Run FreeFlame at multiple inert (N2) dilution levels at constant T_u
            2. Compute burning flux = rho_molar * S_L for each dilution
            3. Perform linear regression on ln(burning_flux) vs 1/T_ad
            4. Extract Ea from slope: Ea = -2 * R * slope

        Returns:
            Ea: Activation energy [J/mol]
            data: Dictionary with Arrhenius plot data for diagnostics
        """
        ct = self._ct
        R = 8.314  # J/(mol·K)

        # Generate dilution amounts
        dilution_amounts = np.linspace(
            self.dilution_range[0],
            self.dilution_range[1],
            self.n_dilution_points
        )

        ln_burning_flux = []
        inv_T_ad = []
        T_ad_values = []
        S_L_values = []
        burning_flux_values = []
        rho_molar_values = []

        print(f"  Computing activation energy (inert dilution method)...")
        print(f"    Inert species: {self.inert_species}")
        print(f"    Dilution range: {self.dilution_range[0]:.2f} - {self.dilution_range[1]:.2f} moles "
              f"({self.n_dilution_points} points)")

        # Get base composition
        self.gas.TP = self.T0, self.P0
        self.gas.set_equivalence_ratio(self.phi0, self.fuel, self.oxidizer)
        base_X = dict(self.gas.mole_fraction_dict())

        # Get base inert amount (may be 0 for pure O2 oxidizer)
        base_inert = base_X.get(self.inert_species, 0.0)

        for dilution in dilution_amounts:
            # Create diluted composition
            X_diluted = base_X.copy()
            X_diluted[self.inert_species] = base_inert + dilution

            try:
                # Set gas state with diluted composition
                self.gas.TPX = self.T0, self.P0, X_diluted

                # Get molar density of unburned gas
                rho_molar = self.gas.density_mole  # mol/m³

                # Solve flame
                flame = ct.FreeFlame(self.gas, width=self.width)
                flame.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
                flame.solve(loglevel=0, auto=True)

                S_L = flame.velocity[0]
                T_ad = flame.T[-1]  # Adiabatic flame temperature

                # Compute burning flux (mass burning rate per unit area in molar terms)
                burning_flux = rho_molar * S_L

                ln_burning_flux.append(np.log(burning_flux))
                inv_T_ad.append(1.0 / T_ad)
                T_ad_values.append(T_ad)
                S_L_values.append(S_L)
                burning_flux_values.append(burning_flux)
                rho_molar_values.append(rho_molar)

                print(f"    {self.inert_species} +{dilution:.2f} mol -> T_ad = {T_ad:.0f} K, "
                      f"S_L = {S_L:.4f} m/s, burning_flux = {burning_flux:.4f} mol/(m²·s)")

            except Exception as e:
                warnings.warn(f"FreeFlame failed at dilution = {dilution:.2f}: {e}")
                continue

        if len(ln_burning_flux) < 2:
            raise RuntimeError("Not enough valid flame solutions for activation energy calculation")

        # Linear regression: ln(burning_flux) = a + b * (1/T_ad)
        # slope b = -Ea / (2R)
        ln_burning_flux = np.array(ln_burning_flux)
        inv_T_ad = np.array(inv_T_ad)

        # Use polyfit for linear regression
        coeffs = np.polyfit(inv_T_ad, ln_burning_flux, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        # Calculate R² for diagnostics
        ln_bf_fit = slope * inv_T_ad + intercept
        ss_res = np.sum((ln_burning_flux - ln_bf_fit) ** 2)
        ss_tot = np.sum((ln_burning_flux - np.mean(ln_burning_flux)) ** 2)
        R_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Extract activation energy: Ea = -2 * R * slope
        Ea = -2.0 * R * slope

        # Package diagnostic data
        data = {
            'dilution_amounts': dilution_amounts,
            'T_ad_values': np.array(T_ad_values),
            'S_L_values': np.array(S_L_values),
            'burning_flux_values': np.array(burning_flux_values),
            'rho_molar_values': np.array(rho_molar_values),
            'ln_burning_flux': ln_burning_flux,
            'inv_T_ad': inv_T_ad,
            'slope': slope,
            'intercept': intercept,
            'R_squared': R_squared,
            'inert_species': self.inert_species,
        }

        return Ea, data

    def _get_burned_temperature(self, T: float, P: float, phi: float) -> float:
        """
        Get burned gas temperature at current conditions using Cantera equilibrate('HP').

        Constant enthalpy and pressure equilibrium.

        Parameters:
            T: Unburned gas temperature [K]
            P: Pressure [Pa]
            phi: Equivalence ratio

        Returns:
            T_b: Burned gas temperature [K]
        """
        # Check cache (with rounding for cache efficiency)
        key = (round(T, 1), round(P, 0), round(phi, 3))
        if key in self._T_b_cache:
            return self._T_b_cache[key]

        # Set unburned state
        self.gas.TP = T, P
        self.gas.set_equivalence_ratio(phi, self.fuel, self.oxidizer)

        # Equilibrate at constant H and P
        self.gas.equilibrate('HP')

        T_b = self.gas.T
        rho_b = self.gas.density

        # Cache results
        self._T_b_cache[key] = T_b
        self._rho_b_cache[key] = rho_b

        return T_b

    def _get_burned_density_from_equilibrate(self, T: float, P: float, phi: float) -> float:
        """
        Get burned gas density at current conditions using Cantera equilibrate('HP').

        Parameters:
            T: Unburned gas temperature [K]
            P: Pressure [Pa]
            phi: Equivalence ratio

        Returns:
            rho_b: Burned gas density [kg/m³]
        """
        # Check cache
        key = (round(T, 1), round(P, 0), round(phi, 3))
        if key in self._rho_b_cache:
            return self._rho_b_cache[key]

        # This will also populate the cache
        _ = self._get_burned_temperature(T, P, phi)

        return self._rho_b_cache[key]

    def _save_flame_profile(self, flame, label: str, T_u: float, P: float,
                            description: Optional[str] = None) -> str:
        """
        Save flame profile data to CSV file with sequenced naming.

        Saves: x[m], T[K], u[m/s], rho[kg/m3], hrr[W/m3]

        Parameters:
            flame: Cantera FreeFlame object after solve
            label: Short label for file naming (e.g., 'reference', 'dilution_0.10')
            T_u: Unburned gas temperature [K]
            P: Pressure [Pa]
            description: Optional longer description for log output

        Returns:
            Path to saved CSV file
        """
        if not self.save_flame_profiles or self.flame_profiles_dir is None:
            return None

        # Get flame profile data
        x = flame.grid
        T = flame.T
        u = flame.velocity
        rho = flame.density
        hrr = flame.heat_release_rate

        # Create filename with sequence number
        csv_filename = f'profiles_step_{self._profile_counter:04d}_{label}.csv'
        csv_path = self.flame_profiles_dir / csv_filename

        # Save to CSV
        data = np.column_stack([x, T, u, rho, hrr])
        header = 'x[m],T[K],u[m/s],rho[kg/m3],hrr[W/m3]'
        np.savetxt(csv_path, data, delimiter=',', header=header, comments='')

        # Also save metadata
        meta_filename = f'metadata_step_{self._profile_counter:04d}_{label}.txt'
        meta_path = self.flame_profiles_dir / meta_filename
        with open(meta_path, 'w') as f:
            f.write(f"# Flame Profile Metadata\n")
            f.write(f"step: {self._profile_counter}\n")
            f.write(f"label: {label}\n")
            f.write(f"T_unburned: {T_u:.2f} K\n")
            f.write(f"P: {P:.2f} Pa\n")
            f.write(f"P_bar: {P / 1e5:.4f} bar\n")
            f.write(f"T_adiabatic: {flame.T[-1]:.2f} K\n")
            f.write(f"S_L: {flame.velocity[0]:.6f} m/s\n")
            f.write(f"rho_unburned: {flame.density[0]:.6f} kg/m3\n")
            f.write(f"rho_burned: {flame.density[-1]:.6f} kg/m3\n")
            f.write(f"max_hrr: {np.max(hrr):.6e} W/m3\n")
            if description:
                f.write(f"description: {description}\n")

        desc_str = f" ({description})" if description else ""
        print(f"    Saved flame profile: {csv_filename}{desc_str}")

        self._profile_counter += 1
        return str(csv_path)

    def get_flame_speed(self, T: float, P: float, phi: float = 1.0) -> float:
        """
        Calculate flame speed using the full Clavin activation energy equation.

        S_L = S_L0 * (T_b/T_b0)^2 * (T_u/T_u0)^(3/2) * exp( Ea/(2R) * (1/T_b0 - 1/T_b) )

        Parameters:
            T: Unburned gas temperature [K] (T_u)
            P: Pressure [Pa]
            phi: Equivalence ratio

        Returns:
            S_L: Laminar flame speed [m/s]
        """
        # Current unburned temperature
        T_u = T

        # Get burned temperature at current conditions
        T_b = self._get_burned_temperature(T, P, phi)

        # Temperature prefactors (from Clavin formulation)
        T_b_ratio = (T_b / self.T_b0) ** 2  # Burned gas temperature ratio
        T_u_ratio = (T_u / self.T_u0) ** (3.0 / 2.0)  # Unburned gas temperature ratio

        # Activation energy exponential term
        exponent = (self.Ea / (2.0 * self.R)) * (1.0 / self.T_b0 - 1.0 / T_b)

        # Full Clavin equation
        S_L = self.S_L0 * T_b_ratio * T_u_ratio * np.exp(exponent)

        # Ensure non-negative
        return max(S_L, 0.0)

    def get_burned_density(self, T: float, P: float, phi: float = 1.0) -> float:
        """
        Get burned gas density from equilibrate('HP').

        Parameters:
            T: Unburned gas temperature [K]
            P: Pressure [Pa]
            phi: Equivalence ratio

        Returns:
            rho_b: Burned gas density [kg/m³]
        """
        return self._get_burned_density_from_equilibrate(T, P, phi)

    def get_diagnostics(self) -> Dict:
        """
        Get diagnostic information about the flame speed provider.

        Returns:
            Dictionary with reference values, activation energy, and Arrhenius fit data.
        """
        return {
            'S_L0': self.S_L0,
            'T_b0': self.T_b0,
            'T_u0': self.T_u0,
            'rho_b0': self.rho_b0,
            'Ea': self.Ea,
            'Ea_kJ_mol': self.Ea / 1000.0,
            'arrhenius_data': self.arrhenius_data,
            'T0': self.T0,
            'P0': self.P0,
            'phi0': self.phi0,
            'inert_species': self.inert_species,
            'dilution_range': self.dilution_range,
        }

    def plot_arrhenius(self, save_path: Optional[str] = None):
        """
        Plot the Arrhenius fit used to determine activation energy.

        Uses burning flux (rho_molar * S_L) vs 1/T_ad for the fit.

        Parameters:
            save_path: Path to save figure (optional)
        """
        import matplotlib.pyplot as plt

        data = self.arrhenius_data
        inv_T_ad = data['inv_T_ad']
        ln_burning_flux = data['ln_burning_flux']
        slope = data['slope']
        intercept = data['intercept']

        # Create fit line
        inv_T_fit = np.linspace(min(inv_T_ad), max(inv_T_ad), 100)
        ln_bf_fit = slope * inv_T_fit + intercept

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(inv_T_ad * 1000, ln_burning_flux, s=80, c='blue', marker='o',
                   label=f'Cantera FreeFlame ({self.inert_species} dilution)', zorder=5)
        ax.plot(inv_T_fit * 1000, ln_bf_fit, 'r--', lw=2,
                label=f'Linear fit (R² = {data["R_squared"]:.4f})')

        ax.set_xlabel('1000/T_ad [1/K]')
        ax.set_ylabel('ln(burning flux) [ln(mol/(m²·s))]')
        ax.set_title(f'Arrhenius Plot for Activation Energy (Inert Dilution Method)\n'
                     f'Ea = {self.Ea / 1000:.2f} kJ/mol')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Arrhenius plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def clear_cache(self):
        """Clear the burned temperature/density cache."""
        self._T_b_cache.clear()
        self._rho_b_cache.clear()


def generate_flame_speed_table(
        mechanism: str = 'gri30.yaml',
        fuel: str = 'CH4',
        oxidizer: str = 'O2:1, N2:3.76',
        T_range: Tuple[float, float, int] = (300, 800, 11),
        P_range: Tuple[float, float, int] = (50000, 500000, 10),
        phi_range: Tuple[float, float, int] = (0.6, 1.4, 9),
        output_file: Optional[str] = None,
        verbose: bool = True
) -> TabularFlameSpeed:
    """
    Generate a flame speed lookup table using Cantera.

    Parameters:
    -----------
    mechanism : str
        Cantera mechanism file
    fuel, oxidizer : str
        Fuel and oxidizer specifications
    T_range : tuple (min, max, n_points)
        Temperature range [K]
    P_range : tuple (min, max, n_points)
        Pressure range [Pa]
    phi_range : tuple (min, max, n_points)
        Equivalence ratio range
    output_file : str, optional
        Save table to this file
    verbose : bool
        Print progress

    Returns:
    --------
    table : TabularFlameSpeed
        The generated lookup table
    """
    T_values = np.linspace(*T_range)
    P_values = np.linspace(*P_range)
    phi_values = np.linspace(*phi_range)

    n_T, n_P, n_phi = len(T_values), len(P_values), len(phi_values)
    total = n_T * n_P * n_phi

    S_L_table = np.zeros((n_T, n_P, n_phi))
    rho_b_table = np.zeros((n_T, n_P, n_phi))

    # Create solver (no caching needed since we compute each point once)
    solver = CanteraFlameSpeed(
        mechanism=mechanism,
        fuel=fuel,
        oxidizer=oxidizer,
        loglevel=0,
        cache_size=1
    )

    count = 0
    for i, T in enumerate(T_values):
        for j, P in enumerate(P_values):
            for k, phi in enumerate(phi_values):
                count += 1
                if verbose and count % 10 == 0:
                    print(f"  Computing {count}/{total}: T={T:.0f}K, P={P / 1000:.0f}kPa, phi={phi:.2f}")

                S_L, rho_b = solver._solve_flame(T, P, phi)
                S_L_table[i, j, k] = S_L
                rho_b_table[i, j, k] = rho_b

    table = TabularFlameSpeed(
        T_values=T_values,
        P_values=P_values,
        phi_values=phi_values,
        S_L_table=S_L_table,
        rho_b_table=rho_b_table
    )

    if output_file:
        table.save(output_file)
        if verbose:
            print(f"Saved table to {output_file}")

    return table
