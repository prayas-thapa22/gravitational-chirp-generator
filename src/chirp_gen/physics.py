from chirp_gen.constants import G, C
import numpy as np
from scipy.integrate import cumulative_trapezoid
# Quadrupole inspiral equations (pure functions)
# These functions are used to compute the derived masses


def compute_derived_masses(m1_kg: float, m2_kg: float) -> dict:
    """Derive total, reduced, symmetric mass ratio, and chirp mass for a binary.

    - M: total mass in kilograms; overall mass scale of the system.
    - mu: reduced mass in kilograms; effective two-body inertia.
    - eta: symmetric mass ratio (mu / M); dimensionless, max 0.25 for equal masses.
    - M_c: chirp mass in kilograms; sets how fast GW frequency rises in the inspiral.

    Args:
        m1_kg: Primary mass in kilograms.
        m2_kg: Secondary mass in kilograms.

    Returns:
        Dict with keys M, mu, eta, M_c (kg except eta, which is dimensionless).
    """
    M = compute_total_mass(m1_kg, m2_kg)
    mu = compute_reduced_mass(m1_kg, m2_kg)
    eta = compute_symmetric_mass_ratio(M, mu)
    M_c = compute_chirp_mass(m1_kg, m2_kg)
    return {"M": M, "mu": mu, "eta": eta, "M_c": M_c}


def compute_total_mass(m1_kg: float, m2_kg: float) -> float:
    """Return the total mass of a binary.

    Args:
        m1_kg: Primary mass in kilograms.
        m2_kg: Secondary mass in kilograms.

    Returns:
        Total mass M = m1 + m2 in kilograms.
    """
    return m1_kg + m2_kg


def compute_reduced_mass(m1_kg: float, m2_kg: float) -> float:
    """Return the reduced mass of a binary.

    Args:
        m1_kg: Primary mass in kilograms.
        m2_kg: Secondary mass in kilograms.

    Returns:
        Reduced mass mu = m1*m2 / (m1+m2) in kilograms.
    """
    return (m1_kg * m2_kg) / (m1_kg + m2_kg)


def compute_symmetric_mass_ratio(M: float, mu: float) -> float:
    """Return the symmetric mass ratio eta = mu / M.

    Args:
        M: Total mass in kilograms.
        mu: Reduced mass in kilograms.

    Returns:
        Symmetric mass ratio eta (dimensionless).
    """
    return mu / M


def compute_chirp_mass(m1_kg: float, m2_kg: float) -> float:
    """Return the chirp mass of a binary.

    Args:
        m1_kg: Primary mass in kilograms.
        m2_kg: Secondary mass in kilograms.

    Returns:
        Chirp mass M_c in kilograms.
    """
    return ((m1_kg * m2_kg) ** (3 / 5)) / ((m1_kg + m2_kg) ** (1 / 5))


# These functions are used to compute the time to merger and the frequency on the time grid
def time_to_merger_from_frequency(f_hz: np.ndarray, m_c_kg: float) -> np.ndarray:
    """Map GW frequency to signed time to merger (merger at t = 0).

    Args:
        f_hz: Gravitational-wave frequency in Hertz (scalar or array).
        m_c_kg: Chirp mass in kilograms.

    Returns:
        Time to merger in seconds (negative before merger).
    """
    return -(
        (5.0 / 256.0)
        * (np.pi * f_hz) ** (-8.0 / 3.0)
        * (G * m_c_kg / C**3) ** (-5.0 / 3.0)
    )


def frequency_on_time_grid(
    time_s: np.ndarray, m_c_kg: float, f_start_hz: float, f_end_hz: float
) -> np.ndarray:
    """Map signed time (merger at t = 0) to GW frequency on a time grid.

    Args:
        time_s: Time samples in seconds (negative before merger).
        m_c_kg: Chirp mass in kilograms.
        f_start_hz: Band start frequency in Hertz (for checks / callers).
        f_end_hz: Band end frequency in Hertz (for checks / callers).

    Returns:
        Gravitational-wave frequency at each time sample in Hertz.
    """
    tau = -np.asarray(time_s, dtype=float)
    theta = G * m_c_kg / C**3
    return (1.0 / np.pi) * ((5.0 / 256.0) / tau * theta ** (-5.0 / 3.0)) ** (3.0 / 8.0)

def strain_amplitude(f_hz: np.ndarray, m_c_kg: float, distance_m: float) -> np.ndarray:
    """Return the leading-order GW strain amplitude A(f).

    Args:
        f_hz: Gravitational-wave frequency in Hertz.
        m_c_kg: Chirp mass in kilograms.
        distance_m: Luminosity distance in meters.

    Returns:
        Dimensionless strain amplitude A.
    """
    return (4.0/distance_m)*((G*m_c_kg/C**2.0)**(5.0/3.0))*(((f_hz*np.pi)/C)**(2.0/3.0))

def gw_phase(time_s: np.ndarray, frequency_hz: np.ndarray) -> np.ndarray:
    """Return GW phase by integrating 2π f(t) along the time grid.
    Convention: Φ(t0) = 0 at the first sample.

    Args:
        time_s: Time samples in seconds (merger at t = 0).
        frequency_hz: Instantaneous GW frequency at each sample in Hertz.

    Returns:
        GW phase Φ(t) in radians.
    """
    integral = cumulative_trapezoid(frequency_hz, time_s, initial=0.0)
    return (2.0 * np.pi) * integral
    
def strain_waveform(time_s: np.ndarray, frequency_hz: np.ndarray, m_c_kg: float, distance_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the time-domain GW strain h(t) = A(f) cos Φ(t).

    Args:
        time_s: Time samples in seconds (merger at t = 0).
        frequency_hz: Instantaneous GW frequency at each sample in Hertz.
        m_c_kg: Chirp mass in kilograms.
        distance_m: Luminosity distance in meters.

    Returns:
        Tuple of (strain, frequency_hz, phase_rad):
        dimensionless h(t), frequency in Hertz, phase in radians.
    """    
    amplitude = strain_amplitude(frequency_hz, m_c_kg, distance_m)
    phase_rad = gw_phase(time_s, frequency_hz)
    strain = amplitude * np.cos(phase_rad)
    return strain, frequency_hz, phase_rad

