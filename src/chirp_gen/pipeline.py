"""Pipeline orchestration: BinaryParameters + WaveformConfig -> WaveformResult.

Wires the pure physics in ``physics.py`` into a single entry point,
``generate_waveform``, that validates inputs, builds a uniform time grid,
and returns the assembled strain waveform with run metadata.

Scope (leading-order Newtonian quadrupole inspiral only):

- Circular, non-spinning binary; single-polarization, face-on strain.
- Band-limited inspiral segment from ``f_start_hz`` to ``f_end_hz``
  (merger at t = 0); no merger, ringdown, spin, or post-Newtonian
  corrections.
"""

import numpy as np

from chirp_gen.constants import msun_to_kg, mpc_to_m
from chirp_gen.models import BinaryParameters, WaveformConfig, WaveformResult
from chirp_gen.physics import (
    compute_derived_masses,
    time_to_merger_from_frequency,
    frequency_on_time_grid,
    strain_waveform,
)


def _require(condition: bool, message: str) -> None:
    """Raise a ValueError with message if condition is False.

    Args:
        condition: The condition that must hold.
        message: Error message to raise if the condition fails.

    Returns:
        None.
    """
    if not condition:
        raise ValueError(message)


def validate_parameters(params: BinaryParameters, config: WaveformConfig) -> None:
    """Validate binary and sampling parameters before waveform generation.

    Args:
        params: Physical parameters of the binary system.
        config: Numerical sampling configuration.

    Returns:
        None. Raises ValueError if any parameter is invalid.
    """
    _require(params.m1_msun > 0, "m1_msun must be positive")
    _require(params.m2_msun > 0, "m2_msun must be positive")
    _require(params.m1_msun >= params.m2_msun, "m1_msun must be >= m2_msun")
    _require(params.distance_mpc > 0, "distance_mpc must be positive")
    _require(params.f_start_hz > 0, "f_start_hz must be positive")
    _require(params.f_end_hz > params.f_start_hz, "f_end_hz must be > f_start_hz")
    _require(config.sample_rate_hz > 0, "sample_rate_hz must be positive")
    _require(
        config.sample_rate_hz >= 2.0 * params.f_end_hz,
        "sample_rate_hz must be >= 2 * f_end_hz (Nyquist)",
    )


def generate_waveform(
    params: BinaryParameters, config: WaveformConfig
) -> WaveformResult:
    """Generate a time-domain GW strain waveform for a circular binary inspiral.

    Args:
        params: Physical parameters of the binary system.
        config: Numerical sampling configuration.

    Returns:
        WaveformResult with time, strain, frequency, phase arrays and metadata.
    """
    validate_parameters(params, config)

    # Convert inputs to SI units.
    m1_kg = msun_to_kg(params.m1_msun)
    m2_kg = msun_to_kg(params.m2_msun)
    distance_m = mpc_to_m(params.distance_mpc)

    # Chirp mass drives the inspiral's frequency evolution.
    derived = compute_derived_masses(m1_kg, m2_kg)
    M_c = derived["M_c"]

    # Map the requested frequency band to signed time to merger (t = 0 at merger).
    t_start = time_to_merger_from_frequency(params.f_start_hz, M_c)
    t_end = time_to_merger_from_frequency(params.f_end_hz, M_c)

    # Uniform time grid at the requested sample rate.
    time_s = np.arange(t_start, t_end, 1.0 / config.sample_rate_hz)
    _require(
        len(time_s) >= 2,
        "Requested band and sample rate produce fewer than 2 samples",
    )

    frequency_hz = frequency_on_time_grid(
        time_s, M_c, params.f_start_hz, params.f_end_hz
    )
    strain, frequency_hz, phase_rad = strain_waveform(
        time_s, frequency_hz, M_c, distance_m
    )

    return WaveformResult(
        time_s=time_s,
        strain=strain,
        frequency_hz=frequency_hz,
        phase_rad=phase_rad,
        metadata={
            "M_c_kg": M_c,
            "eta": derived["eta"],
            "m1_msun": params.m1_msun,
            "m2_msun": params.m2_msun,
            "distance_mpc": params.distance_mpc,
            "f_start_hz": params.f_start_hz,
            "f_end_hz": params.f_end_hz,
            "sample_rate_hz": config.sample_rate_hz,
        },
    )