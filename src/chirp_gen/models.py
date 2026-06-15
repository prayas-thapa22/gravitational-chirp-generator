# Dataclasses: BinaryParameters, WaveformResult
from dataclasses import dataclass
import numpy as np
# BinaryParameters class
@dataclass(frozen=True)
class BinaryParameters:
    """Physical inputs for a circular binary inspiral waveform."""
    # Masses in solar masses
    m1_msun: float
    m2_msun: float
    # Distance in megaparsecs
    distance_mpc: float
    # Start and end frequencies in Hertz
    f_start_hz: float
    # End frequency in Hertz
    f_end_hz: float

# WaveformConfig class
@dataclass(frozen=True)
class WaveformConfig:
    """Numerical settings for time-domain sampling."""
    # Sample rate in Hertz
    sample_rate_hz: float

# WaveformResult class
@dataclass
class WaveformResult:
    """Generated waveform arrays and run metadata."""
    # Time in seconds
    time_s: np.ndarray
    # Strain in dimensionless units
    strain: np.ndarray
    # Frequency in Hertz
    frequency_hz: np.ndarray
    # Phase in radians
    phase_rad: np.ndarray
    # Metadata dictionary
    metadata: dict  # Dictionary of metadata about the waveform 