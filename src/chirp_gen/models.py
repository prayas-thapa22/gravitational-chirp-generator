# Dataclasses: BinaryParameters, WaveformResult
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BinaryParameters:
    """Physical inputs for a circular binary inspiral waveform.

    Attributes:
        m1_msun: Primary mass in solar masses.
        m2_msun: Secondary mass in solar masses.
        distance_mpc: Luminosity distance in megaparsecs.
        f_start_hz: Start frequency of the analysis band in Hertz.
        f_end_hz: End frequency of the analysis band in Hertz.
    """

    m1_msun: float
    m2_msun: float
    distance_mpc: float
    f_start_hz: float
    f_end_hz: float


@dataclass(frozen=True)
class WaveformConfig:
    """Numerical settings for time-domain sampling.

    Attributes:
        sample_rate_hz: Sample rate in Hertz.
    """

    sample_rate_hz: float


@dataclass
class WaveformResult:
    """Generated waveform arrays and run metadata.

    Attributes:
        time_s: Time samples in seconds (merger at t = 0).
        strain: Dimensionless strain h(t).
        frequency_hz: Instantaneous GW frequency in Hertz.
        phase_rad: GW phase in radians.
        metadata: Dictionary of run metadata about the waveform.
    """

    time_s: np.ndarray
    strain: np.ndarray
    frequency_hz: np.ndarray
    phase_rad: np.ndarray
    metadata: dict
