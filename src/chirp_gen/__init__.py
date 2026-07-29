"""Generate synthetic gravitational-wave chirps from binary inspirals."""

from chirp_gen.models import BinaryParameters, WaveformConfig, WaveformResult
from chirp_gen.pipeline import generate_waveform

__version__ = "0.1.0"

__all__ = [
    "BinaryParameters",
    "WaveformConfig",
    "WaveformResult",
    "generate_waveform",
]
