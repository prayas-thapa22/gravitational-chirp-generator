import pytest
from chirp_gen.models import BinaryParameters, WaveformConfig, WaveformResult
import numpy as np
# Test the BinaryParameters class
# Use GW150914 as a test case
def test_gw150914_binary_parameters_construct():
    """Test the construction of the BinaryParameters class for GW150914."""
    params = BinaryParameters(
        m1_msun=36.0,
        m2_msun=29.0,
        distance_mpc=410.0,
        f_start_hz=35.0,
        f_end_hz=250.0,
    )
    # Test the masses
    assert params.m1_msun == 36.0
    # Test the second mass
    assert params.m2_msun == 29.0
    # Test the distance
    assert params.distance_mpc == 410.0
    # Test the start frequency
    assert params.f_start_hz == 35.0
    # Test the end frequency
    assert params.f_end_hz == 250.0
    # f_end_hz should be greater than f_start_hz
    assert params.f_end_hz > params.f_start_hz

def test_waveform_config_construct():
    """Test the construction of the WaveformConfig class."""
    config = WaveformConfig(sample_rate_hz=4096.0)
    # Test the sample rate
    assert config.sample_rate_hz == 4096.0

def test_waveform_result_holds_arrays():
    """Test the construction of the WaveformResult class."""
    # Test the time array
    n = 100
    time_s = np.linspace(-1, 0, n)
    # Test the strain array
    strain = np.zeros(n)
    # Test the frequency array
    frequency_hz = np.linspace(35, 250, n)
    # Test the phase array
    phase_rad = np.zeros(n)
    # Test the metadata
    metadata = {"m_c_msun": 28.6}
    result = WaveformResult(
        time_s=time_s,
        strain=strain,
        frequency_hz=frequency_hz,
        phase_rad=phase_rad,
        metadata=metadata,
    )
    # Test the time array
    assert len(result.time_s) == n
    # Test the strain array
    assert len(result.strain) == n
    # Test the frequency array
    assert len(result.frequency_hz) == n
    # Test the phase array
    assert len(result.phase_rad) == n
    # Test the metadata
    assert result.metadata == metadata