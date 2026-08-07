from chirp_gen.models import BinaryParameters, WaveformConfig, WaveformResult
import numpy as np


def test_gw150914_binary_parameters_construct():
    """GW150914-like BinaryParameters stores masses, distance, and band edges."""
    params = BinaryParameters(
        m1_msun=36.0,
        m2_msun=29.0,
        distance_mpc=410.0,
        f_start_hz=35.0,
        f_end_hz=250.0,
    )
    assert params.m1_msun == 36.0
    assert params.m2_msun == 29.0
    assert params.distance_mpc == 410.0
    assert params.f_start_hz == 35.0
    assert params.f_end_hz == 250.0
    assert params.f_end_hz > params.f_start_hz


def test_waveform_config_construct():
    """WaveformConfig stores the requested sample rate in Hertz."""
    config = WaveformConfig(sample_rate_hz=4096.0)
    assert config.sample_rate_hz == 4096.0


def test_waveform_result_holds_arrays():
    """WaveformResult holds equal-length time, strain, frequency, and phase arrays."""
    n = 100
    time_s = np.linspace(-1, 0, n)
    strain = np.zeros(n)
    frequency_hz = np.linspace(35, 250, n)
    phase_rad = np.zeros(n)
    metadata = {"m_c_msun": 28.6}
    result = WaveformResult(
        time_s=time_s,
        strain=strain,
        frequency_hz=frequency_hz,
        phase_rad=phase_rad,
        metadata=metadata,
    )
    assert len(result.time_s) == n
    assert len(result.strain) == n
    assert len(result.frequency_hz) == n
    assert len(result.phase_rad) == n
    assert result.metadata == metadata
