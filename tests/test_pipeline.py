import numpy as np
import pytest

from chirp_gen.models import BinaryParameters, WaveformConfig
from chirp_gen.pipeline import generate_waveform, validate_parameters


@pytest.fixture
def gw150914_params():
    """GW150914-like binary parameters spanning the 35-250 Hz inspiral band."""
    return BinaryParameters(
        m1_msun=36.0,
        m2_msun=29.0,
        distance_mpc=410.0,
        f_start_hz=35.0,
        f_end_hz=250.0,
    )


@pytest.fixture
def config():
    """Sample rate comfortably above Nyquist for the GW150914-like band."""
    return WaveformConfig(sample_rate_hz=4096.0)


def test_generate_waveform_returns_matching_length_arrays(gw150914_params, config):
    """generate_waveform returns time, strain, frequency, and phase of equal length."""
    result = generate_waveform(gw150914_params, config)
    n = len(result.time_s)
    assert n > 1
    assert len(result.strain) == n
    assert len(result.frequency_hz) == n
    assert len(result.phase_rad) == n


def test_generate_waveform_time_strictly_increasing(gw150914_params, config):
    """Time samples strictly increase toward merger at t = 0."""
    result = generate_waveform(gw150914_params, config)
    assert np.all(np.diff(result.time_s) > 0)
    assert result.time_s[-1] <= 0


def test_generate_waveform_frequency_increases_towards_merger(gw150914_params, config):
    """GW frequency strictly increases as time approaches merger."""
    result = generate_waveform(gw150914_params, config)
    assert np.all(np.diff(result.frequency_hz) > 0)


def test_generate_waveform_metadata_includes_derived_masses(gw150914_params, config):
    """Result metadata includes chirp mass, eta, and the requested band and rate."""
    result = generate_waveform(gw150914_params, config)
    assert result.metadata["M_c_kg"] > 0
    assert 0 < result.metadata["eta"] <= 0.25
    assert result.metadata["f_start_hz"] == gw150914_params.f_start_hz
    assert result.metadata["f_end_hz"] == gw150914_params.f_end_hz
    assert result.metadata["sample_rate_hz"] == config.sample_rate_hz


def test_validate_parameters_rejects_secondary_heavier_than_primary(config):
    """validate_parameters raises when m2 exceeds m1."""
    params = BinaryParameters(
        m1_msun=10.0, m2_msun=20.0, distance_mpc=410.0, f_start_hz=35.0, f_end_hz=250.0
    )
    with pytest.raises(ValueError):
        validate_parameters(params, config)


def test_validate_parameters_rejects_nonpositive_distance(config):
    """validate_parameters raises when distance is not positive."""
    params = BinaryParameters(
        m1_msun=36.0, m2_msun=29.0, distance_mpc=0.0, f_start_hz=35.0, f_end_hz=250.0
    )
    with pytest.raises(ValueError):
        validate_parameters(params, config)


def test_validate_parameters_rejects_f_end_below_f_start(config):
    """validate_parameters raises when f_end_hz does not exceed f_start_hz."""
    params = BinaryParameters(
        m1_msun=36.0, m2_msun=29.0, distance_mpc=410.0, f_start_hz=250.0, f_end_hz=35.0
    )
    with pytest.raises(ValueError):
        validate_parameters(params, config)


def test_validate_parameters_rejects_undersampled_rate(gw150914_params):
    """validate_parameters raises when sample rate is below the Nyquist floor."""
    undersampled = WaveformConfig(sample_rate_hz=100.0)
    with pytest.raises(ValueError):
        validate_parameters(gw150914_params, undersampled)
