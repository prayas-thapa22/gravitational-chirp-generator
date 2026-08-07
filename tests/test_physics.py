import numpy as np
import pytest

from chirp_gen.constants import M_SUN, mpc_to_m, msun_to_kg
from chirp_gen.physics import (
    compute_derived_masses,
    frequency_on_time_grid,
    gw_phase,
    strain_amplitude,
    strain_waveform,
    time_to_merger_from_frequency,
)


def test_compute_derived_masses_equal_masses():
    """Equal component masses give eta = 0.25 exactly."""
    m1_kg = msun_to_kg(10.0)
    m2_kg = msun_to_kg(10.0)
    result = compute_derived_masses(m1_kg, m2_kg)
    assert result["eta"] == 0.25


def test_compute_derived_masses_gw150914():
    """GW150914-like masses (36, 29 Msun) give chirp mass ~28.1 Msun."""
    m1_kg = msun_to_kg(36.0)
    m2_kg = msun_to_kg(29.0)
    result = compute_derived_masses(m1_kg, m2_kg)
    assert result["M_c"] / M_SUN == pytest.approx(28.1, abs=0.1)


def test_compute_derived_masses_scaling():
    """Scaling both masses by k scales chirp mass by k."""
    m1_kg = msun_to_kg(36.0)
    m2_kg = msun_to_kg(29.0)
    base = compute_derived_masses(m1_kg, m2_kg)

    for scale_factor in (2.0, 3.0):
        scaled = compute_derived_masses(
            m1_kg * scale_factor, m2_kg * scale_factor
        )
        assert scaled["M_c"] == pytest.approx(
            scale_factor * base["M_c"], rel=1e-9
        )


@pytest.mark.parametrize(
    "m1_msun, m2_msun",
    [(36.0, 29.0), (100.0, 10.0), (50.0, 20.0)],
)
def test_compute_derived_masses_eta_bounded(m1_msun, m2_msun):
    """Symmetric mass ratio is at most 0.25 for unequal-mass pairs."""
    result = compute_derived_masses(msun_to_kg(m1_msun), msun_to_kg(m2_msun))
    assert result["eta"] <= 0.25


@pytest.fixture
def m_c():
    """GW150914-like chirp mass in kilograms for frequency evolution tests."""
    return compute_derived_masses(msun_to_kg(36.0), msun_to_kg(29.0))["M_c"]


def test_time_to_merger_from_frequency_higher_f_closer_to_merger(m_c):
    """Higher GW frequency gives time closer to merger (t closer to 0 from below)."""
    t_start = time_to_merger_from_frequency(35.0, m_c)
    t_end = time_to_merger_from_frequency(250.0, m_c)
    assert t_end > t_start
    assert t_end < 0 and t_start < 0


def test_frequency_increases_towards_merger(m_c):
    """GW frequency strictly increases as time approaches merger from below."""
    f_start, f_end = 35.0, 250.0
    t_start = time_to_merger_from_frequency(f_start, m_c)
    t_end = time_to_merger_from_frequency(f_end, m_c)
    time_s = np.linspace(t_start, t_end, 50)
    freq = frequency_on_time_grid(time_s, m_c, f_start, f_end)
    assert np.all(np.diff(freq) > 0)


def test_frequency_time_round_trip(m_c):
    """t(f) then f(t) recovers the original frequency within relative tolerance."""
    f0 = 100.0
    t0 = time_to_merger_from_frequency(f0, m_c)
    f_back = frequency_on_time_grid(t0, m_c, 35.0, 250.0)
    assert f_back == pytest.approx(f0, rel=1e-10)


def test_strain_amplitude_halves_when_distance_doubles(m_c):
    """Doubling luminosity distance halves the strain amplitude."""
    f_hz = 100.0
    d = mpc_to_m(410.0)
    a1 = strain_amplitude(f_hz, m_c, d)
    a2 = strain_amplitude(f_hz, m_c, 2.0 * d)
    assert a2 == pytest.approx(0.5 * a1, rel=1e-12)


def test_strain_amplitude_increases_with_frequency(m_c):
    """Higher GW frequency increases strain amplitude at fixed chirp mass and distance."""
    d = mpc_to_m(410.0)
    a_low = strain_amplitude(50.0, m_c, d)
    a_high = strain_amplitude(200.0, m_c, d)
    assert a_high > a_low


def test_strain_amplitude_gw150914_order_of_magnitude(m_c):
    """GW150914-like parameters give strain amplitude of order 1e-21."""
    a = strain_amplitude(100.0, m_c, mpc_to_m(410.0))
    assert 1e-22 < a < 1e-20


def test_gw_phase_starts_at_zero_and_increases(m_c):
    """GW phase starts at zero and strictly increases along a chirp."""
    f_start, f_end = 35.0, 250.0
    t_start = time_to_merger_from_frequency(f_start, m_c)
    t_end = time_to_merger_from_frequency(f_end, m_c)
    time_s = np.linspace(t_start, t_end, 50)
    frequency_hz = frequency_on_time_grid(time_s, m_c, f_start, f_end)
    phase = gw_phase(time_s, frequency_hz)
    assert phase[0] == pytest.approx(0.0, abs=1e-15)
    assert np.all(np.diff(phase) > 0)


def test_strain_waveform_matches_a_cos_phi(m_c):
    """strain_waveform returns h equal to A cos Phi with matching array lengths."""
    f_start, f_end = 35.0, 250.0
    d = mpc_to_m(410.0)
    t_start = time_to_merger_from_frequency(f_start, m_c)
    t_end = time_to_merger_from_frequency(f_end, m_c)
    time_s = np.linspace(t_start, t_end, 50)
    frequency_hz = frequency_on_time_grid(time_s, m_c, f_start, f_end)

    strain, freq_out, phase = strain_waveform(time_s, frequency_hz, m_c, d)
    amplitude = strain_amplitude(frequency_hz, m_c, d)

    assert len(strain) == len(time_s) == len(freq_out) == len(phase)
    assert np.allclose(freq_out, frequency_hz)
    assert np.allclose(strain, amplitude * np.cos(phase))