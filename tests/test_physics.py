import pytest

from chirp_gen.constants import M_SUN, msun_to_kg
from chirp_gen.physics import compute_derived_masses


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
