import pytest
from chirp_gen.constants import G, C, msun_to_kg, mpc_to_m, M_SUN


def test_msun_to_kg_one_solar_mass():
    """1 Msun converts to the nominal solar mass in kg."""
    assert msun_to_kg(1.0) == pytest.approx(1.989e30, rel=1e-6)


def test_msun_kg_round_trip():
    """Converting Msun -> kg -> Msun returns the same mass."""
    for m_msun in (1.0, 36.0, 29.0):
        assert msun_to_kg(m_msun) / M_SUN == pytest.approx(m_msun)


def test_gw150914_gravitational_radius():
    """G*M/c^2 for 36 Msun is ~53 km (~53000 m)."""
    radius_m = G * msun_to_kg(36) / C**2
    assert radius_m == pytest.approx(53_000, rel=0.01)


def test_gw150914_distance():
    """410 Mpc converts to ~1.27e25 m."""
    assert mpc_to_m(410) == pytest.approx(1.27e25, rel=0.01)
