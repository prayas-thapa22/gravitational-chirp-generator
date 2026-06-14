# Constants for the gravitational wave chirp generator 
#G, c, M_sun, unit conversions
G: float = 6.67430e-11          # m^3 kg^-1 s^-2  — gravitational constant
C: float = 299792458          # m s^-1          — speed of light in vacuum
M_SUN: float = 1.989e30      # kg              — IAU nominal solar mass
MPC_TO_M: float = 3.085677581e22   # m per Mpc       — megaparsec to meters

# Unit conversions
def msun_to_kg(m_msun: float) -> float: 
    """Convert solar mass to kilograms"""
    return m_msun * M_SUN

def mpc_to_m(d_mpc: float) -> float: 
    """Convert megaparsec to meters"""
    return d_mpc * MPC_TO_M