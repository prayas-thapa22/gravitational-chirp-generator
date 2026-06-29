# Quadrupole inspiral equations (pure functions)
#these functions are used to compute the derived masses 

def compute_derived_masses(m1_kg: float, m2_kg: float) -> dict:
    """Derive total, reduced, symmetric mass ratio, and chirp mass for a binary.
    - M: total mass in kg of the binary system, overall scale of the system 
    - mu: reduced mass in kg of the binary system, the effective inertia is captured by this quantity
    - eta: symmetric mass ratio, the ratio of the reduced mass to the total mass, it is a dimensionless quantity and is a measure of the mass distribution of the binary system
    - M_c: chirp mass in kg of the binary system, it is a combination of m1 and m2 that determines how fast the GW frequency rises during the inspiral

    Args:
        m1_kg: Primary mass in kilograms.
        m2_kg: Secondary mass in kilograms.
        
    Returns:
        Dict with keys M, mu, eta, M_c (kg except eta, which is dimensionless).
    """

    M = compute_total_mass(m1_kg, m2_kg)

    mu = compute_reduced_mass(m1_kg, m2_kg)

    eta = compute_symmetric_mass_ratio(M, mu)
    
    M_c = compute_chirp_mass(m1_kg, m2_kg)

    return {"M": M, "mu": mu, "eta": eta, "M_c": M_c}

def compute_total_mass(m1_kg: float, m2_kg: float) -> float:
    """input masses in kg and output the total mass in kg"""
    return m1_kg + m2_kg

def compute_reduced_mass(m1_kg: float, m2_kg: float) -> float:
    """input masses in kg and output the reduced mass in kg"""
    return (m1_kg * m2_kg) / (m1_kg + m2_kg)    

def compute_symmetric_mass_ratio(M: float, mu: float) -> float:
    """input total mass and reduced mass and output the symmetric mass ratio"""
    return (mu / M)

def compute_chirp_mass(m1_kg: float, m2_kg: float) -> float:
    """input masses in kg and output the chirp mass in kg"""
    return ((m1_kg * m2_kg) ** (3/5)) / ((m1_kg + m2_kg) ** (1/5))