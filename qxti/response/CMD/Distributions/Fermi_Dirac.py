# Contain a distribution function for fermions, the Fermi-Dirac distribution.

import numpy as np

kB = 8.617333262e-5

def fermi_dirac(E, mu, T):

    beta = 1.0 / (kB * T)

    return 1.0 / (
        np.exp(beta * (E - mu)) + 1.0
    )