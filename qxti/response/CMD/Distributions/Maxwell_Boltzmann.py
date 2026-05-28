#Contain the Maxwell-Boltzmann distribution function for classical particles.

import numpy as np

kB = 8.617333262e-5

def maxwell_boltzmann(E, mu, T):

    beta = 1.0 / (kB * T)

    return np.exp(-beta * (E - mu))