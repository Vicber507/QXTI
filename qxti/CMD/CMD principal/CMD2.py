# Librarías estándar

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np                        # Permite usar matrices y álgebra lineal.
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]     # Permite usar matrices con componentes complejas.

"""Atributos de la clase CMD"""

from qxti.physics import Hamiltonian
from qxti.physics import LaserSystem 
from qxti.physics import KGrid
from qxti.physics import TimeGrid
from qxti.physics import OperatorFactory
from qxti.physics import Solver

class CMD:

     """
    Responsibility:
    First class combining Hamiltonian and LaserSystem.
    Output: rho = {0: rho0, 1: rho1, 2: rho2, 3: rho3}.
    
    Integrates the quantum Liouville-von Neumann equation using an adaptive
    Runge-Kutta-Fehlberg 4(5) solver (RKF45) over the reciprocal space grid.
    """
    
    def __init__(
        self,
        hamiltonian: Hamiltonian,
        laser_system: LaserSystem,
        kgrid: KGrid,
        timegrid: TimeGrid,
        operator_factory: OperatorFactory,
        solver: Solver,
        max_order: int,
        gamma_population: float,
        gamma_coherence: float,
        temperature: float,
        fermi_level: float,
        basis: str,
        gauge: str,
        include_intraband: bool,
        include_interband: bool,
        include_dephasing: bool
    ):
        # --- Attributes ---
        self.hamiltonian: Hamiltonian = hamiltonian
        self.laser_system: LaserSystem = laser_system
        self.kgrid: KGrid = kgrid
        self.timegrid: TimeGrid = timegrid
        self.operator_factory: OperatorFactory = operator_factory
        self.solver: Solver = solver
        self.max_order: int = max_order
        self.gamma_population: float = gamma_population
        self.gamma_coherence: float = gamma_coherence
        self.temperature: float = temperature
        self.fermi_level: float = fermi_level
        self.basis: str = basis               # "orbital" o "band"
        self.gauge: str = gauge               # "velocity" o "length"
        self.include_intraband: bool = include_intraband
        self.include_interband: bool = include_interband
        self.include_dephasing: bool = include_dephasing

# Methods / Inputs and outputs

