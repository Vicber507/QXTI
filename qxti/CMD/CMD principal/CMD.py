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

    # --- Methods / Inputs -> Outputs ---

    # Definición para las distintas distribuciones.

    def 



    def rho_equilibrium(self, k: NDArray[np.float64]) -> ComplexArray:
        """
        Calcula la matriz de densidad en equilibrio térmico usando la estadística 
        de Fermi-Dirac y las energías de banda provistas por el Hamiltoniano.
        
        Input: k: ndarray[3] -> [kx, ky, kz]
        Output: ndarray[Nb, Nb]
        """
        kx, ky, kz = k[0], k[1], k[2]
        
        # Obtener los autovalores (energías de las bandas) del Hamiltoniano
        energias = self.hamiltonian.eigenvalues(kx, ky, kz)
        Nb = self.hamiltonian.basis_size
        
        rho_eq = np.zeros((Nb, Nb), dtype=np.complex128)
        
        # Ocupación térmica de Fermi-Dirac (Unidades complejas elementales)
        # f(E) = 1 / (exp((E - E_f)/k_B*T) + 1)
        # Evitamos la división por cero si T = 0 absoluto
        kBT = max(self.temperature, 1e-12) 
        
        for i in range(Nb):
            factor_exponencial = (energias[i] - self.fermi_level) / kBT
            # Truncamiento de seguridad para evitar desbordamientos numéricos (Overflow)
            factor_exponencial = np.clip(factor_exponencial, -100, 100)
            rho_eq[i, i] = 1.0 / (np.exp(factor_exponencial) + 1.0)
            
        # Si la base de la simulación general es la base orbital (Original),
        # transformamos rho_eq hacia atrás usando los autovectores.
        if self.basis.lower() == "orbital":
            U = self.hamiltonian.eigenvectors(kx, ky, kz)
            rho_eq = U @ rho_eq @ U.conj().T
            
        return rho_eq

    def _evaluar_derivada(self, t: float, rho: ComplexArray, k_punto: NDArray[np.float64]) -> ComplexArray:
        """
        Ecuación cuántica de movimiento acoplada al sistema físico.
        f(t, rho, k) -> d(rho)/dt
        """
        kx, ky, kz = k_punto[0], k_punto[1], k_punto[2]
        
        # 1. Obtener Hamiltoniano base de cristal libre
        H_cristal = self.hamiltonian.H(kx, ky, kz)
        
        # 2. Inicializar interacción de campo externo de radiación-materia
        H_interaccion = np.zeros_like(H_cristal, dtype=np.complex128)
        
        # 3. Acoplamiento Gauge de Velocidad: H_int = -A(t) · v(k)
        if self.gauge.lower() == "velocity":
            # Asumimos que tu LaserSystem provee el Potencial Vector A(t) = [Ax, Ay, Az]
            A_t = self.laser_system.get_vector_potential(t)
            
            direcciones = ["x", "y", "z"]
            for i, dir_eje in enumerate(direcciones):
                if i < self.hamiltonian.dimension:
                    v_op = self.hamiltonian.velocity_operator(kx, ky, kz, dir=dir_eje)
                    H_interaccion += -A_t[i] * v_op
                    
        # Acoplamiento Gauge de Longitud: H_int = -E(t) · r(k)
        elif self.gauge.lower() == "length":
            # Asumimos que tu LaserSystem provee el Campo Eléctrico E(t) = [Ex, Ey, Ez]
            E_t = self.laser_system.get_electric_field(t)
            
            direcciones = ["x", "y", "z"]
            for i, dir_eje in enumerate(direcciones):
                if i < self.hamiltonian.dimension:
                    # En gauge de longitud se usan operadores de dipolo de transición r
                    r_op = self.hamiltonian.dipole_operator(kx, ky, kz, dir=dir_eje)
                    H_interaccion += -E_t[i] * r_op

        # Hamiltoniano Dinámico Total
        H_total = H_cristal + H_interaccion
        
        # 4. Si la simulación requiere explícitamente trabajar en la Base de Bandas:
        if self.basis.lower() == "band":
            H_total = self.hamiltonian.transform_to_band_basis(H_total, kx, ky, kz)
            
        # 5. Evolución coherente (Liouville-von Neumann con hbar = 1)
        conmutador = H_total @ rho - rho @ H_total
        d_rho_coherente = -1j * conmutador
        
        # 6. Evolución Disipativa / Relajaciones fenomenológicas de pérdidas
        d_rho_disipativo = np.zeros_like(rho, dtype=np.complex128)
        if self.include_dephasing:
            rho_eq = self.rho_equilibrium(k_punto)
            
            # Las poblaciones (diagonal) decaen hacia el equilibrio térmico
            np.fill_diagonal(d_rho_disipativo, -self.gamma_population * (np.diag(rho) - np.diag(rho_eq)))
            
            # Las coherencias (fuera de diagonal) decaen a cero de forma neta
            mascara_coherencias = ~np.eye(rho.shape[0], dtype=bool)
            d_rho_disipativo[mascara_coherencias] = -self.gamma_coherence * rho[mascara_coherencias]
            
        return d_rho_coherente + d_rho_disipativo

    def _RKF45_core(self, t0: float, tf: float, rho0: ComplexArray, h_inicial: float, k_punto: NDArray[np.float64]) -> Tuple[List[float], List[ComplexArray]]:
        """
        Algoritmo Runge-Kutta-Fehlberg acoplado y optimizado para tensores complejos matriciales.
        Controla dinámicamente el paso del tiempo h evaluando el error de truncamiento local.
        """
        t = t0 
        rho = rho0.astype(np.complex128)
        
        t_l = [t]  
        rho_l = [rho.copy()] 
        
        h = h_inicial
        tol = 1e-6  # Tolerancia cuántica estándar para conservación de fase
     
        while t < tf:
            if t + h > tf:  
                h = tf - t 
     
            ti, rho_i = t, rho
            
            # Las 6 evaluaciones obligatorias del método embebido RKF45
            k1 = h * self._evaluar_derivada(ti, rho_i, k_punto)
            k2 = h * self._evaluar_derivada(ti + h/4, rho_i + k1/4, k_punto)
            k3 = h * self._evaluar_derivada(ti + 3*h/8, rho_i + 3*k1/32 + 9*k2/32, k_punto)
            k4 = h * self._evaluar_derivada(ti + 12*h/13, rho_i + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197, k_punto)
            k5 = h * self._evaluar_derivada(ti + h, rho_i + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104, k_punto)
            k6 = h * self._evaluar_derivada(ti + h/2, rho_i - 8*k1/27 + 2*k2 + 3544*k3/2565 - 1859*k4/4140 - 11*k5/40, k_punto)
            
            # Solución aproximada de Orden 5
            rho5 = rho_i + 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55
     
            # Estimación del error local absoluto usando Norma de Frobenius
            matriz_error = k1/360 - 128*k3/4275 - 2197*k4/75240 + k5/50 + k6/55
            err = np.linalg.norm(matriz_error) 
     
            if err == 0: 
                err = 1e-16 
     
            # Si el error está en los rangos permitidos, avanzamos el paso temporal
            if err < tol:
                t = t + h
                # Corrección de estabilidad cuántica: Garantizar la propiedad adjunta (Hermiticidad)
                rho = (rho5 + rho5.conj().T) / 2.0 
     
                t_l.append(t) 
                rho_l.append(rho.copy()) 
     
            # Cálculo adaptativo del siguiente paso h (Algoritmo Fehlberg estándar de orden 4)
            factor = 0.84 * (tol / err) ** (1 / 4)  
            factor = min(2.0, max(0.1, factor))  # Evita saltos desproporcionados
            h = h * factor 
     
        return t_l, rho_l

    def solve_time_domain(self) -> Dict[int, NDArray[Any]]:
        """
        Propaga de forma global la matriz de densidad sobre toda la grilla espacial KGrid.
        Output: dict[int, ndarray] -> Dimensiones de salida ordenadas: [Nk, Nt, Nb, Nb]
        """
        # Se asume que kgrid tiene una lista ejecutable de coordenadas mapeadas
        k_puntos = self.kgrid.get_all_points()  # Retorna array de forma [Nk, 3]
        
        t0 = self.timegrid.t0
        tf = self.timegrid.tf
        h_inicial = self.timegrid.initial_h
        
        # Almacenes temporales por puntos
        lista_de_soluciones_completas = []
        
        # Bucle principal sobre el espacio recíproco k (Zona de Brillouin)
        for k_actual in k_puntos:
            # 1. Establecer la condición física inicial (Equilibrio térmico)
            rho_0 = self.rho_equilibrium(k_actual)
            
            # 2. Correr el integrador RKF45 adaptativo temporal
            tiempos, matrices_calculadas = self._RKF45_core(t0, tf, rho_0, h_inicial, k_actual)
            
            # Convertimos la evolución de este punto K en un arreglo de numpy
            lista_de_soluciones_completas.append(matrices_calculadas)
            
        # Transformar a un gran arreglo consolidado de NumPy
        # Dimensiones resultantes: [Nk, Nt, Nb, Nb]
        tensor_global_rho = np.array(lista_de_soluciones_completas, dtype=np.complex128)
        
        # El diagrama UML requiere una separación por "órdenes de perturbación" en un diccionario.
        # Si no ejecutas teoría de perturbaciones analítica explícita por separado,
        # la densidad total calculada numéricamente se asigna a los órdenes mapeados:
        rho_final_dict: Dict[int, NDArray[Any]] = {}
        for orden in range(self.max_order + 1):
            rho_final_dict[orden] = tensor_global_rho
            
        return rho_final_dict

    def compute_rho_order(self, order: int) -> NDArray[Any]:
        """Extrae el tensor completo de un orden específico [Nk, Nt, Nb, Nb]"""
        diccionario_completo = self.solve_time_domain()
        return diccionario_completo.get(order, np.array([]))

    def compute_all_orders(self) -> Dict[int, NDArray[Any]]:
        """Retorna el diccionario de órdenes completo"""
        return self.solve_time_domain()

    def solve_frequency_domain(self) -> Dict[int, NDArray[Any]]:
        """Aplica una Transformada Rápida de Fourier (FFT) sobre las componentes de tiempo"""
        # Esqueleto estructural para compatibilidad UML
        pass

    def save_density_matrices(self, output_dir: str) -> None:
        """Exporta los diccionarios de datos consolidados a disco (.npy)"""
        datos = self.solve_time_domain()
        for orden, tensor in datos.items():
            np.save(f"{output_dir}/rho_order_{orden}.npy", tensor)