from __future__ import annotations

import ast
import configparser
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""

    lowered = stripped.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None

    try:
        return ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return stripped


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_band_indices(value: str) -> tuple[int, ...] | None:
    stripped = value.strip()
    if not stripped or stripped.lower() in {"all", "none"}:
        return None

    parsed = _parse_scalar(stripped)
    if isinstance(parsed, int):
        return (int(parsed),)
    if isinstance(parsed, (list, tuple)):
        return tuple(int(item) for item in parsed)
    if isinstance(parsed, str):
        return tuple(int(item.strip()) for item in parsed.split(",") if item.strip())

    raise ValueError("band_indices must be an int, a sequence of ints, or 'all'.")


def _parse_int_tuple_or_none(value: str) -> tuple[int, ...] | None:
    stripped = value.strip()
    if not stripped or stripped.lower() in {"all", "none"}:
        return None

    parsed = _parse_scalar(stripped)
    if isinstance(parsed, int):
        return (int(parsed),)
    if isinstance(parsed, (list, tuple)):
        return tuple(int(item) for item in parsed)
    if isinstance(parsed, str):
        return tuple(int(item.strip()) for item in parsed.split(",") if item.strip())

    raise ValueError("Expected an int, a sequence of ints, or 'all'.")


def _parse_int_tuple(value: str, *, default: tuple[int, ...] = ()) -> tuple[int, ...]:
    stripped = value.strip()
    if not stripped:
        return tuple(default)

    parsed = _parse_scalar(stripped)
    if isinstance(parsed, int):
        return (int(parsed),)
    if isinstance(parsed, (list, tuple)):
        return tuple(int(item) for item in parsed)
    raise ValueError(f"Expected an integer tuple/list, got {value!r}.")


def _parse_float_list(value: str, *, default: list[float] | None = None) -> list[float]:
    stripped = value.strip()
    if not stripped:
        return list(default or [])

    parsed = _parse_scalar(stripped)
    if isinstance(parsed, (int, float)):
        return [float(parsed)]
    if isinstance(parsed, (list, tuple)):
        return [float(item) for item in parsed]
    raise ValueError(f"Expected a numeric list, got {value!r}.")


def _nested_assignment(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        return

    current = target
    for part in parts[:-1]:
        node = current.get(part)
        if node is None:
            node = {}
            current[part] = node
        elif not isinstance(node, dict):
            raise ValueError(f"Cannot assign nested key '{dotted_key}' because '{part}' is not a dict.")
        current = node
    current[parts[-1]] = value


@dataclass(slots=True)
class HamiltonianConfig:
    source_file: str
    function_name: str = "H"
    model_name: str = "custom-hamiltonian"
    basis_size: int | None = None
    dimension: int | None = None
    basis_type: str = "custom"
    is_periodic: bool = True
    dk_derivative: float = 1.0e-5
    params: dict[str, Any] = field(default_factory=dict)
    lattice: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HamiltonianPlotsConfig:
    enabled: bool = True
    plots: tuple[str, ...] = field(default_factory=tuple)
    output_dir: str = "outputs/hamiltonian"
    path_type: str = "kx"
    manual_path: list[list[float]] = field(default_factory=list)
    plane: str = "kx_ky"
    k_min: float = -0.1
    k_max: float = 0.1
    k1_min: float = -0.1
    k1_max: float = 0.1
    k2_min: float = -0.1
    k2_max: float = 0.1
    nk_path: int = 201
    mesh_points: int = 101
    mesh_points_1: int | None = None
    mesh_points_2: int | None = None
    fixed_kx: float = 0.0
    fixed_ky: float = 0.0
    fixed_kz: float = 0.0
    band_index: int | None = None
    band_indices: tuple[int, ...] | None = None
    surface_style: str = "surface"
    quiver_stride: int = 4


@dataclass(slots=True)
class KGridConfig:
    dimension: int | None = None
    k_points: tuple[int, ...] = field(default_factory=tuple)
    points_per_axis: int | None = None
    kx_values: list[float] = field(default_factory=list)
    ky_values: list[float] = field(default_factory=list)
    kz_values: list[float] = field(default_factory=list)
    shifted: bool = False
    auto_degeneracy_guard: bool = True


@dataclass(slots=True)
class TimeGridConfig:
    dt: float = 0.01
    t_min: float | None = None
    t_max: float | None = None
    Nt: int | None = None
    fft_window: str = "hann"
    zero_padding: bool = False
    padding_factor: int = 2


@dataclass(slots=True)
class LaserConfig:
    omega: float = 1.0
    E0: float = 0.0
    cep: float = 0.0
    ellip: float = 0.0
    ncycles: float = 1.0
    envname: str = "gauss"
    t0: float = 0.0
    phix: float = 0.0
    thetaz: float = 0.0
    phiz: float = 0.0
    ncycles_clipped: float = 0.0
    blaser: float = 0.0
    alaser: float = 0.0
    pulses: list[dict[str, Any]] = field(default_factory=list)

    @property
    def phase(self) -> float:
        return self.cep

    @property
    def ellipticity(self) -> float:
        return self.ellip

    @property
    def num_cycles(self) -> float:
        return self.ncycles

    @property
    def envelope(self) -> str:
        return self.envname

    @property
    def theta(self) -> float:
        return self.thetaz

    @property
    def phi(self) -> float:
        return self.phiz


# Canonical response-engine names (with backward-compatible aliases).  The eje:
#   pfddm = perturbative frequency-domain DM  (old "theory";     the mesh engine)
#   ptddm = perturbative time-domain DM       (old "simulation"; the CMD engine)
#   tddm  = full NON-perturbative time-domain DM (new; velocity gauge)
#   both  = the two perturbative engines (pfddm + ptddm)
#   all   = run all three and emit the comparison
_ENGINE_ALIASES = {
    "theory": "pfddm", "pfddm": "pfddm",
    "simulation": "ptddm", "ptddm": "ptddm",
    "tddm": "tddm", "both": "both", "all": "all",
}
_CANONICAL_ENGINES = ("pfddm", "ptddm", "tddm", "both", "all")


def _canonical_method(value: str | None, *, default: str = "ptddm") -> str:
    """Map an engine name (including deprecated aliases) to its canonical form."""
    s = (str(value).strip().lower() if value is not None else "") or default
    return _ENGINE_ALIASES.get(s, s)


@dataclass(slots=True)
class CMDConfig:
    enabled: bool = False
    output_dir: str = "outputs/cmd"
    max_order: int = 1
    # Response engine (canonical: pfddm | ptddm | tddm | both | all; old names
    # theory/simulation are accepted as aliases). See _ENGINE_ALIASES above.
    response_method: str = "ptddm"
    rho_storage_dtype: str = "complex128"
    scratch_rho_storage_dtype: str = "auto"
    keep_rho_orders: bool = True
    dataset_time_stride: int = 1
    save_population_dataset: bool = True
    save_coherence_dataset: bool = True
    save_xtp_dataset: bool = True
    population_time: float = math.inf
    coherence_time: float = math.inf
    temperature: float = 0.0
    fermi_level: float = 0.0
    distribution: str = "fermi_dirac"
    basis: str = "band"
    gauge: str = "length"
    include_intraband: bool = True
    include_interband: bool = True
    include_dephasing: bool = True
    solver: str = "rkf45"
    solver_tolerance: float = 1.0e-6
    solver_max_iterations: int | None = None
    solver_max_rejections: int = 10000
    solver_h_min: float = 1.0e-12
    solver_h_max: float | None = None
    solver_safety_factor: float = 0.9
    solver_min_factor: float = 0.2
    solver_max_factor: float = 4.0
    save_frequency_domain: bool = False
    # Number of parallel workers for the k-point loop (0 = use all performance cores).
    n_workers: int = 0
    # RAM (GB) the streaming mesh engine always keeps free so the machine never
    # swaps itself to death; the block size adapts to honor it (any OS).
    reserve_gb: float = 1.0
    # tddm (-xtp): field-amplitude ladder for the χ⁽ˢ⁾ polynomial extraction
    # (empty -> auto geometric ladder covering max_order+1 amplitudes).
    tddm_amplitude_ladder: tuple[float, ...] = ()
    # Order of the covariant k-gradient finite-difference stencil in the pfddm/ptddm
    # mesh recursion. 2 = 2-point central (error O(Δk²), default); 4 = 5-point 4th
    # order (error O(Δk⁴), ~1.5x/point but far less grid error, so a coarser mesh
    # reaches the same accuracy). Halo widens to 2·(max_order−1) when 4.
    gradient_stencil: int = 2


@dataclass(slots=True)
class CMDPlotsConfig:
    enabled: bool = False
    output_path: str = "outputs/cmd/population_heatmap.png"
    orders: tuple[int, ...] | None = None
    k_aggregation: str = "mean"
    save_animation: bool = False
    animation_path: str = "outputs/cmd/population_animation.gif"
    fps: int = 10
    frame_stride: int = 1


@dataclass(slots=True)
class XTPConfig:
    bz_mask_enabled: bool = False
    bz_mask_radius_percent: float = 100.0
    bz_mask_sigma: float | None = None
    bz_mask_sigma_percent_legacy: float | None = None
    susceptibility_enabled: bool = False
    susceptibility_output_dir: str = "outputs/susceptibility"
    susceptibility_orders: tuple[int, ...] = (1,)
    susceptibility_omega_min: float = 0.05
    susceptibility_omega_max: float = 0.15
    susceptibility_num_frequencies: int = 11
    susceptibility_omega_values: tuple[float, ...] = field(default_factory=tuple)
    susceptibility_eps: float = 1.0e-14
    # Which engine computes the response:
    #   "simulation" = time-domain CMD pulse + FFT (numerically exact, slow)
    #   "theory"     = analytical frequency-domain Kubo/Hipolito formula (fast)
    #   "both"       = run both and save both for comparison (+ timing report)
    susceptibility_method: str = "ptddm"
    # Parallel processes for the laser-frequency sweep (0 = auto, RAM-bounded).
    susceptibility_n_workers: int = 0
    susceptibility_plot_enabled: bool = False
    susceptibility_plot_output_dir: str = ""
    susceptibility_plot_dataset_file: str = "xtp_susceptibility.npz"
    susceptibility_plot_orders: tuple[int, ...] | None = None
    susceptibility_plot_positive_only: bool = False
    susceptibility_plot_omega_min: float | None = None
    susceptibility_plot_omega_max: float | None = None
    susceptibility_plot_dpi: int = 300
    susceptibility_plot_overview_enabled: bool = True
    susceptibility_plot_grid_enabled: bool = True
    susceptibility_plot_components_enabled: bool = True
    susceptibility_plot_conductivity_enabled: bool = True
    susceptibility_plot_ev_axis: bool = True

@dataclass(slots=True)
class LDOSConfig:
    """Configuration for the density-of-states branch (``main.py ... -ldos``).

    Computes the bulk density of states g(E) of H(k) over the [kgrid] mesh by
    diagonalizing every k-point and broadening the eigenvalues. It is the third
    calculation family, next to ``-hhg`` ([cmd]) and ``-xtp`` ([xtp]).

    Quantities produced (all from the same k-grid pass):
      * total DOS g(E)                         -> integrates to ``basis_size``
      * orbital-projected PDOS g_alpha(E)      -> sum_alpha g_alpha = g
      * cumulative N(E) = \\int_{-inf}^{E} g    -> band filling
      * optional momentum-resolved spectral function, in two display modes that
        mirror the original surface-Green example:
          - E vs k    : A(k, E) along a straight k-path (broadened band structure)
          - kx vs ky  : A(kx, ky; E0) at a fixed energy (constant-energy / Fermi map)
        Both are -Im Tr G/pi for finite H(k) and are drawn with the example's
        Blues + linear-alpha colormap, log scale and fixed log color range.

    All input values in ``[ldos]`` stay in atomic units: energies in Hartree and
    momenta in inverse Bohr. The graphics layer converts them to eV and 1/Ang
    only for visualization.

    Engines now available through ``ldos.method``:
      * ``eigenvalues`` -> periodic bulk DOS / PDOS / A(k,E)
      * ``surface``     -> semi-infinite surface/edge Green function; ``surface_side``
                           selects one face or ``both`` (both faces at once, the
                           exact + fast replacement for a finite-slab/ribbon)
      * ``finite``      -> finite plaque in two directions (no good k; use LDOS(r,E))
    """

    enabled: bool = False
    output_dir: str = "outputs/ldos"
    # Engine:
    #   "eigenvalues" = BULK DOS: diagonalize the fully periodic H(k) and broaden
    #                   (Lorentzian = Green trace -Im Tr[(E+i*eta-H)^-1]/pi).
    #   "surface"     = SURFACE/EDGE: make the crystal semi-infinite along one
    #                   direction (Lopez-Sancho decimation) and compute the
    #                   surface Green function. Reveals surface states (3D) and
    #                   edge states (2D, e.g. the Haldane chiral edge mode) inside
    #                   the projected bulk gap. ``surface_side = both`` returns the
    #                   two opposite faces at once (e.g. all Weyl Fermi arcs),
    #                   which is the exact, fast replacement for a finite slab.
    #   "finite"      = FINITE PLAQUE: build a finite 2-D sample with open
    #                   boundaries in both lattice directions, diagonalize it in
    #                   real space, and produce a finite-system DOS together with
    #                   a site-resolved LDOS map. No good crystal momentum exists
    #                   in this mode, so the relevant observable is LDOS(r, E)
    #                   rather than A(k, E). Configured by the finite_* keys.
    method: str = "eigenvalues"
    # Parallel workers for the (embarrassingly parallel) surface k-loops
    # (0 = all performance cores) and RAM to keep free.
    n_workers: int = 0
    reserve_gb: float = 1.0
    # --- Surface/edge mode (method = surface) ---
    # Direction made semi-infinite: "x"|"y"|"z" or "auto" (z in 3D, y in 2D).
    surface_normal: str = "auto"
    # Which termination of the semi-infinite stack to show:
    #   "bottom" / "top" = the two opposite faces (their surface states are
    #                      complementary -- e.g. a Weyl slab's Fermi arcs route
    #                      differently on each face);
    #   "both"           = sum of both faces (shows ALL arcs at once -- this is
    #                      the natural replacement for a finite-slab calculation).
    surface_side: str = "both"
    # Lopez-Sancho decimation controls.
    surface_max_iter: int = 80
    surface_tol: float = 1.0e-8
    # k-points used to Fourier-transform H along the normal into principal-layer
    # blocks H00 (intra-layer) and H01 (inter-layer). Assumes nearest-layer
    # coupling along the normal (the standard principal-layer approximation).
    surface_kn_points: int = 200
    # Also integrate the surface spectral function over the in-plane grid to get
    # the surface LDOS g_surf(E) (peaks = surface/edge states in the bulk gap).
    surface_ldos_enabled: bool = True
    # Effective slab/ribbon thickness used only when plotting the surface
    # contribution against the bulk DOS. 0 = infer from finite_nx/finite_ny when
    # possible. This turns a local surface LDOS into an O(surface/bulk) contribution.
    surface_compare_layers: int = 0
    # --- Finite-plaque mode (method = finite) ---
    # Number of Bravais cells along the two open directions of the finite 2-D
    # sample. The Hilbert-space size is basis_size * finite_nx * finite_ny, so
    # these should stay moderate for dense diagonalization.
    finite_nx: int = 24
    finite_ny: int = 24
    # Energy where the site-resolved LDOS map is evaluated (a.u.). None -> use
    # fermi_level.
    finite_ldos_energy: float | None = None
    # Number of outermost unit-cell layers classified as "edge" when coloring
    # the finite-state spectrum by edge localization.
    finite_edge_cells: int = 1
    # Broadening kernel applied to each level: "lorentzian" or "gaussian".
    broadening: str = "lorentzian"
    # Broadening width eta (a.u.). 0 -> auto (a few energy-grid steps).
    eta: float = 0.0
    # Energy window (a.u.). None -> auto from the band extrema with padding.
    e_min: float | None = None
    e_max: float | None = None
    num_energies: int = 600
    # Orbital/sublattice-projected PDOS (cheap; needs eigenvectors).
    projected: bool = True
    # Fermi level (a.u.), drawn as a reference line and used for the filling note.
    fermi_level: float = 0.0
    # --- Spectral function mode 1: E vs k along a k-path ---
    spectral_enabled: bool = False
    # Simple single-axis path in a.u.: choose the axis and the scalar endpoints.
    # Example:
    #   spectral_axis = kx
    #   spectral_axis_start = -0.5
    #   spectral_axis_end = 0.5
    # Optional aliases accepted in inputs:
    #   spectral_line_axis, spectral_line_start, spectral_line_end
    spectral_axis: str = ""
    spectral_axis_start: float | None = None
    spectral_axis_end: float | None = None
    # Optional fixed point for the non-varying coordinates in the simple-axis mode.
    # Alias accepted: spectral_line_origin
    spectral_axis_origin: tuple[float, ...] = field(default_factory=tuple)
    # Two-point straight path in a.u. (1/Bohr); used when no multi-segment path
    # is given.
    spectral_k0: tuple[float, ...] = field(default_factory=tuple)
    spectral_k1: tuple[float, ...] = field(default_factory=tuple)
    # Multi-segment high-symmetry path in a.u. (1/Bohr): a list of vertices
    # (each a 3-vector) and optional labels, e.g.
    # spectral_path = 0,0,0 ; 1.28,0,0 ; 0.96,0.55,0  and
    # spectral_path_labels = \Gamma, K, M. Takes precedence over spectral_k0/k1.
    spectral_path: tuple[tuple[float, ...], ...] = field(default_factory=tuple)
    spectral_path_labels: tuple[str, ...] = field(default_factory=tuple)
    # Total number of k-points along the whole path.
    spectral_num_k: int = 200
    # --- Spectral function mode 2: kx vs ky at a fixed energy ---
    spectral_plane_enabled: bool = False
    # Fixed energy of the cut (a.u.). None -> use fermi_level (Fermi-surface map).
    spectral_plane_energy: float | None = None
    # kx/ky window in a.u. (1/Bohr). None -> auto from the BZ box.
    spectral_plane_kx_min: float | None = None
    spectral_plane_kx_max: float | None = None
    spectral_plane_ky_min: float | None = None
    spectral_plane_ky_max: float | None = None
    spectral_plane_kz: float = 0.0
    spectral_plane_nkx: int = 160
    spectral_plane_nky: int = 160
    # --- Spectral plot style (defaults copied from the original example) ---
    spectral_cmap: str = "Blues"
    spectral_log_scale: bool = True
    spectral_log_vmin: float | None = -1.5
    spectral_log_vmax: float | None = 3.3
    spectral_alpha_min: float = 0.0
    spectral_alpha_max: float = 1.0
    # Plot dots-per-inch. (Figures are always drawn in eV vs 1/Angstrom.)
    plot_dpi: int = 300


@dataclass(slots=True)
class QXTIConfig:
    hamiltonian: HamiltonianConfig
    hamiltonian_plots: HamiltonianPlotsConfig = field(default_factory=HamiltonianPlotsConfig)
    kgrid: KGridConfig = field(default_factory=KGridConfig)
    timegrid: TimeGridConfig = field(default_factory=TimeGridConfig)
    laser: LaserConfig = field(default_factory=LaserConfig)
    cmd: CMDConfig = field(default_factory=CMDConfig)
    cmd_plots: CMDPlotsConfig = field(default_factory=CMDPlotsConfig)
    xtp: XTPConfig = field(default_factory=XTPConfig)
    ldos: LDOSConfig = field(default_factory=LDOSConfig)
    susceptibility_solver: CMDConfig = field(default_factory=CMDConfig)
    source_path: Path | None = None

    @classmethod
    def from_file(cls, config_path: str | Path) -> QXTIConfig:
        # inline_comment_prefixes lets values carry trailing "# ..." comments
        # (e.g. `eta = 0.0   # auto`), so the heavily-documented reference inputs
        # parse. A '#' only starts an inline comment when preceded by whitespace.
        parser = configparser.ConfigParser(inline_comment_prefixes=("#",))
        parser.optionxform = str
        path = Path(config_path).expanduser()
        if not parser.read(path):
            raise FileNotFoundError(f"Configuration file not found or unreadable: {path}")

        if "hamiltonian" not in parser:
            raise ValueError("Configuration must include a [hamiltonian] section.")

        hamiltonian = cls._parse_hamiltonian_section(parser["hamiltonian"])
        hamiltonian_plots = cls._parse_hamiltonian_plots_section(parser["hamiltonian_plots"]) if "hamiltonian_plots" in parser else HamiltonianPlotsConfig()
        kgrid = cls._parse_kgrid_section(parser["kgrid"]) if "kgrid" in parser else KGridConfig()
        timegrid = cls._parse_timegrid_section(parser["timegrid"]) if "timegrid" in parser else TimeGridConfig()
        laser = cls._parse_laser_section(parser["laser"]) if "laser" in parser else LaserConfig()
        cmd = cls._parse_cmd_section(parser["cmd"]) if "cmd" in parser else CMDConfig()
        cmd_plots = cls._parse_cmd_plots_section(parser["cmd_plots"]) if "cmd_plots" in parser else CMDPlotsConfig()
        xtp = cls._parse_xtp_section(parser["xtp"]) if "xtp" in parser else XTPConfig()
        ldos = cls._parse_ldos_section(parser["ldos"]) if "ldos" in parser else LDOSConfig()
        if "susceptibility_scan" in parser:
            xtp = replace(
                xtp,
                **cls._parse_legacy_susceptibility_scan_section(parser["susceptibility_scan"]),
            )
        # Susceptibility-sweep solver parameters now live INSIDE [xtp] (read with
        # the same CMD parser, since [xtp] and [cmd] share no key names). The
        # standalone [susceptibility_solver] section is still accepted for
        # backward compatibility and takes precedence when present.
        if "susceptibility_solver" in parser:
            susceptibility_solver = cls._parse_cmd_section(parser["susceptibility_solver"])
        elif "xtp" in parser:
            susceptibility_solver = cls._parse_cmd_section(parser["xtp"])
        else:
            susceptibility_solver = CMDConfig()
        return cls(
            hamiltonian=hamiltonian,
            hamiltonian_plots=hamiltonian_plots,
            kgrid=kgrid,
            timegrid=timegrid,
            laser=laser,
            cmd=cmd,
            cmd_plots=cmd_plots,
            xtp=xtp,
            ldos=ldos,
            susceptibility_solver=susceptibility_solver,
            source_path=path.resolve(),
        )

    def run_name(self) -> str:
        """Short identifier for this run, used for the standard output folder.

        Derived from the config file name: ``inputParams.tblg.cfg`` -> ``tblg``.
        Falls back to the model source-file stem, then ``"run"``.
        """
        if self.source_path is not None:
            stem = Path(self.source_path).stem
            if stem.startswith("inputParams."):
                stem = stem[len("inputParams."):]
            if stem and stem.lower() != "inputparams":
                return stem
        src = self.hamiltonian.source_file
        return Path(src).stem if src else "run"

    def with_standard_output_dirs(self) -> "QXTIConfig":
        """Return a copy whose outputs all live under ``outputs/<run_name>/``.

        HHG / time-domain  ->  outputs/<name>/cmd
        Tensor sweep (xtp) ->  outputs/<name>/xtp
        Hamiltonian plots  ->  outputs/<name>/hamiltonian

        This is the single source of truth for paths: inputs no longer need to
        set any output directory, and graphics reads from the same place.

        Only directories still left at their library DEFAULT are rewritten. A
        config that sets an explicit output_dir keeps it (so callers/tests can
        still pin a custom location). Idempotent.
        """
        root = f"outputs/{self.run_name()}"
        cmd = self.cmd
        if cmd.output_dir == "outputs/cmd":
            cmd = replace(cmd, output_dir=f"{root}/cmd")
        xtp = self.xtp
        if xtp.susceptibility_output_dir == "outputs/susceptibility":
            xtp = replace(xtp, susceptibility_output_dir=f"{root}/xtp")
        ham_plots = self.hamiltonian_plots
        if ham_plots.output_dir == "outputs/hamiltonian":
            ham_plots = replace(ham_plots, output_dir=f"{root}/hamiltonian")
        ldos = self.ldos
        if ldos.output_dir == "outputs/ldos":
            ldos = replace(ldos, output_dir=f"{root}/ldos")
        return replace(self, cmd=cmd, xtp=xtp, ldos=ldos, hamiltonian_plots=ham_plots)

    @staticmethod
    def _parse_hamiltonian_section(section: configparser.SectionProxy) -> HamiltonianConfig:
        reserved = {
            "source_file",
            "function_name",
            "model_name",
            "basis_size",
            "dimension",
            "basis_type",
            "is_periodic",
            "dk_derivative",
            "params",
            "lattice",
        }

        source_file = section.get("source_file", fallback="").strip()
        if not source_file:
            raise ValueError("[hamiltonian] must define source_file.")

        params: dict[str, Any] = {}
        lattice: dict[str, Any] = {}

        if section.get("params", fallback="").strip():
            parsed = _parse_scalar(section["params"])
            if not isinstance(parsed, dict):
                raise ValueError("[hamiltonian] params must be a dict when provided.")
            params.update(parsed)

        if section.get("lattice", fallback="").strip():
            parsed = _parse_scalar(section["lattice"])
            if not isinstance(parsed, dict):
                raise ValueError("[hamiltonian] lattice must be a dict when provided.")
            lattice.update(parsed)

        for key, raw_value in section.items():
            if key in reserved:
                continue
            value = _parse_scalar(raw_value)
            if key.startswith("param."):
                _nested_assignment(params, key[6:], value)
            elif key.startswith("lattice."):
                _nested_assignment(lattice, key[8:], value)
            else:
                params[key] = value

        return HamiltonianConfig(
            source_file=source_file,
            function_name=section.get("function_name", fallback="H").strip() or "H",
            model_name=section.get("model_name", fallback="custom-hamiltonian").strip() or "custom-hamiltonian",
            basis_size=section.getint("basis_size", fallback=None),
            dimension=section.getint("dimension", fallback=None),
            basis_type=section.get("basis_type", fallback="custom").strip() or "custom",
            is_periodic=section.getboolean("is_periodic", fallback=True),
            dk_derivative=section.getfloat("dk_derivative", fallback=1.0e-5),
            params=params,
            lattice=lattice,
        )

    @staticmethod
    def _parse_hamiltonian_plots_section(
        section: configparser.SectionProxy,
    ) -> HamiltonianPlotsConfig:
        manual_path = _parse_scalar(section.get("manual_path", fallback="[]"))
        if not isinstance(manual_path, list):
            manual_path = []

        return HamiltonianPlotsConfig(
            enabled=section.getboolean("enabled", fallback=True),
            plots=_parse_csv(section.get("plots", fallback="")),
            output_dir=section.get("output_dir", fallback="outputs/hamiltonian").strip() or "outputs/hamiltonian",
            path_type=section.get("path_type", fallback="kx").strip() or "kx",
            manual_path=manual_path,
            plane=section.get("plane", fallback="kx_ky").strip() or "kx_ky",
            k_min=section.getfloat("k_min", fallback=-0.1),
            k_max=section.getfloat("k_max", fallback=0.1),
            k1_min=section.getfloat("k1_min", fallback=section.getfloat("k_min", fallback=-0.1)),
            k1_max=section.getfloat("k1_max", fallback=section.getfloat("k_max", fallback=0.1)),
            k2_min=section.getfloat("k2_min", fallback=section.getfloat("k_min", fallback=-0.1)),
            k2_max=section.getfloat("k2_max", fallback=section.getfloat("k_max", fallback=0.1)),
            nk_path=section.getint("nk_path", fallback=201),
            mesh_points=section.getint("mesh_points", fallback=101),
            mesh_points_1=section.getint("mesh_points_1", fallback=None),
            mesh_points_2=section.getint("mesh_points_2", fallback=None),
            fixed_kx=section.getfloat("fixed_kx", fallback=0.0),
            fixed_ky=section.getfloat("fixed_ky", fallback=0.0),
            fixed_kz=section.getfloat("fixed_kz", fallback=0.0),
            band_index=section.getint("band_index", fallback=None),
            band_indices=_parse_band_indices(section.get("band_indices", fallback="")),
            surface_style=section.get("surface_style", fallback="surface").strip() or "surface",
            quiver_stride=section.getint("quiver_stride", fallback=4),
        )

    @staticmethod
    def _parse_kgrid_section(section: configparser.SectionProxy) -> KGridConfig:
        k_points_raw = section.get("k_points", fallback="").strip()
        points_per_axis_raw = section.get("points_per_axis", fallback=section.get("num_points", fallback="")).strip()
        return KGridConfig(
            dimension=section.getint("dimension", fallback=None),
            k_points=_parse_int_tuple(k_points_raw, default=()),
            points_per_axis=None if not points_per_axis_raw else int(_parse_scalar(points_per_axis_raw)),
            kx_values=_parse_float_list(section.get("kx_values", fallback=""), default=[]),
            ky_values=_parse_float_list(section.get("ky_values", fallback=""), default=[]),
            kz_values=_parse_float_list(section.get("kz_values", fallback=""), default=[]),
            shifted=section.getboolean("shifted", fallback=section.getboolean("monkhorst_pack", fallback=False)),
            auto_degeneracy_guard=section.getboolean("auto_degeneracy_guard", fallback=True),
        )

    @staticmethod
    def _parse_timegrid_section(section: configparser.SectionProxy) -> TimeGridConfig:
        dt_raw = section.get("dt", fallback="").strip()
        t_min_raw = section.get("t_min", fallback="").strip()
        t_max_raw = section.get("t_max", fallback="").strip()
        nt_raw = section.get("Nt", fallback="").strip()
        return TimeGridConfig(
            dt=0.01 if not dt_raw else float(_parse_scalar(dt_raw)),
            t_min=None if not t_min_raw else float(_parse_scalar(t_min_raw)),
            t_max=None if not t_max_raw else float(_parse_scalar(t_max_raw)),
            Nt=None if not nt_raw else int(_parse_scalar(nt_raw)),
            fft_window=section.get("fft_window", fallback="hann").strip() or "hann",
            zero_padding=section.getboolean("zero_padding", fallback=False),
            padding_factor=section.getint("padding_factor", fallback=2),
        )

    @staticmethod
    def _parse_laser_section(section: configparser.SectionProxy) -> LaserConfig:
        omega_raw = section.get("omega", fallback=section.get("w0", fallback="1.0")).strip()
        omega = float(_parse_scalar(omega_raw))
        num_cycles_raw = section.get("ncycles", fallback=section.get("num_cycles", fallback="")).strip()
        fwhm_raw = section.get("fwhm", fallback="").strip()
        if num_cycles_raw:
            num_cycles = float(_parse_scalar(num_cycles_raw))
        elif fwhm_raw:
            fwhm = float(_parse_scalar(fwhm_raw))
            num_cycles = fwhm * omega / (2.0 * math.pi)
        else:
            num_cycles = 1.0

        pulses_raw = section.get("pulses", fallback="").strip()
        parsed_pulses: list[dict[str, Any]] = []
        if pulses_raw:
            pulses_value = _parse_scalar(pulses_raw)
            if not isinstance(pulses_value, list) or any(not isinstance(item, dict) for item in pulses_value):
                raise ValueError("[laser] pulses must be a list of dictionaries when provided.")
            parsed_pulses = [dict(item) for item in pulses_value]

        return LaserConfig(
            omega=omega,
            E0=section.getfloat("E0", fallback=0.0),
            cep=section.getfloat("cep", fallback=section.getfloat("phase", fallback=0.0)),
            ellip=section.getfloat("ellip", fallback=section.getfloat("ellipticity", fallback=0.0)),
            ncycles=num_cycles,
            envname=section.get("envname", fallback=section.get("envelope", fallback="gauss")).strip() or "gauss",
            t0=section.getfloat("t0", fallback=0.0),
            phix=section.getfloat("phix", fallback=0.0),
            thetaz=section.getfloat("thetaz", fallback=section.getfloat("theta", fallback=0.0)),
            phiz=section.getfloat("phiz", fallback=section.getfloat("phi", fallback=0.0)),
            ncycles_clipped=section.getfloat("ncycles_clipped", fallback=0.0),
            blaser=section.getfloat("blaser", fallback=0.0),
            alaser=section.getfloat("alaser", fallback=0.0),
            pulses=parsed_pulses,
        )

    @staticmethod
    def _parse_cmd_section(section: configparser.SectionProxy) -> CMDConfig:
        solver_h_max = section.get("solver_h_max", fallback="").strip()
        solver_max_iterations_raw = section.get("solver_max_iterations", fallback="").strip()
        population_time_raw = section.get("population_time", fallback=section.get("T1", fallback="")).strip()
        coherence_time_raw = section.get("coherence_time", fallback=section.get("T2", fallback="")).strip()
        gamma_population_raw = section.get("gamma_population", fallback="").strip()
        gamma_coherence_raw = section.get("gamma_coherence", fallback="").strip()

        if not solver_max_iterations_raw:
            solver_max_iterations: int | None = None
        else:
            parsed_max_iterations = int(_parse_scalar(solver_max_iterations_raw))
            solver_max_iterations = None if parsed_max_iterations <= 0 else parsed_max_iterations

        if population_time_raw:
            population_time = float(_parse_scalar(population_time_raw))
        elif gamma_population_raw:
            gamma_population = float(_parse_scalar(gamma_population_raw))
            population_time = math.inf if gamma_population == 0.0 else 1.0 / gamma_population
        else:
            population_time = math.inf

        if coherence_time_raw:
            coherence_time = float(_parse_scalar(coherence_time_raw))
        elif gamma_coherence_raw:
            gamma_coherence = float(_parse_scalar(gamma_coherence_raw))
            coherence_time = math.inf if gamma_coherence == 0.0 else 1.0 / gamma_coherence
        else:
            coherence_time = math.inf

        return CMDConfig(
            enabled=section.getboolean("enabled", fallback=False),
            output_dir=section.get("output_dir", fallback="outputs/cmd").strip() or "outputs/cmd",
            max_order=section.getint("max_order", fallback=1),
            response_method=_canonical_method(section.get("response_method", fallback="ptddm")),
            rho_storage_dtype=section.get("rho_storage_dtype", fallback="complex128").strip().lower() or "complex128",
            scratch_rho_storage_dtype=section.get("scratch_rho_storage_dtype", fallback="auto").strip().lower() or "auto",
            keep_rho_orders=section.getboolean("keep_rho_orders", fallback=True),
            dataset_time_stride=max(1, section.getint("dataset_time_stride", fallback=1)),
            save_population_dataset=section.getboolean("save_population_dataset", fallback=True),
            save_coherence_dataset=section.getboolean("save_coherence_dataset", fallback=True),
            save_xtp_dataset=section.getboolean("save_xtp_dataset", fallback=True),
            population_time=population_time,
            coherence_time=coherence_time,
            temperature=section.getfloat("temperature", fallback=0.0),
            fermi_level=section.getfloat("fermi_level", fallback=0.0),
            distribution=section.get("distribution", fallback="fermi_dirac").strip() or "fermi_dirac",
            basis=section.get("basis", fallback="band").strip() or "band",
            gauge=section.get("gauge", fallback="length").strip() or "length",
            include_intraband=section.getboolean("include_intraband", fallback=True),
            include_interband=section.getboolean("include_interband", fallback=True),
            include_dephasing=section.getboolean("include_dephasing", fallback=True),
            solver=section.get("solver", fallback="rkf45").strip() or "rkf45",
            solver_tolerance=section.getfloat("solver_tolerance", fallback=1.0e-6),
            solver_max_iterations=solver_max_iterations,
            solver_max_rejections=section.getint("solver_max_rejections", fallback=10000),
            solver_h_min=section.getfloat("solver_h_min", fallback=1.0e-12),
            solver_h_max=None if not solver_h_max else float(_parse_scalar(solver_h_max)),
            solver_safety_factor=section.getfloat("solver_safety_factor", fallback=0.9),
            solver_min_factor=section.getfloat("solver_min_factor", fallback=0.2),
            solver_max_factor=section.getfloat("solver_max_factor", fallback=4.0),
            save_frequency_domain=section.getboolean("save_frequency_domain", fallback=False),
            n_workers=section.getint("n_workers", fallback=0),
            reserve_gb=section.getfloat("reserve_gb", fallback=1.0),
            tddm_amplitude_ladder=tuple(_parse_float_list(section.get("tddm_amplitude_ladder", fallback=""))),
            gradient_stencil=section.getint("gradient_stencil", fallback=2),
        )

    @staticmethod
    def _parse_cmd_plots_section(section: configparser.SectionProxy) -> CMDPlotsConfig:
        return CMDPlotsConfig(
            enabled=section.getboolean("enabled", fallback=False),
            output_path=section.get("output_path", fallback="outputs/cmd/population_heatmap.png").strip()
            or "outputs/cmd/population_heatmap.png",
            orders=_parse_int_tuple_or_none(section.get("orders", fallback="")),
            k_aggregation=section.get("k_aggregation", fallback="mean").strip() or "mean",
            save_animation=section.getboolean("save_animation", fallback=False),
            animation_path=section.get("animation_path", fallback="outputs/cmd/population_animation.gif").strip()
            or "outputs/cmd/population_animation.gif",
            fps=section.getint("fps", fallback=10),
            frame_stride=section.getint("frame_stride", fallback=1),
        )

    @staticmethod
    def _parse_xtp_section(section: configparser.SectionProxy) -> XTPConfig:
        sigma_raw = section.get("bz_mask_sigma", fallback="").strip()
        sigma_percent_legacy_raw = section.get("bz_mask_sigma_percent", fallback="").strip()
        susceptibility_omega_min_raw = section.get("susceptibility_omega_min", fallback="").strip()
        susceptibility_omega_max_raw = section.get("susceptibility_omega_max", fallback="").strip()
        susceptibility_plot_omega_min_raw = section.get("susceptibility_plot_omega_min", fallback="").strip()
        susceptibility_plot_omega_max_raw = section.get("susceptibility_plot_omega_max", fallback="").strip()
        return XTPConfig(
            bz_mask_enabled=section.getboolean("bz_mask_enabled", fallback=False),
            bz_mask_radius_percent=section.getfloat("bz_mask_radius_percent", fallback=100.0),
            bz_mask_sigma=None if not sigma_raw else float(_parse_scalar(sigma_raw)),
            bz_mask_sigma_percent_legacy=None
            if not sigma_percent_legacy_raw
            else float(_parse_scalar(sigma_percent_legacy_raw)),
            susceptibility_enabled=section.getboolean("susceptibility_enabled", fallback=False),
            susceptibility_output_dir=section.get(
                "susceptibility_output_dir",
                fallback="outputs/susceptibility",
            ).strip()
            or "outputs/susceptibility",
            susceptibility_orders=_parse_int_tuple(
                section.get("susceptibility_orders", fallback="1"),
                default=(1,),
            ),
            susceptibility_omega_min=0.05
            if not susceptibility_omega_min_raw
            else float(_parse_scalar(susceptibility_omega_min_raw)),
            susceptibility_omega_max=0.15
            if not susceptibility_omega_max_raw
            else float(_parse_scalar(susceptibility_omega_max_raw)),
            susceptibility_num_frequencies=section.getint("susceptibility_num_frequencies", fallback=11),
            susceptibility_omega_values=tuple(
                _parse_float_list(section.get("susceptibility_omega_values", fallback=""), default=[])
            ),
            susceptibility_eps=section.getfloat("susceptibility_eps", fallback=1.0e-14),
            susceptibility_method=_canonical_method(section.get("susceptibility_method", fallback="ptddm")),
            susceptibility_n_workers=section.getint("susceptibility_n_workers", fallback=0),
            susceptibility_plot_enabled=section.getboolean("susceptibility_plot_enabled", fallback=False),
            susceptibility_plot_output_dir=section.get("susceptibility_plot_output_dir", fallback="").strip(),
            susceptibility_plot_dataset_file=section.get(
                "susceptibility_plot_dataset_file",
                fallback="xtp_susceptibility.npz",
            ).strip()
            or "xtp_susceptibility.npz",
            susceptibility_plot_orders=_parse_int_tuple_or_none(
                section.get("susceptibility_plot_orders", fallback="")
            ),
            susceptibility_plot_positive_only=section.getboolean(
                "susceptibility_plot_positive_only",
                fallback=False,
            ),
            susceptibility_plot_omega_min=None
            if not susceptibility_plot_omega_min_raw
            else float(_parse_scalar(susceptibility_plot_omega_min_raw)),
            susceptibility_plot_omega_max=None
            if not susceptibility_plot_omega_max_raw
            else float(_parse_scalar(susceptibility_plot_omega_max_raw)),
            susceptibility_plot_dpi=section.getint("susceptibility_plot_dpi", fallback=300),
            susceptibility_plot_overview_enabled=section.getboolean(
                "susceptibility_plot_overview_enabled",
                fallback=True,
            ),
            susceptibility_plot_grid_enabled=section.getboolean(
                "susceptibility_plot_grid_enabled",
                fallback=True,
            ),
            susceptibility_plot_components_enabled=section.getboolean(
                "susceptibility_plot_components_enabled",
                fallback=True,
            ),
            susceptibility_plot_conductivity_enabled=section.getboolean(
                "susceptibility_plot_conductivity_enabled",
                fallback=True,
            ),
            susceptibility_plot_ev_axis=section.getboolean(
                "susceptibility_plot_ev_axis",
                fallback=True,
            ),
        )

    @staticmethod
    def _parse_ldos_section(section: configparser.SectionProxy) -> LDOSConfig:
        def _opt_float(key: str) -> float | None:
            raw = section.get(key, fallback="").strip()
            return None if not raw else float(_parse_scalar(raw))

        axis_origin_raw = section.get("spectral_axis_origin", fallback="").strip() or section.get(
            "spectral_line_origin",
            fallback="",
        ).strip()
        axis_origin = tuple(_parse_float_list(axis_origin_raw, default=[]))
        spectral_axis = section.get("spectral_axis", fallback="").strip().lower() or section.get(
            "spectral_line_axis",
            fallback="",
        ).strip().lower()
        spectral_axis_start = _opt_float("spectral_axis_start")
        if spectral_axis_start is None:
            spectral_axis_start = _opt_float("spectral_line_start")
        spectral_axis_end = _opt_float("spectral_axis_end")
        if spectral_axis_end is None:
            spectral_axis_end = _opt_float("spectral_line_end")
        k0 = tuple(_parse_float_list(section.get("spectral_k0", fallback=""), default=[]))
        k1 = tuple(_parse_float_list(section.get("spectral_k1", fallback=""), default=[]))
        # Multi-segment path: vertices separated by ';', coordinates by ','.
        path_raw = section.get("spectral_path", fallback="").strip()
        spectral_path: tuple[tuple[float, ...], ...] = tuple(
            tuple(float(x) for x in vertex.replace("[", "").replace("]", "").split(",") if x.strip())
            for vertex in path_raw.split(";")
            if vertex.strip()
        )
        labels_raw = section.get("spectral_path_labels", fallback="").strip()
        spectral_path_labels = tuple(s.strip() for s in labels_raw.split(",") if s.strip())
        return LDOSConfig(
            enabled=section.getboolean("enabled", fallback=False),
            output_dir=section.get("output_dir", fallback="outputs/ldos").strip() or "outputs/ldos",
            method=section.get("method", fallback="eigenvalues").strip().lower() or "eigenvalues",
            n_workers=section.getint("n_workers", fallback=0),
            reserve_gb=section.getfloat("reserve_gb", fallback=1.0),
            surface_normal=section.get("surface_normal", fallback="auto").strip().lower() or "auto",
            surface_side=section.get("surface_side", fallback="both").strip().lower() or "both",
            surface_max_iter=section.getint("surface_max_iter", fallback=80),
            surface_tol=section.getfloat("surface_tol", fallback=1.0e-8),
            surface_kn_points=section.getint("surface_kn_points", fallback=200),
            surface_ldos_enabled=section.getboolean("surface_ldos_enabled", fallback=True),
            surface_compare_layers=section.getint("surface_compare_layers", fallback=0),
            finite_nx=section.getint("finite_nx", fallback=24),
            finite_ny=section.getint("finite_ny", fallback=24),
            finite_ldos_energy=_opt_float("finite_ldos_energy"),
            finite_edge_cells=section.getint("finite_edge_cells", fallback=1),
            broadening=section.get("broadening", fallback="lorentzian").strip().lower() or "lorentzian",
            eta=section.getfloat("eta", fallback=0.0),
            e_min=_opt_float("e_min"),
            e_max=_opt_float("e_max"),
            num_energies=section.getint("num_energies", fallback=600),
            projected=section.getboolean("projected", fallback=True),
            fermi_level=section.getfloat("fermi_level", fallback=0.0),
            spectral_enabled=section.getboolean("spectral_enabled", fallback=False),
            spectral_axis=spectral_axis,
            spectral_axis_start=spectral_axis_start,
            spectral_axis_end=spectral_axis_end,
            spectral_axis_origin=axis_origin,
            spectral_k0=k0,
            spectral_k1=k1,
            spectral_path=spectral_path,
            spectral_path_labels=spectral_path_labels,
            spectral_num_k=section.getint("spectral_num_k", fallback=200),
            spectral_plane_enabled=section.getboolean("spectral_plane_enabled", fallback=False),
            spectral_plane_energy=_opt_float("spectral_plane_energy"),
            spectral_plane_kx_min=_opt_float("spectral_plane_kx_min"),
            spectral_plane_kx_max=_opt_float("spectral_plane_kx_max"),
            spectral_plane_ky_min=_opt_float("spectral_plane_ky_min"),
            spectral_plane_ky_max=_opt_float("spectral_plane_ky_max"),
            spectral_plane_kz=section.getfloat("spectral_plane_kz", fallback=0.0),
            spectral_plane_nkx=section.getint("spectral_plane_nkx", fallback=160),
            spectral_plane_nky=section.getint("spectral_plane_nky", fallback=160),
            spectral_cmap=section.get("spectral_cmap", fallback="Blues").strip() or "Blues",
            spectral_log_scale=section.getboolean("spectral_log_scale", fallback=True),
            spectral_log_vmin=_opt_float("spectral_log_vmin") if section.get("spectral_log_vmin", fallback="").strip() else -1.5,
            spectral_log_vmax=_opt_float("spectral_log_vmax") if section.get("spectral_log_vmax", fallback="").strip() else 3.3,
            spectral_alpha_min=section.getfloat("spectral_alpha_min", fallback=0.0),
            spectral_alpha_max=section.getfloat("spectral_alpha_max", fallback=1.0),
            plot_dpi=section.getint("plot_dpi", fallback=300),
        )

    @staticmethod
    def _parse_legacy_susceptibility_scan_section(
        section: configparser.SectionProxy,
    ) -> dict[str, Any]:
        return {
            "susceptibility_enabled": section.getboolean("enabled", fallback=False),
            "susceptibility_output_dir": section.get(
                "output_dir",
                fallback="outputs/susceptibility",
            ).strip()
            or "outputs/susceptibility",
            "susceptibility_orders": _parse_int_tuple(section.get("orders", fallback="1"), default=(1,)),
            "susceptibility_omega_min": section.getfloat("omega_min", fallback=0.05),
            "susceptibility_omega_max": section.getfloat("omega_max", fallback=0.15),
            "susceptibility_num_frequencies": section.getint("num_frequencies", fallback=11),
            "susceptibility_omega_values": tuple(
                _parse_float_list(section.get("omega_values", fallback=""), default=[])
            ),
            "susceptibility_eps": section.getfloat("eps", fallback=1.0e-14),
        }
