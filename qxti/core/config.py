from __future__ import annotations

import ast
import configparser
import math
from dataclasses import dataclass, field
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
    points_per_axis: int = 3
    kx_values: list[float] = field(default_factory=list)
    ky_values: list[float] = field(default_factory=list)
    kz_values: list[float] = field(default_factory=list)


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
    phase: float = 0.0
    ellipticity: float = 0.0
    fwhm: float = 1.0
    envelope: str = "gaussian"
    t0: float = 0.0
    theta: float = math.pi / 2.0
    phi: float = 0.0


@dataclass(slots=True)
class CMDConfig:
    enabled: bool = False
    output_dir: str = "outputs/cmd"
    max_order: int = 1
    gamma_population: float = 0.0
    gamma_coherence: float = 0.0
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
    solver_max_iterations: int = 100000
    solver_h_min: float = 1.0e-12
    solver_h_max: float | None = None
    save_frequency_domain: bool = False


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
class QXTIConfig:
    hamiltonian: HamiltonianConfig
    hamiltonian_plots: HamiltonianPlotsConfig = field(default_factory=HamiltonianPlotsConfig)
    kgrid: KGridConfig = field(default_factory=KGridConfig)
    timegrid: TimeGridConfig = field(default_factory=TimeGridConfig)
    laser: LaserConfig = field(default_factory=LaserConfig)
    cmd: CMDConfig = field(default_factory=CMDConfig)
    cmd_plots: CMDPlotsConfig = field(default_factory=CMDPlotsConfig)
    source_path: Path | None = None

    @classmethod
    def from_file(cls, config_path: str | Path) -> QXTIConfig:
        parser = configparser.ConfigParser()
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
        return cls(
            hamiltonian=hamiltonian,
            hamiltonian_plots=hamiltonian_plots,
            kgrid=kgrid,
            timegrid=timegrid,
            laser=laser,
            cmd=cmd,
            cmd_plots=cmd_plots,
            source_path=path.resolve(),
        )

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
        return KGridConfig(
            dimension=section.getint("dimension", fallback=None),
            points_per_axis=section.getint("points_per_axis", fallback=section.getint("num_points", fallback=3)),
            kx_values=_parse_float_list(section.get("kx_values", fallback=""), default=[]),
            ky_values=_parse_float_list(section.get("ky_values", fallback=""), default=[]),
            kz_values=_parse_float_list(section.get("kz_values", fallback=""), default=[]),
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
        return LaserConfig(
            omega=section.getfloat("omega", fallback=1.0),
            E0=section.getfloat("E0", fallback=0.0),
            phase=section.getfloat("phase", fallback=0.0),
            ellipticity=section.getfloat("ellipticity", fallback=0.0),
            fwhm=section.getfloat("fwhm", fallback=1.0),
            envelope=section.get("envelope", fallback="gaussian").strip() or "gaussian",
            t0=section.getfloat("t0", fallback=0.0),
            theta=section.getfloat("theta", fallback=math.pi / 2.0),
            phi=section.getfloat("phi", fallback=0.0),
        )

    @staticmethod
    def _parse_cmd_section(section: configparser.SectionProxy) -> CMDConfig:
        solver_h_max = section.get("solver_h_max", fallback="").strip()
        return CMDConfig(
            enabled=section.getboolean("enabled", fallback=False),
            output_dir=section.get("output_dir", fallback="outputs/cmd").strip() or "outputs/cmd",
            max_order=section.getint("max_order", fallback=1),
            gamma_population=section.getfloat("gamma_population", fallback=0.0),
            gamma_coherence=section.getfloat("gamma_coherence", fallback=0.0),
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
            solver_max_iterations=section.getint("solver_max_iterations", fallback=100000),
            solver_h_min=section.getfloat("solver_h_min", fallback=1.0e-12),
            solver_h_max=None if not solver_h_max else float(_parse_scalar(solver_h_max)),
            save_frequency_domain=section.getboolean("save_frequency_domain", fallback=False),
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
