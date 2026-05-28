from __future__ import annotations

import ast
import configparser
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
class QXTIConfig:
    hamiltonian: HamiltonianConfig
    hamiltonian_plots: HamiltonianPlotsConfig = field(default_factory=HamiltonianPlotsConfig)
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
        return cls(
            hamiltonian=hamiltonian,
            hamiltonian_plots=hamiltonian_plots,
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
