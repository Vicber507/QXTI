from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import importlib.util
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np

from .hamiltonian import ComplexArray, Hamiltonian


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(slots=True)
class CustomHamiltonian(Hamiltonian):
    """Hamiltonian wrapper for one user-defined function loaded from disk.

    The external file is expected to expose a callable with signature
    ``H(kx, ky, kz, params)`` and may optionally define metadata such as
    ``MODEL_NAME`` or ``DEFAULT_PARAMS``.
    """

    model_name: str = "custom-hamiltonian"
    params: dict[str, Any] = field(default_factory=dict)
    lattice: dict[str, Any] = field(default_factory=dict)
    basis_size: int = 1
    dimension: int = 3
    basis_type: str = "custom"
    is_periodic: bool = True
    dk_derivative: float = 1.0e-5
    source_file: str = ""
    function_name: str = "H"
    user_function: Callable[[float, float, float, dict[str, Any]], Any] = field(
        init=False,
        repr=False,
    )
    _module: ModuleType = field(init=False, repr=False)
    _module_path: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source_file, str) or not self.source_file.strip():
            raise ValueError("source_file must be a non-empty string.")
        if not isinstance(self.function_name, str) or not self.function_name.strip():
            raise ValueError("function_name must be a non-empty string.")

        self.user_function = self.load_from_file()
        self._apply_module_metadata()
        Hamiltonian.__post_init__(self)

    def load_from_file(self) -> Callable[[float, float, float, dict[str, Any]], Any]:
        """Load and validate the external Hamiltonian function."""

        source_path = self._resolve_source_path(self.source_file)
        if not source_path.is_file():
            raise FileNotFoundError(f"Custom Hamiltonian source file not found: {source_path}")

        module_name = f"qxti_user_model_{source_path.stem}_{abs(hash(str(source_path)))}"
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not build an import spec for {source_path}.")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        user_function = getattr(module, self.function_name, None)
        if user_function is None:
            raise AttributeError(
                f"Function '{self.function_name}' was not found in {source_path.name}."
            )
        if not callable(user_function):
            raise TypeError(
                f"Object '{self.function_name}' in {source_path.name} is not callable."
            )

        self._validate_user_function_signature(user_function)
        self._module = module
        self._module_path = source_path
        return user_function

    def default_params(self) -> dict[str, Any]:
        """Return default parameters declared by the external model file."""

        provider = getattr(self._module, "default_params", None)
        if callable(provider):
            params = provider()
            if not isinstance(params, dict):
                raise TypeError("default_params() in the external model must return a dict.")
            return dict(params)

        params = getattr(self._module, "DEFAULT_PARAMS", None)
        if params is None:
            return {}
        if not isinstance(params, dict):
            raise TypeError("DEFAULT_PARAMS in the external model must be a dict.")
        return dict(params)

    def default_lattice(self) -> dict[str, Any]:
        """Return default lattice information declared by the external model file."""

        # Prefer a callable provider because it can derive the lattice from the
        # resolved input parameters (for example an anisotropic a2 or a TBLG
        # twist angle). A static DEFAULT_LATTICE remains the fallback.
        provider = getattr(self._module, "default_lattice", None)
        if callable(provider):
            signature = inspect.signature(provider)
            supports_params = False
            try:
                signature.bind(dict(self.params))
            except TypeError:
                supports_params = False
            else:
                supports_params = True

            lattice = provider(dict(self.params)) if supports_params else provider()
            if not isinstance(lattice, dict):
                raise TypeError("default_lattice() in the external model must return a dict.")
            return deepcopy(lattice)

        lattice = getattr(self._module, "DEFAULT_LATTICE", None)
        if lattice is not None:
            if not isinstance(lattice, dict):
                raise TypeError("DEFAULT_LATTICE in the external model must be a dict.")
            return deepcopy(lattice)

        return {}

    def H(self, kx: float, ky: float, kz: float) -> np.ndarray:
        """Evaluate the external function and return a validated complex matrix."""

        matrix = self.user_function(float(kx), float(ky), float(kz), dict(self.params))
        return self.validate_matrix(np.asarray(matrix, dtype=complex))

    def dH_dk(self, kx: float, ky: float, kz: float, direction: str) -> np.ndarray:
        """Velocity operator dH/dk.

        If the external model exposes an ANALYTIC derivative ``dH_dk(kx, ky, kz,
        direction, params)``, use it (exact, faster, no finite-difference error);
        otherwise fall back to the base centered finite difference.
        """
        provider = getattr(self._module, "dH_dk", None)
        if callable(provider):
            matrix = provider(float(kx), float(ky), float(kz), str(direction), dict(self.params))
            return self.validate_matrix(np.asarray(matrix, dtype=complex))
        return Hamiltonian.dH_dk(self, kx, ky, kz, direction)

    def summary(self) -> dict[str, Any]:
        """Return a serializable summary of the custom Hamiltonian loader."""

        data = Hamiltonian.summary(self)
        data.update(
            {
                "source_file": self.source_file,
                "resolved_source_file": str(self._module_path),
                "function_name": self.function_name,
            }
        )
        return data

    def _apply_module_metadata(self) -> None:
        model_name = getattr(self._module, "MODEL_NAME", None)
        if model_name and self.model_name == "custom-hamiltonian":
            self.model_name = str(model_name)
        elif self.model_name == "custom-hamiltonian":
            self.model_name = self._module_path.stem

        module_basis_size = getattr(self._module, "BASIS_SIZE", None)
        if module_basis_size is not None:
            module_basis_size = int(module_basis_size)
            if self.basis_size not in {1, module_basis_size}:
                raise ValueError(
                    "basis_size conflicts with BASIS_SIZE declared by the external model."
                )
            self.basis_size = module_basis_size

        module_dimension = getattr(self._module, "DIMENSION", None)
        if module_dimension is not None:
            module_dimension = int(module_dimension)
            if self.dimension not in {3, module_dimension}:
                raise ValueError(
                    "dimension conflicts with DIMENSION declared by the external model."
                )
            self.dimension = module_dimension

        module_basis_type = getattr(self._module, "BASIS_TYPE", None)
        if module_basis_type is not None:
            if self.basis_type not in {"custom", str(module_basis_type)}:
                raise ValueError(
                    "basis_type conflicts with BASIS_TYPE declared by the external model."
                )
            self.basis_type = str(module_basis_type)

        module_is_periodic = getattr(self._module, "IS_PERIODIC", None)
        if module_is_periodic is not None:
            self.is_periodic = bool(module_is_periodic)

    @staticmethod
    def _resolve_source_path(source_file: str) -> Path:
        source_path = Path(source_file).expanduser()
        if source_path.is_absolute():
            return source_path.resolve()

        if source_path.suffix != ".py":
            source_path = source_path.with_suffix(".py")

        return (MODELS_DIR / source_path).resolve()

    @staticmethod
    def _validate_user_function_signature(
        user_function: Callable[[float, float, float, dict[str, Any]], Any],
    ) -> None:
        try:
            signature = inspect.signature(user_function)
            signature.bind(0.0, 0.0, 0.0, {})
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "The custom Hamiltonian function must accept the signature "
                "H(kx, ky, kz, params)."
            ) from exc
