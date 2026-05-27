from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np

from .hamiltonian import ComplexArray, Hamiltonian


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(slots=True)
class CustomHamiltonian(Hamiltonian):
    """Load one user-defined Hamiltonian function from ``models/``."""

    model_name: str = "custom-hamiltonian"
    params: dict[str, Any] = field(default_factory=dict)
    basis_size: int = 1
    dimension: int = 3
    basis_type: str = "orbital"
    is_periodic: bool = True
    source_file: str = ""
    function_name: str = "H"
    user_function: Callable[[float, float, float, dict[str, Any]], Any] = field(
        init=False,
        repr=False,
    )
    _module: ModuleType = field(init=False, repr=False)
    _module_path: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.source_file:
            raise ValueError("source_file must be a non-empty string.")

        self.user_function = self.load_from_file()
        self._apply_module_metadata()
        super().__post_init__()

    def load_from_file(self) -> Callable[[float, float, float, dict[str, Any]], Any]:
        """Load the external Hamiltonian function from ``models/``."""

        source_path = self._resolve_source_path(self.source_file)
        if not source_path.is_file():
            raise FileNotFoundError(f"Custom Hamiltonian source file not found: {source_path}")

        module_name = f"qxti_user_model_{source_path.stem}_{abs(hash(source_path))}"
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not build an import spec for {source_path}.")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        user_function = getattr(module, self.function_name, None)
        if not callable(user_function):
            raise AttributeError(
                f"Function '{self.function_name}' was not found or is not callable in "
                f"{source_path.name}."
            )

        self._module = module
        self._module_path = source_path
        return user_function

    def default_params(self) -> dict[str, Any]:
        """Return the external model defaults when they are available."""

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

    def H(self, kx: float, ky: float, kz: float) -> ComplexArray:
        """Delegate ``H(kx, ky, kz, params)`` to the user-defined function."""

        try:
            matrix = self.user_function(float(kx), float(ky), float(kz), dict(self.params))
        except TypeError as exc:
            raise TypeError(
                "The custom Hamiltonian function must have signature "
                "H(kx, ky, kz, params)."
            ) from exc

        return np.asarray(matrix, dtype=complex)

    def _apply_module_metadata(self) -> None:
        model_name = getattr(self._module, "MODEL_NAME", None)
        if self.model_name == "custom-hamiltonian":
            if model_name:
                self.model_name = str(model_name)
            else:
                self.model_name = self._module_path.stem

        basis_size = getattr(self._module, "BASIS_SIZE", None)
        if basis_size is not None:
            self.basis_size = int(basis_size)

        dimension = getattr(self._module, "DIMENSION", None)
        if dimension is not None:
            self.dimension = int(dimension)

        basis_type = getattr(self._module, "BASIS_TYPE", None)
        if basis_type is not None:
            self.basis_type = str(basis_type)

        is_periodic = getattr(self._module, "IS_PERIODIC", None)
        if is_periodic is not None:
            self.is_periodic = bool(is_periodic)

    @staticmethod
    def _resolve_source_path(source_file: str) -> Path:
        source_path = Path(source_file).expanduser()
        if source_path.is_absolute():
            return source_path.resolve()

        if source_path.suffix != ".py":
            source_path = source_path.with_suffix(".py")

        return (MODELS_DIR / source_path).resolve()
