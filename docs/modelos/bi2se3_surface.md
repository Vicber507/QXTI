# Bi2Se3 Surface Model

## Purpose

`models/bi2se3_surface.py` stores the user-defined Bi2Se3 surface Hamiltonian
as an external model file compatible with `qxti.physics.CustomHamiltonian`.

This follows the architecture you marked:

- `qxti.physics.custom_hamiltonian.CustomHamiltonian` is only the loader
- the actual user Hamiltonian lives in `models/`
- the external function uses the signature `H(kx, ky, kz, params) -> matrix`

## File Location

- Model file: `models/bi2se3_surface.py`
- Loader class: `qxti/physics/custom_hamiltonian.py`
- Tests: `tests/test_custom_hamiltonian.py`

## Metadata Exported By The Model File

The file exports metadata that `CustomHamiltonian` reads automatically:

```python
MODEL_NAME = "bi2se3-surface"
BASIS_SIZE = 2
DIMENSION = 2
BASIS_TYPE = "spin"
IS_PERIODIC = True
```

## Public Functions

The file provides:

```python
def default_params() -> dict: ...
def H(kx, ky, kz, params) -> np.ndarray: ...
```

`default_params()` returns the built-in equilibrium constants.

`H(...)` is the entry point used by `CustomHamiltonian`.

## Parameters

The model includes the original surface parameters:

```python
{
    "a0": 7.8234655927,
    "A0": -0.000937,
    "B0": 0.00060,
    "A11": 0.00711836,
    "A12": 0.00823187,
    "A14": 0.00202489,
    "B11": 0.00442095,
    "B14": 0.0,
}
```

This keeps the file compatible with the expected external signature, since the
loader only passes `kx`, `ky`, `kz`, and one parameter dictionary.

## Usage Through `CustomHamiltonian`

```python
from qxti.physics import CustomHamiltonian

model = CustomHamiltonian(source_file="bi2se3_surface.py")
Hk = model.H(0.05, -0.02, 0.0)
```

You can override any model parameter from the constructor:

```python
model = CustomHamiltonian(
    source_file="bi2se3_surface.py",
    params={
        "A14": 0.0030,
        "A0": -0.0011,
    },
)
```

## Mathematical Form

The matrix returned by the model keeps the original 2x2 form:

```text
H(k) = h0 sigma0 + h1 sigmax + h2 sigmay + h3 sigmaz
```

with bond-dependent phases evaluated over the three surface vectors of the
hexagonal Bi2Se3 surface lattice.

## Scope

This file contains only the natural equilibrium Hamiltonian.

- there is no built-in laser drive
- there is no explicit time dependence
- `kz` is accepted only to match the shared QXTI Hamiltonian interface
