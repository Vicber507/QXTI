# Hamiltonian Class Guide

## Purpose

`qxti.physics.hamiltonian.Hamiltonian` defines the base interface for tight-binding Hamiltonian models.

Its job is intentionally narrow:

- store model metadata and parameters
- define the expected matrix size and k-space dimension
- expose the Hamiltonian matrix `H(k)`
- provide numerical k-derivatives of `H(k)`
- provide derived operators such as velocity and inverse mass
- diagonalize `H(k)` and transform operators to the band basis

It does not know anything about lasers, external fields, time grids, solvers, or optical-response workflows.

## File Location

- Implementation: `qxti/physics/hamiltonian.py`
- Generic external loader: `qxti/physics/custom_hamiltonian.py`
- Contract tests: `tests/test_hamiltonian.py`
- Concrete model catalog: `docs/modelos/README.md`

## Current Design

The current version of `Hamiltonian` is an abstract base class. That means it defines the shared behavior expected from all Hamiltonian models, but each concrete model must implement its own `H(kx, ky, kz)` method.

This is the public constructor interface:

```python
Hamiltonian(
    model_name,
    params={},
    basis_size=1,
    dimension=3,
    basis_type="orbital",
    is_periodic=True,
)
```

Because `Hamiltonian` is abstract, you do not instantiate it directly. Instead, create a subclass:

```python
import numpy as np

from qxti.physics import Hamiltonian


class TwoBandHamiltonian(Hamiltonian):
    def default_params(self) -> dict[str, float]:
        return {"mass": 1.0, "coupling": 2.0}

    def H(self, kx: float, ky: float, kz: float) -> np.ndarray:
        mass = self.params["mass"]
        coupling = self.params["coupling"]
        return np.array(
            [
                [mass + kx**2, coupling * (kx - 1j * ky)],
                [coupling * (kx + 1j * ky), -mass + ky**2],
            ],
            dtype=complex,
        )


hamiltonian = TwoBandHamiltonian(model_name="two-band", basis_size=2)
```

## Attribute-by-Attribute Explanation

### `model_name`

Human-readable name of the model.

- Must be a non-empty string.
- Useful for logs, result metadata, plots, and exported data.

Examples:

```text
"graphene"
"two-band"
"hubbard-chain"
```

### `params`

Dictionary containing model-specific parameters.

- Stored as a regular Python `dict`.
- Merged with `default_params()` during initialization.
- Intended for physical constants such as hopping amplitudes, onsite energies, masses, gaps, and couplings.

Example:

```python
params = {
    "mass": 1.0,
    "coupling": 2.0,
}
```

### `basis_size`

Number of basis states in the Hamiltonian matrix.

- Must be strictly positive.
- Determines the required shape of `H(k)`.
- If `basis_size = Nb`, then `H(k)` must return an `Nb x Nb` matrix.

Example:

```text
basis_size = 2
H(k) shape = (2, 2)
```

### `dimension`

Number of reciprocal-space dimensions used by the model.

Allowed range:

```text
1 <= dimension <= 3
```

Interpretation:

- `dimension = 1`: only `kx` is active
- `dimension = 2`: `kx` and `ky` are active
- `dimension = 3`: `kx`, `ky`, and `kz` are active

Even for lower-dimensional models, the public methods still receive `kx`, `ky`, and `kz`. The inactive coordinates can be ignored by the concrete model.

### `basis_type`

Text label describing the basis used by the model.

Examples:

```text
"orbital"
"site"
"spin"
"band"
```

This value is metadata. It does not change the numerical behavior by itself.

### `is_periodic`

Boolean flag describing whether the model is periodic in reciprocal space.

- `True`: periodic tight-binding model
- `False`: non-periodic or effective model

This value is metadata in the current implementation. Future grid or validation code may use it to choose boundary behavior.

## Method-by-Method Explanation

### `default_params()`

Returns the model's default parameter dictionary.

Base behavior:

```python
def default_params(self) -> dict[str, Any]:
    return {}
```

Concrete Hamiltonians can override it:

```python
def default_params(self) -> dict[str, float]:
    return {"t": 1.0, "delta": 0.2}
```

During initialization, these defaults are merged with any user-provided `params`.

### `set_params(params)`

Updates the internal parameter dictionary.

```python
hamiltonian.set_params({"mass": 0.5, "coupling": 2.0})
```

This method stores a copy of the dictionary so that later mutations to the input dictionary do not silently alter the object.

### `H(kx, ky, kz)`

Returns the Hamiltonian matrix at one k-point.

This is the only abstract method. Every concrete model must implement it.

Expected output:

```text
ndarray[basis_size, basis_size]
```

The output is converted internally to a complex NumPy array and validated against `basis_size`.

### `dH_dk(kx, ky, kz, dir)`

Returns the first derivative of the Hamiltonian with respect to one k-space direction.

```python
hamiltonian.dH_dk(kx, ky, kz, "x")
```

Supported directions:

```text
"x", "y", "z"
```

The current implementation uses a central finite difference:

```text
dH/dk_i = [H(k_i + step) - H(k_i - step)] / (2 step)
```

The finite-difference step is currently:

```text
1.0e-5
```

### `d2H_dk2(kx, ky, kz, dir1, dir2)`

Returns the second derivative of the Hamiltonian.

Same-direction example:

```python
hamiltonian.d2H_dk2(kx, ky, kz, "x", "x")
```

Mixed-direction example:

```python
hamiltonian.d2H_dk2(kx, ky, kz, "x", "y")
```

The method supports both diagonal and mixed second derivatives using finite differences.

### `velocity_operator(kx, ky, kz, dir)`

Returns the velocity operator in atomic units.

Current convention:

```text
v_i(k) = dH(k) / dk_i
```

Because QXTI uses atomic units internally, no extra `hbar` factor appears here.

### `inverse_mass_operator(kx, ky, kz, dir1, dir2)`

Returns the inverse mass operator in atomic units.

Current convention:

```text
m^{-1}_{ij}(k) = d2H(k) / dk_i dk_j
```

### `diagonalize(kx, ky, kz)`

Diagonalizes the Hamiltonian matrix at one k-point.

Returns:

```text
(energies[Nb], eigenvectors[Nb,Nb])
```

Internally, it uses:

```python
np.linalg.eigh(...)
```

This assumes the Hamiltonian is Hermitian.

### `eigenvalues(kx, ky, kz)`

Returns only the band energies.

```python
energies = hamiltonian.eigenvalues(kx, ky, kz)
```

Expected output:

```text
ndarray[Nb]
```

### `eigenvectors(kx, ky, kz)`

Returns only the eigenvectors.

```python
vectors = hamiltonian.eigenvectors(kx, ky, kz)
```

Expected output:

```text
ndarray[Nb,Nb]
```

Each column corresponds to one eigenvector, following NumPy's `eigh` convention.

### `transform_to_band_basis(op, kx, ky, kz)`

Transforms an operator from the original basis to the band basis.

```python
op_band = hamiltonian.transform_to_band_basis(op, kx, ky, kz)
```

Formula:

```text
O_band = U^\dagger O U
```

where `U` is the eigenvector matrix returned by `eigenvectors(kx, ky, kz)`.

### `validate_hermiticity(kx, ky, kz)`

Checks whether `H(k)` is Hermitian at one k-point.

```python
is_valid = hamiltonian.validate_hermiticity(kx, ky, kz)
```

Returns:

```text
True or False
```

Internally, it checks:

```text
H(k) == H(k)^\dagger
```

within NumPy's default numerical tolerance.

## Direction Validation

The Hamiltonian object accepts direction labels as strings:

```text
"x" -> 0
"y" -> 1
"z" -> 2
```

If the model dimension is lower than the requested direction, the object raises an error.

Example:

```python
hamiltonian = TwoBandHamiltonian(
    model_name="two-band",
    basis_size=2,
    dimension=2,
)

hamiltonian.dH_dk(0.0, 0.0, 0.0, "z")
```

This raises a `ValueError` because `z` is outside a two-dimensional model.

## Matrix Validation

Every matrix returned by `H(k)` is validated internally.

If:

```text
basis_size = 2
```

then `H(k)` must return:

```text
shape = (2, 2)
```

If the matrix has the wrong shape, the object raises a `ValueError`.

## Responsibility Boundary

`Hamiltonian` is the abstract tight-binding interface.

It knows about:

- model parameters
- basis size
- reciprocal-space coordinates
- Hamiltonian matrices
- derivatives of `H(k)`
- operators derived from `H(k)`
- band-basis transformations

It does not know about:

- laser pulses
- electric fields
- vector potentials
- time evolution
- solvers
- observables
- plotting

This separation is intentional. The first layer allowed to combine Hamiltonian and laser information is the response layer, such as `CMD`.

## Minimal Usage Example

```python
hamiltonian = TwoBandHamiltonian(
    model_name="two-band",
    basis_size=2,
    dimension=2,
)

kx = 0.1
ky = 0.2
kz = 0.0

matrix = hamiltonian.H(kx, ky, kz)
energies = hamiltonian.eigenvalues(kx, ky, kz)
velocity_x = hamiltonian.velocity_operator(kx, ky, kz, "x")
mass_xy = hamiltonian.inverse_mass_operator(kx, ky, kz, "x", "y")
```

## Current Limitations

- Derivatives are numerical finite differences, not analytic derivatives.
- The finite-difference step is fixed internally.
- The class assumes `np.linalg.eigh` is appropriate for diagonalization.
- Hermiticity is checked only when explicitly requested.
- `basis_type` and `is_periodic` are metadata for now.

These choices keep the base object simple while allowing concrete Hamiltonian models to grow later.
