# CustomHamiltonian

## Purpose

`qxti.physics.CustomHamiltonian` loads one user-defined Hamiltonian function
from `models/`.

Its responsibility is exactly this:

- store `source_file`
- store `function_name`
- load one external callable from disk
- call that function through the standard QXTI `Hamiltonian` interface

## Expected External Signature

The loaded model file must expose a callable with signature:

```python
H(kx, ky, kz, params) -> matrix
```

Optional helpers supported by the loader:

```python
def default_params() -> dict: ...
```

Optional metadata supported by the loader:

```python
MODEL_NAME
BASIS_SIZE
DIMENSION
BASIS_TYPE
IS_PERIODIC
```

## Constructor

```python
from qxti.physics import CustomHamiltonian

model = CustomHamiltonian(
    source_file="bi2se3_surface.py",
    function_name="H",
)
```

## How It Resolves Files

- relative paths are resolved inside `models/`
- if the `.py` suffix is omitted, the loader adds it automatically
- absolute paths are also accepted

## Example

```python
from qxti.physics import CustomHamiltonian

model = CustomHamiltonian(
    source_file="bi2se3_surface.py",
    params={"A14": 0.0030},
)

matrix = model.H(0.1, 0.0, 0.0)
```

## Visual Verification

You can generate a multi-panel preview for the current custom Hamiltonian with:

```bash
python tests/test_custom_hamiltonian.py
```

That script keeps the assertions and also saves:

```text
tests/custom_hamiltonian_preview.png
```

The preview includes:

- band cut along `kx`
- off-diagonal matrix magnitude along the cut
- band energies versus time at one fixed k-point
- lowest-band map in the `(kx, ky)` plane
- highest-band map in the `(kx, ky)` plane
- direct-gap map
