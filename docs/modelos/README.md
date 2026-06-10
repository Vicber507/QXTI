# Model Catalog

This directory documents the external Hamiltonian model files stored in
`models/` and loaded through `qxti.physics.CustomHamiltonian`.

## Available models

- `CustomHamiltonian`: generic loader for user-defined files in `models/`. See
  [custom_hamiltonian.md](custom_hamiltonian.md).
- `bi2se3_surface.py`: natural 2D surface Hamiltonian for Bi2Se3 loaded
  through `CustomHamiltonian`. See [bi2se3_surface.md](bi2se3_surface.md).
- `graphene.py`: monolayer graphene tight-binding model on the honeycomb
  lattice.
- `graphene_bilayer.py`: 4-band bilayer graphene model with `stacking = AB`,
  `BA` or `AA`. The Bernal branches use `gamma0`, `gamma1`, `gamma3`,
  `gamma4`, `delta_prime` and `u`; the `AA` branch uses the minimal
  `gamma0`-`gamma1`-`u` form.
