# Cross-code validation

The nonredundant verification matrix for the software paper is documented in
[`VERIFICATION_PLAN.md`](VERIFICATION_PLAN.md). It separates per-model contracts
from deep engine benchmarks and defines evidence grades, metrics, artifacts and
the recommended implementation order.

These benchmarks compare QXTI with independent scientific software using the
same Hamiltonian parameters, reciprocal-space coordinates, broadening, filling,
and numerical grids. A comparison is accepted only after conventions have been
made explicit; visual similarity of plots is not a pass criterion.

Install the validation dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[validation]'
```

Run the first benchmark:

```bash
.venv/bin/python -m validation.benchmark_haldane_pythtb
```

It validates the topological and trivial Haldane inputs against PythTB 2.x:

- band energies at deterministic random k-points;
- Gaussian-broadened bulk DOS on identical grids;
- occupied-band Chern number from PythTB's Fukui-Hatsugai-Suzuki routine;
- QXTI's DOS state-count sum rule.

Validate the reciprocal integration domain with independently generated meshes:

```bash
.venv/bin/python -m validation.benchmark_haldane_grid_convergence
```

QXTI samples its Cartesian rectangular reciprocal cell; PythTB independently
samples its native reduced primitive cell. The comparison is repeated on
`11x11`, `21x21`, `41x41`, and `81x81` meshes against a `161x161` PythTB
reference. Agreement must improve with every refinement and reach 1% or better.

Validate first/second Hamiltonian derivatives and the optical operators built
from them:

```bash
.venv/bin/python -m validation.benchmark_haldane_operators
```

This combines separately coded closed-form Haldane derivatives with PythTB's
analytic velocity operator. Raw complex interband elements are phase-gauge
dependent, so the cross-code comparison uses their gauge-invariant magnitudes.

Complete the remaining static Hamiltonian, grid and operator benchmarks from
section 5.1 of the verification plan:

```bash
.venv/bin/python -m validation.benchmark_graphene_gapless
.venv/bin/python -m validation.benchmark_wsm_grid_3d
.venv/bin/python -m validation.benchmark_taas_multiband_8
.venv/bin/python -m validation.benchmark_tblg_truncation
```

These add, respectively, a gapless Dirac model and guarded grid, an anisotropic
three-dimensional Weyl grid and DOS, an eight-band real-space hopping
reconstruction with symmetry checks, and convergence of TBLG's 76-band cutoff
against a 148-band reference. The TaAs and TBLG references are separately coded
evaluators, not experimental or ab-initio validations; the generated report
states that limitation explicitly.

All command-line benchmarks update:

- `validation/results/registry.json`: machine-readable cumulative registry;
- `validation/VALIDATION_REPORT.md`: human-readable report generated from it;
- one raw JSON artifact per benchmark under `validation/results/`.

The next validation layers should keep separate reports for:

1. bands, gaps, DOS/LDOS, topology and symmetry;
2. linear optical conductivity against WannierBerri or PYATB;
3. second-order response against PYATB SHG/shift-current calculations;
4. time-dependent current and HHG against CUED using an identical SBE model;
5. qualitative ab-initio comparisons with Octopus only after the tight-binding
   cross-code benchmarks pass.

For optical and HHG comparisons, record the field convention, Fourier sign,
current sign, spin/valley degeneracy, cell-volume normalization, occupation,
dephasing, gauge, pulse envelope, time window, and windowing function. A mismatch
in any one of these can create a constant factor or phase difference without a
physics error.
