# QXTI verification plan for the software paper

## 1. Purpose and scope

This plan organizes the evidence required to support a bounded claim that QXTI
correctly and robustly implements its documented tight-binding workflows. It
separates:

- **code verification**: the implementation solves the stated equations;
- **numerical verification**: discretization and solver errors are controlled;
- **cross-code verification**: observables agree with an independent program;
- **physical validation**: comparison with experiment, which is outside the core
  software-verification claim unless explicitly added later.

Passing one model does not validate every model, and passing unit tests does not
by itself validate a scientific observable. The paper must state exactly which
models, algorithms, observables, and parameter ranges are covered.

## 2. Rule for avoiding redundancy

QXTI is verified on two orthogonal axes:

1. **Every shipped model receives one lightweight model-contract benchmark.**
   This checks that its particular formula, metadata, units, lattice and known
   spectral landmarks are implemented correctly.
2. **Each generic calculation engine receives one deep benchmark on the smallest
   representative model that exercises its difficult case.** The same deep
   benchmark is not repeated for every model unless the model introduces a new
   numerical risk.

A new benchmark is justified only if it adds at least one of these capabilities:

- a new dimension (2-D versus 3-D);
- a new spectral class (gapped, gapless, degenerate or Weyl-node);
- a new basis scale (2, 4, 8 or many bands);
- a new observable or response order;
- a new boundary condition (bulk, ribbon, surface or finite);
- a new numerical method (k integration, time integration or basis truncation);
- a new execution path (serial, parallel, streamed, checkpointed or reduced
  precision).

The same calculation may support several checks, but evidence that shares the
same implementation is not described as independent. For example, agreement
between two QXTI engines is an internal consistency check; agreement with PythTB,
PYATB, WannierBerri, Kwant or CUED is cross-code evidence.

## 3. Evidence grades

| grade | evidence | role in the paper |
| --- | --- | --- |
| A | exact or closed-form solution | strongest code-verification evidence |
| B | independent external software | strongest cross-code evidence |
| C | independent QXTI formulation or solver | internal consistency and limit checks |
| D | invariant, symmetry, sum rule or regression test | necessary supporting evidence |

Every headline scientific capability should have at least one grade-A or grade-B
benchmark. Grades C and D alone support robustness but are not sufficient for a
broad correctness claim.

## 4. Model-contract layer

The following contract is applied once to every production model:

1. load the model and validate metadata, basis size, dimensionality and units;
2. test Hermiticity at deterministic generic k-points;
3. test spectral periodicity under primitive reciprocal translations (matrix
   periodicity is used only when the chosen tight-binding gauge permits it);
4. compare selected high-symmetry energies, gaps, nodes or analytic limits with
   an independent reference;
5. verify any declared symmetry or block reduction;
6. compare declared analytic derivatives with an independent finite difference,
   when analytic derivatives are supplied;
7. save the configuration, reference values, errors and software version.

This layer prevents a correct generic solver from hiding a wrong model formula or
unit conversion. It does **not** repeat full DOS, susceptibility and HHG studies
for every model.

| ID | production model | bands | role added by the contract | preferred reference |
| --- | --- | ---: | --- | --- |
| MC-HAL | Haldane | 2 | 2-D gapped/topological model | closed form + PythTB |
| MC-GRA | graphene | 2 | 2-D gapless Dirac points | closed form + PythTB |
| MC-BLG | bilayer graphene | 4 | 2-D multiband and AB/BA/AA limits | analytic reductions + PythTB |
| MC-BI2 | Bi2Se3 surface | 2 | surface Dirac Hamiltonian and spin basis | closed-form dispersion |
| MC-W2 | two-band WSM | 2 | 3-D BZ and Weyl nodes | analytic node positions |
| MC-WO | Orenstein WSM | 4 | inversion-broken 3-D multiband model | paper/reference implementation |
| MC-NP4 | NatPhys TaAs | 4 | independent four-band Weyl implementation | antelope/reference implementation |
| MC-TA4 | symmetry-exact TaAs | 4 | explicit space-group constraints | symmetry identities + reference bands |
| MC-FZ8 | Frank-Zhang TaAs | 8 | spinful eight-band model | cited model/reference code |
| MC-TB8 | realistic TaAs TB | 8 | eV/Angstrom to atomic-unit conversion | independent TB/Wannier bands |
| MC-TBLG | BMD TBLG | 76 default | large continuum basis and gauge covariance | independent BM implementation |

The example Hamiltonian template is tested as an API example, not claimed as a
physical production model.

## 5. Deep benchmark matrix

### 5.1 Static Hamiltonian, grids and operators

| ID | capability | representative case | independent evidence | principal metrics | status |
| --- | --- | --- | --- | --- | --- |
| STA-2D | 2-D bands and bulk DOS | Haldane topological/trivial | PythTB | pointwise band max/RMS; DOS relative L2; state-count sum rule | **PASS** |
| GRID-2D | independent 2-D BZ integration | Haldane | native QXTI and PythTB meshes | convergence curve; final cross-code L2 | **PASS** |
| OP-2D | dH, d2H, velocity and dipole | Haldane | closed derivatives + PythTB velocity | matrix max error; gauge-invariant interband magnitudes | **PASS** |
| GAPLESS-2D | degeneracy/gapless handling | graphene | analytic Dirac dispersion + PythTB | node position; cone velocity; grid-guard stability | **PASS** |
| GRID-3D | 3-D BZ weights and DOS | two-band WSM | independent 3-D integration | DOS L2; state count; low-energy scaling; grid convergence | **PASS** |
| MULTI-8 | multiband eigenvectors/operators | realistic TaAs TB | independent hopping reconstruction | selected bands; velocity invariants; symmetry residuals | **PASS** |
| TRUNC-MB | many-band basis truncation | TBLG | independent BM implementation | flat-band energies; bandwidth; N-rings convergence | **PASS** |

Every completed row has a versioned raw artifact and record in
`validation/results/registry.json`. MULTI-8 verifies the shipped TaAs hopping
model and its symmetries; it is not yet a comparison with ab-initio or
experimental bands. TRUNC-MB uses a separately coded higher-cutoff BM evaluator,
not an exact infinite-basis solution.

### 5.2 Topology and boundaries

| ID | capability | representative case | independent evidence | principal metrics | status |
| --- | --- | --- | --- | --- | --- |
| TOP-C | QXTI topological invariant | Haldane topological/trivial | PythTB Fukui result + analytic phase boundary | QXTI Chern error; mesh convergence; integer stability | blocked until QXTI exposes the observable |
| EDGE-2D | edge spectrum and LDOS | Haldane ribbon, both phases | Kwant or independently built PythTB ribbon | energy-resolved L2; in-gap edge weight; LDOS sum rule | planned |
| SURF-3D | semi-infinite surface solver | two-band WSM | finite-thickness/Kwant reference | surface spectral L2; Fermi-arc locus; iteration convergence | planned |
| FINITE | finite sample and projected LDOS | Haldane or graphene | direct dense diagonalization | eigenvalues; site LDOS; total-state sum rule | planned |

PythTB's Chern number currently classifies the Haldane input but does not verify a
QXTI Chern implementation. The paper must preserve this distinction.

### 5.3 Linear and nonlinear response

| ID | capability | representative case | independent evidence | principal metrics | status |
| --- | --- | --- | --- | --- | --- |
| RESP-L1 | linear conductivity/susceptibility | gapped Haldane plus graphene limit | independent Kubo implementation, PYATB or WannierBerri | complex-curve L2; peak positions; f-sum/Kramers-Kronig residual; k/eta convergence | planned |
| RESP-CV | relation between current and polarization outputs | same RESP-L1 run | analytic Fourier relation | residual of the documented sigma-chi convention | planned |
| RESP-L2 | second-order SHG/shift response | Orenstein WSM | PYATB or independent sum-over-states code | complex tensor L2; 2omega location; field-scaling exponent | planned |
| RESP-NULL | symmetry-forbidden response | inversion-symmetric control model | exact symmetry prediction | forbidden/allowed component ratio | planned |
| RESP-L3 | third-order perturbative response | gapped two-band control | independent recursion/sum-over-states result | complex amplitude/phase; E0 cubed scaling | planned |
| TENSOR | tensor reconstruction and coordinate transforms | synthetic known tensors + one physical case | exact constructed tensors | component max error; permutation and point-group residuals | partial internal tests |

`RESP-L2` includes a nonzero inversion-broken case, while `RESP-NULL` is its
negative control. This is one complementary pair, not a repetition across all
models.

### 5.4 Density matrix, time propagation and HHG

| ID | capability | representative case | independent evidence | principal metrics | status |
| --- | --- | --- | --- | --- | --- |
| TIME-EX | time integrators | exactly solvable two-level systems | analytic matrix exponential | global convergence order; trace; Hermiticity; positivity | partial internal tests |
| TIME-WF | weak-field limit | gapped Haldane | perturbative engine + external SBE solver | J(t) L2; harmonic amplitude/phase; convergence in E0 | planned |
| TIME-REL | T1/T2 relaxation | two-level control | closed-form relaxation | population/coherence decay rates; equilibrium limit | internal tests, paper benchmark planned |
| HHG-SEL | harmonic selection rules | graphene/Haldane controls | symmetry prediction | forbidden/allowed harmonic ratio; helicity rules | planned |
| HHG-X | complete HHG cross-code result | Haldane or graphene with identical pulse | CUED or independent SBE implementation | J(t) L2; spectrum L2; peak amplitude/phase | planned |
| FFT | time-to-frequency pipeline | synthetic sinusoids and known pulses | analytic Fourier content | frequency, amplitude, phase, leakage and window normalization | planned |

Agreement among PFDDM, PTDDM and TDDM is grade C. At least one complete HHG case
must also have grade-B evidence before the paper claims externally verified HHG.

### 5.5 Execution robustness and reproducibility

| ID | capability | representative workload | evidence | principal metrics | status |
| --- | --- | --- | --- | --- | --- |
| EXEC-P | serial/process/thread/block equivalence | small Haldane and one multiband case | same mathematical workload | relative L2 and max error | internal tests |
| EXEC-D | storage precision | complex128, complex64 and float16-complex scratch | complex128 reference | observable error and file-size reduction | planned paper benchmark |
| EXEC-R | checkpoint/resume and streaming | interrupted CMD calculation | uninterrupted run | byte/numeric equality; completed work count | planned |
| EXEC-M | memory guard and block planner | synthetic and production grids | analytic allocation bounds | peak memory; no invalid/empty chunks | internal tests |
| EXEC-X | platform reproducibility | Linux workstation and cluster/SLURM | archived environments | tolerance-bounded observable equality | planned |
| PERF | performance scaling | 2-, 8- and 76-band workloads | repeatable timing protocol | strong scaling; memory versus Nk, Nt and Nb | planned |

Performance is reported separately from correctness. A faster result is not
evidence of a more accurate result.

## 6. Standard metrics

Every benchmark uses only the metrics appropriate to its observable and records
the complete arrays needed to recompute them.

### Pointwise scalar or matrix error

For samples indexed by `s`:

```text
delta_s = QXTI_s - reference_s
max_abs = max_s |delta_s|
rms     = sqrt(mean_s |delta_s|^2)
```

For eigenvectors or interband matrices with arbitrary phases, compare projectors,
singular values, oscillator strengths or absolute matrix elements rather than raw
complex components.

### Complete-curve error

```text
relative_L2 = ||QXTI - reference||_2 / ||reference||_2
```

Peak positions, integrated spectral weight and a maximum pointwise difference are
also recorded, because an L2 norm alone can hide a shifted narrow resonance.

### Convergence

For discretization `h`, record the observable at a minimum of four refinements and
estimate the observed order from the slope of `log(error)` versus `log(h)`. A
benchmark passes only if it approaches a stable reference and its final error is
below the predeclared tolerance. Monotonic error is required only where the method
and metric theoretically imply it.

### Symmetry/null residual

```text
leakage = norm(forbidden components) / max(norm(allowed components), scale_floor)
```

An absolute floor is always included so that two nearly zero arrays cannot produce
a meaningless relative pass.

## 7. Acceptance-policy rules

1. Tolerances are committed before production results are generated.
2. Exact/closed references use tolerances derived from floating-point and
   discretization error; cross-code spectral curves use physically justified
   numerical tolerances.
3. A plot that merely looks similar is never a pass criterion.
4. Failed cases remain in the registry with their configuration and raw artifact.
5. Changing a convention or tolerance creates a new benchmark version; it does
   not silently overwrite the historical method.
6. Each PASS is restricted to the scope printed in its registry entry.
7. Production paper results must come from a clean commit
   (`working_tree_dirty: false`).

## 8. Required artifact for every benchmark

Each benchmark stores:

- stable benchmark ID and method version;
- QXTI commit and dirty/clean state;
- external program and version;
- complete model and numerical configuration;
- coordinate, unit, Fourier, current and degeneracy conventions;
- raw QXTI and reference arrays or a durable link to them;
- error formulas, tolerances and measured values;
- random seed and deterministic sampling rule;
- runtime, hardware and worker configuration when performance is relevant;
- PASS/FAIL and an explicit limitation statement.

The human-readable report is generated from the machine-readable registry. Paper
tables and figures should be generated from the same raw artifacts, not copied by
hand.

## 9. Minimal nonredundant paper claim set

The following completed set is sufficient for a broad but bounded software claim:

1. all eleven model contracts;
2. the existing three Haldane static benchmarks;
3. graphene gapless handling;
4. one 3-D WSM grid/DOS benchmark;
5. one eight-band TaAs benchmark;
6. one TBLG truncation benchmark;
7. Haldane edge/LDOS and WSM surface verification;
8. linear response with grade-B evidence;
9. nonzero second-order response plus a symmetry-null control;
10. exact time-integrator convergence and weak-field engine agreement;
11. one complete HHG comparison with an external code;
12. serial/parallel/storage/checkpoint robustness and a separate scaling study.

Additional models or response orders are added only when the paper claims them or
when they introduce a numerical class not represented above.

## 10. Recommended implementation order

1. Automate `MC-*` model contracts and finish graphene (`GAPLESS-2D`).
2. Add `GRID-3D` using the two-band WSM.
3. Validate `EDGE-2D` before the more expensive 3-D surface solver.
4. Implement `RESP-L1` and `RESP-CV` as the first full response benchmark.
5. Add `RESP-L2` plus `RESP-NULL`.
6. Promote current integrator tests into `TIME-EX`, then add `TIME-WF`.
7. Complete external `HHG-X`.
8. Add TaAs, TBLG, reproducibility and performance evidence.

This order validates foundational quantities before using them inside more derived
observables, so a response failure can be localized instead of forcing the entire
stack to be debugged at once.
