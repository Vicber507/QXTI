# QXTI Validation Report

This report is generated from `validation/results/registry.json`. A passing entry applies only to the scope stated in that entry; it does not validate unrelated QXTI observables.

Each entry records how the benchmark was implemented, the provenance and independence level of its reference, the error calculation, and any production-code change required by the check. A separately coded formula is an independent reference, but it is not labeled as external software.

Recorded validations: **7** · Passed: **7** · Failed: **0**

## Validation workflow

- Each `validation/benchmark_*.py` module is an executable, deterministic benchmark that builds QXTI and its reference through separate code paths.
- Every run writes a raw JSON artifact, replaces its stable entry in `validation/results/registry.json`, and regenerates this report.
- `tests/test_external_validation.py` invokes the same benchmark functions so scientific checks fail in the normal pytest regression suite.
- The `validation` optional dependency group pins the external PythTB dependency used by the cross-code cases; closed-form and reconstruction references are explicitly classified as internal independent evidence.

## Graphene gapless spectrum and degeneracy-safe sampling

- Status: **PASS**
- Date: `2026-08-21`
- Scope: Gapless 2-D bands, Dirac cone, reciprocal periodicity and finite Berry-connection sampling for graphene.
- Reference: PythTB 2.0.2 graphene model and closed-form Dirac limit
- Evidence type: external cross-code plus analytic Dirac identities
- QXTI commit: `df53f4508eb96e0c87d28d44a212456a9fbc06b8`
- Working tree dirty: `True`

| band max error (Ha) | Dirac energy error (Ha) | cone velocity relative error | periodicity error (Ha) | minimum shifted-grid gap (Ha) |
| --- | --- | --- | --- | --- |
| 6.9388939e-17 | 1.2669255e-16 | 1.1621259e-06 | 1.110223e-16 | 0.033479058 |

Implementation:

- Build PythTB's graphene model with the same nearest-neighbour hopping and onsite mass, then compare sorted bands after an independent Cartesian-to-reduced momentum conversion.
- Evaluate both closed-form Dirac points, estimate the cone slope radially, test spectral reciprocal periodicity, and run every point of a shifted QXTI grid through the Berry-connection operator.

Reference provenance:

- PythTB is the external band reference; zero node energy and $v_F=3|t|a_0/2$ are analytic nearest-neighbour graphene results.

Production-code changes motivated by this benchmark:

- Replace rounded decimal real-space vectors in `models/graphene.py` with expressions derived from the configured lattice constant. The rounded metadata corrupted Cartesian-to-reduced conversion and produced a spurious cross-code error of about 1.36e-5 Ha even though H(k) was correct.
- No graphene Hamiltonian term was changed; only the lattice metadata was made numerically consistent with the Hamiltonian parameters.

Error calculation:

- Compare sorted bands point by point at 128 deterministic random Cartesian k-points after an independent Cartesian-to-reduced conversion for PythTB.
- Evaluate both analytic Dirac points and compare their energies with zero; estimate the radial cone slope in 12 directions around each valley and compare with $v_F=3|t|a_0/2$.
- Translate generic k-points by both primitive reciprocal vectors and compare spectra, which is gauge invariant.
- Build QXTI's shifted guarded grid and require a finite nonzero sampled gap and finite off-diagonal Berry connections at every grid point.

Acceptance criteria:

- Maximum PythTB band error <= 1e-12 Ha.
- Maximum Dirac-node energy <= 1e-12 Ha.
- Maximum cone-velocity relative error <= 1e-4.
- Maximum reciprocal-periodicity spectral error <= 1e-12 Ha.
- The shifted grid avoids exact degeneracies and every sampled connection is finite.

Conclusion: QXTI reproduces the gapless graphene spectrum and samples the Dirac model without injecting singular grid observables.

Limitations:

- This verifies the static gapless model and grid guard, not graphene optical conductivity or HHG.
- Berry connection exactly at a Dirac point is undefined and is deliberately not evaluated.

Raw artifact: `validation/results/graphene_gapless_2d.json`

## Haldane bands and bulk DOS against PythTB

- Status: **PASS**
- Date: `2026-08-21`
- Scope: Pointwise bands and same-grid bulk DOS for topological and trivial Haldane inputs.
- Reference: PythTB 2.0.2
- Evidence type: external cross-code reference plus state-count invariant
- QXTI commit: `df53f4508eb96e0c87d28d44a212456a9fbc06b8`
- Working tree dirty: `True`

| case | band max error (Ha) | DOS relative L2 | DOS integral | PythTB Chern |
| --- | --- | --- | --- | --- |
| topological | 8.3266727e-17 | 6.1758694e-15 | 1.9989831 | -1 |
| trivial | 1.110223e-16 | 7.2135924e-15 | 1.99877 | 0 |

Implementation:

- Load the topological and trivial QXTI configurations, instantiate PythTB with the same Haldane hoppings and onsite mass, and convert QXTI Cartesian momenta independently to PythTB reduced coordinates.
- Diagonalize both models point by point; build the reference DOS from PythTB eigenvalues using the same Gaussian width and energy axis used by QXTI.

Reference provenance:

- The physical model is the Haldane honeycomb Hamiltonian; the executable comparison is against the independently distributed PythTB implementation.

Production-code changes motivated by this benchmark:

- None. This benchmark passed without modifying the Haldane production Hamiltonian or DOS engine.

Error calculation:

- Bands are compared point by point at 128 deterministic k-points and for each sorted band: $\Delta E_{kn}=E^{QXTI}_{kn}-E^{PythTB}_{kn}$. The report gives $\max_{kn}|\Delta E_{kn}|$; the raw artifact also stores $\sqrt{\mathrm{mean}_{kn}|\Delta E_{kn}|^2}$.
- DOS arrays are compared point by point on the same 501-energy axis. The reported relative error is $\|g_{QXTI}-g_{PythTB}\|_2/\|g_{PythTB}\|_2$, where the norm sums over every energy node. This is a whole-curve metric, not the error at only one selected energy.
- The raw artifact also stores the largest pointwise DOS difference $\max_E|g_{QXTI}(E)-g_{PythTB}(E)|$.
- The DOS sum rule is evaluated by composite trapezoidal integration over the energy grid and compared with the expected value 2.

Acceptance criteria:

- Maximum pointwise band error <= 1e-12 Ha.
- Same-grid DOS relative L2 error <= 1e-11.
- PythTB Chern magnitude matches the expected input classification.
- QXTI DOS integral differs from two bands by <= 0.02.

Conclusion: QXTI and PythTB implement the same Haldane band spectrum; QXTI's same-grid bulk DOS agrees to floating-point precision.

Limitations:

- The same-grid DOS comparison does not independently validate QXTI's reciprocal integration domain; that is covered by the separate convergence benchmark.
- The Chern number is calculated only by PythTB and validates the input classification, not a QXTI Chern implementation.

Raw artifact: `validation/results/haldane_pythtb_pointwise.json`

## Haldane derivatives and optical operators

- Status: **PASS**
- Date: `2026-08-21`
- Scope: Pointwise first/second Hamiltonian derivatives, band velocities, and off-diagonal dipole/Berry magnitudes for topological and trivial Haldane inputs.
- Reference: PythTB 2.0.2 analytic velocity and closed-form Haldane derivatives
- Evidence type: external cross-code plus separately coded closed form
- QXTI commit: `df53f4508eb96e0c87d28d44a212456a9fbc06b8`
- Working tree dirty: `True`

| case | dH error | d2H error | band velocity error | interband v magnitude error | dipole A magnitude error |
| --- | --- | --- | --- | --- | --- |
| topological | 2.4827668e-10 | 1.1544558e-06 | 1.8396906e-10 | 2.7547042e-10 | 1.771614e-09 |
| trivial | 2.466039e-10 | 1.1544558e-06 | 4.0061732e-10 | 2.1536939e-10 | 1.1382477e-09 |

Implementation:

- Code the first and second Cartesian derivatives of the Haldane Hamiltonian separately from QXTI and evaluate them at deterministic generic momenta.
- Use PythTB's analytic velocity operator as the external band-basis reference; compare gauge-invariant interband magnitudes because eigenvector phases are arbitrary.

Reference provenance:

- Closed derivatives follow directly from the trigonometric Haldane Hamiltonian, while band velocities are supplied by the independently distributed PythTB implementation.

Production-code changes motivated by this benchmark:

- None. QXTI's existing finite-difference derivatives and optical-operator construction met the declared tolerances.

Error calculation:

- All quantities are compared at 128 deterministic random Cartesian k-points and in both x and y directions.
- For first and second Hamiltonian derivatives, the error is the maximum absolute difference over every k-point and every matrix element relative to separately coded closed-form Haldane derivatives.
- For band velocities, each code diagonalizes its own Hamiltonian. Diagonal elements are compared directly; interband elements are compared as $||v^{QXTI}_{01}|-|v^{PythTB}_{01}||$ to remove arbitrary eigenvector phases.
- The PythTB dipole/Berry reference is $A_{nm}=iv_{nm}/(E_m-E_n)$. QXTI and PythTB are compared through the gauge-invariant magnitude $|A_{01}|$ point by point.

Acceptance criteria:

- Maximum first-derivative matrix error <= 1e-9 Ha*Bohr.
- Maximum second-derivative matrix error <= 1e-5 Ha*Bohr^2.
- Maximum PythTB band-velocity and interband-magnitude error <= 1e-9 Ha*Bohr.
- Maximum off-diagonal dipole/Berry magnitude error <= 1e-8 Bohr.

Conclusion: QXTI's finite-difference Hamiltonian derivatives and the gauge-invariant optical matrix elements derived from them agree with independent closed-form and PythTB references.

Limitations:

- Raw complex interband elements are not compared because independently diagonalized eigenvectors carry arbitrary phases.
- The diagonal Berry connection is gauge dependent and QXTI fixes it to zero; this benchmark validates only the physical off-diagonal connection/dipole magnitude.
- This does not yet validate integrated optical conductivity, nonlinear response, LDOS, or HHG.

Raw artifact: `validation/results/haldane_operators.json`

## Haldane DOS convergence with independent reciprocal meshes

- Status: **PASS**
- Date: `2026-08-21`
- Scope: Bulk DOS integration domain and k-grid convergence for topological and trivial Haldane inputs.
- Reference: PythTB 2.0.2
- Evidence type: external cross-code integration reference
- QXTI commit: `df53f4508eb96e0c87d28d44a212456a9fbc06b8`
- Working tree dirty: `True`

| case | grid | QXTI vs PythTB | QXTI vs reference | PythTB vs reference | DOS integral |
| --- | --- | --- | --- | --- | --- |
| topological | 11x11 | 0.728086 | 0.58941287 | 0.93965091 | 2 |
| topological | 21x21 | 0.27504425 | 0.23716594 | 0.41008017 | 2 |
| topological | 41x41 | 0.075390113 | 0.053159806 | 0.10304745 | 2 |
| topological | 81x81 | 0.0031148725 | 0.0020367126 | 0.0030990446 | 2 |
| trivial | 11x11 | 0.66586612 | 0.50469848 | 0.80955501 | 2 |
| trivial | 21x21 | 0.23902723 | 0.22070521 | 0.35699717 | 2 |
| trivial | 41x41 | 0.075530057 | 0.056078717 | 0.11157427 | 2 |
| trivial | 81x81 | 0.0080973894 | 0.0060361319 | 0.011338105 | 2 |

Implementation:

- Let QXTI generate its Cartesian reciprocal meshes while PythTB independently generates reduced primitive-cell meshes; do not reuse QXTI k-points in the reference path.
- Repeat the complete DOS calculation at four refinements and compare both sequences with a native high-resolution PythTB mesh.

Reference provenance:

- PythTB supplies the external Hamiltonian and reduced-coordinate sampling; the 161x161 convergence mesh and acceptance thresholds are choices of this verification study.

Production-code changes motivated by this benchmark:

- None. The existing QXTI reciprocal grid and DOS integration paths passed the independent-mesh convergence test.

Error calculation:

- For every mesh size $N$, QXTI and PythTB independently generate their own $N\times N$ k-mesh. Their DOS arrays are then compared at the same 501 energy nodes.
- A PythTB native $161\times161$ calculation defines $g_{ref}$. The common normalization is $\|g_{ref}\|_2$ so every row is comparable across mesh sizes.
- The cross-code error is $\epsilon_{QP}(N)=\|g_{QXTI,N}-g_{PythTB,N}\|_2/\|g_{ref}\|_2$.
- Individual convergence errors are $\epsilon_Q(N)=\|g_{QXTI,N}-g_{ref}\|_2/\|g_{ref}\|_2$ and $\epsilon_P(N)=\|g_{PythTB,N}-g_{ref}\|_2/\|g_{ref}\|_2$.
- These are whole-curve L2 errors over the energy axis, not pointwise errors at a single energy. The DOS integral uses the composite trapezoidal rule.

Acceptance criteria:

- Cross-code relative L2 error decreases at every grid refinement.
- Final QXTI-vs-PythTB relative L2 error <= 1%.
- Final QXTI-vs-reference error <= 1% and PythTB-vs-reference error <= 2%.
- Final QXTI DOS integral differs from two bands by <= 0.02.

Conclusion: The Cartesian QXTI integration cell and the independently generated PythTB reduced cell converge to the same bulk DOS.

Limitations:

- This validates bulk DOS integration, not QXTI Berry curvature, optical response, surface LDOS, or HHG.
- The high-resolution reference is numerical PythTB data, not a closed-form analytic DOS.

Raw artifact: `validation/results/haldane_dos_grid_convergence.json`

## Eight-band TaAs hopping reconstruction and symmetries

- Status: **PASS**
- Date: `2026-08-21`
- Scope: Eight-band matrix assembly, unit conversion, velocities, reciprocal periodicity, time reversal, space-group spectra and Kramers degeneracy.
- Reference: Independent real-space hopping-sum evaluator and exact symmetry identities
- Evidence type: in-repository reconstruction and invariants; not external software
- QXTI commit: `df53f4508eb96e0c87d28d44a212456a9fbc06b8`
- Working tree dirty: `True`

| matrix error (Ha) | band error (Ha) | velocity error (Ha Bohr) | periodicity error (Ha) | TRS error (Ha) | space-group error (Ha) | Gamma Kramers splitting (Ha) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 6.7867975e-11 | 5.5511151e-17 | 4.8572257e-17 | 7.3439899e-17 | 2.0816682e-17 |

Implementation:

- Read the finalized real-space hopping terms and rebuild each 8x8 Bloch matrix in a separate loop implementing the Fourier sum, rather than calling the QXTI wrapper's H(k).
- Differentiate the phase factors analytically for the velocity reference and evaluate reciprocal, time-reversal, space-group and Kramers spectral identities independently of QXTI's finite-difference operator.

Reference provenance:

- The lattice constants and I4_1md structure describe TaAs, but the hopping amplitudes in `models/taas_tb.py` are explicitly illustrative project parameters rather than a traced published Wannier parametrization.
- Consequently the reference is the exact Fourier reconstruction of the shipped hopping model; it validates implementation and units, not agreement with real TaAs bands.

Production-code changes motivated by this benchmark:

- None. The TaAs wrapper, unit conversions and declared spectral symmetries passed without a production-code modification.
- Only the validation benchmark and automated regression test were added.

Error calculation:

- Reassemble every 8x8 matrix directly from the finalized real-space hopping list, including the independent 1/Bohr-to-1/Angstrom and eV-to-Hartree factors, then compare all matrix elements and sorted bands point by point.
- Differentiate the real-space phase factors analytically and compare the full x/y/z velocity matrices with QXTI's finite differences.
- Compare spectra at k and k+G, k and -k, and all space-group-rotated momenta; compare Kramers pairs at Gamma.

Acceptance criteria:

- Matrix and band errors <= 1e-12 Ha.
- Velocity matrix error <= 1e-8 Ha Bohr.
- Reciprocal-periodicity and time-reversal spectral errors <= 1e-12 Ha.
- Space-group spectral residual <= 1e-10 Ha and Gamma Kramers splitting <= 1e-12 Ha.

Conclusion: The QXTI wrapper preserves the eight-band real-space hopping Hamiltonian, physical units, analytic velocities and declared TaAs spectral symmetries.

Limitations:

- This is exact code verification against an independently assembled hopping sum, not validation against ab-initio or experimental TaAs bands.
- The shipped conventional tetragonal sampling box is explicitly approximate for the body-centered tetragonal primitive BZ, so absolute integrated response is not validated here.

Raw artifact: `validation/results/taas_multiband_8.json`

## TBLG 76-band implementation and basis truncation

- Status: **PASS**
- Date: `2026-08-21`
- Scope: Bistritzer--MacDonald matrix assembly and flat-band convergence of the fixed N=2 (76-band) reciprocal cutoff.
- Reference: Separately coded BM plane-wave evaluator at N=1, N=2 and N=3
- Evidence type: in-repository independent implementation and convergence reference; not external software
- QXTI commit: `df53f4508eb96e0c87d28d44a212456a9fbc06b8`
- Working tree dirty: `True`

| matrix error (Ha) | band error (Ha) | N2-N3 normalized error | N1/N2 improvement | bandwidth relative error |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0.00014613574 | 74.265934 | 0.00092288046 |

Implementation:

- Implement a second BM constructor that independently generates the hexagonal reciprocal cutoff, rotated layer Dirac blocks and T1/T2/T3 interlayer couplings, without calling private helpers from `models/tblg_bmd.py`.
- First compare the complete N=2 matrix and all 76 eigenvalues at random momenta; then compute N=1, N=2 and N=3 central bands at 49 points including Gamma, K and M.

Reference provenance:

- The continuum construction follows Bistritzer and MacDonald, PNAS 108, 12233 (2011), with corrugation-inspired unequal interlayer couplings as in later continuum work such as Koshino et al., Phys. Rev. X 8, 031087 (2018).
- The current hbar*vF is consistent with Koshino, but the exact pair wAA=81.7 meV and wAB=110 meV is a project-specific hybrid and is not the 79.7/97.5 meV pair reported by Koshino.
- The N=1/2/3 cutoffs, use of N=3 as numerical reference and acceptance thresholds are choices of this benchmark, not external published band data.

Production-code changes motivated by this benchmark:

- None. The shipped 76-band TBLG Hamiltonian matched the separate implementation without changing production code.
- The benchmark path construction was corrected during validation to include the Gamma, K and M vertices exactly; that change affects validation code only.

Error calculation:

- Rebuild the BM Hamiltonian independently at deterministic random momenta and compare every matrix element and every sorted eigenvalue point by point with QXTI's 76-band model.
- Along Gamma-K-M-Gamma, compare the central valence/conduction energies from N=1 and N=2 cutoffs against N=3; normalize the maximum error by the natural continuum scale hbar*vF*k_theta.
- Compute the combined two-flat-band energy range for N=2 and N=3 and report their relative difference; require the N=2 error to improve materially over N=1.

Acceptance criteria:

- QXTI has 76 bands and matrix/band implementation errors are <= 1e-12 Ha.
- N=2 maximum central-band error relative to N=3 is <= 5% of hbar*vF*k_theta.
- N=2 improves the maximum error by at least a factor of two over N=1.
- N=2 versus N=3 flat-bandwidth relative error is <= 10%.

Conclusion: The shipped 76-band BM Hamiltonian matches an independent assembly and its central bands are converged against the 148-band cutoff within the stated thresholds.

Limitations:

- N=3 is a higher-cutoff numerical reference, not an exact infinite-basis solution or external ab-initio validation.
- Only the central bands on Gamma-K-M-Gamma at theta=1.2 degrees are covered; nonlinear response and the approximate boundary gauge are not validated here.

Raw artifact: `validation/results/tblg_truncation.json`

## Three-dimensional WSM grid and DOS convergence

- Status: **PASS**
- Date: `2026-08-21`
- Scope: Three-dimensional reciprocal bounds, independent mesh construction, bulk DOS normalization and convergence for the two-band WSM.
- Reference: Separately coded closed-form WSM eigenvalues on an independent reduced 3-D mesh
- Evidence type: in-repository closed-form reference; not external software
- QXTI commit: `df53f4508eb96e0c87d28d44a212456a9fbc06b8`
- Working tree dirty: `True`

| grid | same-grid relative L2 | QXTI vs reference | independent vs reference | DOS integral |
| --- | --- | --- | --- | --- |
| 7^3 | 1.1074354e-15 | 0.085134118 | 0.085134118 | 2 |
| 11^3 | 2.1982895e-15 | 0.025906945 | 0.025906945 | 2 |
| 15^3 | 2.4005608e-15 | 0.010415515 | 0.010415515 | 2 |
| 21^3 | 3.2853145e-15 | 0.0025568653 | 0.0025568653 | 2 |

Implementation:

- Resolve the WSM parameters through QXTI, but construct the reference grid independently in reduced coordinates and evaluate the two eigenvalues from $B_0$ and the norm of $(B_1,B_2,B_3)$ without calling the model module.
- Construct the Gaussian reference DOS directly from those eigenvalues, compare same-size grids to isolate implementation error, and use a separate 41^3 calculation to measure discretization convergence.

Reference provenance:

- The Hamiltonian form is an anisotropic adaptation of Eq. 11 of McCormick, Kimchi and Trivedi, Phys. Rev. B 95, 075133 (2017).
- The current M0=0.014 and a2=12 Bohr are project-specific adjusted parameters, not a parameter table copied unchanged from that paper.
- The exact 2x2 eigenvalue formula, reciprocal bounds, two-state DOS sum rule, 41^3 reference grid and numerical thresholds define this verification benchmark; they are not published comparison data.

Production-code changes motivated by this benchmark:

- Add a parameter-aware `default_lattice(params)` to `models/wsm_two_weyl.py`, so a0, a1 and a2 determine both the Hamiltonian and reciprocal cell.
- Change `CustomHamiltonian.default_lattice()` to prefer a callable lattice provider over static `DEFAULT_LATTICE`. Previously the static metadata silently ignored an input-dependent a2 and could integrate the correct Hamiltonian over the wrong 3-D Brillouin zone.
- Add a regression test with a0=6, a1=7 and a2=12 that checks real-space lengths and bounds $[-pi/a_i,pi/a_i]$.

Error calculation:

- QXTI independently builds a shifted Cartesian $N^3$ grid while the reference builds a reduced $[-1/2,1/2)^3$ grid and converts each axis using its own lattice constant.
- At each N, compare the complete DOS arrays with an L2 norm normalized by the independent $41^3$ reference DOS.
- Verify every reciprocal bound against plus/minus pi/a_i and integrate the DOS by the composite trapezoidal rule.
- Fit log(g(E)) versus log(E) over 0.004--0.012 Ha; an isolated linear Weyl cone predicts an exponent near two.

Acceptance criteria:

- QXTI-to-reference errors decrease at every refinement.
- Final same-grid relative L2 error <= 1e-11.
- Final QXTI-to-41^3-reference relative L2 error <= 1%.
- Final DOS integral differs from two states by <= 0.02.
- Fitted low-energy DOS exponent is between 1.7 and 2.3.

Conclusion: QXTI's anisotropic 3-D reciprocal cell, weights and DOS converge to an independently generated WSM reference.

Limitations:

- The reference is an independent closed-form evaluator, not a second external software package.
- This benchmark does not verify Weyl-node chirality, surface Fermi arcs or optical response.

Raw artifact: `validation/results/wsm_grid_3d.json`
