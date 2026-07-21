---
tags: [package, physics]
updated: 2026-07-21
---

# 📦 qxti.physics — Hamiltoniano, operadores, láser

> La capa de física "estática": qué es el material y qué es el pulso. **No se mezclan**
> hasta [[qxti.response|CMD]]. [[Home]] · [[Architecture Map]]

**Depende de:** [[qxti.grids]] (solo `band_gauge`). El resto son hojas.
**Usado por:** [[qxti.response]], [[qxti.analytics]], [[qxti.data]], [[qxti.core]]

## Archivos

| Archivo | Rol | Símbolos |
| --- | --- | --- |
| [hamiltonian.py](../../qxti/physics/hamiltonian.py) | ABC del H(k); derivadas FD, diagonalización | `Hamiltonian` |
| [custom_hamiltonian.py](../../qxti/physics/custom_hamiltonian.py) | Carga `models/*.py` (H(kx,ky,kz,params)) | `CustomHamiltonian` |
| [operators.py](../../qxti/physics/operators.py) | Fábrica de velocidad/corriente/dipolo/masa/Berry | `OperatorFactory` |
| [band_gauge.py](../../qxti/physics/band_gauge.py) | Autovectores con fase suave en la malla k | `BandGaugeFrame` |
| [laser.py](../../qxti/physics/laser.py) | Un pulso: E(t), A(t), polarización | `Laser` |
| [laser_system.py](../../qxti/physics/laser_system.py) | Suma de pulsos | `LaserSystem` |
| [observables.py](../../qxti/physics/observables.py) | ⚠️ **stub vacío** | — |

Guías detalladas: [[HAMILTONIAN]] y [[LASER]] (en `docs/`).

## Hamiltonian (ABC)

Interfaz: `H(kx,ky,kz)`, `dH_dk` (FD centrado, paso `dk_derivative`), `d2H_dk2`,
`diagonalize`, `transform_to_band_basis`, `reciprocal_box_bounds`.
`CustomHamiltonian` lo implementa cargando dinámicamente un módulo de `models/` que declara
`H`, y opcionalmente `MODEL_NAME`, `BASIS_SIZE`, `DIMENSION`, `H_batch`/`dH_dk` analíticos.

⛔ **Invariantes**
- **Unidades atómicas** ([[Concept - Atomic Units]]). El wrapper del modelo convierte de eV/Å a a.u.
- `H(k)` **hermítico** antes de diagonalizar (`validate_hermiticity`).
- **Signo:** velocidad `v = ∂H/∂k` (sin menos); la **corriente** lleva el menos en `operators.current` (`j = −v`).
- Un `H_batch(kpts,params)` vectorizado debe ser **bit-exacto** vs el `H` escalar (los motores mesh lo asumen).

## OperatorFactory

`velocity`, `current` (`=−v`), `dipole`/`position` (conexión de Berry), `inverse_mass`,
`berry_connection` (`A_nm = i·v_nm/(E_m−E_n)`, diagonal=0, pares casi degenerados con
`|ΔE|<dipole_regularization` se saltan). Ver [[Concept - Occupation Gauge Basis]].

## BandGaugeFrame

Alinea la fase de los autovectores a lo largo de las líneas de la malla (continuidad de fase) y
precalcula `energies`, `connection(dir)`, `velocity(dir)`, `current(dir)` en base de banda.
Lo consume [[qxti.response|XTP]] (operadores cacheados) y la ruta *streaming* de `CMD`.
⚠️ La alineación es **secuencial por eje**, no globalmente suave en 2D/3D (gauge no-Abeliano
en subespacios degenerados no está resuelto — ver [[Playbook - Invariants Not to Break]]).

## Laser / LaserSystem

Un `Laser` = envolvente × portadora con polarización (mayor `x̂` amplitud `E0`, menor `ŷ`
amplitud `E0·ε`), en a.u. `LaserSystem` suma pulsos y resta el valor inicial para forzar
E(atmin)=0. **Multi-pulso** ⇒ los motores usan la rama tiempo (`time_domain_currents`).

⛔ El láser **no** importa nada del Hamiltoniano (regla de oro 2, ver [[Architecture Map]]).

➕ **Extender:**
- Nuevo modelo → [[Playbook - Add a Model]] (editas `models/`, no esta capa).
- Nueva envolvente → método `_xxx_envelope` + registro en `_ENVELOPE_ALIASES` (`laser.py`).
- Nuevo operador → método en `OperatorFactory`.

---

Relacionado: [[Concept - Perturbative Recursion]] · [[Models and Inputs]] · [[qxti.response]]
