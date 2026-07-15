"""NatPhys_TaAs 4-band Weyl model (the antelope material) — QXTI port.

Exact transcription of antelope's NatPhys_TaAs.h GenHamiltonian:
    Bx  = t (cos kx + my (1-cos ky) + mz (1-cos kz))
    By1 = t sin(ky)
    By2 = Delta t cos(ky)
    Bz  = t sin(kz)
    H =  [[0,        Bx-iBy1,  Bz,       -iBy2   ],
          [Bx+iBy1,  0,        iBy2,     -Bz     ],
          [Bz,      -iBy2,     0,        Bx-iBy1 ],
          [iBy2,     Bz,       Bx+iBy1,  0       ]]
Everything in atomic units (a0 in Bohr, energies in Hartree), same as the RCP run.
Used to check the C4v-type selection rule chi^(2)_xxx = 0 (M_x mirror: H is even
in kx, so chi_xxx must vanish).
"""
from __future__ import annotations

import numpy as np

MODEL_NAME = "natphys-taas-4band"
BASIS_SIZE = 4
DIMENSION = 3
IS_PERIODIC = True

_AU_ANGSTROM = 0.529177210903
DEFAULT_PARAMS = {"a": 3.4, "t": 0.03, "delta": 0.5, "my": 1.0, "mz": 5.0}


def _a0(p):
    return float(p["a"]) / _AU_ANGSTROM


def default_params():
    return dict(DEFAULT_PARAMS)


def _bz():
    a0 = _a0(DEFAULT_PARAMS)
    km = np.pi / a0
    return [[2 * km, 0, 0], [0, 2 * km, 0], [0, 0, 2 * km]]


DEFAULT_LATTICE = {
    "lattice_type": "BCC-like Weyl (antelope NatPhys_TaAs 4-band)",
    "BZorigin": [0.0, 0.0, 0.0],
    "BZaxis": _bz(),
    "notes": "H even in kx -> M_x mirror -> chi_xxx must be 0.",
}


def H(kx, ky, kz, params=None):
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update({k: v for k, v in params.items() if k in DEFAULT_PARAMS})
    t, my, mz, delta = float(p["t"]), float(p["my"]), float(p["mz"]), float(p["delta"])
    a0 = _a0(p)
    Bx = t * (np.cos(kx * a0) + my * (1 - np.cos(ky * a0)) + mz * (1 - np.cos(kz * a0)))
    By1 = t * np.sin(ky * a0)
    By2 = delta * t * np.cos(ky * a0)
    Bz = t * np.sin(kz * a0)
    return np.array(
        [
            [0.0,          Bx - 1j * By1, Bz,           -1j * By2],
            [Bx + 1j * By1, 0.0,          1j * By2,     -Bz],
            [Bz,          -1j * By2,      0.0,          Bx - 1j * By1],
            [1j * By2,     Bz,            Bx + 1j * By1, 0.0],
        ],
        dtype=np.complex128,
    )
