from __future__ import annotations

import numpy as np

from qxti.response.susceptibility_tensor_calculator import SusceptibilityTensorCalculator


class FakeXTP:
    def __init__(self, omega_axis, chi_column=None, field_vector=None, p2_vector=None):
        self.omega_axis = omega_axis
        self.chi_column = chi_column
        self.field_vector = field_vector
        self.p2_vector = p2_vector

    def linear_susceptibility(self, *, input_direction: str, eps: float = 1.0e-14):
        return self.omega_axis, self.chi_column

    def polarization_frequency_domain(self, order: int):
        polarization = np.zeros((len(self.omega_axis), 3), dtype=np.complex128)
        polarization[1, :] = self.p2_vector
        return self.omega_axis, polarization

    def electric_field_frequency_domain(self):
        field = np.zeros((len(self.omega_axis), 3), dtype=np.complex128)
        field[0, :] = self.field_vector
        return self.omega_axis, field


def test_chi1_reconstructs_tensor_columns():
    omega_axis = np.array([1.0, 2.0, 3.0], dtype=float)

    xtp_by_label = {
        "x": FakeXTP(
            omega_axis,
            chi_column=np.array(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                ],
                dtype=np.complex128,
            ),
        ),
        "y": FakeXTP(
            omega_axis,
            chi_column=np.array(
                [
                    [10.0, 20.0, 30.0],
                    [40.0, 50.0, 60.0],
                    [70.0, 80.0, 90.0],
                ],
                dtype=np.complex128,
            ),
        ),
        "z": FakeXTP(
            omega_axis,
            chi_column=np.array(
                [
                    [100.0, 200.0, 300.0],
                    [400.0, 500.0, 600.0],
                    [700.0, 800.0, 900.0],
                ],
                dtype=np.complex128,
            ),
        ),
    }

    calculator = SusceptibilityTensorCalculator(xtp_by_label)
    omega, chi1 = calculator.chi1()

    assert np.allclose(omega, omega_axis)
    assert chi1.shape == (3, 3, 3)

    assert np.allclose(chi1[:, :, 0], xtp_by_label["x"].chi_column)
    assert np.allclose(chi1[:, :, 1], xtp_by_label["y"].chi_column)
    assert np.allclose(chi1[:, :, 2], xtp_by_label["z"].chi_column)


def test_chi2_reconstructs_known_tensor():
    omega_axis = np.array([1.0, 2.0], dtype=float)

    true_chi2 = np.arange(27, dtype=float).reshape(3, 3, 3).astype(np.complex128)

    field_vectors = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
        "xy": np.array([1.0, 1.0, 0.0]),
        "xz": np.array([1.0, 0.0, 1.0]),
        "yz": np.array([0.0, 1.0, 1.0]),
        "xyz": np.array([1.0, 1.0, 1.0]),
        "x2y": np.array([2.0, 1.0, 0.0]),
        "xy2": np.array([1.0, 2.0, 0.0]),
    }

    xtp_by_label = {}

    for label, field in field_vectors.items():
        p2 = np.einsum("ijk,j,k->i", true_chi2, field, field)

        xtp_by_label[label] = FakeXTP(
            omega_axis,
            field_vector=field.astype(np.complex128),
            p2_vector=p2,
        )

    calculator = SusceptibilityTensorCalculator(xtp_by_label)
    chi2 = calculator.chi2_at_frequency(input_omega=1.0, output_omega=2.0)

    assert chi2.shape == (3, 3, 3)
    assert np.allclose(chi2, true_chi2)