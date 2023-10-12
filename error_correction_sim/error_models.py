"""Contains different error models.

Includes qubit errors (Pauli errors including depolarizing, bit-flip and phase-flip) and
measurement errors.


This module provides classes that model different types of quantum errors,
such as Pauli errors (X, Y, Z), and measurement errors. Each error model class
allows users to specify probabilities for each type of error. The models can then
simulate the occurrence of these errors on quantum states.

Classes:
    PauliError: Models the X, Y, and Z Pauli errors with specified probabilities.
    DepolarizingError: Equal probability for each Pauli error.
    BitFlipError: Models only the X Pauli error.
    PhaseFlipError: Models only the Z Pauli error.
    MeasurementError: Models measurement outcome flips with a given probability.
"""

from __future__ import annotations

import random as rng
from typing import List, Optional

import numpy as np

from . import tableau


class PauliError:
    """Pauli error model where X, Y and Z errors occur with different probabilities."""

    def __init__(self, p_x: float, p_y: float, p_z: float):
        """
        Args:
            p_x: probability of X error
            p_y: probability of Y error
            p_z: probability of Z error
        """
        #: Probability of X error
        self.p_x: float = p_x
        #: Probability of Y error
        self.p_y: float = p_y
        #: Probability of Z error
        self.p_z: float = p_z
        #: Probability of no error
        self.p_i: float = 1 - (p_x + p_y + p_z)

    def error_str(self, no_qubits: int, reps: int = 1) -> List[str]:
        """Generates random error strings

        Args:
            no_qubits: Number of qubits over which the error acts
            reps: Number of measurement repitions

        Returns:
            List of error strings
        """

        def _single_error(self, qubits: int) -> str:
            """
            Args:
                qubits: Number of qubits

            Returns:
                Single error string
            """
            arr = rng.choices(
                "IXYZ", k=qubits, weights=[self.p_i, self.p_x, self.p_y, self.p_z]
            )
            return "".join(arr)

        pauli_errors = [_single_error(self, no_qubits) for i in range(reps)]
        return pauli_errors

    def targets(self, err_prob: float, qubits: List[int]) -> np.ndarray:
        """Returns list of target qubits upon which the error gate should act.

        Args:
            err_prob: Probability of each qubit getting picked
            qubits: List of all qubits that are subject to a given error

        Returns:
            All the target qubits chosen with probability err_prob
        """
        choices = np.random.choice([0, 1], len(qubits), p=[1 - err_prob, err_prob])
        locs = np.nonzero(choices)[0]
        target_qubits = np.take(qubits, locs)
        return target_qubits

    def act_errors(
        self, state: tableau.QuantumState, qubits: List[int]
    ) -> List[np.ndarray]:
        """Acts error gates on a state

        Args:
            state: state (in tableau representation) on which errors act
            qubits: Qubits that are subject to the error in principle

        Returns:
            list of indices where X, Y and Z errors act.
            Ideally the user should not know it, but useful while debugging.
        """

        x_targets = self.targets(self.p_x, qubits)
        y_targets = self.targets(self.p_y, qubits)
        z_targets = self.targets(self.p_z, qubits)

        state.X(x_targets)
        state.Z(z_targets)
        state.X(y_targets)
        state.Z(y_targets)

        return [x_targets, y_targets, z_targets]


class DepolarizingError(PauliError):
    """Equal probability of X, Y and Z Pauli errors."""

    # TODO see https://github.com/sphinx-doc/sphinx/issues/9884
    p_i: float
    p_x: float
    p_y: float
    p_z: float

    def __init__(self, prob: float):
        """
        Args:
            prob: Total probability of error. Each Pauli error with probability prob/3.
        """
        p_x = p_y = p_z = prob / 3
        super().__init__(p_x, p_y, p_z)


class BitFlipError(PauliError):
    """Only X Pauli error."""

    # TODO see https://github.com/sphinx-doc/sphinx/issues/9884
    p_i: float
    p_x: float
    p_y: float
    p_z: float

    def __init__(self, prob: float):
        """
        Args:
            prob: Probability of X error
        """
        p_x = prob
        p_y = p_z = 0
        super().__init__(p_x, p_y, p_z)


class PhaseFlipError(PauliError):
    """Only Z Pauli error."""

    # TODO see https://github.com/sphinx-doc/sphinx/issues/9884
    p_i: float
    p_x: float
    p_y: float
    p_z: float

    def __init__(self, prob: float):
        """
        Args:
            prob: Probability of Z error
        """
        p_z = prob
        p_x = p_y = 0
        super().__init__(p_x, p_y, p_z)


class MeasurementError:
    """Measurement outcome flips with a given probability."""

    def __init__(self, prob: float):
        """
        Args:
            prob: Probability of measurement error
        """
        #: Probability of measurement error
        self.p_m: float = prob

    def m_errors(self, reps: int, no_stabilizers: int) -> np.ndarray:
        """Generates an array of measurement errors.

        Args:
            reps: Number of measurement repetitions
            no_stabilizers: Number of stabilizers measured

        Returns:
            Binary array with the 1's indicating the faulty measurement outcomes
        """
        measurement_error = np.random.choice(
            [0, 1], size=(reps, no_stabilizers), p=[1 - self.p_m, self.p_m]
        )
        measurement_error[-1, :] = 0
        # Because pymatching requires this. No error in last measurement.
        return measurement_error
