"""Helper functions.

This module provides utility functions for tableau operations involving Pauli matrices
and their binary representations.
"""
from __future__ import annotations

import numpy as np

### Pauli functions


def binary_to_pauli(bin_op: np.ndarray, show_identities: bool = False) -> str:
    """Converts a single binary form to a Pauli string.

    Args:
        bin_arr: One-dimensional binary array

    Returns:
        Pauli form of the binary array while not mentioning the Is
    """
    sign = "-" if bin_op[-1] else "+"

    xs, zs = np.hsplit(bin_op[:-1], 2)
    arr = xs + 2 * zs

    trans = str.maketrans("0123", "IXZY", "[ ,\n]")
    strarr = str(arr.tolist())
    pauli_op_main = strarr.translate(trans)

    if show_identities:
        pauli_op = sign + pauli_op_main
    else:
        alpha_str = ""
        for location, bin_op_str in enumerate(pauli_op_main):
            if bin_op_str != "I":
                alpha_str = alpha_str + " " + bin_op_str + str(location)
        pauli_op = sign + alpha_str

    return pauli_op


def commutator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculates the symplectic product of two operators in their binary form.
    If the operators commute, returns 0 else 1

    Args:
        a: Binary array
        b: Binary array

    Returns:
        Symplectic product of the two binary arrays
    """
    op1 = a[:-1] if a.ndim == 1 else a[:, :-1]
    op2 = b[:-1] if b.ndim == 1 else b[:, :-1]

    xs, zs = np.hsplit(op2, 2)
    op2_reversed = np.hstack([zs, xs])
    return op1.dot(op2_reversed.T) % 2


def single_qubit_op(pauli: str, qubit: int, no_qubits: int) -> np.ndarray:
    """Constructs a single qubit operator in binary representation given the pauli operator, its
    position and the total number of qubits.

    Args:
        pauli: Single qubit Pauli operator. Eg. "Z" or "-X" or "+Y"
        qubit: Position of the operator
        no_qubits: Total number of qubits

    Returns:
        Operator in binary representation
    """
    # acceptable formats of pauli: Z, -Z, +Z and corresponding X and Y
    meas_op = np.zeros(2 * no_qubits + 1, dtype=int)
    meas_op[-1] = 1 if pauli[0] == "-" else 0

    shift: int | np.ndarray
    if pauli[-1] == "X":
        shift = 0
    elif pauli[-1] == "Z":
        shift = no_qubits
    elif pauli[-1] == "Y":
        shift = np.array([0, no_qubits])

    meas_op[shift + qubit] = 1
    return meas_op
