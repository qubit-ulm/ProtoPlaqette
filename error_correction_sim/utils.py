"""Helper functions.

This module provides functions to perform quantum error correction operations.

Functions:
1. pauli_to_binary - Converts a list of Pauli operators to binary representation.
2. binary_to_pauli - Converts a binary form to its corresponding Pauli string.
3. symplectic_product - Computes the symplectic product of two operators in binary form.
4. single_qubit_op - Constructs a single qubit operator in binary representation.
5. errors2syndrome - Extracts the error syndrome and final state from the given quantum
 code and errors.
6. corrected_binary - Returns the state after a recovery operator is applied to an error.
7. get_success - Determines if a recovery operator successfully corrected an error using
 a quantum code.

"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import codes

### Pauli functions


def pauli_to_binary(pauli_list: list[str]) -> np.ndarray:
    """Converts a list of Pauli strings to their binary form.

    Args:
        pauli_list: Pauli strings

    Returns:
        Binary form of the Pauli strings
    """

    def _single_pauli2bin(s: str):
        """Converts a single string of Pauli operators to its binary form"""
        sarr = np.array(list(s))
        xs = (sarr == "X") + (sarr == "Y")
        zs = (sarr == "Z") + (sarr == "Y")
        return np.hstack((xs, zs)).astype(int)

    errors = np.vstack([_single_pauli2bin(p) for p in pauli_list])
    return np.reshape(errors, [len(pauli_list), 2 * len(pauli_list[0])])


def binary_to_pauli(bin_arr: np.ndarray) -> str:
    """Converts a single binary form to a Pauli string.

    Args:
        bin_arr: One-dimensional binary array

    Returns:
        Pauli form of the binary array
    """
    xs, zs = np.hsplit(bin_arr, 2)
    arr = xs + 2 * zs
    strarr = str(arr.tolist())
    trans = str.maketrans("0123", "IXZY", "[ ,\n]")
    return strarr.translate(trans)


def symplectic_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculates the symplectic product of two operators in their binary form

    Args:
        a: Binary array
        b: Binary array

    Returns:
        Symplectic product of the two binary arrays
    """
    xs, zs = np.hsplit(b, 2)
    b_reversed = np.hstack([zs, xs])
    return a.dot(b_reversed.T) % 2


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


### Syndrome and correction


def errors2syndrome(
    code: codes.StabilizerCode,
    qubit_errors: np.ndarray,
    measurement_error: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, str]:
    """Returns the error syndrome and the final state from the code and the errors that occured.

    Args:
        code: An instance of any of the classes in itsqec.codes
        qubit_errors: Error operator in the binary form
        measurement_error: Array indicating positions of measurement errors

    Returns:
        The difference syndrome and the final state in the Pauli form. The difference syndrome is
        the difference of the syndromes between consecutive measurements if many measurements are performed
        and is the error syndrome when only one measurement is performed.
    """
    current_state = (np.cumsum(qubit_errors, 0) % 2).astype(np.uint8)
    perfect_syndrome = symplectic_product(current_state, code.stabilizers)

    if type(measurement_error) is np.ndarray:
        actual_syndrome = perfect_syndrome ^ measurement_error
    elif type(measurement_error) == type(None):
        actual_syndrome = perfect_syndrome

    diff_syndrome = np.copy(actual_syndrome)
    diff_syndrome[1:, :] = (actual_syndrome[1:, :] - actual_syndrome[0:-1, :]) % 2

    final_state = current_state[-1, :]

    return diff_syndrome, binary_to_pauli(final_state)


def corrected_binary(recovery: str, error: str) -> np.ndarray:
    """Returns the state after recovery operator acts on the error.

    Args:
        recovery: Recovery operator
        error: Error operator

    Returns:
        The final state in binary form
    """
    rec_bin = pauli_to_binary([recovery])
    err_bin = pauli_to_binary([error])
    return rec_bin ^ err_bin


def get_success(recovery: str, error: str, code: codes.StabilizerCode) -> bool:
    """Finds whether the recovery operator successfully corrects the error.

    Args:
        recovery: Recovery operator
        error: Error operator
        code: An instance of any of the classes in itsqec.codes

    Returns:
        True if error correction was successful
    """
    corrected = corrected_binary(recovery, error)

    # if not in code space
    if not np.all(symplectic_product(corrected, code.stabilizers) == 0):
        return False

    arr = symplectic_product(corrected, code.logicals)
    return bool(np.all(arr == 0))
