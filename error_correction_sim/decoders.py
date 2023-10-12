"""
Decoders
--------
Provides base and specific quantum error correction decoders.

Classes:
    DecoderBase: Abstract base class for all decoders.
        Defines the required methods and attributes for decoders.
    LookupDecoder: An error correction decoder that searches for matching syndromes by generating all
        possible errors up to a certain weight.
    MWPMDecoder: Min-weight perfect matching decoder that uses the pymatching library to decode
        errors given a syndrome.
"""
from __future__ import annotations

import abc
import itertools as it
from typing import Iterator, Optional, Tuple

import numpy as np
import pymatching as pm  # type: ignore

from . import codes, utils


class DecoderBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def decode_wrapper(
        self,
        code: codes.StabilizerCode,
        syndrome: np.ndarray,
        p_qubit: float = 0.0,
        p_meas: float = 0.0,
    ) -> str:
        """Returns the correction operator given the QEC code and measurement syndrome."""
        raise NotImplementedError


class LookupDecoder(DecoderBase):
    """The lookup decoder looks at all possible error configurations in order of weight of the configuration
    and compares their syndrome with the given syndrome. When it finds a match, it returns the error configuration.
    """

    #: The maximum weight of the error that the decoder checks for matching syndrome
    max_weight: int

    def __init__(self, max_weight: int = 3):
        """
        Args:
            max_weight: The maximum weight of the error that the decoder checks for matching syndrome
        """
        self.max_weight = max_weight

    def decode_wrapper(
        self,
        code: codes.StabilizerCode,
        syndrome: np.ndarray,
        p_qubit: float = 0.0,
        p_meas: float = 0.0,
    ) -> str:
        # called decode_wrapper because decode is a pymatching function
        """Returns the correction operator given the QEC code and measurement syndrome.

        Args:
            code: An instance of any of the classes in itsqec.codes
            syndrome: The error syndrome to decode
            p_qubit: ¯\\\\_(ツ)_/¯
            p_meas: ¯\\\\_(ツ)_/¯

        Returns:
            The error that led to the syndrome. The error is also the correction operator here.

        .. todo:: remove useless arguments. The useless arguments makes the usage similar to the other decoders.
        """
        error = "I" * code.no_qubits
        for error_try in self.all_errors(code.no_qubits):
            error_try_bin = utils.pauli_to_binary([error_try])
            if np.array_equal(utils.errors2syndrome(code, error_try_bin)[0], syndrome):
                error = error_try
                break
        return error

    def all_errors(self, no_qubits: int) -> Iterator[str]:
        """Yields an iterator over all error strings up to the maximum weight.

        Args:
            no_qubits: Number of qubits on which to generate all errors

        Yields:
            iterator over all error strings in Pauli form
        """
        for weight in range(0, self.max_weight + 1):
            for selected_qubits in it.combinations(range(no_qubits), weight):
                for selected_xyzs in it.product("XZY", repeat=weight):
                    pauli = ["I"] * no_qubits
                    for selected_qubit, selected_xyz in zip(
                        selected_qubits, selected_xyzs
                    ):
                        pauli[selected_qubit] = selected_xyz
                    yield ("".join(pauli))


class MWPMDecoder(DecoderBase):
    """Min-weight perfect matching decoder."""

    def decode_wrapper(
        self,
        code: codes.StabilizerCode,
        diff_syndrome: np.ndarray,
        p_qubit: float = 1e-15,
        p_meas: float = 1e-15,
    ) -> str:
        """Returns the correction operator given the QEC code and measurement syndrome. Qubit and measurement
        probabilities are required for assigning appropriate weights in the MWPM algorithm.

        Args:
            code: An instance of codes.SurfaceCode
            diff_syndrome: The error syndrome when only one measurement is performed. If many
                measurements are performed, this is the difference of the syndromes between consecutive measurements.
            p_qubit: Probability of qubit error
            p_meas: Probability of measurement error

        Returns:
            recovery operator in Pauli form

        .. todo:: p_qubit should depend on error model.

        .. todo::

           The default values for ``p_qubit`` and ``p_meas`` are ``1e-15`` instead
           of ``0`` (from above) because the code currently cannot handle ``0``.
        """
        Xm, Zm = self.stabilizer2matching(code.stabilizers)

        Z_diff_syndrome = np.hsplit(diff_syndrome, 2)[0]
        X_diff_syndrome = np.hsplit(diff_syndrome, 2)[1]
        reps = diff_syndrome.shape[0]

        X_mat = pm.Matching(
            Xm,
            spacelike_weights=np.log((1 - p_qubit) / p_qubit),
            repetitions=reps,
            timelike_weights=np.log((1 - p_meas) / p_meas),
        )
        Z_mat = pm.Matching(
            Zm,
            spacelike_weights=np.log((1 - p_qubit) / p_qubit),
            repetitions=reps,
            timelike_weights=np.log((1 - p_meas) / p_meas),
        )

        recoveryX = X_mat.decode(X_diff_syndrome.T)
        recoveryZ = Z_mat.decode(Z_diff_syndrome.T)

        recovery = np.hstack([recoveryZ, recoveryX])
        return utils.binary_to_pauli(recovery)

    def stabilizer2matching(self, stabilizers: np.ndarray) -> Tuple[list, list]:
        """Converts the stabilizers to separate X and Z stabilizer arrays.

        Args:
            stabilizers: Stabilizers of a code in binary form

        Returns:
            x and z stabilizer arrays
        """
        x_stab_all, z_stab_all = np.hsplit(stabilizers, 2)
        x_stab = np.vsplit(x_stab_all, 2)[1]
        z_stab = np.vsplit(z_stab_all, 2)[0]
        return x_stab, z_stab
