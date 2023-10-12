"""
Defines different stabilizer codes


This module provides a structure for defining stabilizer codes used in quantum error
correction. The primary class, `StabilizerCode`, represents a generic stabilizer code
with subclasses `FiveQubitCode` and `SurfaceCode` providing implementations of
specific quantum error correcting codes.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from . import utils


class StabilizerCode:
    """Stabilizer codes"""

    #: Logical operators in binary form
    logicals: np.ndarray
    #: Name of the code
    name: str
    #: Number of data qubits in the code
    no_qubits: int
    #: Stabilizers in binary form
    stabilizers: np.ndarray

    def __init__(
        self, name: str, stabilizers: np.ndarray, logicals: np.ndarray, no_qubits: int
    ):
        """Summary

        Args:
            name: Name of the code
            stabilizers: Stabilizers in binary form
            logicals: Logical operators in binary form
            no_qubits: number of data qubits in the code
        """
        self.name = name
        self.stabilizers = stabilizers
        self.logicals = logicals
        self.no_qubits = no_qubits


class FiveQubitCode(StabilizerCode):
    """Five qubit stabilizer code. It is the smallest code that can correct for all Pauli errors."""

    # TODO see https://github.com/sphinx-doc/sphinx/issues/9884
    logicals: np.ndarray
    name: str
    no_qubits: int
    stabilizers: np.ndarray

    def __init__(self):
        stabilizers_pauli = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
        logicals_pauli = ["XXXXX", "ZZZZZ"]
        no_qubits = 5

        stabilizers = utils.pauli_to_binary(stabilizers_pauli)
        logicals = utils.pauli_to_binary(logicals_pauli)

        super().__init__("5-qubit", stabilizers, logicals, no_qubits)


class SurfaceCode(StabilizerCode):
    """Planar surface code."""

    #: Size of the surface code defined by the number of plaquettes in either direction
    size: int

    # TODO see https://github.com/sphinx-doc/sphinx/issues/9884
    logicals: np.ndarray
    name: str
    no_qubits: int
    stabilizers: np.ndarray

    def __init__(self, size: int):
        """
        Args:
            size: Size of the surface code defined by the number of plaquettes in either direction
        """
        self.size = size
        self.no_qubits = (size + 1) ** 2 + size**2
        self.no_ancillas = len(self.ancilla_x_index()) + len(self.ancilla_z_index())

        stabilizers_pauli = self.z_stabilizers() + self.x_stabilizers()
        logicals_pauli = self.logicals_surface()

        stabilizers = utils.pauli_to_binary(stabilizers_pauli)
        logicals = utils.pauli_to_binary(logicals_pauli)

        super().__init__("Surface", stabilizers, logicals, self.no_qubits)

    def stab_qubits(self) -> Tuple[list, list]:
        """Returns the list of x and z stabilizer qubits.

        Returns:
            list of qubits in x and z stabilizer generators
        """
        x_stab_q = [self.plaqs2qubit(x_plaq) for x_plaq in self.x_plaqs()]
        z_stab_q = [self.plaqs2qubit(z_plaq) for z_plaq in self.z_plaqs()]
        return x_stab_q, z_stab_q

    def xy2index(self, xy: Tuple[int, int]) -> int:
        """Flattens the labelling of qubits. Converts coordinates to index.

        Args:
            xy: Coordinates of the qubit

        Returns:
            Index of the qubit
        """
        return int((xy[0] + xy[1] * (2 * self.size + 1)) / 2)

    def index2xy(self, i: int) -> Tuple[int, int]:
        """Converts index of a qubit to its coordinates.

        Args:
            i: Index of the qubit

        Returns:
            Coordinates of the qubit
        """
        x = (2 * i) % (2 * self.size + 1)
        y = (2 * i) // (2 * self.size + 1)
        return (x, y)

    def x_plaqs(self) -> list:
        """Returns the coordinates of the x plaquettes

        Returns:
            Coordinates of the x plaquettes
        """
        return [
            (x, y)
            for y in range(0, 2 * self.size + 1, 2)
            for x in range(1, 2 * self.size, 2)
        ]

    def z_plaqs(self) -> list:
        """Returns the coordinates of the z plaquettes

        Returns:
            Coordinates of the z plaquettes
        """
        return [
            (x, y)
            for y in range(1, 2 * self.size, 2)
            for x in range(0, 2 * self.size + 1, 2)
        ]

    def ancilla_x_index(self) -> list:
        """Returns a list of the indices of x ancilla qubits

        Returns:
            indices of x ancilla qubits
        """
        return [
            int((xy[0] + xy[1] * self.size) / 2 + self.no_qubits)
            for xy in self.x_plaqs()
        ]

    def ancilla_z_index(self) -> list:
        """Returns a list of the indices of z ancilla qubits

        Returns:
            indices of z ancilla qubits
        """
        return [
            int((xy[0] * self.size + xy[1]) / 2 + len(self.x_plaqs()) + self.no_qubits)
            for xy in self.z_plaqs()
        ]

    def inlattice_qubits(self, qubit_list: list) -> list:
        """Cleans up qubit coordinates by removing those qubits not in the lattice.

        Args:
            qubit_list: all qubit coordinates around a plaquette

        Returns:
            qubit coordinates of only those inside the lattice
        """
        return [
            qubit
            for qubit in qubit_list
            if (min(qubit) >= 0 and max(qubit) <= 2 * self.size)
        ]

    def plaqs2qubit(self, plaq: Tuple[int, int]) -> list:
        """Returns a list of qubit indices around the input plaquette

        Args:
            plaq: coordinates of the centre of a plaquette

        Returns:
            indices of the qubits around the given plaquette
        """
        qubit_xys = self.plaqs2qubit_xys(plaq)
        qubit_indices = [self.xy2index(qubitxy) for qubitxy in qubit_xys]
        return qubit_indices

    def plaqs2qubit_xys(self, plaq: Tuple[int, int]) -> list:
        """Returns a list of qubit coordinates around the input plaquette

        Args:
            plaq: coordinates of the centre of a plaquette

        Returns:
            coordinates of the qubits around the given plaquette
        """
        up = (plaq[0], plaq[1] + 1)
        down = (plaq[0], plaq[1] - 1)
        left = (plaq[0] - 1, plaq[1])
        right = (plaq[0] + 1, plaq[1])
        qubit_xys = self.inlattice_qubits([up, left, down, right])
        return qubit_xys

    def qubits2stab(self, qubits: list, pauli: str) -> str:
        """Returns the stabilizer operator in Pauli form for the given qubits

        Args:
            qubits: indices of the qubits around a plaquette
            pauli: Pauli operator (e.g. 'X', 'Z')

        Returns:
            stabilizer operator in Pauli form
        """
        stab_list = list("I" * self.no_qubits)
        for qubit in qubits:
            stab_list[qubit] = pauli
        return "".join(stab_list)

    def x_stabilizers(self) -> list:
        """Returns a list of all the X stabilizers

        Returns:
            x stabilizers in the Pauli form
        """
        x_stabs = []
        for plaq in self.x_plaqs():
            qubits = self.plaqs2qubit(plaq)
            stab = self.qubits2stab(qubits, "X")
            x_stabs.append(stab)
        return x_stabs

    def z_stabilizers(self) -> list:
        """Returns a list of all the Z stabilizers

        Returns:
            z stabilizers in the Pauli form
        """
        z_stabs = []
        for plaq in self.z_plaqs():
            qubits = self.plaqs2qubit(plaq)
            stab = self.qubits2stab(qubits, "Z")
            z_stabs.append(stab)
        return z_stabs

    def logicals_surface(self) -> list:
        """Returns a list of the logical operators

        Returns:
            X and Z logical operators in Pauli form
        """
        xbar_qubits = [(0, i) for i in range(0, 2 * self.size + 1, 2)]
        zbar_qubits = [(i, 0) for i in range(0, 2 * self.size + 1, 2)]

        xbar_qubit_indices = [self.xy2index(qubitxy) for qubitxy in xbar_qubits]
        zbar_qubit_indices = [self.xy2index(qubitxy) for qubitxy in zbar_qubits]

        xbar = self.qubits2stab(xbar_qubit_indices, "X")
        zbar = self.qubits2stab(zbar_qubit_indices, "Z")

        return [xbar, zbar]
