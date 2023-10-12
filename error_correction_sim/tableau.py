"""Tools for simulating stabilizer circuits via tableau formalism

This module provides the QuantumState class for representing and manipulating quantum states
in the stabilizer formalism. The class implements methods for initialization to a zero state,
common quantum gates (like X, Z, Hadamard, Phase, C-X, and C-Z), and operations for
multiplying Pauli operators.

Main Components:
- QuantumState: Class representing the state of qubits in the stabilizer representation.
  - init_to_zeros: Initializes the quantum state to |00...0>.
  - X: Implements the X (bit-flip) gate.
  - Z: Implements the Z (phase-flip) gate.
  - hadamard: Implements the Hadamard gate.
  - phase: Implements the phase gate.
  - cx: Implements the C-NOT or C-X gate.
  - cz: Implements the C-PHASE or C-Z gate.
  - g: Helps in calculating the sign of operator when multiplying Pauli matrices.
  - rowsum: Multiplies two Pauli operators from the tableau.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from . import tab_utils


class QuantumState:
    """Describes the state of the qubits in the stabilizer representation"""

    def __init__(self, no_qubits: int):
        """
        Args:
            no_qubits: Total number of qubits in the state
        """
        #: Total number of qubits in the state
        self.no_qubits: int = no_qubits
        #: Random number generator
        self.rng = np.random.default_rng()
        #: Full tableau of the state including sign, destabilizer and stabiliser
        #: generators in the binary picture
        self.state: np.ndarray = self.init_to_zeros()

    def init_to_zeros(self) -> np.ndarray:
        """Initializes the quantum state to ``|00...0>``

        Returns:
            The state ``|00...0>`` in the tableau representation
        """
        a = np.eye(2 * self.no_qubits, dtype=int)
        b = np.zeros([2 * self.no_qubits, 1], dtype=int)
        return np.hstack([a, b])

    def X(self, targets: int | list[int] | np.ndarray):
        """Performs the X (bit-flip) gate on a given target qubit. In the stabilizer
        representation, X -> X and Z -> -Z

        Args:
            targets: The indices of the target qubit
        """
        if isinstance(targets, int):
            targets = [targets]

        for target in targets:
            # Set r = r ^ z_target
            self.state[:, 2 * self.no_qubits] = (
                self.state[:, 2 * self.no_qubits]
                ^ self.state[:, target + self.no_qubits]
            )

    def Z(self, targets: int | list[int] | np.ndarray):
        """Performs the Z (phase-flip) gate on a given target qubit. In the stabilizer
        representation, X -> -X and Z -> Z

        Args:
            targets: The indices of the target qubit
        """
        if isinstance(targets, int):
            targets = [targets]

        for target in targets:
            # Set r = r ^ x_target
            self.state[:, 2 * self.no_qubits] = (
                self.state[:, 2 * self.no_qubits] ^ self.state[:, target]
            )

    def hadamard(self, targets: int | list[int] | np.ndarray):
        """Performs the Hadamard gate on a given target qubit. In the stabilizer
        representation, X -> Z and Z -> X

        Args:
            targets: The indices of the target qubit
        """
        if isinstance(targets, int):
            targets = [targets]

        for target in targets:
            # Set r = r ^ (x_target*z_target)
            self.state[:, 2 * self.no_qubits] = self.state[:, 2 * self.no_qubits] ^ (
                self.state[:, target] * self.state[:, target + self.no_qubits]
            )
            # Swap x_target and z_target
            self.state[:, [target, target + self.no_qubits]] = self.state[
                :, [target + self.no_qubits, target]
            ]

    def phase(self, targets: int | list[int] | np.ndarray):
        """Performs the phase gate on a given target qubit. In the stabilizer
        representation, X -> Y and Z -> Z

        Args:
            targets: The indices of the target qubit
        """
        if isinstance(targets, int):
            targets = [targets]

        for target in targets:
            # Set r = r ^ (x_target*z_target)
            self.state[:, 2 * self.no_qubits] = self.state[:, 2 * self.no_qubits] ^ (
                self.state[:, target] * self.state[:, target + self.no_qubits]
            )
            # z_target = z_target ^ x_target
            self.state[:, target + self.no_qubits] = (
                self.state[:, target + self.no_qubits] ^ self.state[:, target]
            )

    def cx(self, targets: Union[tuple, list[tuple]]):
        """Performs the C-NOT or C-X gate on a given target qubit based on a control qubit

        Args:
            targets: Each tuple contains the indices of the control
                qubit followed by the target qubit
        """
        if isinstance(targets, tuple):
            targets = [targets]

        for ctrl, tgt in targets:
            # Calculate intermediate results for readability
            int_res1 = self.state[:, ctrl] * self.state[:, tgt + self.no_qubits]
            int_res2 = self.state[:, tgt] ^ self.state[:, ctrl + self.no_qubits] ^ 1

            # Compute the modified r value
            self.state[:, 2 * self.no_qubits] ^= int_res1 * int_res2

            # Update x_target value
            self.state[:, tgt] ^= self.state[:, ctrl]

            # Modify z_control value
            self.state[:, ctrl + self.no_qubits] ^= self.state[:, tgt + self.no_qubits]

    def cz(self, targets: Union[tuple, list[tuple]]):
        """Performs the C-PHASE or C-Z gate on a given target qubit based on a control qubit

        Args:
            targets: Each tuple contains the indices of the control
                qubit followed by the target qubit
        """
        if isinstance(targets, tuple):
            targets = [targets]

        for ctrl_qubit, tgt_qubit in targets:
            # Calculate intermediate results for clarity
            int_x_mult = self.state[:, ctrl_qubit] * self.state[:, tgt_qubit]
            int_z_xor = self.state[:, tgt_qubit + self.no_qubits] ^ self.state[:,
                                                                    ctrl_qubit + self.no_qubits]

            # Compute the updated r value
            self.state[:, 2 * self.no_qubits] ^= int_x_mult * int_z_xor

            # Modify z_target value
            self.state[:, tgt_qubit + self.no_qubits] ^= self.state[:, ctrl_qubit]

            # Update z_control value
            self.state[:, ctrl_qubit + self.no_qubits] ^= self.state[:, tgt_qubit]

    def g(self, x1: int, z1: int, x2: int, z2: int) -> int:
        """Helps calculates the sign of the operator when multiplying Pauli matrices.

        Args:
            x1: x bit of operator 1
            z1: z bit of operator 1
            x2: x bit of operator 2
            z2: z bit of operator 2

        Returns:
            exponent of i when two operators are multiplied
        """
        if x1 == 0 and z1 == 0:
            g = 0
        elif x1 == 1 and z1 == 1:
            g = z2 - x2
        elif x1 == 1 and z1 == 0:
            g = z2 * (2 * x2 - 1)
        elif x1 == 0 and z1 == 1:
            g = x2 * (1 - 2 * z2)
        # Error if any other case
        return g

    def rowsum(self, i: int, j: int) -> np.ndarray:
        """Multiplies two Pauli operators from the tableau.

        Args:
            i: Row index of first operator
            j: Row index of second operator

        Returns:
            Multiplied result in binary form with the sign in the last bit
        """

        final = self.state[i] ^ self.state[j]

        g_tot = 0
        for qubit in range(self.no_qubits):
            g_tot = g_tot + self.g(
                self.state[i, qubit],
                self.state[i, qubit + self.no_qubits],
                self.state[j, qubit],
                self.state[j, qubit + self.no_qubits],
            )

        r = ((2 * self.state[i, -1] + 2 * self.state[j, -1] + g_tot) % 4) // 2
        # Ideally the function is 0 or 2. Need error message if 1 or 3.
        final[-1] = r

        return final

    def measure_nondest(
        self, pauli: str, targets: int | list[int] | np.ndarray
    ) -> list[int]:
        """Measurement of the state in the operator given by the Pauli operator on a specified qubit.
        Here, the measured qubit is not destroyed, but collapses to the measured state.

        Args:
            pauli: Measurement operator. Eg. "Z" or "-X"
            targets: The indices of the qubits to be measured

        Returns:
            List of measurement outcomes (0 or 1) in the order of targets
        """
        if isinstance(targets, int):
            targets = [targets]
        meas_outcomes = []

        for qubit in targets:
            meas_op = tab_utils.single_qubit_op(pauli, qubit, self.no_qubits)

            comm = tab_utils.commutator(self.state, meas_op)
            nonzero = np.nonzero(comm.flatten())[0]

            if nonzero[-1] >= self.no_qubits:
                meas_outcome = self.rng.choice([0, 1])
                meas_op[-1] = meas_op[-1] ^ meas_outcome

                first_nc_stab = nonzero[nonzero >= self.no_qubits][0]
                index_pos = np.where(nonzero == first_nc_stab)[0][0]
                other_rows = np.delete(nonzero, index_pos)

                for row_no in other_rows:
                    self.state[row_no] = self.rowsum(first_nc_stab, row_no)

                self.state[first_nc_stab - self.no_qubits] = self.state[first_nc_stab]
                self.state[first_nc_stab] = meas_op

            else:
                scratch = np.zeros([1, 2 * self.no_qubits + 1], dtype=int)
                self.state = np.concatenate((self.state, scratch))

                for row_no in nonzero:
                    self.state[2 * self.no_qubits, :] = self.rowsum(
                        2 * self.no_qubits, row_no + self.no_qubits
                    )

                meas_outcome = int(self.state[-1, -1] != meas_op[-1])
                self.state = self.state[:-1, :]

            meas_outcomes.append(meas_outcome)

        return meas_outcomes

    def measure_nondest_operator(self, meas_op: np.ndarray) -> list[int]:
        """Measurement of the state in the operator given by the Pauli operator on a specified qubit.
        Here, the measured qubit is not destroyed, but collapses to the measured state.

        Args:
            meas_op: Measurement operator in binary form

        Returns:
            List of measurement outcomes (0 or 1) in the order of targets
        """

        comm = tab_utils.commutator(self.state, meas_op)
        nonzero = np.nonzero(comm.flatten())[0]

        if nonzero[-1] >= self.no_qubits:
            meas_outcome = self.rng.choice([0, 1])
            meas_op[-1] = meas_op[-1] ^ meas_outcome

            first_nc_stab = nonzero[nonzero >= self.no_qubits][0]
            index_pos = np.where(nonzero == first_nc_stab)[0][0]
            other_rows = np.delete(nonzero, index_pos)

            for row_no in other_rows:
                self.state[row_no] = self.rowsum(first_nc_stab, row_no)

            self.state[first_nc_stab - self.no_qubits] = self.state[first_nc_stab]
            self.state[first_nc_stab] = meas_op

        else:
            scratch = np.zeros([1, 2 * self.no_qubits + 1], dtype=int)
            self.state = np.concatenate((self.state, scratch))

            for row_no in nonzero:
                self.state[2 * self.no_qubits, :] = self.rowsum(
                    2 * self.no_qubits, row_no + self.no_qubits
                )

            meas_outcome = int(self.state[-1, -1] != meas_op[-1])
            self.state = self.state[:-1, :]

        return meas_outcome

    def measure_dest(
        self, pauli: str, targets: int | list[int] | np.ndarray
    ) -> list[int]:
        """Destructive measurement of the state in the operator given by the Pauli operator on a specified qubit.
        Here, the measured qubit is traced out.

        Args:
            pauli: Measurement operator. Eg. "Z" or "-X"
            targets: The indices of the qubits to be measured

        Returns:
            List of measurement outcomes (0 or 1) in the order of targets
        """
        if isinstance(targets, int):
            targets = [targets]
        meas_outcomes: list[int] = []

        for qubit in targets:

            meas_outcome = self.measure_nondest(pauli, qubit)
            meas_outcomes = meas_outcomes + meas_outcome

            if pauli == "X":
                pos = qubit
            elif pauli == "Z":
                pos = qubit + self.no_qubits
            # Can't do Y yet. Do we even want to?

            arr = self.state[self.no_qubits :, pos]
            ones = np.where(arr > 0)[0]
            first_stab = ones[0]

            for i in ones[1:]:
                self.state[i + self.no_qubits] = self.rowsum(
                    i + self.no_qubits, first_stab + self.no_qubits
                )
                self.state[first_stab] = self.rowsum(i, first_stab)

            a = np.delete(self.state, [first_stab, first_stab + self.no_qubits], axis=0)
            b = np.delete(
                a, [pos, (pos + self.no_qubits) % (2 * self.no_qubits)], axis=1
            )
            self.state = b
            self.no_qubits = self.no_qubits - 1

        return meas_outcomes

    def reset_qubit(self, pauli: str, targets: int | list[int] | np.ndarray):
        """Resets given target qubits to +1 eigenstate of given Pauli operator

        Args:
            pauli: Pauli operator. Eg. "Z" or "-X"
            targets: The indices of the qubits to be reset
        """
        # No Ys yet please
        if isinstance(targets, int):
            targets = [targets]

        for qubit in targets:
            meas_outcome = self.measure_nondest(pauli, qubit)
            if meas_outcome[0]:
                if pauli == "Z" or pauli == "-Z":
                    self.X(qubit)
                elif pauli == "X" or pauli == "-X":
                    self.Z(qubit)

    def add_qubit(self, pos: Optional[int] = None):
        """Adds a qubit at a specified position in state ``|0>``

        Args:
            pos: Position of qubit to be added. Default position is at the end.

        """
        if pos is None:
            pos = self.no_qubits

        self.state = np.insert(self.state, [pos, pos + self.no_qubits], 0, axis=1)
        self.state = np.insert(
            self.state, [self.no_qubits, 2 * self.no_qubits], 0, axis=0
        )
        self.no_qubits = self.no_qubits + 1
        self.state[self.no_qubits - 1, pos] = 1
        self.state[2 * self.no_qubits - 1, pos + self.no_qubits] = 1

    def display_state_pauli(
        self,
        full_tableu: bool = False,
        show_identities: bool = False,
    ):
        """Displays the state in terms of the stabilizer (and optionally destabilizer) generators.

        Args:
            full_tableu: Whether to also display the destabilizer generators
            show_identities: Whether to display identities on qubits
        """
        if full_tableu:
            print("Destabilizer generators:")
            for generator in self.state[: self.no_qubits]:
                print(tab_utils.binary_to_pauli(generator, show_identities))

        print("Stabilizer generators:")
        for generator in self.state[self.no_qubits :]:
            print(tab_utils.binary_to_pauli(generator, show_identities))
