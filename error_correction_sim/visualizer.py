"""
Module for visualizing quantum surface codes using plotting functions.

This module provides functionalities to plot surface codes along with their
errors and recovery procedures. The visualizations are designed to show
gridlines, plaquettes (areas around logical qubits), data qubits, and ancilla qubits.
Additionally, users can visualize errors and recovery operations on these qubits.

Classes:
    SurfacePlot: A class designed to visualize the surface code, errors, and recovery
    operations in a 2D plane.
"""
from __future__ import annotations

from typing import Optional

import matplotlib as mpl  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import seaborn as sns  # type: ignore
from matplotlib.patches import Polygon  # type: ignore

from . import codes


class SurfacePlot:
    """Class to help visualizing a surface code."""

    def __init__(
        self, size: int, error: Optional[str] = None, recovery: Optional[str] = None
    ):
        """
        Args:
            size: Size of surface code (number of plaquettes in either direction)
            error: Pauli string of errors on data qubits
            recovery: Pauli string of recovery operator on data qubits
        """
        #: Matplotlib figure
        self.fig: plt.Figure = plt.figure()
        #: Matplotlib axes
        self.ax: mpl.axes.SubplotBase = self.fig.add_subplot()
        #: The surface code we want to plot
        self.code: codes.SurfaceCode = codes.SurfaceCode(size)
        #: Pauli error string to plot
        self.errs: Optional[str] = error
        #: Pauli recovery operator string to plot
        self.recos: Optional[str] = recovery

    def draw_gridlines(self):
        """Draws gridlines."""
        size = self.code.size
        for i in range(size):
            self.ax.hlines(2 * i, 0, 2 * size, colors="silver")
            self.ax.vlines(2 * i + 1, 0, 2 * size, colors="silver")
        self.ax.hlines(2 * size, 0, 2 * size, colors="silver")

    def draw_plaquettes(self):
        """Draws the X and Z plaquettes."""
        for plaq in self.code.x_plaqs():
            qubit_xys = self.code.plaqs2qubit_xys(plaq)
            polygon = Polygon(qubit_xys, color="khaki", alpha=0.15)
            self.ax.add_patch(polygon)
        for plaq in self.code.z_plaqs():
            qubit_xys = self.code.plaqs2qubit_xys(plaq)
            polygon = Polygon(qubit_xys, color="palevioletred", alpha=0.15)
            self.ax.add_patch(polygon)

    def calc_xys(self) -> np.ndarray:
        """Returns a list of coordinates of all the data qubits

        Returns:
            coordinates of all data qubits
        """
        xys = np.zeros([self.code.no_qubits, 2], dtype=int)
        for index in range(self.code.no_qubits):
            xys[index] = self.code.index2xy(index)
        return xys

    def draw_sites(self):
        """Draws the data qubits."""
        xys = self.calc_xys()
        self.ax.scatter(xys[:, 0], xys[:, 1], c="darkblue", alpha=0.3, zorder=3)

    def draw_site_labels(self):
        """Labels the data qubits with their indices."""
        xys = self.calc_xys()
        for index in range(self.code.no_qubits):
            x, y = xys[index]
            name = str(index)
            self.ax.annotate(name, (x + 0.06, y + 0.06))

    def draw_ancilla_labels(self):
        """Labels the ancilla qubits with their indices."""
        x_xys = self.code.x_plaqs()
        z_xys = self.code.z_plaqs()
        for i, index in enumerate(self.code.ancilla_x_index()):
            x, y = x_xys[i]
            name = str(index)
            self.ax.annotate(name, (x + 0.06, y + 0.06))
        for i, index in enumerate(self.code.ancilla_z_index()):
            x, y = z_xys[i]
            name = str(index)
            self.ax.annotate(name, (x + 0.06, y + 0.06))

    def positions_index(self, errs_recos: str) -> tuple[list, list, list]:
        """Converts a Pauli string into lists of indices of X, Y and Z operators.

        Args:
            errs_recos: Pauli string of errors or recovery operation on data qubits

        Returns:
            positions of X, Y and Z operators in the Pauli string
        """
        x_pos = []
        y_pos = []
        z_pos = []
        for pos, char in enumerate(errs_recos):
            if char == "X":
                x_pos.append(pos)
            elif char == "Y":
                y_pos.append(pos)
            elif char == "Z":
                z_pos.append(pos)
        return x_pos, y_pos, z_pos

    def positions_xy(self, errs_recos: str) -> tuple[list, list, list]:
        """Converts a Pauli string into lists of coordinates of X, Y and Z operators.

        Args:
            errs_recos: Pauli string of errors or recovery operation on data qubits

        Returns:
            coordinates of X, Y and Z operators from the Pauli string
        """
        x_pos, y_pos, z_pos = self.positions_index(errs_recos)
        X_xy = [self.code.index2xy(x_p) for x_p in x_pos]
        Y_xy = [self.code.index2xy(y_p) for y_p in y_pos]
        Z_xy = [self.code.index2xy(z_p) for z_p in z_pos]
        return X_xy, Y_xy, Z_xy

    def draw_errs(self):
        """Draws the error instance on the surface code."""
        if self.errs != None:  # Replace with exception later
            for x_pos in self.positions_xy(self.errs)[0]:
                self.ax.annotate("X", (x_pos[0] + 0.06, x_pos[1] - 0.2), color="r")
            for x_pos in self.positions_xy(self.errs)[1]:
                self.ax.annotate("Y", (x_pos[0] + 0.06, x_pos[1] - 0.2), color="r")
            for x_pos in self.positions_xy(self.errs)[2]:
                self.ax.annotate("Z", (x_pos[0] + 0.06, x_pos[1] - 0.2), color="r")

    def draw_recos(self):
        """Draws the recovery operator on the surface code."""
        if self.recos != None:  # Replace with exception later
            for x_pos in self.positions_xy(self.recos)[0]:
                self.ax.annotate("X", (x_pos[0] - 0.2, x_pos[1] - 0.2), color="g")
            for x_pos in self.positions_xy(self.recos)[1]:
                self.ax.annotate("Y", (x_pos[0] - 0.2, x_pos[1] - 0.2), color="g")
            for x_pos in self.positions_xy(self.recos)[2]:
                self.ax.annotate("Z", (x_pos[0] - 0.2, x_pos[1] - 0.2), color="g")

    def draw_ancillas(self):
        """Draws the ancilla qubits."""
        x_xys = self.code.x_plaqs()
        self.ax.scatter(
            [xy[0] for xy in x_xys],
            [xy[1] for xy in x_xys],
            c="darkgreen",
            alpha=0.3,
            zorder=3,
        )
        z_xys = self.code.z_plaqs()
        self.ax.scatter(
            [xy[0] for xy in z_xys],
            [xy[1] for xy in z_xys],
            c="darkgreen",
            alpha=0.3,
            zorder=3,
        )

    def plot(self, save_fig: bool):
        """Plotting function that sets style, axis limits."""
        size = self.code.size
        self.ax.set_xlim(0 - 0.3, 2 * size + 0.3)
        self.ax.set_ylim(0 - 0.3, 2 * size + 0.3)
        self.ax.axis("off")
        self.ax.set_aspect(1)
        sns.set_style("white")
        if save_fig:
            plt.savefig("Surface_viz.pdf")
        else:
            plt.show()

    def plot_all(
        self,
        draw_gridlines_YN: bool = True,
        draw_plaquettes_YN: bool = True,
        draw_sites_YN: bool = True,
        draw_site_labels_YN: bool = False,
        draw_err_rec: bool = True,
        draw_anc_YN: bool = True,
        save_fig: bool = False,
    ):
        """Plotting function to draw each part of the surface code plot.

        Args:
            draw_gridlines_YN: Option to draw gridlines
            draw_plaquettes_YN: Option to draw plaquettes
            draw_sites_YN: Option to draw data qubits
            draw_site_labels_YN: Option to draw site labels for data qubits
            draw_err_rec: Option to draw error and recovery operators
            draw_anc_YN: Option to draw ancilla qubits and their site labels
        """
        if draw_gridlines_YN:
            self.draw_gridlines()
        if draw_plaquettes_YN:
            self.draw_plaquettes()
        if draw_sites_YN:
            self.draw_sites()
        if draw_site_labels_YN:
            self.draw_site_labels()
        if draw_err_rec:
            self.draw_errs()
            self.draw_recos()
        if draw_anc_YN:
            self.draw_ancillas()
            self.draw_ancilla_labels()
        self.plot(save_fig)
