# ProtoPlaqette

This is a python software with tools for error correction simulations.

# Functionality

- The simulations are performed via the Clifford formalism of keeping track of the stabilizers. The Tableau approach for simulating circuits introduced by Aaronson and Gottesman is also available.
- Error correction codes: By default, the surface code and 5-qubit code are included, but other error correction codes can easily be introduced.
- Available error models include Pauli errors and measurement errors.
- Decoding is possible via PyMatching and also via a lookup decoder.
- A visualizer is included to easily understand where the errors occurred and how they were corrected.

# Installation
The software can be performed by cloning from Github via
```
git clone https://github.com/qubit-ulm/ProtoPlaqette
```
and installing using `setup.py` as
```
python setup.py install
```

# Examples and documentation

To explore how to use ProtoPlaqette, we recommend exploring the Jupyter notebooks in the examples folder. Detailed in-file docstrings are available describing all functionality in the source code.

# Supporters

`ProtoPlaqette` was developed by Dres. Shreya Kumar and Ish Dhand under guidance of Prof. Dr. Martin B Plenio with funding from the [BMBF project PhoQuant](https://www.quantentechnologien.de/forschung/foerderung/quantencomputer-demonstrationsaufbauten/phoquant.html). The concepts explored here were developed further in the all-encompassing fault-tolerance software package [Plaquette](https://github.com/qc-design/plaquette), which was created by [QCDesign](https://www.qc.design/) and funded by the [BMBF project PhotoQ](https://www.photonq.de/). Note that ProtoPlaqette and Plaquette are different packages and share no code.
