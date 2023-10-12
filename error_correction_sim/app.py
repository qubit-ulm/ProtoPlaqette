"""Module to integrate functionality and obtain logical error rates.
"""
from __future__ import annotations

from typing import Optional, Union

import joblib as jl  # type: ignore
import numpy as np

from . import codes, decoders, error_models, utils


def one_shot(
    code: codes.StabilizerCode,
    decoder: decoders.DecoderBase,
    q_error_model: error_models.PauliError,
    p_qubit: float,
    m_error_model: Optional[error_models.MeasurementError] = None,
    p_meas: float = 1e-15,
    m_reps: int = 1,
) -> bool:
    """One shot of an error correction run. Given the choice of code, decoder, error models, probability of errors
    and measurement repetitions, one instance of an error is generated and decoded. Returns a success if decoding was
    successful.

    Args:
        code: An instance of any of the classes in itsqec.codes
        decoder: An instance of any of the classes in itsqec.decoders
        q_error_model: An instance of any of the qubit error classes in itsqec.error_models
        p_qubit: Probability of qubit error
        m_error_model: An instance of any of the measurement error classes in itsqec.error_models
        p_meas: Probability of measurement error
        m_reps: Number of measurement repetitions

    Returns:
        True if decoding was successful
    """
    pauli_qubit_errors = q_error_model.error_str(code.no_qubits, reps=m_reps)
    qubit_errors = utils.pauli_to_binary(pauli_qubit_errors)

    if m_error_model is not None:
        measurement_error = m_error_model.m_errors(m_reps, len(code.stabilizers))
    else:
        measurement_error = None

    diff_syndrome, final_state = utils.errors2syndrome(
        code, qubit_errors, measurement_error
    )
    correction = decoder.decode_wrapper(code, diff_syndrome, p_qubit, p_meas)

    return utils.get_success(correction, final_state, code)


def logical_error_rate(
    code: codes.StabilizerCode,
    decoder: decoders.DecoderBase,
    q_error_model: error_models.PauliError,
    p_qubit: float,
    m_error_model: Optional[error_models.MeasurementError] = None,
    p_meas: float = 1e-15,
    m_reps: int = 1,
    reps: int = 10,
    num_cores: int = 7,
) -> float:
    """Returns the logical error rate by performing many shots of an error correction run. Given the parameters of the
    error correction procedure, the logical error rate is estimated through repeated runs.

    Args:
        code: An instance of any of the classes in itsqec.codes
        decoder: An instance of any of the classes in itsqec.decoders
        q_error_model: An instance of any of the qubit error classes in itsqec.error_models
        p_qubit: Probability of qubit error
        m_error_model: An instance of any of the measurement error classes in itsqec.error_models
        p_meas: Probability of measurement error
        m_reps: Number of measurement repetitions
        reps: Number of shots of the error correction procedure
        num_cores: Number of cores for parallelization

    Returns:
        Logical error rate
    """
    results = jl.Parallel(n_jobs=num_cores)(
        jl.delayed(one_shot)(
            code, decoder, q_error_model, p_qubit, m_error_model, p_meas, m_reps
        )
        for i in range(reps)
    )
    logical_error = 1 - np.mean(results)
    return logical_error
