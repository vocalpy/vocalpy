"""Metrics for segmentation adapted from information retrieval."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ... import validators


def compute_true_positives(hypothesis: npt.NDArray, reference: npt.NDArray,
                           tolerance: float | int, decimals: int | bool = 3):
    """Helper function to compute number of true positives.

    Parameters
    ----------
    hypothesis : numpy.ndarray
        Boundaries, e.g., onsets or offsets of segments,
        as computed by some method.
    reference : numpy.ndarray
        Ground truth boundaries that the hypothesized
        boundaries ``hypothesis`` are compared to.
    tol : float or int
        Tolerance, in seconds. Default is ``0.01``.
    decimals: int
        The number of decimal places to round both
        ``hypothesis`` and ``reference`` to, using
        :func:`numpy.round`. This mitigates inflated
        error rates due to floating point error.
        Rounding is only applied
        if both ``hypothesis`` and ``reference``
        are floating point values. To avoid rounding,
        e.g. to compute strict precision and recall,
        pass in the value ``False``. Default is 3, which
        assumes that the values are in seconds
        and should be rounded to milliseconds.

    Returns
    -------
    n_tp : int
        The number of true positives.
    hits : numpy.ndarray
        The indices of the true positives.

    Notes
    -----
    Adapted from this post by https://github.com/droyed under CC BY-SA 4.0 license.
    https://stackoverflow.com/a/51747164/4906855
    """
    validators.is_valid_boundaries_array(hypothesis)  # 1-d, non-negative, strictly increasing
    validators.is_valid_boundaries_array(reference)
    validators.have_same_dtype(hypothesis, reference)

    if tolerance < 0:
        raise ValueError(
            f"``tolerance`` must be a non-negative number but was: {tolerance}"
        )

    if decimals is not False and not isinstance(decimals, int):
        raise ValueError(
            f"``decimals`` must either be ``False`` or an integer but was: {decimals}"
        )

    if decimals < 0:
        raise ValueError(
            f"``decimals`` must be a non-negative number but was: {decimals}"
        )

    if issubclass(reference.dtype.type, float):
        if not isinstance(tolerance, float):
            raise TypeError(
                "If ``hypothesis`` and ``reference`` are floating, tolerance must be a float also, "
                f"but type was: {type(tolerance)}"
            )
        if decimals is not False:
            # we assume float values are in units of seconds and round to ``decimals``,
            # the default is 3 to indicate "milliseconds"
            reference = np.round(reference, decimals=decimals)
            hypothesis = np.round(hypothesis, decimals=decimals)

    if issubclass(reference.dtype.type, int):
        if not isinstance(tolerance, int):
            raise TypeError(
                "If ``hypothesis`` and ``reference`` are integers, tolerance must be an integer also, "
                f"but type was: {type(tolerance)}"
            )

    # ---- "algorithm" ----
    # To determine whether elements in ``hypothesis`` are in ``reference`` with some ``tolerance``
    # (plus or minus some maximum difference),  we find where we would insert elements
    # from ``hypothesis`` in ``reference`` to maintain ,
    # and then find the minimum distance between the inserted elements from hypothesis
    # and the nearest elements in reference
    # ---------------------
    # determine where to insert elements from ``hypothesis`` in sorted order with ``reference``
    insert_indices = np.searchsorted(reference, hypothesis)

    # we special case values in ``hypothesis`` that would be inserted *after* the last element of reference,
    # so that below we can say ``reference[these_indices] - hypothesis`` without raising an IndexError
    left_invalid_mask = insert_indices == len(reference)
    insert_indices[left_invalid_mask] = len(reference) - 1
    left_differences = reference[insert_indices] - hypothesis
    # we need to multiply any special cased values so that they are positive instead of negative
    # (they are negative because we put them on the "wrong" side using ``left_invalid_mask``
    left_differences[left_invalid_mask] *= -1

    # we do the same thing now for values from the right
    right_invalid_mask = insert_indices == 0
    insert_indices_minus_one = insert_indices - 1
    insert_indices_minus_one[right_invalid_mask] = 0
    right_differences = hypothesis - reference[insert_indices_minus_one]
    right_differences[right_invalid_mask] *= -1

    is_within_tolerance = np.minimum(left_differences, right_differences) <= tolerance
    hits = np.nonzero(is_within_tolerance)
    n_tp = is_within_tolerance.sum()

    return n_tp, hits


def compute_tp(onsets_hyp, offsets_hyp, onsets_ref, offsets_ref, tol=0.01, method="combine"):
    if method == "combine":
        boundaries_hyp = np.concatenate((onsets_hyp, offsets_hyp))
        boundaries_ref = np.concatenate((onsets_ref, offsets_ref))
        tp, all_hits = _compute_tp(boundaries_hyp, boundaries_ref)
    elif method == "separate":
        tp = 0
        all_hits = []
        for boundaries_hyp, boundaries_ref in zip((onsets_hyp, offsets_hyp), (onsets_ref, offsets_ref)):
            tp_, all_hits_ = _compute_tp(boundaries_hyp, boundaries_ref)
            tp += tp_
            all_hits.append(all_hits_)
    return tp, all_hits


def _precision(tp, pp):
    return tp / pp


# FIXME
def precision():
    pass


def _recall(tp, p):
    return tp / p
