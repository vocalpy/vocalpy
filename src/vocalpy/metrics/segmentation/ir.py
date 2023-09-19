"""Metrics for segmentation adapted from information retrieval."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.spatial.distance

tol = 0.01


def _compute_tp(boundaries_hyp: npt.NDArray, boundaries_ref: npt.NDArray, tol: float = 0.01) -> int:
    """Helper function to compute number of true positives.

    Parameters
    ----------
    boundaries_hyp : numpy.ndarray
        Boundaries, e.g., onsets or offsets of segments,
        as computed by some method.
    boundaries_ref : numpy.ndarray
        Ground truth boundaries that the hypothesized
        boundaries ``boundaries_hyp`` are compared to.
    tol : float
        Tolerance, in seconds. Default is ``0.01``.

    Returns
    -------
    tp : int
        The number of true positives.
    """
    if tol < 0.0:
        raise ValueError(f"Tolerance value ``tol`` must be a non-negative float but was: {tol}")

    tp = 0
    for _, boundary_time_ref in enumerate(boundaries_ref):
        # FIXME: do we ignore multiple hits here?
        hits = (
            np.abs(scipy.spatial.distance.cdist(np.array([[boundary_time_ref]]), boundaries_hyp[:, np.newaxis])) < tol
        ).sum()
        if hits > 0:
            tp += 1
    return tp


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
