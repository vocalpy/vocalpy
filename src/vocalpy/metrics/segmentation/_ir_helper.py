"""Helper functions for metrics from information.

Adapted from of ``mir_eval``, under MIT license: 
https://github.com/craffel/mir_eval
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import numpy.typing as npt

from ... import validators


__all__ = [
    'bipartite_match',
    'fast_hit_windows',
    'match_event_times',
]


def fast_hit_windows(reference: npt.NDArray, hypothesis: npt.NDArray, tolerance: float) -> tuple[list, list]:
    """Fast calculation of toleranceed hits for time events.

    Given two lists of event times ``reference`` and ``hypothesis``, and a
    tolerance window, computes a list of pairings
    ``(i, j)`` where ``|reference[i] - hypothesis[j]| <= tolerance``.

    This is equivalent to, but more efficient than the following:

    >>> hit_reference, hit_hypothesis = np.where(np.abs(np.subtract.outer(reference, hypothesis)) <= tolerance)

    Parameters
    ----------
    reference : np.ndarray, shape=(n,)
        Array of reference values
    hypothesis : np.ndarray, shape=(m,)
        Array of hypothesis values
    tolerance : float >= 0
        Tolerance, used to create a window
        around times :math:`t_0` in ``reference``.

    Returns
    -------
    hit_reference : list
    hit_hypothesis : list
        indices such that ``|hit_reference[i] - hit_hypothesis[i]| <= tolerance``

    Notes
    -----
    This function is adapted from of ``mir_eval``, under MIT license:
    https://github.com/craffel/mir_eval
    """
    reference = np.asarray(reference)
    hypothesis = np.asarray(hypothesis)
    ref_idx = np.argsort(reference)
    ref_sorted = reference[ref_idx]

    left_idx = np.searchsorted(ref_sorted, hypothesis - tolerance, side='left')
    right_idx = np.searchsorted(ref_sorted, hypothesis + tolerance, side='right')

    hit_ref, hit_hyp = [], []
    # TODO: test whether we can vectorize
    # something like
    # diff = right_idx - left_idx
    # ``hit_hyp = np.where(diff != 0)`` would give us the ID of hits,
    # but wouldn't count multiple hits for a single estimated ref;
    # ``np.unique(ref_sorted[hit_hyp])`` gives us ID of hits in ref, but not matched with hits in hyp.
    # see also https://stackoverflow.com/a/51747164/4906855 for a similar approach
    for j, (start, end) in enumerate(zip(left_idx, right_idx)):
        hit_ref.extend(ref_idx[start:end])
        hit_hyp.extend([j] * (end - start))

    return hit_ref, hit_hyp


def bipartite_match(graph: dict) -> dict:
    """Find maximum cardinality matching of a bipartite graph (U,V,E).
    The input format is a dictionary mapping members of U to a list
    of their neighbors in V.

    The output is a dict M mapping members of V to their matches in U.

    Parameters
    ----------
    graph : dictionary : left-vertex -> list of right vertices
        The input bipartite graph.  Each edge need only be specified once.

    Returns
    -------
    matching : dictionary : right-vertex -> left vertex
        A maximal bipartite matching.

    Notes
    -----
    This function is adapted from of ``mir_eval``, under MIT license:
    https://github.com/craffel/mir_eval

    The implementation of :func:`bipartite_match` is adapted from:
    Hopcroft-Karp bipartite max-cardinality matching and max independent set
    David Eppstein, UC Irvine, 27 Apr 2002
    """
    # initialize greedy matching (redundant, but faster than full search)
    matching = {}
    for u in graph:
        for v in graph[u]:
            if v not in matching:
                matching[v] = u
                break

    while True:
        # structure residual graph into layers
        # pred[u] gives the neighbor in the previous layer for u in U
        # preds[v] gives a list of neighbors in the previous layer for v in V
        # unmatched gives a list of unmatched vertices in final layer of V,
        # and is also used as a flag value for pred[u] when u is in the first
        # layer
        preds = {}
        unmatched = []
        pred = dict([(u, unmatched) for u in graph])
        for v in matching:
            del pred[matching[v]]
        layer = list(pred)

        # repeatedly extend layering structure by another pair of layers
        while layer and not unmatched:
            new_layer = {}
            for u in layer:
                for v in graph[u]:
                    if v not in preds:
                        new_layer.setdefault(v, []).append(u)
            layer = []
            for v in new_layer:
                preds[v] = new_layer[v]
                if v in matching:
                    layer.append(matching[v])
                    pred[matching[v]] = v
                else:
                    unmatched.append(v)

        # did we finish layering without finding any alternating paths?
        if not unmatched:
            unlayered = {}
            for u in graph:
                for v in graph[u]:
                    if v not in preds:
                        unlayered[v] = None
            return matching

        def recurse(v):
            """Recursively search backward through layers to find alternating
            paths.  recursion returns true if found path, false otherwise
            """
            if v in preds:
                L = preds[v]
                del preds[v]
                for u in L:
                    if u in pred:
                        pu = pred[u]
                        del pred[u]
                        if pu is unmatched or recurse(pu):
                            matching[v] = u
                            return True
            return False

        for v in unmatched:
            recurse(v)


def match_event_times(hypothesis: npt.NDArray, reference: npt.NDArray,
                      tolerance: float | int | None = None, decimals: int | None = None
                      ) -> list[tuple]:
    """Compute a maximum matching between reference and estimated event times,
    where matches are found with a specified tolerance.

    Given two lists of event times ``ref`` and ``hypothesis``, we seek the largest set
    of correspondences ``:math:(ref_i, hypothesis_j)`` such that
    :math:`abs(ref_i - hypothesis_j) <= tolerance``, and each
    :math:`ref_i` and :math:`hypothesis_j` is matched at most once.

    This is useful for computing precision/recall metrics in beat tracking,
    onset detection, and segmentation.

    Parameters
    ----------
    hypothesis : numpy.ndarray
        Array of estimated event times with shape :math:`m`.
    reference : numpy.ndarray
        Array of reference event times with shape :math:`n`.
    tolerance : float
        Size of the tolerance around values in ``reference``, non-negative number.

    Returns
    -------
    matches : list of tuples
        A list of matched reference and event numbers.
        ``matches[i] == (i, j)`` where ``ref[i]`` is matched with ``hyp[j]``.

    Notes
    -----
    This function is adapted from of ``mir_eval``, under MIT license:
    https://github.com/craffel/mir_eval

    The addition of a tolerance parameter is based on [1]_.
    This is also sometimes known as a "collar" [2]_ or "forgiveness collar" [3]_.
    The value for the tolerance is usually determined by visual inspection
    of the distribution; see for example [4]_.

    References
    ----------
    .. [1] Kemp, T., Schmidt, M., Whypphal, M., & Waibel, A. (2000, June).
    Strategies for automatic segmentation of audio data.
    In 2000 ieee international conference on acoustics, speech, and signal processing.
    proceedings (cat. no. 00ch37100) (Vol. 3, pp. 1423-1426). IEEE.

    .. [2] Jordán, P. G., & Giménez, A. O. (2023).
    Advances in Binary and Multiclass Audio Segmentation with Deep Learning Techniques.

    .. [3] NIST. (2009). The 2009 (RT-09) Rich Transcription Meeting Recognition Evaluation Plan.
    https://web.archive.org/web/20100606041157if_/http://www.itl.nist.gov/iad/mig/thyps/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf

    .. [4] Du, P., & Troyer, T. W. (2006).
    A segmentation algorithm for zebra finch song at the note level.
    Neurocomputing, 69(10-12), 1375-1379.
    """
    validators.is_valid_boundaries_array(hypothesis)  # 1-d, non-negative, strictly increasing
    validators.is_valid_boundaries_array(reference)
    validators.have_same_dtype(hypothesis, reference)

    if tolerance is None:
        if issubclass(reference.dtype.type, np.floating):
            tolerance = 0.
        elif issubclass(reference.dtype.type, np.integer):
            tolerance = 0

    if tolerance < 0:
        raise ValueError(
            f"``tolerance`` must be a non-negative number but was: {tolerance}"
        )

    if decimals and (decimals is not False and not isinstance(decimals, int)):
        raise ValueError(
            f"``decimals`` must either be ``False`` or an integer but was: {decimals}"
        )

    if issubclass(reference.dtype.type, np.floating):
        if not isinstance(tolerance, float):
            raise TypeError(
                "If ``hypothesis`` and ``reference`` are floating, tolerance must be a float also, "
                f"but type was: {type(tolerance)}"
            )
        if decimals is None:
            decimals = 3

        if decimals < 0:
            raise ValueError(
                f"``decimals`` must be a non-negative number but was: {decimals}"
            )

        if decimals is not False:
            # we assume float values are in units of seconds and round to ``decimals``,
            # the default is 3 to indicate "milliseconds"
            reference = np.round(reference, decimals=decimals)
            hypothesis = np.round(hypothesis, decimals=decimals)

    if issubclass(reference.dtype.type, np.integer):
        if not isinstance(tolerance, int):
            raise TypeError(
                "If ``hypothesis`` and ``reference`` are integers, tolerance must be an integer also, "
                f"but type was: {type(tolerance)}"
            )
        if decimals is not None:
            raise ValueError(
                f"Cannot specify a ``decimals`` value when dtype of arrays is int"
            )

    # TODO: test whether it would be faster to just use np.where(np.abs(np.subtract.outer(ref, hyp)) <= tolerance)
    hits: tuple[list, list] = fast_hit_windows(reference, hypothesis, tolerance)

    # TODO: test whether we can vectorize with ``scipy.sparse.csgraph.maximum_bipartite_matching``
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.maximum_bipartite_matching
    # Construct the graph input
    G = defaultdict(list)
    for ref_i, hypothesis_i in zip(*hits):
        G[hypothesis_i].append(ref_i)

    # Compute the maximum matching
    matches: dict = bipartite_match(G)  # dict that maps hypothesis -> reference
    matches: list = sorted(matches.items())  # tuples mapping reference -> hypothesis

    return matches
