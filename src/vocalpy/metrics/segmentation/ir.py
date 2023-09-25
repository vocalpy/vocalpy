"""Metrics for segmentation adapted from information retrieval."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ... import validators


__all__ = [
    'compute_true_positives'
]


def compute_true_positives(hypothesis: npt.NDArray, reference: npt.NDArray,
                           tolerance: float | int | None = None, decimals: int | bool | None = None):
    r"""Helper function to compute number of true positives.
    This is used by :func:`~vocalpymetrics.segmentation.ir.precision_recall_fscore`.

    Parameters
    ----------
    hypothesis : numpy.ndarray
        Boundaries, e.g., onsets or offsets of segments,
        as computed by some method.
    reference : numpy.ndarray
        Ground truth boundaries that the hypothesized
        boundaries ``hypothesis`` are compared to.
    tolerance : float or int
        Tolerance :math:`\Delta t` in seconds.
        Elements in ``hypothesis`` are considered
        a true positive if they are within a time interval
        around any reference boundary :math:`t_0`
        in ``reference`` plus or minus
        the ``tolerance``, i.e.,
        if a hypothesized boundary :math:`t_h`
        is within the interval
        :math:`t_0 - \Delta t < t < t_0 + \Delta t`.
        Default is None,
        in which case it is set to ``0``
        (either float or int, depending on the
        dtype of ``hypothesis`` and ``reference``).
        See notes for more detail.
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

    The addition of a tolerance parameter is based on [1]_.
    This is also sometimes known as a "collar" [2]_ or "forgiveness collar" [3]_.
    The value for the tolerance is usually determined by visual inspection
    of the distribution; see for example [4]_.

    References
    ----------
    .. [1] Kemp, T., Schmidt, M., Westphal, M., & Waibel, A. (2000, June).
    Strategies for automatic segmentation of audio data.
    In 2000 ieee international conference on acoustics, speech, and signal processing.
    proceedings (cat. no. 00ch37100) (Vol. 3, pp. 1423-1426). IEEE.

    .. [2] Jordán, P. G., & Giménez, A. O. (2023).
    Advances in Binary and Multiclass Audio Segmentation with Deep Learning Techniques.

    .. [3] NIST. (2009). The 2009 (RT-09) Rich Transcription Meeting Recognition Evaluation Plan.
    https://web.archive.org/web/20100606041157if_/http://www.itl.nist.gov/iad/mig/tests/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf

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
    hits = np.nonzero(is_within_tolerance)[0]
    n_tp = is_within_tolerance.sum()

    return n_tp, hits


def precision_recall_fscore(hypothesis: npt.NDArray, reference: npt.NDArray, metric: str,
               tolerance: float | int | None = None, decimals: int | bool | None = None):
    r"""Helper function that computes precision, recall, and the F-score.

    Since all these metrics require computing the number of true positives,
    and F-score is a combination of precision and recall,
    we rely on this helper function to compute them.
    You can compute each directly without needing the ``metric`` argument
    that this function requires by calling the appropriate function:
    :func:`~vocalpy.metrics.segmentation.ir.precision`,
    :func:`~vocalpy.metrics.segmentation.ir.recall`, and
    :func:`~vocalpy.metrics.segmentation.ir.fscore`.
    See those functions for definitions of the metrics
    in terms of segmentation algorithms.

    Parameters
    ----------
    hypothesis : numpy.ndarray
        Boundaries, e.g., onsets or offsets of segments,
        as computed by some method.
    reference : numpy.ndarray
        Ground truth boundaries that the hypothesized
        boundaries ``hypothesis`` are compared to.
    metric : str
        The name of the metric to compute.
        One of: ``{"precision", "recall", "fscore"}``.
    tolerance : float or int
        Tolerance, in seconds.
        Elements in ``hypothesis`` are considered
        a true positive if they are within a time interval
        around any reference boundary :math:`t_0`
        in ``reference`` plus or minus
        the ``tolerance``, i.e.,
        if a hypothesized boundary :math:`t_h`
        is within the interval
        :math:`t_0 - \Delta t < t < t_0 + \Delta t`.
        Default is None,
        in which case it is set to ``0``
        (either float or int, depending on the
        dtype of ``hypothesis`` and ``reference``).
        See notes for more detail.
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
    metric_value : float
        Value for ``metric``.
    n_tp : int
        The number of true positives.
    hits : numpy.ndarray
        The indices of the true positives.

    Notes
    -----
    The addition of a tolerance parameter is based on [1]_.
    This is also sometimes known as a "collar" [2]_ or "forgiveness collar" [3]_.
    The value for the tolerance is usually determined by visual inspection
    of the distribution; see for example [4]_.

    References
    ----------
    .. [1] Kemp, T., Schmidt, M., Westphal, M., & Waibel, A. (2000, June).
    Strategies for automatic segmentation of audio data.
    In 2000 ieee international conference on acoustics, speech, and signal processing.
    proceedings (cat. no. 00ch37100) (Vol. 3, pp. 1423-1426). IEEE.

    .. [2] Jordán, P. G., & Giménez, A. O. (2023).
    Advances in Binary and Multiclass Audio Segmentation with Deep Learning Techniques.

    .. [3] NIST. (2009). The 2009 (RT-09) Rich Transcription Meeting Recognition Evaluation Plan.
    https://web.archive.org/web/20100606041157if_/http://www.itl.nist.gov/iad/mig/tests/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf

    .. [4] Du, P., & Troyer, T. W. (2006).
    A segmentation algorithm for zebra finch song at the note level.
    Neurocomputing, 69(10-12), 1375-1379.
    """
    if metric not in {"precision", "recall", "fscore"}:
        raise ValueError(
            f"``metric`` must be one of: {{\"precision\", \"recall\", \"fscore\"}} but was: {metric}"
        )

    n_tp, hits = compute_true_positives(hypothesis, reference, tolerance, decimals)
    if metric == "precision":
        precision_ = n_tp / hypothesis.size
        return precision_, n_tp, hits
    elif metric == "recall":
        recall_ = n_tp / reference.size
        return recall_, n_tp, hits
    elif metric == "fscore":
        precision_ = n_tp / hypothesis.size
        recall_ = n_tp / reference.size
        if np.isclose(precision_, 0.0) and np.isclose(recall_, 0.0):
            # avoids divide-by-zero that would give NaN
            return 0., n_tp, hits
        fscore_ = 2 * (precision_ * recall_) / (precision_ + recall_)
        return fscore_, n_tp, hits


    # avoid divide-by-zero that would give NaN
def _precision(hypothesis: npt.NDArray, reference: npt.NDArray,
               tolerance: float | int | None = None, decimals: int | bool | None = None):
    r"""Helper function to compute precision :math:`P`
    given a hypothesized vector of boundaries ``hypothesis``
    returned by a segmentation algorithm
    and a reference vector of boundaries ``reference``,
    e.g., boundaries cleaned by a human expert
    or boundaries from a benchmark dataset.

    Precision is defined as the number of true positives (:math:`T_p`)
    over the number of true positives
    plus the number of false positives (:math:`F_p`).

    :math:`P = \\frac{T_p}{T_p+F_p}`.

    The number of true positives ``n_tp`` is computed by calling
    :func:`vocalpy.metrics.segmentation.ir.compute_true_positives`.
    This function then computes the precision as
    ``precision = n_tp / hypothesis.size``.

    Parameters
    ----------
    hypothesis : numpy.ndarray
        Boundaries, e.g., onsets or offsets of segments,
        as computed by some method.
    reference : numpy.ndarray
        Ground truth boundaries that the hypothesized
        boundaries ``hypothesis`` are compared to.
    tolerance : float or int
        Tolerance, in seconds.
        Elements in ``hypothesis`` are considered
        a true positive if they are within a time interval
        around any reference boundary :math:`t_0`
        in ``reference`` plus or minus
        the ``tolerance``, i.e.,
        if a hypothesized boundary :math:`t_h`
        is within the interval
        :math:`t_0 - \Delta t < t < t_0 + \Delta t`.
        Default is None,
        in which case it is set to ``0``
        (either float or int, depending on the
        dtype of ``hypothesis`` and ``reference``).
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
    precision : float
        Value for precision, computed as described above.
    n_tp : int
        The number of true positives.
    hits : numpy.ndarray
        The indices of the true positives.

    Notes
    -----
    The addition of a tolerance parameter is based on [1]_.
    This is also sometimes known as a "collar" [2]_ or "forgiveness collar" [3]_.
    The value for the tolerance is usually determined by visual inspection
    of the distribution; see for example [4]_.

    References
    ----------
    .. [1] Kemp, T., Schmidt, M., Westphal, M., & Waibel, A. (2000, June).
    Strategies for automatic segmentation of audio data.
    In 2000 ieee international conference on acoustics, speech, and signal processing.
    proceedings (cat. no. 00ch37100) (Vol. 3, pp. 1423-1426). IEEE.

    .. [2] Jordán, P. G., & Giménez, A. O. (2023).
    Advances in Binary and Multiclass Audio Segmentation with Deep Learning Techniques.

    .. [3] NIST. (2009). The 2009 (RT-09) Rich Transcription Meeting Recognition Evaluation Plan.
    https://web.archive.org/web/20100606041157if_/http://www.itl.nist.gov/iad/mig/tests/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf

    .. [4] Du, P., & Troyer, T. W. (2006).
    A segmentation algorithm for zebra finch song at the note level.
    Neurocomputing, 69(10-12), 1375-1379.
    """
    return precision_recall_fscore(hypothesis, reference, "precision", tolerance, decimals)


def _recall(hypothesis: npt.NDArray, reference: npt.NDArray,
            tolerance: float | int | None = None, decimals: int | bool | None = None):
    r"""Helper function to compute recall :math:`R`
    given a hypothesized vector of boundaries ``hypothesis``
    returned by a segmentation algorithm
    and a reference vector of boundaries ``reference``,
    e.g., boundaries cleaned by a human expert
    or boundaries from a benchmark dataset.

    Recall (:math:`R`) is defined as the number of true positives (:math:`T_p`)
    over the number of true positives plus the number of false negatives
    (:math:`F_n`).

    :math:`R = \\frac{T_p}{T_p + F_n}`

    The number of true positives ``n_tp`` is computed by calling
    :func:`vocalpy.metrics.segmentation.ir.compute_true_positives`.
    This function then computes the recall as
    ``recall = n_tp / reference.size``.

    Parameters
    ----------
    hypothesis : numpy.ndarray
        Boundaries, e.g., onsets or offsets of segments,
        as computed by some method.
    reference : numpy.ndarray
        Ground truth boundaries that the hypothesized
        boundaries ``hypothesis`` are compared to.
    tolerance : float or int
        Tolerance, in seconds.
        Elements in ``hypothesis`` are considered
        a true positive if they are within a time interval
        around any reference boundary :math:`t_0`
        in ``reference`` plus or minus
        the ``tolerance``, i.e.,
        if a hypothesized boundary :math:`t_h`
        is within the interval
        :math:`t_0 - \Delta t < t < t_0 + \Delta t`.
        Default is None,
        in which case it is set to ``0``
        (either float or int, depending on the
        dtype of ``hypothesis`` and ``reference``).
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
    recall : float
        Value for recall, computed as described above.
    n_tp : int
        The number of true positives.
    hits : numpy.ndarray
        The indices of the true positives.

    Notes
    -----
    The addition of a tolerance parameter is based on [1]_.
    This is also sometimes known as a "collar" [2]_ or "forgiveness collar" [3]_.
    The value for the tolerance is usually determined by visual inspection
    of the distribution; see for example [4]_.

    References
    ----------
    .. [1] Kemp, T., Schmidt, M., Westphal, M., & Waibel, A. (2000, June).
    Strategies for automatic segmentation of audio data.
    In 2000 ieee international conference on acoustics, speech, and signal processing.
    proceedings (cat. no. 00ch37100) (Vol. 3, pp. 1423-1426). IEEE.

    .. [2] Jordán, P. G., & Giménez, A. O. (2023).
    Advances in Binary and Multiclass Audio Segmentation with Deep Learning Techniques.

    .. [3] NIST. (2009). The 2009 (RT-09) Rich Transcription Meeting Recognition Evaluation Plan.
    https://web.archive.org/web/20100606041157if_/http://www.itl.nist.gov/iad/mig/tests/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf

    .. [4] Du, P., & Troyer, T. W. (2006).
    A segmentation algorithm for zebra finch song at the note level.
    Neurocomputing, 69(10-12), 1375-1379.
    """
    return precision_recall_fscore(hypothesis, reference, "recall", tolerance, decimals)


def _fscore(hypothesis: npt.NDArray, reference: npt.NDArray,
             tolerance: float | int | None = None, decimals: int | bool | None = None):
    r"""Helper function to compute the F-score
    given a hypothesized vector of boundaries ``hypothesis``
    returned by a segmentation algorithm
    and a reference vector of boundaries ``reference``,
    e.g., boundaries cleaned by a human expert
    or boundaries from a benchmark dataset.

    The F-score can be interpreted as a harmonic mean of the precision and
    recall, where an F-score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F-score are
    equal. The formula for the F-score is:

    ``f_score = 2 * (precision * recall) / (precision + recall)``

    Parameters
    ----------
    hypothesis : numpy.ndarray
        Boundaries, e.g., onsets or offsets of segments,
        as computed by some method.
    reference : numpy.ndarray
        Ground truth boundaries that the hypothesized
        boundaries ``hypothesis`` are compared to.
    tolerance : float or int
        Tolerance, in seconds.
        Elements in ``hypothesis`` are considered
        a true positive if they are within a time interval
        around any reference boundary :math:`t_0`
        in ``reference`` plus or minus
        the ``tolerance``, i.e.,
        if a hypothesized boundary :math:`t_h`
        is within the interval
        :math:`t_0 - \Delta t < t < t_0 + \Delta t`.
        Default is None,
        in which case it is set to ``0``
        (either float or int, depending on the
        dtype of ``hypothesis`` and ``reference``).
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
    f_score : float
        Value for F-score, computed as described above.
    n_tp : int
        The number of true positives.
    hits : numpy.ndarray
        The indices of the true positives.

    Notes
    -----
    The addition of a tolerance parameter is based on [1]_.
    This is also sometimes known as a "collar" [2]_ or "forgiveness collar" [3]_.
    The value for the tolerance is usually determined by visual inspection
    of the distribution; see for example [4]_.

    References
    ----------
    .. [1] Kemp, T., Schmidt, M., Westphal, M., & Waibel, A. (2000, June).
    Strategies for automatic segmentation of audio data.
    In 2000 ieee international conference on acoustics, speech, and signal processing.
    proceedings (cat. no. 00ch37100) (Vol. 3, pp. 1423-1426). IEEE.

    .. [2] Jordán, P. G., & Giménez, A. O. (2023).
    Advances in Binary and Multiclass Audio Segmentation with Deep Learning Techniques.

    .. [3] NIST. (2009). The 2009 (RT-09) Rich Transcription Meeting Recognition Evaluation Plan.
    https://web.archive.org/web/20100606041157if_/http://www.itl.nist.gov/iad/mig/tests/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf

    .. [4] Du, P., & Troyer, T. W. (2006).
    A segmentation algorithm for zebra finch song at the note level.
    Neurocomputing, 69(10-12), 1375-1379.
    """
    return precision_recall_fscore(hypothesis, reference, "fscore", tolerance, decimals)


def precision(onsets_hyp: npt.NDArray, offsets_hyp: npt.NDArray, onsets_ref: npt.NDArray, offsets_ref: npt.NDArray,
              tolerance: float | int | None = None, decimals: int | bool | None = None, method: str = "combine"
              ) -> tuple[float, int, npt.NDArray] | tuple[float, float, int, int, npt.NDArray, npt.NDArray]:
    r"""Compute precision :math:`P`
    given a hypothesized segmentation with
    onsets and offsets ``onset_hyp`` and offsets_hyp``
    and a reference segmentation
    with onsets and offsets ``onsets_ref`` and ``offsets_ref``

    For example: a hypothesized segmentation would be one
    with boundaries returned by a segmentation algorithm,
    and a reference segmentation would be one with
    boundaries cleaned by a human expert,
    or from a benchmark dataset.

    Precision is defined as the number of true positives (:math:`T_p`)
    over the number of true positives
    plus the number of false positives (:math:`F_p`).

    :math:`P = \\frac{T_p}{T_p+F_p}`.

    The number of true positives ``n_tp`` is computed by calling
    :func:`vocalpy.metrics.segmentation.ir.compute_true_positives`.
    This function then computes the precision as
    ``precision = n_tp / hypothesis.size``.

    Parameters
    ----------
    onsets_hyp : numpy.ndarray
        Onsets of segments, as computed by some method.
    offsets_hyp : numpy.ndarray
        Offsets of segments, as computed by some method.
    onsets_ref : numpy.ndarray
        Ground truth onsets of segments that the hypothesized
        ``onsets_hyp`` are compared to.
    offsets_ref : numpy.ndarray
        Ground truth offsets of segments that the hypothesized
        ``offsets_hyp`` are compared to.
    tolerance : float or int
        Tolerance, in seconds.
        Elements in ``hypothesis`` are considered
        a true positive if they are within a time interval
        around any reference boundary :math:`t_0`
        in ``reference`` plus or minus
        the ``tolerance``, i.e.,
        if a hypothesized boundary :math:`t_h`
        is within the interval
        :math:`t_0 - \Delta t < t < t_0 + \Delta t`.
        Default is None,
        in which case it is set to ``0``
        (either float or int, depending on the
        dtype of ``hypothesis`` and ``reference``).
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
    method : str
        One of ``{"combine", "separate"}``. If ``"combine"``,
        the onsets and offsets are concatenated,
        and a single precision value is returned.
        If ``"separate"``, precision is computed separately
        for the onsets and offsets, and two values are returned.

    Returns
    -------
    precision : float
        Value for precision, computed as described above.
        If ``method`` is ``"separate"`` then two values for
        precision will be returned, one for onsets
        and one for offsets.
    n_tp : int
        The number of true positives.
    hits : numpy.ndarray
        The indices of the true positives.

    Notes
    -----
    The addition of a tolerance parameter is based on [1]_.
    This is also sometimes known as a "collar" [2]_ or "forgiveness collar" [3]_.
    The value for the tolerance is usually determined by visual inspection
    of the distribution; see for example [4]_.

    References
    ----------
    .. [1] Kemp, T., Schmidt, M., Westphal, M., & Waibel, A. (2000, June).
    Strategies for automatic segmentation of audio data.
    In 2000 ieee international conference on acoustics, speech, and signal processing.
    proceedings (cat. no. 00ch37100) (Vol. 3, pp. 1423-1426). IEEE.

    .. [2] Jordán, P. G., & Giménez, A. O. (2023).
    Advances in Binary and Multiclass Audio Segmentation with Deep Learning Techniques.

    .. [3] NIST. (2009). The 2009 (RT-09) Rich Transcription Meeting Recognition Evaluation Plan.
    https://web.archive.org/web/20100606041157if_/http://www.itl.nist.gov/iad/mig/tests/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf

    .. [4] Du, P., & Troyer, T. W. (2006).
    A segmentation algorithm for zebra finch song at the note level.
    Neurocomputing, 69(10-12), 1375-1379.
    """
    if method == "combine":
        boundaries_hyp = np.sort(np.concatenate((onsets_hyp, offsets_hyp)))
        boundaries_ref = np.sort(np.concatenate((onsets_ref, offsets_ref)))
        precision_, n_tp, all_hits = _precision(boundaries_hyp, boundaries_ref, tolerance, decimals)
        return precision_, n_tp, all_hits
    elif method == "separate":
        precision_on, n_tp_on, all_hits_on = _precision(onsets_hyp, onsets_ref, tolerance, decimals)
        precision_off, n_tp_off, all_hits_off = _precision(offsets_hyp, offsets_ref, tolerance, decimals)
        return precision_on, precision_off, n_tp_on, n_tp_off, all_hits_on, all_hits_off


def recall(onsets_hyp: npt.NDArray, offsets_hyp: npt.NDArray, onsets_ref: npt.NDArray, offsets_ref: npt.NDArray,
           tolerance: float | int | None = None, decimals: int | bool | None = None, method: str = "combine"
           ) -> tuple[float, int, npt.NDArray] | tuple[float, float, int, int, npt.NDArray, npt.NDArray]:
    r"""Compute recall :math:`R`
    given a hypothesized segmentation with
    onsets and offsets ``onset_hyp`` and offsets_hyp``
    and a reference segmentation
    with onsets and offsets ``onsets_ref`` and ``offsets_ref``

    For example: a hypothesized segmentation would be one
    with boundaries returned by a segmentation algorithm,
    and a reference segmentation would be one with
    boundaries cleaned by a human expert,
    or from a benchmark dataset.

    Recall (:math:`R`) is defined as the number of true positives (:math:`T_p`)
    over the number of true positives plus the number of false negatives
    (:math:`F_n`).

    :math:`R = \\frac{T_p}{T_p + F_n}`

    The number of true positives ``n_tp`` is computed by calling
    :func:`vocalpy.metrics.segmentation.ir.compute_true_positives`.
    This function then computes the recall as
    ``recall = n_tp / reference.size``.

    Parameters
    ----------
    onsets_hyp : numpy.ndarray
        Onsets of segments, as computed by some method.
    offsets_hyp : numpy.ndarray
        Offsets of segments, as computed by some method.
    onsets_ref : numpy.ndarray
        Ground truth onsets of segments that the hypothesized
        ``onsets_hyp`` are compared to.
    offsets_ref : numpy.ndarray
        Ground truth offsets of segments that the hypothesized
        ``offsets_hyp`` are compared to.
    tolerance : float or int
        Tolerance, in seconds.
        Elements in ``hypothesis`` are considered
        a true positive if they are within a time interval
        around any reference boundary :math:`t_0`
        in ``reference`` plus or minus
        the ``tolerance``, i.e.,
        if a hypothesized boundary :math:`t_h`
        is within the interval
        :math:`t_0 - \Delta t < t < t_0 + \Delta t`.
        Default is None,
        in which case it is set to ``0``
        (either float or int, depending on the
        dtype of ``hypothesis`` and ``reference``).
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
    method : str
        One of ``{"combine", "separate"}``. If ``"combine"``,
        the onsets and offsets are concatenated,
        and a single precision value is returned.
        If ``"separate"``, precision is computed separately
        for the onsets and offsets, and two values are returned.

    Returns
    -------
    recall : float
        Value for recall, computed as described above.
    n_tp : int
        The number of true positives.
    hits : numpy.ndarray
        The indices of the true positives.

    Notes
    -----
    The addition of a tolerance parameter is based on [1]_.
    This is also sometimes known as a "collar" [2]_ or "forgiveness collar" [3]_.
    The value for the tolerance is usually determined by visual inspection
    of the distribution; see for example [4]_.

    References
    ----------
    .. [1] Kemp, T., Schmidt, M., Westphal, M., & Waibel, A. (2000, June).
    Strategies for automatic segmentation of audio data.
    In 2000 ieee international conference on acoustics, speech, and signal processing.
    proceedings (cat. no. 00ch37100) (Vol. 3, pp. 1423-1426). IEEE.

    .. [2] Jordán, P. G., & Giménez, A. O. (2023).
    Advances in Binary and Multiclass Audio Segmentation with Deep Learning Techniques.

    .. [3] NIST. (2009). The 2009 (RT-09) Rich Transcription Meeting Recognition Evaluation Plan.
    https://web.archive.org/web/20100606041157if_/http://www.itl.nist.gov/iad/mig/tests/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf

    .. [4] Du, P., & Troyer, T. W. (2006).
    A segmentation algorithm for zebra finch song at the note level.
    Neurocomputing, 69(10-12), 1375-1379.
    """
    if method == "combine":
        boundaries_hyp = np.sort(np.concatenate((onsets_hyp, offsets_hyp)))
        boundaries_ref = np.sort(np.concatenate((onsets_ref, offsets_ref)))
        recall_, n_tp, all_hits = _recall(boundaries_hyp, boundaries_ref, tolerance, decimals)
        return recall_, n_tp, all_hits
    elif method == "separate":
        recall_on, n_tp_on, all_hits_on = _recall(onsets_hyp, onsets_ref, tolerance, decimals)
        recall_off, n_tp_off, all_hits_off = _recall(offsets_hyp, offsets_ref, tolerance, decimals)
        return recall_on, recall_off, n_tp_on, n_tp_off, all_hits_on, all_hits_off


def fscore(onsets_hyp: npt.NDArray, offsets_hyp: npt.NDArray, onsets_ref: npt.NDArray, offsets_ref: npt.NDArray,
           tolerance: float | int | None = None, decimals: int | bool | None = None, method: str = "combine"
           ) -> tuple[float, int, npt.NDArray] | tuple[float, float, int, int, npt.NDArray, npt.NDArray]:
    r"""Compute precision :math:`P`
    given a hypothesized segmentation with
    onsets and offsets ``onset_hyp`` and offsets_hyp``
    and a reference segmentation
    with onsets and offsets ``onsets_ref`` and ``offsets_ref``

    For example: a hypothesized segmentation would be one
    with boundaries returned by a segmentation algorithm,
    and a reference segmentation would be one with
    boundaries cleaned by a human expert,
    or from a benchmark dataset.

    Recall (:math:`R`) is defined as the number of true positives (:math:`T_p`)
    over the number of true positives plus the number of false negatives
    (:math:`F_n`).

    :math:`R = \\frac{T_p}{T_p + F_n}`

    The number of true positives ``n_tp`` is computed by calling
    :func:`vocalpy.metrics.segmentation.ir.compute_true_positives`.
    This function then computes the recall as
    ``recall = n_tp / reference.size``.

    Parameters
    ----------
    onsets_hyp : numpy.ndarray
        Onsets of segments, as computed by some method.
    offsets_hyp : numpy.ndarray
        Offsets of segments, as computed by some method.
    onsets_ref : numpy.ndarray
        Ground truth onsets of segments that the hypothesized
        ``onsets_hyp`` are compared to.
    offsets_ref : numpy.ndarray
        Ground truth offsets of segments that the hypothesized
        ``offsets_hyp`` are compared to.
    tolerance : float or int
        Tolerance, in seconds.
        Elements in ``hypothesis`` are considered
        a true positive if they are within a time interval
        around any reference boundary :math:`t_0`
        in ``reference`` plus or minus
        the ``tolerance``, i.e.,
        if a hypothesized boundary :math:`t_h`
        is within the interval
        :math:`t_0 - \Delta t < t < t_0 + \Delta t`.
        Default is None,
        in which case it is set to ``0``
        (either float or int, depending on the
        dtype of ``hypothesis`` and ``reference``).
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
    method : str
        One of ``{"combine", "separate"}``. If ``"combine"``,
        the onsets and offsets are concatenated,
        and a single precision value is returned.
        If ``"separate"``, precision is computed separately
        for the onsets and offsets, and two values are returned.

    Returns
    -------
    f_score : float
        Value for F-score, computed as described above.
    n_tp : int
        The number of true positives.
    hits : numpy.ndarray
        The indices of the true positives.

    Notes
    -----
    The addition of a tolerance parameter is based on [1]_.
    This is also sometimes known as a "collar" [2]_ or "forgiveness collar" [3]_.
    The value for the tolerance is usually determined by visual inspection
    of the distribution; see for example [4]_.

    References
    ----------
    .. [1] Kemp, T., Schmidt, M., Westphal, M., & Waibel, A. (2000, June).
    Strategies for automatic segmentation of audio data.
    In 2000 ieee international conference on acoustics, speech, and signal processing.
    proceedings (cat. no. 00ch37100) (Vol. 3, pp. 1423-1426). IEEE.

    .. [2] Jordán, P. G., & Giménez, A. O. (2023).
    Advances in Binary and Multiclass Audio Segmentation with Deep Learning Techniques.

    .. [3] NIST. (2009). The 2009 (RT-09) Rich Transcription Meeting Recognition Evaluation Plan.
    https://web.archive.org/web/20100606041157if_/http://www.itl.nist.gov/iad/mig/tests/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf

    .. [4] Du, P., & Troyer, T. W. (2006).
    A segmentation algorithm for zebra finch song at the note level.
    Neurocomputing, 69(10-12), 1375-1379.
    """
    if method == "combine":
        boundaries_hyp = np.sort(np.concatenate((onsets_hyp, offsets_hyp)))
        boundaries_ref = np.sort(np.concatenate((onsets_ref, offsets_ref)))
        fscore_, n_tp, all_hits = _fscore(boundaries_hyp, boundaries_ref, tolerance, decimals)
        return fscore_, n_tp, all_hits
    elif method == "separate":
        fscore_on, n_tp_on, all_hits_on = _fscore(onsets_hyp, onsets_ref, tolerance, decimals)
        fscore_off, n_tp_off, all_hits_off = _fscore(offsets_hyp, offsets_ref, tolerance, decimals)
        return fscore_on, fscore_off, n_tp_on, n_tp_off, all_hits_on, all_hits_off
