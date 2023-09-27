"""Metrics for segmentation adapted from information retrieval."""
from __future__ import annotations

import attr
import numpy as np
import numpy.typing as npt

from ... import validators


def find_hits(hypothesis: npt.NDArray, reference: npt.NDArray,
              tolerance: float | int | None = None, decimals: int | None = None
              ) -> tuple[npt.NDArray, npt.NDArray]:
    """Find hits in arrays of event times.

    This is a helper function used to compute information retrieval metrics.
    Specifically, this function is called by
    :func:`~vocalpy.metrics.segmentation.ir.precision_recall_fscore`.

    An element in ``hypothesis``, is considered a hit
    if its value :math:`t_h` falls within an interval around
    any value in ``reference``, :math:`t_0`, plus or minus ``tolerance``

    :math:`t_0 - \Delta t < t < t_0 + \Delta t`

    This function only allows there to be zero or one hit
    for each element in ``reference``, but not more than one.
    If the condition :math:`|ref_i - hyp_j| < tolerance`
    is true for multiple values :math:`hyp_j` in ``hypothesis``,
    then the value with the smallest difference from :math:`ref_i`
    is considered a hit.

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
    hits_ref : numpy.ndarray
        The indices of hits in ``reference``.
    hits_hyp : numpy.ndarray
        The indices of hits in ``hypothesis``.
    diffs : numpy.ndarray
        Absolute differences :math:`|hit^{ref}_i - hit^{hyp}_i|`,
        i.e., ``np.abs(reference[hits_ref] - hypothesis[hits_hyp])``.
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

    diffs = np.abs(np.subtract.outer(reference, hypothesis))
    in_window = diffs <= tolerance
    hits_ref, hits_hyp = np.where(in_window)

    # now force there to be only one hit in hyp for each hit in ref;
    # we do this by choosing the hit that has the smallest absolute difference
    diffs_in_window = diffs[hits_ref, hits_hyp]
    hits_ref_uniq, uniq_idx_ref, counts_ref = np.unique(hits_ref, return_index=True, return_counts=True)
    have_duplicates_ref = counts_ref > 1
    if not np.any(have_duplicates_ref):
        # no duplicates, we can return
        return hits_ref, hits_hyp, diffs_in_window
    else:
        # For each unique value in ref, we need to find the value in
        # hyp that has the smallest distance from it. This is the hit we keep.
        hits_hyp_out = []
        diffs_out = []
        for uniq_ind, ref_uniq in enumerate(hits_ref_uniq):
            if not have_duplicates_ref[uniq_ind]:
                ind_in_hits = uniq_idx_ref[uniq_ind]
                hits_hyp_out.append(hits_hyp[ind_in_hits])
                diffs_out.append(diffs_in_window[ind_in_hits])
            else:
                dupe_inds = np.where(hits_ref == ref_uniq)[0]
                min_diff_ind = np.argmin(diffs_in_window[dupe_inds])
                hits_hyp_out.append(hits_hyp[dupe_inds][min_diff_ind])
                diffs_out.append(diffs_in_window[dupe_inds][min_diff_ind])
        return hits_ref_uniq, np.array(hits_hyp_out), np.array(diffs_out)


@attr.define
class IRMetricData:
    """Class representing data used to compute 
    an information retrieval metric.
    
    This class contains data 
    needed to compute metrics like precision and recall
    for estimated event times
    compared to reference event times.
    
    The class attributes are the variables 
    returned by 
    :func:`vocalpy.metrics.segmentation.find_hits`.
    Instances of this class are returned by 
    along with the value of the computed metrics.

    The values can be useful when computing 
    additional statistics,
    e.g., the classes of segments that had higher 
    or lower precision or recall, 
    or the distribution of 
    differences between reference times 
    and estimated times for some class of events. 

    Attributes
    ----------
    hits_ref : numpy.ndarray
        The indices of hits in ``reference``.
    hits_hyp : numpy.ndarray
        The indices of hits in ``hypothesis``.
    diffs : numpy.ndarray
        Absolute differences :math:`|hit^{ref}_i - hit^{hyp}_i|`,
        i.e., ``np.abs(reference[hits_ref] - hypothesis[hits_hyp])``.
    """
    hits_ref: npt.NDArray
    hits_hyp: npt.NDArray
    diffs: npt.NDArray



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
    See docstrings of those functions for definitions of the metrics
    in terms of segmentation algorithms.

    Precision, recall, and F-score are computed using matches found with
    :func:`vocalpy.metrics.segmentation._ir_helper.match_even_times`.
    See docstring of that function for more detail on how matches are computed.

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
    metric_data : IRMetricData
        Instance of :class:`IRMetricData` 
        with indices of hits in both 
        ``hypothesis`` and ``reference``,
        and the absolute difference between times
        in ``hypothesis`` and ``reference``
        for the hits.

    Notes
    -----
    The addition of a tolerance parameter is based on [1]_.
    This is also sometimes known as a "collar" [2]_ or "forgiveness collar" [3]_.
    The value for the tolerance can be determined by visual inspection
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
    if metric not in {"precision", "recall", "fscore"}:
        raise ValueError(
            f"``metric`` must be one of: {{\"precision\", \"recall\", \"fscore\"}} but was: {metric}"
        )

    # If we have no boundaries, we get no score.
    if len(reference) == 0 or len(hypothesis) == 0:
        return 0.0, 0, IRMetricData()

    hits_ref, hits_hyp, diffs = find_hits(hypothesis, reference, tolerance, decimals)
    metric_data = IRMetricData(hits_ref, hits_hyp, diffs)
    n_tp = hits_hyp.size
    if metric == "precision":
        precision_ = n_tp / hypothesis.size
        return precision_, n_tp, metric_data
    elif metric == "recall":
        recall_ = n_tp / reference.size
        return recall_, n_tp, metric_data
    elif metric == "fscore":
        precision_ = n_tp / hypothesis.size
        recall_ = n_tp / reference.size
        if np.isclose(precision_, 0.0) and np.isclose(recall_, 0.0):
            # avoids divide-by-zero that would give NaN
            return 0., n_tp, metric_data
        fscore_ = 2 * (precision_ * recall_) / (precision_ + recall_)
        return fscore_, n_tp, metric_data


def _precision(hypothesis: npt.NDArray, reference: npt.NDArray,
               tolerance: float | int | None = None, decimals: int | bool | None = None
               ) -> tuple[float, int, IRMetricData]:
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
    metric_data : IRMetricData
        Instance of :class:`IRMetricData`
        with indices of hits in both
        ``hypothesis`` and ``reference``,
        and the absolute difference between times
        in ``hypothesis`` and ``reference``
        for the hits.

    Notes
    -----
    The addition of a tolerance parameter is based on [1]_.
    This is also sometimes known as a "collar" [2]_ or "forgiveness collar" [3]_.
    The value for the tolerance can be determined by visual inspection
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
    return precision_recall_fscore(hypothesis, reference, "precision", tolerance, decimals)


def _recall(hypothesis: npt.NDArray, reference: npt.NDArray,
            tolerance: float | int | None = None, decimals: int | bool | None = None
            ) -> tuple[float, int, IRMetricData]:
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
    metric_data : IRMetricData
        Instance of :class:`IRMetricData`
        with indices of hits in both
        ``hypothesis`` and ``reference``,
        and the absolute difference between times
        in ``hypothesis`` and ``reference``
        for the hits.

    Notes
    -----
    The addition of a tolerance parameter is based on [1]_.
    This is also sometimes known as a "collar" [2]_ or "forgiveness collar" [3]_.
    The value for the tolerance can be determined by visual inspection
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
    return precision_recall_fscore(hypothesis, reference, "recall", tolerance, decimals)


def _fscore(hypothesis: npt.NDArray, reference: npt.NDArray,
             tolerance: float | int | None = None, decimals: int | bool | None = None
            ) -> tuple[float, int, IRMetricData]:
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
    metric_data : IRMetricData
        Instance of :class:`IRMetricData`
        with indices of hits in both
        ``hypothesis`` and ``reference``,
        and the absolute difference between times
        in ``hypothesis`` and ``reference``
        for the hits.

    Notes
    -----
    The addition of a tolerance parameter is based on [1]_.
    This is also sometimes known as a "collar" [2]_ or "forgiveness collar" [3]_.
    The value for the tolerance can be determined by visual inspection
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
    return precision_recall_fscore(hypothesis, reference, "fscore", tolerance, decimals)


def precision(onsets_hyp: npt.NDArray, offsets_hyp: npt.NDArray, onsets_ref: npt.NDArray, offsets_ref: npt.NDArray,
              tolerance: float | int | None = None, decimals: int | bool | None = None, method: str = "combine"
              ) -> tuple[float, int, IRMetricData] | tuple[float, float, int, int, IRMetricData, IRMetricData]:
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
        If ``method`` is ``"separate"`` then two values
        will be returned, one for onsets and one for offsets.
    n_tp : int
        The number of true positives.
        If ``method`` is ``"separate"`` then two values
        will be returned, one for onsets and one for offsets.
    metric_data : IRMetricData
        Instance of :class:`IRMetricData`
        with indices of hits in both
        ``hypothesis`` and ``reference``,
        and the absolute difference between times
        in ``hypothesis`` and ``reference``
        for the hits.
        If ``method`` is ``"separate"`` then two values
        will be returned, one for onsets and one for offsets.

    Notes
    -----
    The addition of a tolerance parameter is based on [1]_.
    This is also sometimes known as a "collar" [2]_ or "forgiveness collar" [3]_.
    The value for the tolerance can be determined by visual inspection
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
    if method == "combine":
        boundaries_hyp = np.sort(np.concatenate((onsets_hyp, offsets_hyp)))
        boundaries_ref = np.sort(np.concatenate((onsets_ref, offsets_ref)))
        precision_, n_tp, metric_data = _precision(boundaries_hyp, boundaries_ref, tolerance, decimals)
        return precision_, n_tp, metric_data
    elif method == "separate":
        precision_on, n_tp_on, metric_data_on = _precision(onsets_hyp, onsets_ref, tolerance, decimals)
        precision_off, n_tp_off, metric_data_off = _precision(offsets_hyp, offsets_ref, tolerance, decimals)
        return precision_on, precision_off, n_tp_on, n_tp_off, metric_data_on, metric_data_off


def recall(onsets_hyp: npt.NDArray, offsets_hyp: npt.NDArray, onsets_ref: npt.NDArray, offsets_ref: npt.NDArray,
           tolerance: float | int | None = None, decimals: int | bool | None = None, method: str = "combine"
           ) -> tuple[float, int, IRMetricData] | tuple[float, float, int, int, IRMetricData, IRMetricData]:
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
        If ``method`` is ``"separate"`` then two values
        will be returned, one for onsets and one for offsets.
    n_tp : int
        The number of true positives.
        If ``method`` is ``"separate"`` then two values
        will be returned, one for onsets and one for offsets.
    metric_data : IRMetricData
        Instance of :class:`IRMetricData`
        with indices of hits in both
        ``hypothesis`` and ``reference``,
        and the absolute difference between times
        in ``hypothesis`` and ``reference``
        for the hits.
        If ``method`` is ``"separate"`` then two values
        will be returned, one for onsets and one for offsets.

    Notes
    -----
    The addition of a tolerance parameter is based on [1]_.
    This is also sometimes known as a "collar" [2]_ or "forgiveness collar" [3]_.
    The value for the tolerance can be determined by visual inspection
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
    if method == "combine":
        boundaries_hyp = np.sort(np.concatenate((onsets_hyp, offsets_hyp)))
        boundaries_ref = np.sort(np.concatenate((onsets_ref, offsets_ref)))
        recall_, n_tp, metric_data = _recall(boundaries_hyp, boundaries_ref, tolerance, decimals)
        return recall_, n_tp, metric_data
    elif method == "separate":
        recall_on, n_tp_on, metric_data_on = _recall(onsets_hyp, onsets_ref, tolerance, decimals)
        recall_off, n_tp_off, metric_data_off = _recall(offsets_hyp, offsets_ref, tolerance, decimals)
        return recall_on, recall_off, n_tp_on, n_tp_off, metric_data_on, metric_data_off


def fscore(onsets_hyp: npt.NDArray, offsets_hyp: npt.NDArray, onsets_ref: npt.NDArray, offsets_ref: npt.NDArray,
           tolerance: float | int | None = None, decimals: int | bool | None = None, method: str = "combine"
           ) -> tuple[float, int, IRMetricData] | tuple[float, float, int, int, IRMetricData, IRMetricData]:
    r"""Compute the F-score
    given a hypothesized segmentation with
    onsets and offsets ``onset_hyp`` and offsets_hyp``
    and a reference segmentation
    with onsets and offsets ``onsets_ref`` and ``offsets_ref``.

    For example: a hypothesized segmentation would be one
    with boundaries returned by a segmentation algorithm,
    and a reference segmentation would be one with
    boundaries cleaned by a human expert,
    or from a benchmark dataset.

    The F-score can be interpreted as a harmonic mean of the precision and
    recall, where an F-score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F-score are
    equal. The formula for the F-score is:

    ``f_score = 2 * (precision * recall) / (precision + recall)``

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
        If ``method`` is ``"separate"`` then two values
        will be returned, one for onsets and one for offsets.
    n_tp : int
        The number of true positives.
        If ``method`` is ``"separate"`` then two values
        will be returned, one for onsets and one for offsets.
    metric_data : IRMetricData
        Instance of :class:`IRMetricData`
        with indices of hits in both
        ``hypothesis`` and ``reference``,
        and the absolute difference between times
        in ``hypothesis`` and ``reference``
        for the hits.
        If ``method`` is ``"separate"`` then two values
        will be returned, one for onsets and one for offsets.

    Notes
    -----
    The addition of a tolerance parameter is based on [1]_.
    This is also sometimes known as a "collar" [2]_ or "forgiveness collar" [3]_.
    The value for the tolerance can be determined by visual inspection
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
    if method == "combine":
        boundaries_hyp = np.sort(np.concatenate((onsets_hyp, offsets_hyp)))
        boundaries_ref = np.sort(np.concatenate((onsets_ref, offsets_ref)))
        fscore_, n_tp, metric_data = _fscore(boundaries_hyp, boundaries_ref, tolerance, decimals)
        return fscore_, n_tp, metric_data
    elif method == "separate":
        fscore_on, n_tp_on, metric_data_on = _fscore(onsets_hyp, onsets_ref, tolerance, decimals)
        fscore_off, n_tp_off, metric_data_off = _fscore(offsets_hyp, offsets_ref, tolerance, decimals)
        return fscore_on, fscore_off, n_tp_on, n_tp_off, metric_data_on, metric_data_off
