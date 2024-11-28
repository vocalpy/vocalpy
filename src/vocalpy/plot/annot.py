"""Functions for plotting annotations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy.typing as npt

from ..annotation import Annotation


def labels(
    labels: npt.ArrayLike,
    t: npt.NDArray,
    t_shift_label: float = 0.01,
    y: float = 0.4,
    ax: plt.Axes | None = None,
    text_kwargs: dict | None = None,
):
    """Plot labels of segments on an axis.

    Parameters
    ----------
    labels : list, numpy.ndarray
    t : numpy.ndarray
        Times (in seconds) at which to plot labels
    t_shift_label : float
        Amount (in seconds) that labels should be shifted to the left, for centering.
        Necessary because width of text box isn't known until rendering.
    y : float, int
        Height on y-axis at which segments should be plotted.
        Default is 0.5.
    ax : matplotlib.axes.Axes
        Axes on which to plot segment. Default is None,
        in which case a new Axes instance is created.
    text_kwargs : dict
        Keyword arguments passed to the `Axes.text` method
        that plots the labels. Default is None.

    Returns
    -------
    text_list : list
        of text objections, the matplotlib.Text instances for each label
    """
    if text_kwargs is None:
        text_kwargs = {}

    if ax is None:
        fig, ax = plt.subplots

    text_list = []

    for label, t_lbl in zip(labels, t):
        t_lbl -= t_shift_label
        text = ax.text(t_lbl, y, label, **text_kwargs, label="label")
        text_list.append(text)

    return text_list


def segments(
    onsets: npt.NDArray,
    offsets: npt.NDArray,
    lbl: npt.NDArray | None,
    tlim: list | tuple | None = None,
    y_segments: float = 0.4,
    h_segments: float = 0.4,
    y_labels=0.3,
    ax: plt.Axes | None = None,
    label_color_map: dict | None = None,
    text_kwargs: dict | None = None,
) -> tuple[list[plt.Rectangle], list[plt.Text] | None]:
    """Plot segments on an axis.

    Creates rectangles
    with the specified `onsets` and `offsets`
    all at height `y_labels`
    and places them on the axes `ax`.
    If `labels` are supplied,
    these are plotted in the rectangles.

    Parameters
    ----------
    onsets : numpy.ndarray
        Onset times of segments.
    offsets : numpy.ndarray
        Offset times of segments.
    lbl : list, numpy.ndarray
        Labels of segments.
    y_segments : float, int
        Height on y-axis at which segments should be plotted.
        Default is 0.4.
    h_segments : float, int
        Height of rectangles that represent segments.
        Default is 0.4.
    y_labels : float, int
        Height on y-axis at which segment labels (if any) are plotted.
        Default is 0.4.
    ax : matplotlib.axes.Axes
        axes on which to plot segment. Default is None,
        in which case a new Axes instance is created
    label_color_map : dict, optional
        A :class:`dict` that maps string labels to colors
        (that are valid `color` arguments for matplotlib).
    text_kwargs : dict
        Keyword arguments passed to the `Axes.text` method
        that plots the labels. Default is None.
    """
    if ax is None:
        fig, ax = plt.subplots

    if label_color_map is None:
        if lbl is not None:
            labelset = set(lbl)
            cmap = plt.get_cmap("tab20")
            colors = [cmap(ind) for ind in range(len(labelset))]
            label_color_map = {
                label: color for label, color in zip(labelset, colors)
            }

    labels_to_plot = []
    label_plot_times = []
    rectangles = []
    if lbl is not None:
        zipped = zip(lbl, onsets, offsets)
    else:
        zipped = zip(onsets, offsets)

    for a_tuple in zipped:
        if lbl is not None:
            label, onset_s, offset_s = a_tuple
        else:
            onset_s, offset_s = a_tuple

        if tlim:
            if offset_s < tlim[0] or onset_s > tlim[1]:
                continue

        kwargs = {
            "width": offset_s - onset_s,
            "height": h_segments,
        }
        if lbl is not None:
            labels_to_plot.append(label)
            label_plot_times.append(onset_s + ((offset_s - onset_s) / 2))
            kwargs["facecolor"] = label_color_map[label]

        rectangle = plt.Rectangle((onset_s, y_segments), **kwargs)
        ax.add_patch(rectangle)
        rectangles.append(rectangle)

    if labels_to_plot is not None:
        text_list = labels(
            labels_to_plot,
            t=label_plot_times,
            y=y_labels,
            text_kwargs=text_kwargs,
            ax=ax,
        )
    else:
        text_list = None

    if tlim:
        ax.set_xlim(tlim)

    return rectangles, text_list


def annotation(
    annot: Annotation,
    tlim: tuple | list | None = None,
    y_segments: float = 0.5,
    h_segments: float = 0.4,
    y_labels: float = 0.3,
    text_kwargs: dict | None = None,
    ax: plt.Axes | None = None,
    label_color_map: dict | None = None,
) -> tuple[list[plt.Rectangle], list[plt.Text] | None]:
    """Plot a :class:`vocalpy.Annotation`.

    Parameters
    ----------
    annot : crowsetta.Annotation
        Annotation that has segments to be plotted
        (the `Annotation.data.seq.segments` attribute).
    tlim : tuple, list
        Limits of time axis (tmin, tmax) (i.e., x-axis).
        Default is None, in which case entire range of ``t`` will be plotted.
    y_segments : float
        Height at which segments should be plotted.
        Default is 0.5 (assumes y-limits of 0 and 1).
    h_segments : float, int
        Height of rectangles that represent segments.
        Default is 0.4.
    y_labels : float
        Height on y-axis at which segment labels (if any) are plotted.
        Default is 0.4.
    text_kwargs : dict
        Keyword arguments for :meth:`matplotlib.axes.Axes.text`.
        Passed to the function :func:`vocalpy.plot.annot.labels` that plots labels
        using ``Axes.text`` method. Default is None.
    ax : matplotlib.axes.Axes
        Axes on which to plot segments.
        Default is None, in which case
        a new figure with a single axes is created.
    label_color_map : dict, optional
        A :class:`dict` that maps string labels to colors
        (that are valid `color` arguments for matplotlib).
    text_kwargs : dict
        Keyword arguments passed to the `Axes.text` method
        that plots the labels. Default is None.
    """
    if not hasattr(annot.data, "seq"):
        raise ValueError(
            "Currently only annotations in sequence-like formats are supported.\n"
            "Please see this issue and give it a 'thumbs up' if support for bounding boxes would help you:\n"
            "https://github.com/vocalpy/vocalpy/issues/34"
        )

    if ax is None:
        fig, ax = plt.subplots()
        ax.set_ylim(0, 1)

    rectangles, text_list = segments(
        onsets=annot.data.seq.onsets_s,
        offsets=annot.data.seq.offsets_s,
        lbl=annot.data.seq.labels,
        tlim=tlim,
        y_segments=y_segments,
        h_segments=h_segments,
        y_labels=y_labels,
        ax=ax,
        label_color_map=label_color_map,
        text_kwargs=text_kwargs,
    )

    # FIXME: if we plot bounding boxes then we actually want yticks
    ax.set_yticks([])

    return rectangles, text_list
