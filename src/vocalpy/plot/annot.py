"""functions for plotting annotations for vocalizations"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.collections import LineCollection

from ..annotation import Annotation


def segments(
    onsets: npt.NDArray,
    offsets: npt.NDArray,
    y: float = 0.5,
    ax: plt.Axes | None = None,
    line_kwargs: dict | None = None,
) -> None:
    """Plot segments on an axis.

    Creates a collection of horizontal lines
    with the specified `onsets` and `offsets`
    all at height `y` and places them on the axes `ax`.

    Parameters
    ----------
    onsets : numpy.ndarray
        onset times of segments
    offsets : numpy.ndarray
        offset times of segments
    y : float, int
        height on y-axis at which segments should be plotted.
        Default is 0.5.
    ax : matplotlib.axes.Axes
        axes on which to plot segment. Default is None,
        in which case a new Axes instance is created
    line_kwargs : dict
        keyword arguments passed to the `LineCollection`
        that represents the segments. Default is None.
    """
    if line_kwargs is None:
        line_kwargs = {}

    if ax is None:
        fig, ax = plt.subplots
    segments = []
    for on, off in zip(onsets, offsets):
        segments.append(((on, y), (off, y)))
    lc = LineCollection(segments, **line_kwargs)
    ax.add_collection(lc)


def labels(labels: npt.ArrayLike, t: npt.NDArray, y=0.6, ax: plt.Axes | None = None, text_kwargs: dict | None = None):
    """Plot labels on an axis.

    Parameters
    ----------
    labels : list, numpy.ndarray
        Lables for units in sequence.
    t : numpy.ndarray
        Times at which to plot labels
    y : float, int
        Height on y-axis at which labels should be plotted.
        Default is 0.6 (in range (0., 1.)).
    ax : matplotlib.axes.Axes
        Axes on which to plot segment. Default is None,
        in which case a new Axes instance is created
    text_kwargs : dict
        Keyword arguments passed to the `Axes.text` method
        that plots the labels. Default is None.
    """
    if text_kwargs is None:
        text_kwargs = {}

    if ax is None:
        fig, ax = plt.subplots
    for label, t_lbl in zip(labels, t):
        ax.text(t_lbl, y, label, **text_kwargs)


def annotation(
    annot: Annotation,
    tlim: tuple | list | None = None,
    y_segments: float = 0.5,
    y_labels: float = 0.6,
    line_kwargs: dict | None = None,
    text_kwargs: dict | None = None,
    ax: plt.Axes | None = None,
) -> None:
    """Plot a :class:`vocalpy.Annotation`.

    Parameters
    ----------
    annot : crowsetta.Annotation
        Annotation that has segments to be plotted
        (the `Annotation.data.seq.segments` attribute).
    t : numpy.ndarray
        Vector of centers of time bins from spectrogram.
    tlim : tuple, list
        Limits of time axis (tmin, tmax) (i.e., x-axis).
        Default is None, in which case entire range of `t` will be plotted.
    y_segments : float
        Height at which segments should be plotted.
        Default is 0.5 (assumes y-limits of 0 and 1).
    y_labels : float
        Height at which labels should be plotted.
        Default is 0.6 (assumes y-limits of 0 and 1).
    line_kwargs : dict
        Keyword arguments for `LineCollection`.
        Passed to the function :func:`vocalpy.plot.annot.segments` that plots segments
        as a :class:`matplotlib.collections.LineCollection` instance. Default is None.
    text_kwargs : dict
        Keyword arguments for :meth:`matplotlib.axes.Axes.text`.
        Passed to the function :func:`vocalpy.plot.annot.labels` that plots labels
        using ``Axes.text`` method. Default is None.
    ax : matplotlib.axes.Axes
        Axes on which to plot segments.
        Default is None, in which case
        a new figure with a single axes is created.
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

    segment_centers = []
    for on, off in zip(annot.data.seq.onsets_s, annot.data.seq.offsets_s):
        segment_centers.append(np.mean([on, off]))
    segments(
        onsets=annot.data.seq.onsets_s,
        offsets=annot.data.seq.offsets_s,
        y=y_segments,
        ax=ax,
        line_kwargs=line_kwargs,
    )

    if tlim:
        ax.set_xlim(tlim)
        tmin, tmax = tlim

        labels_ = []
        segment_centers_tmp = []
        for label, segment_center in zip(annot.data.seq.labels, segment_centers):
            if tmin < segment_center < tmax:
                labels_.append(label)
                segment_centers_tmp.append(segment_center)
        segment_centers = segment_centers_tmp
    else:
        labels_ = annot.data.seq.labels

    segment_centers = np.array(segment_centers)
    labels(labels=labels_, t=segment_centers, y=y_labels, ax=ax, text_kwargs=text_kwargs)
