"""Functions for plotting spectrograms"""
from __future__ import annotations

import matplotlib.pyplot as plt

from .._spectrogram.data_type import Spectrogram
from ..annotation import Annotation
from .annot import annotation


def spectrogram(
    spect: Spectrogram,
    tlim: tuple | list | None = None,
    flim: tuple | list | None = None,
    ax: plt.Axes | None = None,
    pcolormesh_kwargs: dict | None = None,
) -> None:
    """Plot a spectrogram.

    Parameters
    ----------
    spectrogram : vocalpy.Spectrogram
    tlim : tuple, list
        limits of time axis (min, max) (i.e., x-axis).
        Default is None, in which case entire range of t will be plotted.
    flim : tuple, list
        limits of frequency axis (min, max) (i.e., x-axis).
        Default is None, in which case entire range of f will be plotted.
        limits of time axis (min, max) (i.e., x-axis).
        Default is None, in which case entire range of t will be plotted.
    flim : tuple, list
        limits of frequency axis (min, max) (i.e., x-axis).
        Default is None, in which case entire range of f will be plotted.
    ax : matplotlib.axes.Axes
        axes on which to plot spectrogram
    pcolormesh_kwargs : dict
        keyword arguments passed to :meth:`matplotlib.axes.Axes.pcolormesh`
        method used to plot spectrogram. Default is None.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if pcolormesh_kwargs is None:
        pcolormesh_kwargs = {}

    ax.pcolormesh(spect.times, spect.frequencies, spect.data, **pcolormesh_kwargs)

    if tlim is not None:
        ax.set_xlim(tlim)

    if flim is not None:
        ax.set_ylim(flim)


def annotated_spectrogram(
    spect: Spectrogram,
    annot: Annotation,
    tlim: tuple | list | None = None,
    flim: tuple | list | None = None,
    y_segments: float = 0.5,
    h_segments: float = 0.4,
    y_labels: float = 0.3,
    label_color_map: dict | None = None,
    pcolormesh_kwargs: dict | None = None,
    text_kwargs=None,
) -> tuple[plt.Figure, dict]:
    """Plot a :class:`vocalpy.Spectrogram` with a :class:`vocalpy.Annotation` below it.

    Convenience function that calls :func:`vocalpy.plot.spectrogram` and :func:`vocalpy.plot.annotation`.

    Parameters
    ----------
    spect : vocalpy.Spectrogram
    annot : vocalpy.Annotation
        Annotation that has segments to be plotted
        (the `annot.seq.segments` attribute)
    tlim : tuple, list
        Limits of time axis (min, max) (i.e., x-axis).
        Default is None, in which case entire range of t will be plotted.
    flim : tuple, list
        limits of frequency axis (min, max) (i.e., x-axis).
        Default is None, in which case entire range of f will be plotted.
    y_segments : float
        Height at which segments should be plotted.
        Default is 0.5 (assumes y-limits of 0 and 1).
    h_segments : float, int
        Height of rectangles that represent segments.
        Default is 0.4.
    y_labels : float
        Height on y-axis at which segment labels (if any) are plotted.
        Default is 0.4.
    label_color_map : dict, optional
        A :class:`dict` that maps string labels to colors
        (that are valid `color` arguments for matplotlib).
    pcolormesh_kwargs : dict
        Keyword arguments that will get passed to :meth:`matplotlib.axes.Axes.pcolormesh`
        when using that method to plot spectrogram.
    text_kwargs : dict
        keyword arguments for :meth:`matplotlib.axes.Axes.text`.
        Passed to the function :func:`vocalpy.plot.annot.labels` that plots labels
        using Axes.text method.
        Defaults are defined as :data:`vocalpy.plot.annot.DEFAULT_TEXT_KWARGS`.

    Returns
    -------
    fig, axes :
        Matplotlib Figure and :class:`dict` of Axes instances.
        The axes containing the spectrogram is ``axes['spect']``
        and the axes containing the annotated segments
        is ``axes['annot']``.
    """
    fig, axs = plt.subplot_mosaic(
        [
            ["spect"],
            ["spect"],
            ["annot"],
        ],
        layout="constrained",
        sharex=True,
    )

    spectrogram(spect, tlim, flim, ax=axs["spect"], pcolormesh_kwargs=pcolormesh_kwargs)

    annotation(
        annot,
        tlim,
        y_segments=y_segments,
        h_segments=h_segments,
        y_labels=y_labels,
        text_kwargs=text_kwargs,
        ax=axs["annot"],
        label_color_map=label_color_map,
    )

    return fig, axs
