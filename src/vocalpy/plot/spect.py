"""Functions for plotting spectrograms"""
from __future__ import annotations

import matplotlib.pyplot as plt

from ..annotation import Annotation
from ..spectrogram import Spectrogram
from .annot import annotation


def spectrogram(
    spect: Spectrogram,
    tlim: tuple | list | None = None,
    flim: tuple | list | None = None,
    ax: plt.Axes | None = None,
    imshow_kwargs: dict | None = None,
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
        axes on which to plot spectrgraom
    imshow_kwargs : dict
        keyword arguments passed to matplotlib.axes.Axes.imshow method
        used to plot spectrogram. Default is None.
    """
    if imshow_kwargs is None:
        imshow_kwargs = {}

    if ax is None:
        fig, ax = plt.subplots()

    s, t, f = spect.data, spect.times, spect.frequencies

    extent = [t.min(), t.max(), f.min(), f.max()]

    ax.imshow(s, aspect="auto", origin="lower", extent=extent, **imshow_kwargs)

    if tlim is not None:
        ax.set_xlim(tlim)

    if flim is not None:
        ax.set_ylim(flim)


def annotated_spectrogram(
    spect: Spectrogram,
    annot: Annotation,
    tlim: tuple | list | None = None,
    flim: tuple | list | None = None,
    fig: plt.Figure | None = None,
    imshow_kwargs: dict | None = None,
    line_kwargs=None,
    text_kwargs=None,
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Plot a :class:`vocalpy.Spectrogram` with a :class:`vocalpy.Annotation` below it.

    Convenience function that calls :func:`vocalpy.plot.spectrogram` and :func:`vocalpy.plot.annotation`.

    Parameters
    ----------
    spect : vocalpy.Spectrogram
    annotation : vocalpy.Annotation
        annotation that has segments to be plotted
        (the `annot.seq.segments` attribute)
    tlim : tuple, list
        limits of time axis (min, max) (i.e., x-axis).
        Default is None, in which case entire range of t will be plotted.
    flim : tuple, list
        limits of frequency axis (min, max) (i.e., x-axis).
        Default is None, in which case entire range of f will be plotted.
    fig : matplotlib.pyplot.Figure
        A :class:`matplotlib.pyplot.Figure` instance on which
        the spectrogram and annotation should be plotted.
    imshow_kwargs : dict
        keyword arguments that will get passed to `matplotlib.axes.Axes.imshow`
        when using that method to plot spectrogram.
    line_kwargs : dict
        keyword arguments for `LineCollection`.
        Passed to the function `vocalpy.plot.annot.segments` that plots segments
        as a `LineCollection` instance. Default is None.
    text_kwargs : dict
        keyword arguments for `matplotlib.axes.Axes.text`.
        Passed to the function `vocalpy.plot.annot.labels` that plots labels
        using Axes.text method.
        Defaults are defined as `vocalpy.plot.annot.DEFAULT_TEXT_KWARGS`.

    Returns
    -------
    fig, spect_ax, annot_ax :
        Matplotlib Figure and Axes instances.
        The spect_ax is the axes containing the spectrogram
        and the annot_ax is the axes containing the
        annotated segments.
    """
    fig = plt.figure()
    gs = fig.add_gridspec(3, 3)
    spect_ax = fig.add_subplot(gs[:2, :])
    annot_ax = fig.add_subplot(gs[2, :])

    spectrogram(spect, tlim, flim, ax=spect_ax, imshow_kwargs=imshow_kwargs)

    annotation(annot, tlim, ax=annot_ax, line_kwargs=line_kwargs, text_kwargs=text_kwargs)

    return fig, spect_ax, annot_ax
