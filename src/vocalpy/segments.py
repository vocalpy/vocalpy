"""Classes that represent line segments returned by segmenting algorithms."""
from __future__ import annotations

import io
import json
import numbers
import pathlib
import reprlib
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from .sound import Sound

if TYPE_CHECKING:
    import vocalpy


class Segments:
    """Class that represents a set of line segments
    returned by a segmenting algorithm.

    A set of segments returned by a segmenting algorithm,
    where each segment is defined by a start index and length.

    More precisely, this class represents the result of algorithms that segment
    a signal into a series of consecutive, non-overlapping 2-D line segments :math:`S`.
    For a list of such algorithms, call :func:`vocalpy.segment.line.list`.
    For algorithms that segment spectrograms into boxes, see :class:`Boxes`.

    Each segment :math:`s_i` in a :class:`Segments` instance
    has an integer start index and length.
    The start index is computed by the segmenting algorithm.
    a segmenting algorithm. For algorithms that find segments by thresholding energy,
    the length will be equal to

    Attributes
    ----------
    start_inds : numpy.ndarray
    lengths: numpy.ndarray
    labels: list, optional
        A :class:`list` of strings,
        where each string is the label for each segment.
    sound : vocalpy.Sound
        The sound that was segmented to produce this set of line segments.

    Examples
    --------

    :class:`Segments` are returned by the segmenting algorithms that return a set of line segments
    (as opposed to segmenting algorithms that return a set of boxes).

    >>> sounds = voc.example('bfsongrepo', return_type='sound')
    >>> segments = voc.segment.meansquared(sounds[0], threshold=1500, min_dur=0.2, min_silent_dur=0.02)
    >>> segments
    Segments(start_inds=array([ 13050...3127, 225700]), lengths=array([1032, ..., 2074, 2640]), labels=array(['', ''..., dtype='<U1'))  # noqa

    :class:`~vocalpy.Segments` can be used to compute metrics for segmentation

    >>> sounds = voc.example('bfsongrepo', return_type='sound')
    >>> segments = voc.segment.meansquared(sound, threshold=1500, min_dur=0.2, min_silent_dur=0.02)
    >>> annots = voc.example('bfsongrepo', return_type='annotation')
    >>> ref = np.sorted(np.concatenate(annots[0].seq.onsets, annot[0].seq.offsets))
    >>> hyp = segments.all_times
    >>> prec, _ = voc.metrics.segmentation.ir.precision(reference=ref, hypothesis=hyp)

    The segments can be saved to a json file.


    See Also
    --------
    Boxes
    """
    def __init__(
        self,
        start_inds: npt.NDArray,
        lengths: npt.NDArray,
        samplerate: int,
        labels: list[str] | None = None,
    ) -> None:
        if not isinstance(start_inds, np.ndarray):
            raise TypeError(f"`start_inds` must be a numpy array but type was: {type(start_inds)}")
        if not isinstance(lengths, np.ndarray):
            raise TypeError(f"`lengths` must be a numpy array but type was: {type(lengths)}")

        if not issubclass(start_inds.dtype.type, numbers.Integral):
            raise ValueError(f"`start_inds` must have an integer dtype, but dtype was: {start_inds.dtype}")
        if not issubclass(lengths.dtype.type, numbers.Integral):
            raise ValueError(f"`lengths` must have an integer dtype, but dtype was: {lengths.dtype}")

        if start_inds.size == lengths.size == 0:
            # no need to validate
            pass
        else:
            if not start_inds.ndim == 1:
                raise ValueError(
                    "`start_inds` for `Segments` should be 1-dimensional array "
                    f"but start_inds.ndim was: {start_inds.ndim}"
                )
            if not lengths.ndim == 1:
                raise ValueError(
                    f"`lengths` for `Segments` should be 1-dimensional array but lengths.ndim was: {lengths.ndim}"
                )
            if start_inds.size != lengths.size:
                raise ValueError(
                    "`start_inds` and `lengths` of `Segments` must have same number of elements."
                    f"`start_inds` has {start_inds.size} elements and `lengths` has {lengths.size} elements."
                )
            if not np.all(start_inds >= 0):
                raise ValueError("Values of `start_inds` for `Segments` must all be non-negative.")

            if not np.all(start_inds[1:] > start_inds[:-1]):
                raise ValueError("Values of `start_inds` for `Segments` must be strictly increasing.")

            if not np.all(lengths >= 1):
                raise ValueError("Values of `lengths` for `Segments` must all be positive.")

        if not isinstance(samplerate, int):
            raise TypeError

        if labels is not None:
            if not isinstance(labels, list):
                raise TypeError(f"`labels` must be a list but type was: {type(labels)}")
            if not all([isinstance(lbl, str) for lbl in labels]):
                types = set([type(lbl) for lbl in labels])
                raise ValueError(
                    f"`labels` of `Segments` must be a list of strings, but found the following types: {types}"
                )
            if len(labels) != start_inds.size:
                raise ValueError(
                    "`labels` for `Segments` must have same number of elements as `start_inds`. "
                    f"`labels` has {len(labels)} elements but `start_inds` has {start_inds.size} elements."
                )
        else:  # if labels is None
            # then default to empty strings
            labels = [""] * start_inds.shape[0]

        self.start_inds = start_inds
        self.lengths = lengths
        self.samplerate = samplerate
        self.labels = labels

    @property
    def stop_inds(self):
        """Indices of where segments stop.

        Returns ``self.start_inds + self.lengths``.
        """
        return self.start_inds + self.lengths

    @property
    def all_inds(self):
        """Start and stop indices of segments.

        Returns the following:

        .. code-block: python

           np.unique(np.concatenate(self.start_inds, self.stop_inds)
        """
        return np.unique(np.concatenate((self.start_inds, self.stop_inds)))

    @property
    def start_times(self):
        """Start times of segments.

        Returns ``self.start_inds / self.sound.samplerate``.
        """
        return self.start_inds / self.samplerate

    @property
    def durations(self):
        """Durations of segments.

        Returns ``self.lengths / self.sound.samplerate``.
        """
        return self.lengths / self.samplerate

    @property
    def stop_times(self):
        """Stop times of segments.

        Returns ``self.start_times + self.durations``.
        """
        return self.start_times + self.durations

    @property
    def all_times(self):
        return np.unique(np.concatenate((self.start_times, self.stop_times)))

    def __repr__(self):
        return (
            f"Segments(start_inds={reprlib.repr(self.start_inds)}, lengths={reprlib.repr(self.lengths)}, "
            f"samplerate={self.samplerate!r}, labels={reprlib.repr(self.labels)})"
        )

    def to_json(self, path: str | pathlib.Path) -> None:
        """Save :class:`Segments` to a json file.

        Parameters
        ----------
        json_path : str, pathlib.Path
            The path where the json file should be saved
            with these :class:`Segments`.
        """
        path = pathlib.Path(path)
        json_dict = {
            "start_inds": self.start_inds.tolist(),
            "lengths": self.lengths.tolist(),
            "samplerate": self.samplerate,
            "labels": self.labels,
        }
        with path.open("w") as fp:
            json.dump(json_dict, fp)

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> "Segments":
        """Load :class:`Segments` from a json file.

        Parameters
        ----------
        path : str, pathlib.Path
            The path to the json file to load the :class:`Segments` from.

        Returns
        -------
        segments : Segments
        """
        path = pathlib.Path(path)
        with path.open("r") as fp:
            json_dict = json.load(fp)
        start_inds = np.array(json_dict["start_inds"], type=int)
        lengths = np.array(json_dict["lengths"], type=int)
        samplerate = json_dict["samplerate"]
        labels = json_dict["labels"]
        return cls(start_inds, lengths, samplerate, labels)

    def __len__(self):
        return len(self.start_times)

    def __eq__(self, other: "Segments") -> bool:
        if not isinstance(other, Segments):
            return False
        return (
            np.array_equal(self.start_inds, other.start_inds)
            and np.array_equal(self.lengths, other.lengths)
            and self.samplerate == other.samplerate
            and self.labels == other.labels
        )
