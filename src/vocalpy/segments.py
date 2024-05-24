"""Classes that represent line segments returned by segmenting algorithms."""
from __future__ import annotations

import json
import numbers
import pathlib
import reprlib
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

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
    >>> sound = sounds[0]
    >>> segments = voc.segment.meansquared(sound, threshold=1500, min_dur=0.2, min_silent_dur=0.02)
    >>> segments
    Segments(start_inds=array([ 22293...4425, 220495]), lengths=array([ 8012,... 6935,  7896]), samplerate=32000, labels=['', '', '', '', '', '', ...])  # noqa

    Because audio data is a digital signal with discrete samples,
    segments are defined in terms of start indices and lengths.
    Thus, the start index of each segment is the index of the sample
    where it starts--also known as a "boundary"--and the length
    is given in number of samples.

    However, we often want to think of segments times in terms of seconds.
    We can get the start times of segments in seconds with the :attr:`~Segments.start_times`
    property, and we can get the duration of segments in seconds with the
    :attr:`~Segments.durations` property.

    >>> segments.start_times
    array([0.69665625, 1.801375  , 2.26390625, 2.7535625 , 3.5885    ,
           6.38828125, 6.89046875])
    >>> segments.durations
    array([0.250375  , 0.33278125, 0.31      , 0.23625   , 0.308625  ,
           0.21671875, 0.24675   ])

    This is possible because each set of :class:`Segments` has a
    :attr:`~Segments.samplerate` attribute, that can be used to convert
    from sample numbers to seconds.
    This attribute is taken from the :class:`vocalpy.Sound` that
    was segmented to produce the :class:`Segments` in the first place.

    Depending on the segmenting algorithm,
    the start of one segment may not be the same as the end of
    the segment that precedes it.
    In this case we may want to find where the segments stop.
    We can do so with the :attr:`~Segments.stop_ind`
    and :attr:`~Segments.stop_ind` properties.

    To actually get a :class:`Sound` for every segment in a set of :class:`Segments`,
    we can pass the :class:`Segments` into to the :meth:`vocalpy.Sound.segment` method.

    >>> segment_sounds = sound.segment(segments)

    This might seem verbose, but it has a couple of advantages.
    The first is that the :class:`Segments` can be saved in a json file,
    so they can be loaded again and used to segment a sound
    without needed to re-run the segmentation.
    You can use a naming convention so that each sound file
    has a segments file paired with it: e.g., if the
    sound file is named ``"mouse1-day1-bout1.wav"``,
    then the json file could be named
    ``"mouse1-day1-bout1.segments.json"``.

    >>> segments.to_json(path='mouse1-day1-bout1.segments.json')

    A set of :class:`Segments` is then loaded with the
    :meth:`~Segments.from_json` method.

    >>> segments = voc.Segments.from_json(path='mouse1-day1-bout1.segments.json')

    The second advantage of representing :class:`Segments` separately
    is that they can then be used to compute metrics for segmentation.
    Note that here we are using the :attr:`~Segments.all_times` property,
    that gives us all the boundary times in seconds.

    >>> sounds = voc.example('bfsongrepo', return_type='sound')
    >>> segments = voc.segment.meansquared(sound, threshold=1500, min_dur=0.2, min_silent_dur=0.02)
    >>> annots = voc.example('bfsongrepo', return_type='annotation')
    >>> ref = np.sorted(np.concatenate(annots[0].seq.onsets, annot[0].seq.offsets))
    >>> hyp = segments.all_times
    >>> prec, _ = voc.metrics.segmentation.ir.precision(reference=ref, hypothesis=hyp)


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
            raise TypeError(f"Type of ``samplerate`` must be int but was: {type(samplerate)}")
        if not samplerate > 0:
            raise ValueError(f"Value of ``samplerate`` must be a positive integer, but was {samplerate}.")

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
