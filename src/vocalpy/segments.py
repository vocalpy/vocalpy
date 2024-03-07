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


class Segment:
    def __init__(
            self, start_ind: int, length: int, data: xr.DataArray, label: str = '',
    ) -> None:
        if not isinstance(start_ind, numbers.Integral):
            raise TypeError(
                f"Type of `start_ind` for `Segment` must be int but was: {type(start_ind)}"
            )
        if not isinstance(length, numbers.Integral):
            raise TypeError(
                f"Type of `length` for `Segment` must be int but was: {type(start_ind)}"
            )
        if not isinstance(label, str):
            raise TypeError(
                f"`label` for `Segment` should be an instance of `str`, but type was: {type(label)}"
            )
        # explicitly convert type numpy.str_ to a str instance so we can save as an attribute
        label = str(label)
        if not start_ind >= 0:
            raise ValueError(
                f"`start_ind` for `Segment` must be a non-negative number but was: {start_ind}"
            )
        if not length >= 0:
            raise ValueError(
                f"`length` for `Segment` must be a non-negative number but was: {start_ind}"
            )
        if not isinstance(data, xr.DataArray):
            raise TypeError(
                f"`data` for `Segment` should be an instance of xarray.DataArray, but type was: {type(data)}"
            )

        if not data.ndim == 1:
            raise ValueError(
                f"`data` for `Segment` should be have one dimension but `data.ndim` was: {data.ndim}"
            )

        if not data.size == length:
            raise ValueError(
                f"`data.size` for `Segment` should equal `length` but `data.size` was {data.size} "
                f"and `length` was {length}."
            )

        self.start_ind = start_ind
        self.length = length
        self.label = label
        self.data = data

    def __repr__(self):
        return f"Segment(start_ind={self.start_ind!r}, length={self.length!r}, "\
               f"label={self.label!r}, data={self.data!r}"

    def write(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        # we make a new DataArray that points to the same underlying ``values``,
        # so we can mutate the new DataArray's ``attrs`` without affecting
        # the ``attrs`` of this instance's ``data``
        data = xr.DataArray(self.data)
        data.attrs = {
            'start_ind': self.start_ind,
            'length': self.length,
            'label': self.label,
        }
        data.to_netcdf(path, engine="h5netcdf")

    @classmethod
    def read(cls, path: str | pathlib.Path) -> Segment:
        path = pathlib.Path(path)
        data = xr.load_dataarray(path)
        start_ind = data.attrs['start_ind']
        length = data.attrs['length']
        label = data.attrs['label']
        # throw away metadata; feels weird but want round-trip to give us back
        # an instance that is __eq__ual
        data.attrs = {}
        return cls(
            start_ind=start_ind,
            length=length,
            label=label,
            data=data,
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Segment):
            return False
        return (
            (self.start_ind == other.start_ind) and
            (self.length == other.length) and
            (self.label == other.label) and
            (self.data.equals(other.data))
        )


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

    Examples
    --------

    :class:`Segments` are returned by the segmenting algorithms that return a set of line segments
    (as opposed to segmenting algorithms that return a set of boxes).

    >>> sounds = voc.example('bfsongrepo', return_type='sound')
    >>> segments = voc.segment.meansquared(sounds[0], threshold=1500, min_dur=0.2, min_silent_dur=0.02)
    >>> segments
    Segments(start_inds=array([ 13050...3127, 225700]), lengths=array([1032, ..., 2074, 2640]), labels=array(['', ''..., dtype='<U1'), sound=vocalpy.Sound(data=array([[-209,..., dtype=int16), samplerate=32000, path=/home/pimienta/.cache/vocalpy/bfsongrepo.tar.gz.untar/gy6or6_baseline_220312_0836.3.wav))  # noqa

    :class:`~vocalpy.Segments` can be used to compute metrics for segmentation

    >>> sounds = voc.example('bfsongrepo', return_type='sound')
    >>> segments = voc.segment.meansquared(sound, threshold=1500, min_dur=0.2, min_silent_dur=0.02)
    >>> annots = voc.example('bfsongrepo', return_type='annotation')
    >>> ref = np.sorted(np.concatenate(annots[0].seq.onsets, annot[0].seq.offsets))
    >>> hyp = segments.all_times
    >>> prec, _ = voc.metrics.segmentation.ir.precision(reference=ref, hypothesis=hyp)

    We can also use :class:`~vocalpy.Segments` to write the data in each :class:`Segment` to disk.
    This is a common step in workflows, such as those that extract acoustic features from each segment
    for further analysis.

    >>> sounds = voc.example('bfsongrepo', return_type='sound')
    >>> segments = voc.segment.meansquared(sounds[0], threshold=1500, min_dur=0.2, min_silent_dur=0.02)
    >>> import pathlib, shutil
    >>> dst = pathlib.Path('./tmp')
    >>> dst.mkdir()
    >>> for ind, segment in enumerate(segments, 1):
    ...     print(f"Segment {ind}: {segment}")
    ...     segment.write(dst / f"{sounds[0].stem}-segment-{ind}.h5")
    Segment 1: Segment(start_time=..., duration=...,)
    Segment 2: Segment(start_time=..., duration=...,)

    Like a Python :class:`list`, :class:`Segments` can be iterated and sliced.
    Iterating through :class:`Segments` results in :class:`Segment` instances.
    This also allows for filtering by list comprehension.

    >>> sounds = voc.example('bfsongrepo', return_type='sound')
    >>> segments = voc.segment.meansquared(sounds[0], threshold=1500, min_dur=0.2, min_silent_dur=0.02)
    >>> # only keep segments longer than 100 ms
    >>> long_segments = [segment for segment in segments if segment.duration > 0.1]
    >>> len(long_segments)
    10

    A :class:`slice` returns a new :class:`~vocalpy.Segments`

    >>> sounds = voc.example('bfsongrepo', return_type='sound')
    >>> segments = voc.segment.meansquared(sounds[0], threshold=1500, min_dur=0.2, min_silent_dur=0.02)
    >>> len(segments)
    150
    >>> five_segments = Segments[5:10]
    >>> type(segments)
    vocalpy.Segments
    >>> len(five_segments)
    5

    The segments can be converted to a :class:`pandas.DataFrame`
    or saved directly to a csv file.

    >>> sounds = voc.example('bfsongrepo', return_type='sound')
    >>> segments = voc.segment.meansquared(sounds[0], threshold=1500, min_dur=0.2, min_silent_dur=0.02)
    >>> df = segments.to_df
    >>> segments.to_csv('gy6or6-segments.csv')

    :class:`Segments` can be loaded from a csv as well.
    Note that this requires passing in audio or the path to audio,
    and no validation is done that guarantees the
    segments are the results of applying a segmentation algorithm
    to the audio.

    >>> sounds = voc.example('bfsongrepo', return_type='sound')
    >>> segments = voc.segment.meansquared(sounds[0], threshold=1500, min_dur=0.2, min_silent_dur=0.02)
    >>> df = segments.to_df
    >>> segments.to_csv('gy6or6-segments.csv')
    >>> segments_from_csv = voc.Segments.from_csv('gy6or6-segments.csv', sound_path=sounds[0].path)

    See Also
    --------
    Boxes

    """

    def __init__(self,
                 start_inds: npt.NDArray,
                 lengths: npt.NDArray,
                 sound: vocalpy.Sound | None = None,
                 labels: list[str] | None = None,
                 ) -> None:
        if sound is not None:
            if not isinstance(sound, Sound):
                raise TypeError(
                    f"`sound` should be an instance of vocalpy.Sound, but type was: {type(sound)}"
                )
        if not isinstance(start_inds, np.ndarray):
            raise TypeError(
                f"`start_inds` must be a numpy array but type was: {type(start_inds)}"
            )
        if not isinstance(lengths, np.ndarray):
            raise TypeError(
                f"`lengths` must be a numpy array but type was: {type(lengths)}"
            )

        if not issubclass(start_inds.dtype.type, numbers.Integral):
            raise ValueError(
                f"`start_inds` must have an integer dtype, but dtype was: {start_inds.dtype}"
            )
        if not issubclass(lengths.dtype.type, numbers.Integral):
            raise ValueError(
                f"`lengths` must have an integer dtype, but dtype was: {lengths.dtype}"
            )

        if start_inds.size == lengths.size == 0:
            # no need to validate
            pass
        else:
            if not start_inds.ndim == 1:
                raise ValueError(
                    f"`start_inds` for `Segments` should be 1-dimensional array but start_inds.ndim was: {start_inds.ndim}"
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
                raise ValueError(f"Values of `start_inds` for `Segments` must all be non-negative.")

            if not np.all(start_inds[1:] > start_inds[:-1]):
                raise ValueError(f"Values of `start_inds` for `Segments` must be strictly increasing.")

            if not np.all(lengths >= 1):
                raise ValueError(f"Values of `lengths` for `Segments` must all be positive.")

            if sound is not None:
                if start_inds[-1] + lengths[-1] > sound.data.shape[-1]:
                    raise ValueError(
                        # TODO: check for off-by-one errors here and elsewhere where we use lengths
                        "Length of last segment is longer than number of samples in sound. "
                        f"Last segment ends at {start_inds[-1] + lengths[-1]} and sound has {sound.data.shape[-1]} samples."
                    )

        if labels is not None:
            if not isinstance(labels, list):
                raise TypeError(
                    f"`labels` must be a list but type was: {type(labels)}"
                )
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
            labels = [''] * start_inds.shape[0]

        self.start_inds = start_inds
        self.lengths = lengths
        self.sound = sound
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
        return np.unique(
            np.concatenate(
                (self.start_inds, self.stop_inds)
            )
        )

    @property
    def start_times(self):
        """Start times of segments.

        Returns ``self.start_inds / self.sound.samplerate``.
        """
        return self.start_inds / self.sound.samplerate

    @property
    def durations(self):
        """Durations of segments.

        Returns ``self.lengths / self.sound.samplerate``.
        """
        return self.lengths / self.sound.samplerate

    @property
    def stop_times(self):
        """Stop times of segments.

        Returns ``self.start_times + self.durations``.
        """
        return self.start_times + self.durations

    @property
    def all_times(self):
        return np.unique(
            np.concatenate(
                (self.start_times, self.stop_times)
            )
        )

    def __repr__(self):
        return f"Segments(start_inds={reprlib.repr(self.start_inds)}, lengths={reprlib.repr(self.lengths)}, "\
               f"labels={reprlib.repr(self.labels)}, sound={self.sound!r})"

    def __str__(self):
        return f"Segments(start_times={self.start_times!r}, durations={self.durations!r}, labels={self.labels!r}, sound={self.sound!r})"

    def to_df(self):
        """Convert :class:`Segments` to a :class:`pandas.DataFrame`.

        Examples
        --------
        >>> sounds = voc.example('bfsongrepo', return_type='sound')
        >>> segments = voc.segment.meansquared(sounds[0], threshold=1500, min_dur=0.2, min_silent_dur=0.02)
        >>> df = segments.to_df()
        >>> df['start_time'] = df['start_ind'] / segments.sound.samplerate
        """
        d = {
            'start_ind': self.start_inds,
            'length': self.lengths,
            'label': self.labels,
        }
        df = pd.DataFrame(d)
        return df

    def to_csv(self, csv_path: str | pathlib.Path, **to_csv_kwargs) -> None:
        """Write :class:`Segments` to a csv file."""
        df = self.to_df()
        df.to_csv(csv_path, **to_csv_kwargs)

    @classmethod
    def from_csv(cls, csv_path: str | pathlib.Path,
                 sound: vocalpy.Sound | None = None,
                 sound_path: str | pathlib.Path | None = None) -> Segments:
        """Read :class:`Segments` from a csv file."""
        if sound is not None and sound_path is not None:
            raise ValueError(
                f"`Segments.from_csv` can accept either `sound` or `sound_path`,"
                f"but not both"
            )
        if sound_path:
            sound = Sound.read(sound_path)
        df = pd.read_csv(
            csv_path,
            # passing a converter is the only way to make sure that 'label'
            # is an object array with strings:
            # if we don't pass a converter, we get NaNs for empty strings,
            converters={'label': str},
            # and if we instead specify its dtype as 'string', we get weird StringType "<NA>"s
            # even when we convert to list (I think). converters take precedence over dtype
            dtype={'start_ind': int, 'length': int},
        )
        start_inds = df['start_ind'].values
        lengths = df['length'].values
        labels = df['label'].values.tolist()
        return cls(
            start_inds, lengths, sound, labels
        )

    def to_json(self, json_path: str | pathlib.Path) -> None:
        """Save :class:`Segments` to a json file

        Parameters
        ----------
        json_path : str, pathlib.Path
        """
        json_path = pathlib.Path(json_path)
        df = self.to_df()
        json_dict = {}
        json_dict['data'] = df.to_json(orient="table")
        json_dict['metadata'] = {
            'sound_path': str(self.sound.path)
        }
        with json_path.open('w') as fp:
            json.dump(json_dict, fp)

    @classmethod
    def from_json(cls, json_path: str | pathlib.Path) -> Segments:
        """Load :class:`Segments` from a json file

        Parameters
        ----------
        json_path

        Returns
        -------
        segments : Segments
        """
        json_path = pathlib.Path(json_path)
        with json_path.open('r') as fp:
            json_dict = json.load(fp)

        if json_dict['metadata']['sound_path'] == "None":
            sound = None
        else:
            sound_path = pathlib.Path(
                json_dict['metadata']['sound_path']
            )
            if sound_path.exists():
                sound = Sound.read(sound_path)
            else:
                sound = None

        df = pd.read_json(
            io.StringIO(json_dict['data']),
            orient="table",
        )
        start_inds = df['start_ind'].values
        lengths = df['length'].values
        labels = df['label'].values.tolist()
        return cls(
            start_inds, lengths, sound, labels
        )

    def __len__(self):
        return len(self.start_times)

    def __eq__(self, other: Segments) -> bool:
        if not isinstance(other, Segments):
            return False
        return (
            np.array_equal(self.start_inds, other.start_inds) and
            np.array_equal(self.lengths, other.lengths) and
            self.labels == other.labels
        )

    def __iter__(self):
        if self.sound is None:
            raise ValueError(
                "This `Segments` instance does not have a `sound`, "
                "unable to iterate through each `Segment`."
            )
        for start_ind, length, label in zip(self.start_inds, self.lengths, self.labels):
            data = xr.DataArray(
                data=self.sound.data[..., start_ind: start_ind + length].squeeze(0)
            )
            segment = Segment(
                start_ind=start_ind, length=length, label=label, data=data
                )
            yield segment

    def __getitem__(self, key):
        cls = type(self)
        if isinstance(key, numbers.Integral):
            if self.sound is None:
                raise ValueError(
                    "This `Segments` instance does not have a `sound`, "
                    "unable to iterate through each `Segment`."
                )
            start_ind = self.start_inds[key]
            length = self.lengths[key]
            label = self.labels[key]
            data = xr.DataArray(
                data=self.sound.data[..., start_ind: start_ind + length].squeeze(0)
            )
            return Segment(
                start_ind=start_ind, length=length, label=label, data=data
            )
        elif isinstance(key, slice):
            start_inds = self.start_inds[key]
            lengths = self.lengths[key]
            labels = self.labels[key]
            return cls(
                start_inds,
                lengths,
                self.sound,
                labels,
            )
        else:
            raise TypeError(
                f"{cls.__name__} indices must be integers or slice"
            )

