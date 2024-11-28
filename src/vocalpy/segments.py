"""Classes that represent line segments returned by segmenting algorithms."""

from __future__ import annotations

import json
import numbers
import pathlib
import reprlib

import numpy as np
import numpy.typing as npt
import pandas as pd


class Segments:
    """Class that represents a set of line segments
    returned by a segmenting algorithm.

    This class represents the result of algorithms that segment
    a signal into a series of consecutive, non-overlapping 2-D line segments :math:`S`.
    Each segment :math:`s_i` in a :class:`Segments` instance
    has an integer start index and length.
    The start index is computed by the segmenting algorithm.
    For algorithms that find segments by thresholding energy,
    the length will be equal to the stop index computed by the algorithm
    minus the start index, plus one (to account for how Python indexes).
    The stop index is the last index above threshold
    for a segment.
    For a list of such algorithms, call :func:`vocalpy.segment.line.list`.
    For algorithms that segment spectrograms into boxes, see :class:`Boxes`.

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

    >>> bfsongrep = voc.example('bfsongrepo')
    >>> sound = bfsongrepo.sounds[0]
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
                raise ValueError(
                    "Values of `start_inds` for `Segments` must all be non-negative."
                )

            if not np.all(start_inds[1:] > start_inds[:-1]):
                raise ValueError(
                    "Values of `start_inds` for `Segments` must be strictly increasing."
                )

            if not np.all(lengths >= 1):
                raise ValueError(
                    "Values of `lengths` for `Segments` must all be positive."
                )

        if not isinstance(samplerate, int):
            raise TypeError(
                f"Type of ``samplerate`` must be int but was: {type(samplerate)}"
            )
        if not samplerate > 0:
            raise ValueError(
                f"Value of ``samplerate`` must be a positive integer, but was {samplerate}."
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
        start_inds = np.array(json_dict["start_inds"], dtype=int)
        lengths = np.array(json_dict["lengths"], dtype=int)
        samplerate = json_dict["samplerate"]
        labels = json_dict["labels"]
        return cls(start_inds, lengths, samplerate, labels)

    VALID_COLUMNS_MAP_VALUES = ["start_s", "stop_s", "start_ind", "length"]

    @classmethod
    def from_csv(
        cls,
        csv_path: str | pathlib.Path,
        samplerate: int,
        columns_map: dict | None = None,
        default_label: str | None = None,
        read_csv_kwargs: dict | None = None,
    ):
        """Create a :class:`~vocalpy.Segments` instance from a csv file.

        The csv file can either have the column names
        ``{"start_ind", "length", "label"}``, that will be used directly
        as the :class:`~vocalpy.Segment` attributes
        ``start_inds``, ``lengths``, and ``labels``, respectively,
        or it can have the column names
        ``{"start_s", "stop_s", "label"}``,
        where ``"start_s"`` and ``"stop_s""`` refer to times in seconds.
        The ``label`` column is not required, and if it is not found,
        the ``labels`` will default to empty strings.
        You can change this behavior by specifying a ``default_label``
        that will be used for all the segments if no ``labels`` column
        is found, instead of an empty string.
        If one of these sets of columns (``{"start_ind", "length"``}``
        or ``{"start_s", "stop_s"}``) is not found in the csv,
        then an error will be raised.
        You can have the :meth:`vocalpy.Segments.from_csv` method
        rename columns for you after it loads the csv file into a
        :class:`pandas.DataFrame` using the ``columns_map`` argument;
        see example below. All other columns are ignored;
        you do not need to somehow remove them to load the file.

        Parameters
        ----------
        csv_path : string or pathlib.Path
            Path to csv file.
        samplerate : int
            The sampling rate of the audio signal that was segmented
            to produce these segments.
        columns_map : dict, optional
            Mapping that will be used to rename columns in the csv
        default_label : str, optional
            String, a default that is assigned as the label to all segments.
        read_csv_kwargs, dict, optional
            Keyword arguments to pass to :func:`pandas.read_csv` function.

        Returns
        -------
        segments : vocalpy.Segments

        Examples
        --------

        The main use of this method is to load a set of line segments
        from a csv file created by another library or a script.

        If the column names in the csv do not match the column names
        that `vocalpy.Segments` expects, you can have the
        `vocalpy.Segments.from_csv` method rename the columns for you
        after loading the csv, using the `columns_map` argument.

        Here is an example of renaming columns to the expected names
        "start_s" and "stop_s". After renaming, the values in these columns
        are then converted to the starting indices and lengths of segments
        using the `samplerate`.

        >>> jourjine = voc.example("jourjine-et-al-2023", return_path=True)
        >>> sound = voc.Sound.read(jourjine.sound)
        >>> csv_path = jourjine.segments
        >>> columns_map = {"start_seconds": "start_s", "stop_seconds": "stop_s"}
        >>> segments = voc.Segments.from_csv(csv_path, samplerate=sound.samplerate, columns_map=columns_map)
        >>> print(segments)
        Segments(start_inds=array([   131...   149767168]), lengths=array([40447,...29696, 25087]),
        samplerate=250000, labels=['', '', '', '', '', '', ...])

        Notes
        -----
        This method is provided as a convenience for the case where
        you have a segmentation saved in a csv file,
        e.g., from a :class:`pandas.DataFrame`,
        that was created by another library or script.
        If you are working mainly with :mod:`vocalpy`, you should
        prefer to load a set of segments with :meth:`~vocalpy.Segments.from_json`,
        and to save the set of segments with :meth:`~vocalpy.Segments.to_json`,
        since this avoids needing to keep track of the `samplerate` value separately.
        """
        if not isinstance(samplerate, int):
            raise TypeError(
                f"The `samplerate` argument must be an int but type was: {type(samplerate)}"
            )
        if samplerate < 1:
            raise ValueError(
                f"The `samplerate` argument must be a positve number but value was: {samplerate}"
            )

        if read_csv_kwargs is not None:
            if not isinstance(read_csv_kwargs, dict):
                raise TypeError(
                    f"The `read_csv_kwargs` must be a `dict` but type was: {type(read_csv_kwargs)}"
                )
        else:
            read_csv_kwargs = {}
        df = pd.read_csv(csv_path, **read_csv_kwargs)

        if columns_map is not None:
            if not isinstance(columns_map, dict):
                raise TypeError(
                    f"The `columns_map` argument must be a `dict` but type was: {type(dict)}"
                )
            if not all(
                (
                    isinstance(k, str) and isinstance(v, str)
                    for k, v in columns_map.items()
                )
            ):
                raise ValueError(
                    "The `columns_map` argument must be a dict that maps string keys to string values, "
                    "but not all keys and values were strings."
                )
            if not all(
                v in cls.VALID_COLUMNS_MAP_VALUES for v in columns_map.values()
            ):
                invalid_values = [
                    v
                    for v in columns_map.values()
                    if v not in cls.VALID_COLUMNS_MAP_VALUES
                ]
                raise ValueError(
                    f"The `columns_map` argument must map keys (column names in the csv) "
                    'to either {"start_seconds", "stop_seconds"} or {"start_ind", "length"}. '
                    f"The following values are invalid: {invalid_values}"
                )
            df.columns = [
                (
                    columns_map[column_name]
                    if column_name in columns_map
                    else column_name
                )
                for column_name in df.columns
            ]

        if "label" not in df.columns and default_label is not None:
            if not isinstance(default_label, str):
                raise TypeError(
                    f"The `default_label` argument must be a string but type was: {type(default_label)}"
                )
            df["label"] = default_label

        if "start_ind" in df.columns and "length" in df.columns:
            return cls(
                start_inds=df["start_ind"].values,
                lengths=df["length"].values,
                labels=(
                    df["label"].values.tolist()
                    if "label" in df.columns
                    else None
                ),
                samplerate=samplerate,
            )
        elif "start_s" in df.columns and "stop_s" in df.columns:
            start_inds = (df["start_s"].values * samplerate).astype(int)
            lengths = (
                (df["stop_s"].values - df["start_s"].values) * samplerate
            ).astype(int)
            return cls(
                start_inds=start_inds,
                lengths=lengths,
                labels=(
                    df["label"].values.tolist()
                    if "label" in df.columns
                    else None
                ),
                samplerate=samplerate,
            )
        else:
            raise ValueError(
                "The csv file loaded from `csv_path must either have columns {'start_ind', 'length'} "
                "or {'start_s', 'stop_s'}, but neither pair was found. "
                f"Columns in the `pandas.DataFrame` loaded from the csv file are: {df.columns}\n"
                "To have the `vocalpy.Segments.from_csv` method rename the columns for you, "
                "use the `columns_map` argument. Type `help(voc.Segments)` or, in iPython, `voc.Segments?`, "
                "to see examples of using this and other arguments."
            )

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
