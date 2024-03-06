"""Class that represents the segmenting step in a pipeline."""
from __future__ import annotations

from typing import Callable, TYPE_CHECKING

import dask
import dask.diagnostics

from .audio_file import AudioFile
from .sound import Sound
from .spectrogram_maker import validate_sound

if TYPE_CHECKING:
    from .segments import Segments


DEFAULT_SEGMENT_PARAMS = {
    "threshold": 5000,
    "min_dur": 0.02,
    "min_silent_dur": 0.002,
}


class Segmenter:
    """Class that represents the segmenting step in a pipeline.

    Attributes
    ----------
    callback : callable, optional
        The function or :class:`Callable` class instance
        that is used to segment.
        If not specified, defaults to
        :func:`vocalpy.segment.meansquared`.
    segment_params : dict, optional.
        If not specified, defaults to
        :const:`vocalpy.segmenter.DEFAULT_SEGMENT_PARAMS`.
    """

    def __init__(self, callback: Callable | None = None, segment_params: dict | None = None):
        """Initialize a new :class:`vocalpy.Segmenter` instance.

        Parameters
        ----------
        callback : callable, optional
            The function or :class:`Callable` class instance
            that is used to segment.
            If not specified, defaults to
            :func:`vocalpy.segment.meansquared`.
        segment_params : dict, optional.
            If not specified, defaults to
            :data:`vocalpy.segmenter.DEFAULT_SEGMENT_PARAMS`.
        """
        if callback is None:
            from vocalpy.segment import meansquared as default_segment_func

            callback = default_segment_func

        if not callable(callback):
            raise ValueError(f"`callback` should be callable, but `callable({callback})` returns False")

        self.callback = callback

        if segment_params is None:
            segment_params = DEFAULT_SEGMENT_PARAMS

        if not isinstance(segment_params, dict):
            raise TypeError(f"`segment_params` should be a `dict` but type was: {type(segment_params)}")

        self.segment_params = segment_params

    def segment(
        self,
        sound: Sound | AudioFile | list[Sound | AudioFile],
        parallelize: bool = True,
    ) -> Segments | list[Segments]:
        """Segment sound.

        Parameters
        ----------
        sound : vocalpy.Sound or list of Sound
            A `class`:vocalpy.Sound` instance
            or list of :class:`vocalpy.Sound` instances
            to segment.
        parallelize : bool
            If True, parallelize segmentation using :mod:`dask`.

        Returns
        -------
        segments : vocalpy.Segments, list
            If a :class:`~vocalpy.Sound` is passed in,
            a single set of :class:`~vocalpy.Segments` will be returned.
            If a list of :class:`~vocalpy.Sound` is passed in,
            a list of :class:`~vocalpy.Segments` will be returned.
        """
        validate_sound(sound)

        # define nested function so vars are in scope and ``dask`` can call it
        def _to_segments(sound_: Sound | AudioFile) -> Segments:
            if isinstance(sound_, AudioFile):
                sound_ = Sound.read(sound_.path)
            segments = self.callback(sound_, **self.segment_params)
            return segments

        if isinstance(sound, (Sound, AudioFile)):
            return _to_segments(sound)

        segments = []
        for sound_ in sound:
            if parallelize:
                segments.append(dask.delayed(_to_segments(sound_)))
            else:
                segments.append(_to_segments(sound_))

        if parallelize:
            graph = dask.delayed()(segments)
            with dask.diagnostics.ProgressBar():
                return graph.compute()
        else:
            return segments
