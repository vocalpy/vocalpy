"""Class that represents the segmenting step in a pipeline."""

from __future__ import annotations

import collections.abc
import inspect
from typing import TYPE_CHECKING, Callable, Mapping

import dask
import dask.diagnostics

from .audio_file import AudioFile
from .params import Params
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
    params : Mapping or Params, optional.
        Parameters passed to ``callback``.
        A :class:`Mapping` of keyword arguments,
        or one of the :class:`Params` classes that
        represents parameters, e.g.,
        class:`vocalpy.segment.MeanSquaredParams`.
        If not specified, defaults to
        :const:`vocalpy.segmenter.DEFAULT_SEGMENT_PARAMS`.
    """

    def __init__(
        self,
        callback: Callable | None = None,
        params: Mapping | Params | None = None,
    ):
        """Initialize a new :class:`vocalpy.Segmenter` instance.

        Parameters
        ----------
        callback : callable, optional
            The function or :class:`Callable` class instance
            that is used to segment.
            If not specified, defaults to
            :func:`vocalpy.segment.meansquared`.
        params : Mapping or Params, optional.
            Parameters passed to ``callback``.
            A :class:`Mapping` of keyword arguments,
            or one of the :class:`Params` classes that
            represents parameters, e.g.,
            class:`vocalpy.segment.MeanSquaredParams`.
            If not specified, defaults to
            :data:`vocalpy.segmenter.DEFAULT_SEGMENT_PARAMS`.
        """
        if callback is None:
            from vocalpy.segment import meansquared

            callback = meansquared
            # if callback was None and we use the default,
            # **and** params is None, we set these default params
            if params is None:
                params = DEFAULT_SEGMENT_PARAMS
        else:
            # if we *don't* use the default callback **and** params is None,
            # then we instead get the defaults for the specified callback
            if params is None:
                params = {}
                signature = inspect.signature(callback)
                for name, param in signature.parameters.items():
                    if param.default is not inspect._empty:
                        params[name] = param.default

        if not callable(callback):
            raise ValueError(
                f"`callback` should be callable, but `callable({callback})` returns False"
            )

        self.callback = callback

        if not isinstance(params, (collections.abc.Mapping, Params)):
            raise TypeError(
                f"`params` should be a `Mapping` or `Params` but type was: {type(params)}"
            )

        if isinstance(params, Params):
            # coerce to dict
            params = {**params}

        signature = inspect.signature(callback)
        if not all([param in signature.parameters for param in params]):
            invalid_params = [
                param for param in params if param not in signature.parameters
            ]
            raise ValueError(
                f"Invalid params for callback: {invalid_params}\n"
                f"Callback parameters are: {signature.parameters}"
            )

        self.params = params

    def __repr__(self):
        return f"Segmenter(callback={self.callback.__qualname__}, params={self.params})"

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
            segments = self.callback(sound_, **self.params)
            return segments

        if isinstance(sound, (Sound, AudioFile)):
            return _to_segments(sound)

        segments = []
        for sound_ in sound:
            if parallelize:
                segments.append(dask.delayed(_to_segments)(sound_))
            else:
                segments.append(_to_segments(sound_))

        if parallelize:
            graph = dask.delayed()(segments)
            with dask.diagnostics.ProgressBar():
                return graph.compute()
        else:
            return segments
