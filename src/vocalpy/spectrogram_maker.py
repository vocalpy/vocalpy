"""Class that represents the step in a pipeline that makes spectrograms from audio."""
from __future__ import annotations

import collections.abc
import inspect
import pathlib
from typing import Callable, List, Mapping, Sequence, Union

import dask
import dask.diagnostics

import vocalpy.constants

from ._spectrogram.data_type import Spectrogram
from .audio_file import AudioFile
from .params import Params
from .sound import Sound
from .spectrogram_file import SpectrogramFile


def default_spect_fname_func(audio_path: Union[str, pathlib.Path]):
    """Default function for naming spectrogram files.
    Adds the extension `.spect.npz` to an audio path.

    Parameters
    ----------
    audio_path : str, pathlib.Path
        A path to an audio file.

    Returns
    -------
    spect_fname : pathlib.Path
        Sound filename with extension added.
        Default extension is :data:`vocalpy.constants.SPECT_FILE_EXT`.

    Notes
    -----
    Adding an extension to the audio path
    (instead of changing it)
    makes it possible to recover the audio path
    from the spectrogram path.
    Adding a longer extension `.spect.npz`
    makes it less likely that the spectrogram file
    will overwrite an existing `.npz` file.
    """
    audio_path = pathlib.Path(audio_path)
    return audio_path.name + vocalpy.constants.SPECT_FILE_EXT


def validate_sound(sound: Sound | AudioFile | Sequence[Sound | AudioFile]) -> None:
    if not isinstance(sound, (Sound, AudioFile, list, tuple)):
        raise TypeError(
            "`sound` must be a `vocalpy.Sound` instance, "
            "a `vocalpy.AudioFile` instance, "
            "or a list/tuple of such instances, "
            f"but type was : {type(sound)}"
        )

    if isinstance(sound, list) or isinstance(sound, tuple):
        if not (
            all([isinstance(item, Sound) for item in sound]) or all([isinstance(item, AudioFile) for item in sound])
        ):
            types_in_sound = set([type(sound) for sound in sound])
            raise TypeError(
                "If `sound` is a list or tuple, "
                "then items in `sound` must either "
                "all be instances of `vocalpy.Sound`"
                "or all be instances of `vocalpy.AudioFile`."
                f"Instead found the following types: {types_in_sound}."
                f"Please make sure only `vocalpy.Sound instances are in the list/tuple."
            )


DEFAULT_SPECT_PARAMS = {"n_fft": 512, "hop_length": 64}


class SpectrogramMaker:
    """Class that represents the step in a pipeline that makes spectrograms from audio.

    Attributes
    ----------
    callback : Callable
        Callable that accepts a :class:`Sound`
        and returns a :class:`Spectrogram`.
        Default is :func:`vocalpy.spectrogram`.
    params : dict
        Parameters for making spectrograms.
        Passed as keyword arguments to ``callback``.
    """

    def __init__(self, callback: Callable | None = None, params: Mapping | Params | None = None):
        if callback is None:
            import vocalpy.spectrogram

            callback = vocalpy.spectrogram
            # if callback was None and we use the default,
            # **and** params is None, we set these default params
            if params is None:
                params = DEFAULT_SPECT_PARAMS
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
            raise ValueError(f"`callback` should be callable, but `callable({callback})` returns False")
        self.callback = callback

        if not isinstance(params, (collections.abc.Mapping, Params)):
            raise TypeError(f"`params` should be a `Mapping` or `Params` but type was: {type(params)}")

        if isinstance(params, Params):
            # coerce to dict
            params = {**params}

        signature = inspect.signature(callback)
        if not all([param in signature.parameters for param in params]):
            invalid_params = [param for param in params if param not in signature.parameters]
            raise ValueError(
                f"Invalid params for callback: {invalid_params}\n" f"Callback parameters are: {signature.parameters}"
            )

        self.params = params

    def __repr__(self):
        return f"FeatureExtractor(callback={self.callback.__qualname__}, params={self.params})"

    def make(
        self,
        sound: Sound | AudioFile | Sequence[Sound | AudioFile],
        parallelize: bool = True,
    ) -> Spectrogram | List[Spectrogram]:
        """Make spectrogram(s) from audio.

        Makes the spectrograms with `self.callback`
        using the parameters `self.params`.

        Takes as input :class:`vocalpy.Sound` or :class:`vocalpy.AudioFile`,
        or a sequence of either
        and returns either a :class:`vocalpy.Spectrogram`
        (given a single :class:`vocalpy.Sound` or :class:`vocalpy.AudioFile` instance)
        or a list of :class:`vocalpy.Spectrogram` instances (given a sequence).

        Parameters
        ----------
        sound: vocalpy.Sound, vocalpy.AudioFile, or a sequence of either
            Source of audio used to make spectrograms.

        Returns
        -------
        spectrogram : vocalpy.Spectrogram or list of vocalpy.Spectrogram
        """
        validate_sound(sound)

        # define nested function so vars are in scope and ``dask`` can call it
        def _to_spect(sound_):
            """Make a ``Spectrogram`` from an ``Sound`` instance,
            using self.callback"""
            if isinstance(sound_, AudioFile):
                sound_ = Sound.read(sound_.path)
            spect = self.callback(sound_, **self.params)
            spect.audio_path = sound_.path
            return spect

        if isinstance(sound, (Sound, AudioFile)):
            return _to_spect(sound)

        spects = []
        for sound_ in sound:
            if parallelize:
                spects.append(dask.delayed(_to_spect)(sound_))
            else:
                spects.append(_to_spect(sound_))

        if parallelize:
            graph = dask.delayed()(spects)
            with dask.diagnostics.ProgressBar():
                return graph.compute()
        else:
            return spects

    def write(
        self,
        sound: Sound | AudioFile | Sequence[Sound | AudioFile],
        dir_path: str | pathlib.Path,
        parallelize: bool = True,
        namer: Callable = default_spect_fname_func,
    ) -> SpectrogramFile | List[SpectrogramFile]:
        """Make spectrogram(s) from audio, and write to file.
        Writes directly to file without returning the spectrograms,
        so that datasets can be generated that are too big
        to fit in memory.

        Makes the spectrograms with `self.callback`
        using the parameters `self.params`.

        Takes as input :class:`vocalpy.Sound` or :class:`vocalpy.AudioFile`,
        or a sequence of either,
        and returns either a :class:`vocalpy.SpectrogramFile`
        (given a single :class:`vocalpy.Sound` or :class:`vocalpy.AudioFile` instance)
        or a list of :class:`vocalpy.Spectrogram` instances (given a sequence).

        Parameters
        ----------
        sound: vocalpy.Sound, vocalpy.AudioFile, a sequence of either, or a Dataset
            Source of audio used to make spectrograms.
        dir_path : string, pathlib.Path
            The directory where the spectrogram files should be saved.
        namer : callable
            Function or class that determines spectrogram file name
            from audio file name. Default is
            :func:`vocalpy.domain_model.services.spectrogram_maker.default_spect_name_func`.

        Returns
        -------
        spectrogram_file : SpectrogramFile, list of SpectrogramFile
            The file(s) containing the spectrogram(s).
        """
        validate_sound(sound)
        dir_path = pathlib.Path(dir_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise NotADirectoryError(f"`dir_path` not found or not recognized as a directory:\n{dir_path}")

        # define nested function so vars are in scope and ``dask`` can call it
        def _to_spect_file(sound_):
            """Compute a `Spectrogram` from an `Sound` instance,
            using self.callback"""
            if isinstance(sound_, AudioFile):
                sound_ = Sound.read(sound_.path)
            spect = self.callback(sound_, **self.params)
            spect_fname = namer(sound_.path)
            spect_path = dir_path / spect_fname
            spect_file = spect.write(spect_path)
            return spect_file

        if isinstance(sound, (Sound, AudioFile)):
            return _to_spect_file(sound)

        spect_files = []
        for sound_ in sound:
            if parallelize:
                spect_files.append(dask.delayed(_to_spect_file(sound_)))
            else:
                spect_files.append(_to_spect_file(sound_))

        if parallelize:
            graph = dask.delayed()(spect_files)
            with dask.diagnostics.ProgressBar():
                return graph.compute()
        else:
            return spect_files
