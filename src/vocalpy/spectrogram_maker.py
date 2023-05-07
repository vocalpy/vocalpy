from __future__ import annotations

import pathlib
from typing import Callable, List, Sequence, Union

import dask
import dask.diagnostics

import vocalpy.constants

from .audio import Audio
from .audio_file import AudioFile
from .spectrogram import Spectrogram
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
        Audio filename with extension added.
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


def validate_audio(audio: Audio | AudioFile | Sequence[Audio | AudioFile]) -> None:
    if not isinstance(audio, (Audio, AudioFile, list, tuple)):
        raise TypeError(
            "`audio` must be a `vocalpy.Audio` instance, "
            "a `vocalpy.AudioFile` instance, "
            "or a list/tuple of such instances, "
            f"but type was : {type(audio)}"
        )

    if isinstance(audio, list) or isinstance(audio, tuple):
        if not (
            all([isinstance(item, Audio) for item in audio]) or all([isinstance(item, AudioFile) for item in audio])
        ):
            types_in_audio = set([type(audio) for audio in audio])
            raise TypeError(
                "If `audio` is a list or tuple, "
                "then items in `audio` must either "
                "all be instances of `vocalpy.Audio`"
                "or all be instances of `vocalpy.AudioFile`."
                f"Instead found the following types: {types_in_audio}."
                f"Please make sure only `vocalpy.Audio instances are in the list/tuple."
            )


DEFAULT_SPECT_PARAMS = {"fft_size": 512, "step_size": 64}


class SpectrogramMaker:
    """Class that makes spectrograms from audio.

    Attributes
    ----------
    callback : Callable
        Callable that takes audio and returns spectrograms.
        Default is :func:`vocalpy.signal.spectrogram.spectrogram`.
    spect_params : dict
        Parameters for making spectrograms.
        Passed as keyword arguments to ``callback``.
    """

    def __init__(self, callback: Callable | None = None, spect_params: dict | None = None):
        if callback is None:
            from vocalpy.signal.spectrogram import spectrogram as default_spect_func

            callback = default_spect_func
        if not callable(callback):
            raise ValueError(f"`callback` should be callable, but `callable({callback})` returns False")
        self.callback = callback

        if spect_params is None:
            spect_params = DEFAULT_SPECT_PARAMS
        if not isinstance(spect_params, dict):
            raise TypeError(f"`spect_params` should be a `dict` but type was: {type(spect_params)}")
        self.spect_params = spect_params

    def make(
        self,
        audio: Audio | AudioFile | Sequence[Audio | AudioFile],
        parallelize: bool = True,
    ) -> Spectrogram | List[Spectrogram]:
        """Make spectrogram(s) from audio.

        Makes the spectrograms with `self.callback`
        using the parameters `self.params`.

        Takes as input :class:`vocalpy.Audio` or :class:`vocalpy.AudioFile`,
        a sequence of either, or a :class:`vocalpy.Dataset` with an
        ``audio_files`` attribute,
        and returns either a :class:`vocalpy.Spectrogram`
        (given a single :class:`vocalpy.Audio` or :class:`vocalpy.AudioFile` instance)
        or a list of :class:`vocalpy.Spectrogram` instances (given a sequence).

        Parameters
        ----------
        audio: vocalpy.Audio, vocalpy.AudioFile, or a sequence of either
            Source of audio used to make spectrograms.

        Returns
        -------
        spectrogram : vocalpy.Spectrogram or list of vocalpy.Spectrogram
        """
        validate_audio(audio)

        # define nested function so vars are in scope and ``dask`` can call it
        def _to_spect(audio_):
            """Make a ``Spectrogram`` from an ``Audio`` instance,
            using self.callback"""
            if isinstance(audio_, AudioFile):
                audio_ = Audio.read(audio_.path)
            spect = self.callback(audio_, **self.spect_params)
            spect.source_audio_path = audio_.path
            return spect

        if isinstance(audio, (Audio, AudioFile)):
            return _to_spect(audio)

        spects = []
        for audio_ in audio:
            if parallelize:
                spects.append(dask.delayed(_to_spect(audio_)))
            else:
                spects.append(_to_spect(audio_))

        if parallelize:
            graph = dask.delayed()(spects)
            with dask.diagnostics.ProgressBar():
                return graph.compute()
        else:
            return spects

    def write(
        self,
        audio: Audio | AudioFile | Sequence[Audio | AudioFile],
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

        Takes as input :class:`vocalpy.Audio` or :class:`vocalpy.AudioFile`,
        a sequence of either, or a :class:`vocalpy.Dataset` with an
        ``audio_files`` attribute,
        and returns either a :class:`vocalpy.SpectrogramFile`
        (given a single :class:`vocalpy.Audio` or :class:`vocalpy.AudioFile` instance)
        or a list of :class:`vocalpy.Spectrogram` instances (given a sequence).

        Parameters
        ----------
        audio: vocalpy.Audio, vocalpy.AudioFile, a sequence of either, or a Dataset
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
        validate_audio(audio)
        dir_path = pathlib.Path(dir_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise NotADirectoryError(f"`dir_path` not found or not recognized as a directory:\n{dir_path}")

        # define nested function so vars are in scope and ``dask`` can call it
        def _to_spect_file(audio_):
            """Compute a `Spectrogram` from an `Audio` instance,
            using self.callback"""
            if isinstance(audio_, AudioFile):
                audio_ = Audio.read(audio_.path)
            spect = self.callback(audio_, **self.spect_params)
            spect_fname = namer(audio_.path)
            spect_path = dir_path / spect_fname
            spect_file = spect.write(spect_path)
            return spect_file

        if isinstance(audio, (Audio, AudioFile)):
            return _to_spect_file(audio)

        spect_files = []
        for audio_ in audio:
            if parallelize:
                spect_files.append(dask.delayed(_to_spect_file(audio_)))
            else:
                spect_files.append(_to_spect_file(audio_))

        if parallelize:
            graph = dask.delayed()(spect_files)
            with dask.diagnostics.ProgressBar():
                return graph.compute()
        else:
            return spect_files
