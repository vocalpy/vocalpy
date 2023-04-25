import pathlib
from typing import Callable, List, Sequence, Union

import dask
import dask.diagnostics

import vocalpy.constants
from vocalpy.domain_model import (
    Audio,
    AudioFile,
    Dataset,
    Spectrogram,
    SpectrogramFile,
    SpectrogramParameters,
)
from vocalpy.signal.spectrogram import spectrogram as default_spect_func


def default_spect_fname_func(audio_path: Union[str, pathlib.Path]):
    """Default function for naming spectrogram files.
    Adds the extension `.spect.npz` to an audio path.

    Parameters
    ----------
    audio_path : str, pathlib.Path

    Returns
    -------
    spect_path : pathlib.Path
        audio path with the extension `.spect.npz` added.

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
    return audio_path.parent / (audio_path.name + vocalpy.constants.SPECT_FILE_EXT)


def validate_audio_source(audio_source):
    if not isinstance(audio_source, (Audio, AudioFile, Dataset, list, tuple)):
        raise TypeError(
            "`audio_source` must be a `vocalpy.Audio` instance, "
            "or a list/tuple of vocalpy.Audio instances, "
            f"but type was : {type(audio_source)}"
        )

    if isinstance(audio_source, list) or isinstance(audio_source, tuple):
        if not all([isinstance(item, (Audio, AudioFile)) for item in audio_source]):
            types_in_audio = set([type(item) for item in audio_source])
            raise TypeError(
                "if ``audio`` is a list or tuple, "
                "then all items in ``audio`` must be instances of vocalpy.Audio."
                f"Instead found the following types: {types_in_audio}."
                f"Please make sure only vocalpy.Audio instances are in the list/tuple."
            )


class SpectrogramMaker:
    """Class that makes spectrograms from audio.

    Attributes
    ----------
    spectrogram_callable : Callable
        Callable that takes audio and returns spectrograms.
        Default is :func:`vocalpy.signal.spectrogram.spectrogram`.
    params : SpectrogramConfig, dict
        Parameters for making spectrograms.
    """

    def __init__(self, callback: Callable = default_spect_func,
                 params: SpectrogramParameters | dict = None):
        self.spectrogram_callable = callback

        if params is None:
            # FIXME: fix magic number -- default kwarg?
            params = SpectrogramParameters(fft_size=512)
        elif isinstance(params, dict):
            params = SpectrogramParameters(**params)
        self.params = params

    def make(
        self,
        audio_source: Audio | AudioFile | Sequence[Audio | AudioFile] | Dataset,
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
        audio_source: vocalpy.Audio, vocalpy.AudioFile, a sequence of either, or a Dataset
            Source of audio used to make spectrograms.

        Returns
        -------
        spectrogram : vocalpy.Spectrogram or list of vocalpy.Spectrogram
        """
        validate_audio_source(audio_source)

        # define nested function so vars are in scope and ``dask`` can call it
        def _to_spect(audio_):
            """Make a ``Spectrogram`` from an ``Audio`` instance,
            using self.callback"""
            if isinstance(audio_, AudioFile):
                audio_ = Audio.read(audio_.path)
            spect = self.spectrogram_callable(audio_, **self.params)
            spect.source_audio_path = audio.path
            return spect

        if isinstance(audio_source, (Audio, AudioFile)):
            return _to_spect(audio_source)

        if isinstance(audio_source, Dataset):
            if not hasattr(audio_source, 'audio_files'):
                raise AttributeError(
                    f"`audio_source` was a `vocalpy.Dataset` but it does "
                    f"not have an `audio_files` attribute. Please supply "
                    f"a dataset with `audio_files` or pass audio "
                    f"or audio files directly into `make` method"
                )
            audios = audio_source.audio_files
        else:
            audios = audio_source

        spects = []
        for audio in audios:
            if parallelize:
                spects.append(
                    dask.delayed(_to_spect(audio))
                )
            else:
                spects.append(
                    _to_spect(audio)
                )

        if parallelize:
            graph = dask.delayed()(spects)
            with dask.diagnostics.ProgressBar():
                return graph.compute()
        else:
            return spects

    def write(self,
              audio_source: Audio | AudioFile | Sequence[Audio | AudioFile] | Dataset,
              dir_path : str | pathlib.Path,
              parallelize: bool = True,
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
        audio_source: vocalpy.Audio, vocalpy.AudioFile, a sequence of either, or a Dataset
            Source of audio used to make spectrograms.
        dir_path : string, pathlib.Path
            The directory where the spectrogram files should be saved.

        Returns
        -------
        spectrogram_file : SpectrogramFile, list of SpectrogramFile
            The file(s) containing the spectrogram(s).
        """
        validate_audio_source(audio_source)
        dir_path = pathlib.Path(dir_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise NotADirectoryError(
                f"`dir_path` not found or not recognized as a directory:\n{dir_path}"
            )

        # define nested function so vars are in scope and ``dask`` can call it
        def _to_spect_file(audio_):
            """compute a ``Spectrogram`` from an ``Audio`` instance,
            using self.callback"""
            if isinstance(audio_, AudioFile):
                audio_ = Audio.read(audio_.path)
            spect = self.spectrogram_callable(audio_, **self.params)
            spect_file = spect.write(dir_path)
            return spect_file

        if isinstance(audio_source, (Audio, AudioFile)):
            return _to_spect_file(audio_source)

        if isinstance(audio_source, Dataset):
            if not hasattr(audio_source, 'audio_files'):
                raise AttributeError(
                    f"`audio_source` was a `vocalpy.Dataset` but it does "
                    f"not have an `audio_files` attribute. Please supply "
                    f"a dataset with `audio_files` or pass audio "
                    f"or audio files directly into `make` method"
                )
            audios = audio_source.audio_files

        spect_files = []
        for audio in audios:
            if parallelize:
                spects.append(
                    dask.delayed(_to_spect(audio))
                )
            else:
                spects.append(
                    _to_spect(audio)
                )

        if parallelize:
            graph = dask.delayed()(spects)
            with dask.diagnostics.ProgressBar():
                return graph.compute()
        else:
            return spects

        # TODO: if Dataset, add spectrogram_files to dataset
        return spects