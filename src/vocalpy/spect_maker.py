from typing import Callable, List, Sequence, Union
import pathlib

import dask.bag
import dask.diagnostics

import vocalpy.constants
from vocalpy.dataclasses import Audio, Spectrogram
from vocalpy.signal.spectrogram import spectrogram as default_spect_func


def default_spect_fname_func(audio_path: Union[str, pathlib.Path]):
    """default function for naming spectrogram files.
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


class SpectMaker:
    """class that makes spectrograms from audio

    Attributes
    ----------
    spect_func : Callable
    default_spect_kwargs : dict
        of keyword arguments
    """
    def __init__(self,
                 spect_func: Callable = default_spect_func,
                 default_spect_kwargs: dict = None):
        self.spect_func = spect_func

        if default_spect_kwargs is None:
            default_spect_kwargs = {}  # avoids mutable as arg default
        self.default_spect_kwargs = default_spect_kwargs

    def make(self,
             audio: Union[Audio, Sequence[Audio], Sequence[pathlib.Path]],
             spect_kwargs: dict = None,
             output_dir: Union[str, pathlib.Path] = None,
             spect_fname_func: Callable = default_spect_func) -> Union[Spectrogram, List[Spectrogram]]:
        """make spectrograms from audio.

        takes as input ``vocalpy.Audio``,
        a sequence of ``vocalpy.Audio``,
        or a sequence of paths to audio files,
        and returns either ``vocalpy.Spectrogram``
        or a list of ``vocalpy.Spectrogram``s.

        Parameters
        ----------
        audio: vocalpy.Audio, sequence of vocalpy.Audio, or sequence of paths to audio files
        spect_kwargs : dict
            keyword arguments to pass in to SpectMaker.spect_func.
            Default is None, in which case
            SpectMaker.default_spect_kwargs will be used
        output_dir : str, pathlib.Path
            directory where spectrograms should be saved.
            Default is None, in which case spectrograms
            are not saved,
        spect_fname_func : callable
            function that creates filename for spectrogram file,
            given audio.path as an input.
            Default is ``vocalpy.spect_maker.default_spect_fname_func``.

        Returns
        -------
        spect : vocalpy.Spectrogram or list of vocalpy.Spectrogram
        """
        # ---- pre-conditions ----
        if not isinstance(audio, Audio) and not (
                isinstance(audio, list) or isinstance(audio, tuple)
        ):
            raise TypeError(
                'audio must be a vocalpy.Audio instance, '
                'or a list/tuple of vocalpy.Audio instances, '
                f'but type was : {type(audio)}'
            )

        if isinstance(audio, list) or isinstance(audio, tuple):
            if not all([isinstance(item, Audio) for item in audio]):
                types_in_audio = set([type(item) for item in audio])
                raise TypeError(
                    'if ``audio`` is a list or tuple, '
                    'then all items in ``audio`` must be instances of vocalpy.Audio.'
                    f'Instead found the following types: {types_in_audio}.'
                    f'Please make sure only vocalpy.Audio instances are in the list/tuple.'
                )

        if output_dir is not None:
            output_dir = pathlib.Path(output_dir)
            if not output_dir.exists():
                raise NotADirectoryError(
                    f"`output_dir` not found or recognized as a directory: {output_dir}"
                )

        # ---- actually make the spectrograms ----
        # define nested function so vars are in scope and ``dask`` can call it
        def _to_spect(audio_):
            """compute a ``Spectrogram`` from an ``Audio`` instance,
            using self.spect_func"""
            if isinstance(audio_, str) or isinstance(audio_, pathlib.Path):
                try:
                    audio_ = Audio.from_file(audio_)
                except FileNotFoundError as e:
                    raise FileNotFoundError(f"did not find audio file: {audio_}") from e

            if spect_kwargs:
                s, t, f = self.spect_func(audio_.data, **spect_kwargs)
            else:
                s, t, f = self.spect_func(audio_.data, **self.default_spect_kwargs)
            spect = Spectrogram(s=s, t=t, f=f, audio_path=audio_.path)
            if output_dir is not None:
                spect_path = spect_fname_func(audio_.path)
                spect.to_file(spect_path)
            return spect

        if isinstance(audio, Audio):
            return _to_spect(audio)
        else:
            audio_bag = dask.bag.from_sequence(audio)
            with dask.diagnostics.ProgressBar():
                spects = list(audio_bag.map(_to_spect))
            return spects
