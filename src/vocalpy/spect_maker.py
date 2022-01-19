from typing import Callable
import pathlib

import dask.bag
import dask.diagnostics

from vocalpy.dataclasses import Audio, Spectrogram


class SpectMaker:
    """class that makes spectrograms from audio

    Attributes
    ----------
    spect_func : Callable
    default_spect_kwargs : dict
        of keywoard arguments
    """
    def __init__(self,
                 spect_func: Callable = default_spect_func,
                 default_spect_kwargs: dict = None):
        self.spect_func = spect_func
        self.default_spect_kwargs = default_spect_kwargs

    def make(self,
             audio: [Audio, list[Audio], tuple[Audio]],
             spect_kwargs: dict=None,
             output_dir: [str, pathlib.Path] = None,
             key: Callable = None) -> [Spectrogram, list[Spectrogram]]:
        """make spectrograms from audio

        Parameters
        ----------
        audio: vocalpy.Audio or list of vocalpy.Audio
        spect_kwargs : dict
            keyword arguments to pass in to SpectMaker.spect_func.
            Default is None, in which case
            SpectMaker.default_spect_kwargs will be used
        dir : str, pathlib.Path
            directory where

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

        # ---- actually make the spectrograms ----
        # define nested function so vars are in scope and ``dask`` can call it
        def _to_spect(audio_):
            """compute a ``Spectrogram`` from an ``Audio`` instance,
            using self.spect_func"""
            if spect_kwargs:
                s, t, f = self.spect_func(audio_.data, **spect_kwargs)
            else:
                s, t, f = self.spect_func(audio_.data, **self.default_spect_kwargs)
            return Spectrogram(s=s, t=t, f=f, audio_path=audio_.path)

        if isinstance(audio, Audio):
            return _to_spect(audio)
        else:
            audio_bag = dask.bag.from_sequence(audio)
            with dask.diagnostics.ProgressBar():
                spects = list(audio_bag.map(_to_spect))
            return spects
