from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from .. import validators
from ..signal.spectrogram import spectrogram
from .spectrogram import Spectrogram


class Audio:
    def __init__(self,
                 path: [str, Path],
                 data: Optional[np.array] = None,
                 samplerate: Optional[int] = None,
                 ):
        if data and samplerate is None:
            raise ValueError(
                f'must provide sampling rate with audio data'
            )
        if samplerate and data is None:
            raise ValueError(
                f'must provide audio data with sampling rate'
            )

        self.data = data
        self.samplerate = samplerate
        self.path = path

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'path={self.path!r}, data={self.data!r}), samplerate={self.samplerate}')

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        value = Path(value)
        self._path = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if value is not None:
            value = np.array(value)
            if not validators.is_1d_or_row_or_column(value):
                raise ValueError(
                    f'data for audio should be a 1-dimensional array, '
                    f'but number of dimensions was {value.ndim}'
                )
        self._data = value

    @property
    def samplerate(self):
        return self._samplerate

    @samplerate.setter
    def samplerate(self, value):
        if value is not None:
            if not isinstance(value, int):
                raise TypeError(f'sampling rate should be an integer but was {type(value)}')
            if value < 0:
                raise ValueError(f'sampling rate should be a positive integer but was {value}')
        self._samplerate = value

    @classmethod
    def from_file(cls, audio_path):
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(
                f'file not found: {audio_path}'
            )
        data, samplerate = sf.read(audio_path)
        return cls(data, samplerate, audio_path)

    def to_spect(self,
                 spect_kwargs,
                 spect_maker=None,
                 ):
        if spect_maker is None:
            spect_maker = spectrogram
        s, t, f = spect_maker(self.data, self.samplerate, **spect_kwargs)
        spect = Spectrogram(s, t, f, audio_path=self.path)
        return spect
