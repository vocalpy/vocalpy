from pathlib import Path

import numpy as np
import soundfile as sf

from .. import validators
from ..signal.spectrogram import spectrogram
from .spectrogram import Spectrogram


class Audio:
    def __init__(self,
                 data,
                 samplerate,
                 audio_path=None):
        data = np.array(data)
        if not validators.is_1d_or_row_or_column(data):
            raise ValueError(
                f'data for audio should be a 1-dimensional array, '
                f'but number of dimensions was {data.ndim}'
            )

        if not isinstance(samplerate, int):
            raise TypeError(f'sampling rate should be an integer but was {type(samplerate)}')
        if samplerate < 0:
            raise TypeError(f'sampling rate should be an integer but was {type(samplerate)}')

        self.data = data
        self.samplerate = samplerate
        self.audio_path = audio_path

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
        spect = Spectrogram(s, t, f, audio_path=self.audio_path)
        return spect
