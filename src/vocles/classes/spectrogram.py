from pathlib import Path

import numpy as np


class Spectrogram:
    def __init__(self,
                 s,
                 t,
                 f,
                 spect_path=None,
                 audio_path=None):
        s = np.array(s)
        if s.ndim < 2:
            raise ValueError(
                f'spectrogram should have at least 2 dimensions, '
                f'but number of dimensions was {s.ndim}'
            )

        if not validators.is_1d_or_row_or_column(data):
            raise ValueError(
                f'data for audio should be a 1-dimensional array, '
                f'but number of dimensions was {data.ndim}'
            )

        if not validators.is_1d_or_row_or_column(data):
            raise ValueError(
                f'data for audio should be a 1-dimensional array, '
                f'but number of dimensions was {data.ndim}'
            )

        self.s = s
        self.t = t
        self.f = f
        self.spect_path = spect_path
        self.audio_path = audio_path

    @classmethod
    def from_file(cls, spect_path):
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(
                f'file not found: {audio_path}'
            )
        data, samplerate = sf.read(audio_path)
