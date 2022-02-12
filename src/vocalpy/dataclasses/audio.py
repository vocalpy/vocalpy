import pathlib
import reprlib

import evfuncs
import soundfile


class Audio:
    """class that represents audio saved in a file.

    Attributes
    ----------
    data : numpy.ndarray
        audio data
    samplerate : int
        sampling rate in Hertz
    audio_path : str, pathlib.Path
        path to a file containing saved arrays.

    Examples
    --------
    >>> audio = vocalpy.Audio("1291.WAV")
    >>> audio
    Audio(data=array([ 0.   ... -0.00115967]), samplerate=44100, audio_path=PosixPath('1291.WAV'))
    """

    def __init__(self,
                 audio_path: [str, pathlib.Path],
                 ):
        self.audio_path = pathlib.Path(audio_path)

        self._data = None
        self._samplerate = None

    def __repr__(self):
        if any([getattr(self, attr) is None for attr in ('_data', '_samplerate')]):
            self._load()

        return (f'{self.__class__.__name__}('
                f'data={reprlib.repr(self._data)}, '
                f'samplerate={reprlib.repr(self._samplerate)}, '
                f'audio_path={self.audio_path!r})'
                )

    def _load(self):
        """function that lazy loads"""
        if not self.audio_path.exists():
            raise FileNotFoundError(
                f'audio file not found: {self.audio_path}'
            )

        try:
            self._data, self._samplerate = soundfile.read(self.audio_path)
        except RuntimeError:
            self._data, self._samplerate = evfuncs.load_cbin(self.audio_path)

    @property
    def data(self):
        if self._data is None:
            self._load()

        return self._data

    @property
    def samplerate(self):
        if self._samplerate is None:
            self._load()

        return self._samplerate

    @classmethod
    def from_file(cls, audio_path):
        return cls(audio_path)
