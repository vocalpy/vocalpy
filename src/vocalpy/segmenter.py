"""Class that segments audio into sequences of units"""
from __future__ import annotations

from typing import Callable

import dask
import dask.diagnostics


from .audio import Audio
from .audio_file import AudioFile
from .sequence import Sequence
from .spectrogram_maker import validate_audio


DEFAULT_SEGMENT_PARAMS = {
    'threshold': 5000,
    'min_dur': 0.02,
    'min_silent_dur': 0.002,
}


class Segmenter:
    def __init__(self, callback: Callable | None = None,
                 segment_params: dict | None = None):
        if callback is None:
            from vocalpy.signal.segment import segment as default_segment_func
            callback = default_segment_func
        if not callable(callback):
            raise ValueError(
                f"`callback` should be callable, but `callable({callback})` returns False"
            )
        self.callback = callback

        if segment_params is None:
            segment_params = DEFAULT_SEGMENT_PARAMS
        if not isinstance(segment_params, dict):
            raise TypeError(
                f"`segment_params` should be a `dict` but type was: {type(segment_params)}"
            )

        self.segment_params = segment_params

    def segment(self,
                audio: Audio | AudioFile | Sequence[Audio | AudioFile],
                parallelize: bool = True,
                ) -> Sequence | list[Sequence]:
        validate_audio(audio)

        # define nested function so vars are in scope and ``dask`` can call it
        def _to_sequence(audio_):
            if isinstance(audio_, AudioFile):
                audio_ = Audio.read(audio_.path)
            seq = self.callback(audio_, **self.segment_params)
            # TODO: add source path(s?) attribute to Sequence
            seq.source_audio_path = audio_.source_path
            return seq

        if isinstance(audio, (Audio, AudioFile)):
            return _to_sequence(audio)

        seqs = []
        for audio_ in audio:
            if parallelize:
                seqs.append(
                    dask.delayed(_to_sequence(audio_))
                )
            else:
                seqs.append(
                    _to_sequence(audio_)
                )

        if parallelize:
            graph = dask.delayed()(seqs)
            with dask.diagnostics.ProgressBar():
                return graph.compute()
        else:
            return seqs
