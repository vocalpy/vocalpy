"""Class that segments audio into sequences of units"""
from __future__ import annotations

from typing import Callable

import dask
import dask.diagnostics

from .audio import Audio
from .audio_file import AudioFile
from .sequence import Sequence
from .spectrogram_maker import validate_audio
from .unit import Unit

DEFAULT_SEGMENT_PARAMS = {
    "threshold": 5000,
    "min_dur": 0.02,
    "min_silent_dur": 0.002,
}


class Segmenter:
    def __init__(self, callback: Callable | None = None, method: str | None = None, segment_params: dict | None = None):
        if callback and method:
            raise ValueError(f"Cannot specify both `callback` and `method`, only one or the other.")

        if method:
            import vocalpy.signal.segment

            # TODO: fix this
            try:
                callback = getattr(vocalpy.signal.segment, method)
            except AttributeError:
                raise AttributeError(
                    f"Method was '{method}' but `vocalpy.signal.segment` has no function named `{method}`"
                )

        if callback is None:
            from vocalpy.signal.segment import audio_amplitude as default_segment_func

            callback = default_segment_func

        if callback is not None and not callable(callback):
            raise ValueError(f"`callback` should be callable, but `callable({callback})` returns False")

        self.callback = callback

        if segment_params is None:
            segment_params = DEFAULT_SEGMENT_PARAMS
        if not isinstance(segment_params, dict):
            raise TypeError(f"`segment_params` should be a `dict` but type was: {type(segment_params)}")

        self.segment_params = segment_params

    def segment(
        self,
        audio: Audio | AudioFile | Sequence[Audio | AudioFile],
        parallelize: bool = True,
    ) -> Sequence | list[Sequence]:
        validate_audio(audio)

        # define nested function so vars are in scope and ``dask`` can call it
        def _to_sequence(audio_: Audio):
            if isinstance(audio_, AudioFile):
                audio_ = Audio.read(audio_.path)
            onsets, offsets = self.callback(audio_.data, audio_.samplerate, **self.segment_params)

            units = []
            for onset, offset in zip(onsets, offsets):
                units.append(Unit(onset=onset, offset=offset))

            return Sequence(
                units=units,
                # note we make a new audio instance **without** data loaded
                audio=Audio(path=audio_.path),
                method=self.callback.__name__,
                segment_params=self.segment_params,
            )

        if isinstance(audio, (Audio, AudioFile)):
            return _to_sequence(audio)

        seqs = []
        for audio_ in audio:
            if parallelize:
                seqs.append(dask.delayed(_to_sequence(audio_)))
            else:
                seqs.append(_to_sequence(audio_))

        if parallelize:
            graph = dask.delayed()(seqs)
            with dask.diagnostics.ProgressBar():
                return graph.compute()
        else:
            return seqs
