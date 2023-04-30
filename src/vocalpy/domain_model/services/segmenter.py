from typing import Callable

from vocalpy.domain_model.entities import Dataset, Sequence


class Segmenter:
    def __init__(self, callback: Callable, params: dict):
        self.callback = callback
        self.params = params

    def segment(self, dataset: Dataset) -> list[Sequence]:
        # TODO: figure out how to handle case where we segment with features
        # how will Segmenter know whether to use spectrogram, audio, etc.?
        pass
