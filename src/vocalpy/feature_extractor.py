"""Class that represents the step in a pipeline that extracts features."""
from __future__ import annotations

import collections.abc
import inspect
import pathlib
from typing import Mapping, Union, TYPE_CHECKING

import dask
import dask.diagnostics

from . import Features

if TYPE_CHECKING:
    import Params, Sound, Segment, Segments

    FeatureSource = Union[
        Sound,
        list[Sound],
        Segment,
        list[Segment],
        Segments
    ]


class FeatureExtractor:
    """Class that represents the step in a pipeline
    that extracts features.

    Attributes
    ----------
    callback : Callable
        Callable that takes a :class:`Sound` or :class:`Segment`
        and returns :class:`Features`.
    params : dict
        Parameters for extracting :class:`Features`.
        Passed as keyword arguments to ``callback``.
    """
    def __init__(self, callback: callable, params: Mapping | Params | None = None):
        if not callable(callback):
            raise ValueError(
                f"`callback` should be callable, but `callable({callback})` returns False"
            )

        self.callback = callback

        if params is None:
            params = {}
            signature = inspect.signature(callback)
            for name, param in signature.parameters.items():
                if param.default is not inspect._empty:
                    params[name] = param.default

        from . import Params  # avoid circular import
        if not isinstance(params, (collections.abc.Mapping, Params)):
            raise TypeError(f"`params` should be a `Mapping` or `Params` but type was: {type(params)}")

        if isinstance(params, Params):
            # coerce to dict
            params = {**params}

        signature = inspect.signature(callback)
        if not all([param in signature.parameters for param in params]):
            invalid_params = [param for param in params if param not in signature.parameters]
            raise ValueError(
                f"Invalid params for callback: {invalid_params}\n" f"Callback parameters are: {signature.parameters}"
            )

        self.params = params

    def extract(self, source: FeatureSource, parallelize: bool = True) -> Features | list[Features]:

        # define nested function so vars are in scope and ``dask`` can call it
        def _to_features(source_: FeatureSource) -> Features:
            features: Features = self.callback(source_, **self.params)
            return features

        from . import Sound, Segment
        if isinstance(source, (Sound, Segment)):
            return _to_features(source)

        features = []
        for source_ in source:
            if parallelize:
                features.append(dask.delayed(_to_features(source_)))
            else:
                features.append(_to_features(source_))

        if parallelize:
            graph = dask.delayed()(features)
            with dask.diagnostics.ProgressBar():
                return graph.compute()
        else:
            return features
