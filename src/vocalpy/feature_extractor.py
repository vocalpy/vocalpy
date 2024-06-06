"""Class that represents the step in a pipeline that extracts features."""
from __future__ import annotations

import collections.abc
import inspect
from typing import TYPE_CHECKING, Mapping, Union

import dask
import dask.diagnostics

if TYPE_CHECKING:
    from . import Features, Params, Segment, Segments, Sound

    FeatureSource = Union[Sound, list[Sound], Segment, list[Segment], Segments]


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
            raise ValueError(f"`callback` should be callable, but `callable({callback})` returns False")

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

    def __repr__(self):
        return f"FeatureExtractor(callback={self.callback.__qualname__}, params={self.params})"

    def extract(self, source: FeatureSource, parallelize: bool = True) -> Features | list[Features]:
        from . import Features, Segment, Segments, Sound

        if not isinstance(source, (list, Segment, Segments, Sound)):
            raise TypeError(
                "`source` to extract features from must be an instance of a `Sound`, a `Segment`, "
                f"a `list` of `Sound` or `Segment` instances, or a `Segments` instance, "
                f"but type was: {type(source)}"
            )

        if isinstance(source, list):
            if not (
                all([isinstance(source_, Segment) for source_ in source])
                or all([isinstance(source_, Segments) for source_ in source])
                or all([isinstance(source_, Sound) for source_ in source])
            ):
                types = set(type(el) for el in source)
                raise TypeError(
                    "A `list` passed to `FeatureExtract.extract` must be either all `Segment` instances, "
                    f"all `Segments` instances, or all `Sound` instances, but found the following types: {types}"
                )

        # define nested function so vars are in scope and ``dask`` can call it
        def _to_features(source_: FeatureSource) -> Features:
            if isinstance(source_, Segment):
                features = Features(data=self.callback(source_.sound, **self.params))
            elif isinstance(source_, Sound):
                features = Features(self.callback(source_, **self.params))
            return features

        if isinstance(source, (Sound, Segment)):
            return _to_features(source)

        elif (
            isinstance(source, Segments)
            or (isinstance(source, list) and all([isinstance(source_, Sound) for source_ in source]))
            or (isinstance(source, list) and all([isinstance(source_, Segment) for source_ in source]))
        ):
            features = []
            for source_ in source:
                if parallelize:
                    features.append(dask.delayed(_to_features)(source_))
                else:
                    features.append(_to_features(source_))

            if parallelize:
                graph = dask.delayed()(features)
                with dask.diagnostics.ProgressBar():
                    return graph.compute()
            else:
                return features

        elif isinstance(source, list) and all([isinstance(source, Segments)]):
            features_list = []
            for segments in source:
                features = []
                for segment in segments:
                    if parallelize:
                        features.append(dask.delayed(_to_features)(segment))
                    else:
                        features.append(_to_features(segment))

            if parallelize:
                graph = dask.delayed()(features_list)
                with dask.diagnostics.ProgressBar():
                    return graph.compute()
            else:
                return features_list
