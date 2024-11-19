"""Class that represents the step in a pipeline that extracts features."""

from __future__ import annotations

import collections.abc
import inspect
from typing import TYPE_CHECKING, Mapping

import dask
import dask.diagnostics

if TYPE_CHECKING:
    from . import Features, Params, Sound


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

    def __init__(
        self, callback: callable, params: Mapping | Params | None = None
    ):
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
            raise TypeError(
                f"`params` should be a `Mapping` or `Params` but type was: {type(params)}"
            )

        if isinstance(params, Params):
            # coerce to dict
            params = {**params}

        signature = inspect.signature(callback)
        if not all([param in signature.parameters for param in params]):
            invalid_params = [
                param for param in params if param not in signature.parameters
            ]
            raise ValueError(
                f"Invalid params for callback: {invalid_params}\n"
                f"Callback parameters are: {signature.parameters}"
            )

        self.params = params

    def __repr__(self):
        return f"FeatureExtractor(callback={self.callback.__qualname__}, params={self.params})"

    def extract(
        self, sound: Sound | list[Sound], parallelize: bool = True
    ) -> Features | list[Features]:
        from . import Features, Sound

        if not isinstance(sound, (list, Sound)):
            raise TypeError(
                "`sound` must be an instance of a `Sound` "
                f"or a `list` of `Sound` instances, "
                f"but type was: {type(sound)}"
            )

        if isinstance(sound, list):
            if not all([isinstance(sound_, Sound) for sound_ in sound]):
                types = set(type(el) for el in sound)
                raise TypeError(
                    "A `list` passed to `FeatureExtract.extract` must be all `Sound` instances, "
                    f"but found the following types: {types}"
                )

        # define nested function so vars are in scope and ``dask`` can call it
        def _to_features(sound_: Sound) -> Features:
            return self.callback(sound_, **self.params)

        if isinstance(sound, Sound):
            return _to_features(sound)

        elif isinstance(sound, list) and all(
            [isinstance(sound_, Sound) for sound_ in sound]
        ):
            features = []
            for sound_ in sound:
                if parallelize:
                    features.append(dask.delayed(_to_features)(sound_))
                else:
                    features.append(_to_features(sound_))

            if parallelize:
                graph = dask.delayed()(features)
                with dask.diagnostics.ProgressBar():
                    return graph.compute()
            else:
                return features
