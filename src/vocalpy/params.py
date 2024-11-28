"""A class to represent the parameters used with a method,
that can be unpacked with the ``**`` operator.

Dataclasses that represent parameters for a method
should inherit from this class,
so that instances of those dataclasses can be passed
into the classes that represent steps in workflows.
"""

from dataclasses import asdict


class Params:
    """A class to represent the parameters used with a method,
    that can be unpacked with the ``**`` operator.

    Dataclasses that represent parameters for a method
    should inherit from this class,
    so that instances of those dataclasses can be passed
    into the classes that represent steps in workflows.
    """

    def keys(self):
        return asdict(self).keys()

    def __getitem__(self, item):
        if getattr(self, "_dict", None) is None:
            self._dict = asdict(self)
        return self._dict[item]
