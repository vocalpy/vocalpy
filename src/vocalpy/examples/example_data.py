"""A :class:`dict`-like container for example data."""

import reprlib


class ExampleData(dict):
    """A :class:`dict`-like container for example data.

    Returned by :func:`vocalpy.example` for any example
    that is more than a single file.
    The :class:`ExampleData` class extends :class:`dict`
    by enabling values to be accessed by key, ``example["data"]``,
    or by attribute with dot notation, ``example.data``.

    Examples
    --------

    >>> from vocalpy.examples import ExampleData
    >>> bells = voc.example("bells.wav")
    >>> samba = voc.example("samba.wav")
    >>> zb_examples = ExampleData(bells=bells, samba=samba)
    >>> zb_examples["bells"]
    vocalpy.Sound(data=array([[-6.10...0000000e+00]]), samplerate=44100)
    >>> zb_examples.samba
    vocalpy.Sound(data=array([[0.003... 0.        ]]), samplerate=44100)
    >>> zb_examples.flashcam = voc.example("flashcam.wav")
    >>> zb_examples["flashcam"]
    vocalpy.Sound(data=array([[0.000...5527344e-05]]), samplerate=44100)

    Notes
    -----

    Adapted from the scikit-learn ``Bunch`` class: 
    https://github.com/scikit-learn/scikit-learn/blob/d5082d32d/sklearn/utils/_bunch.py
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __repr__(self):
        inside_parens = ", ".join(
            [f"{k}={reprlib.repr(v)}" for k, v in self.items()]
        )
        return f"ExampleData({inside_parens})"

    def __getitem__(self, key):
        return super().__getitem__(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)
