"""Registry for example data."""
from __future__ import annotations

import importlib.resources
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import vocalpy


@dataclass
class Example:
    name: str
    metadata: str
    type: str


EXAMPLES = [
    Example(
        name="bells.wav",
        metadata="Zebra finch song from Sound Analysis Pro website: http://soundanalysispro.com/",
        type="audio",
    ),
    Example(
        name="flashcam.wav",
        metadata="Zebra finch song from Sound Analysis Pro website: http://soundanalysispro.com/",
        type="audio",
    ),
    Example(
        name="samba.wav",
        metadata="Zebra finch song from Sound Analysis Pro website: http://soundanalysispro.com/",
        type="audio",
    ),
    Example(
        name="simple.wav",
        metadata="Zebra finch song from Sound Analysis Pro website: http://soundanalysispro.com/",
        type="audio",
    ),
    Example(
        name="BM003.wav",
        metadata="""Mouse ultrasonic vocalization from:
Goffinet, J., Brudner, S., Mooney, R., & Pearson, J. (2021).
Data from: Low-dimensional learned feature spaces quantify individual and group differences
in vocal repertoires. Duke Research Data Repository. https://doi.org/10.7924/r4gq6zn8w.
Adapted under Creative Commons License 1.0: https://creativecommons.org/publicdomain/zero/1.0/.
File name in dataset: BM003_day9_air_20s_sparse_chunk007_0297.wav""",
        type="audio",
    ),
]


REGISTRY = {example.name: example for example in EXAMPLES}


def example(name: str) -> vocalpy.Audio:
    """Get an example from :mod:`vocalpy.examples`.

    Returns a single example, e.g., a :class:`vocalpy.Audio`.

    Parameters
    ----------
    name : str
        Name of the example.
        To see names of examples and the associated metadata,
        call :func:`vocalpy.examples.list`.

    Returns
    -------
    example : vocalpy.Audio
        A :class:`vocalpy.Audio` instance with the example data
        read into it.

    See Also
    --------
    vocalpy.examples.list

    Notes
    -----
    To see all built-in examples, call :func:`vocalpy.examples.list`.

    Examples
    --------
    >>> audio = voc.example('bells.wav')
    >>> spect = voc.spectrogram(audio)
    >>> voc.plot.spect(spect)
    """
    if name not in REGISTRY:
        raise ValueError(f"No example found with name: {name}. " f"For a list of examples, call ")
    example_ = REGISTRY[name]
    path = importlib.resources.files("vocalpy.examples").joinpath(name)
    if example_.type == "audio":
        import vocalpy

        return vocalpy.Audio.read(path)
    else:
        raise ValueError(f"The ``type`` for the example was invalid: {example_.type}")


def list():
    """List example data in :mod:`vocalpy`.

    Examples are listed in the form
    ``{name: metadata}``,
    where ``name`` is the string you would pass to
    :func:`vocalpy.example` to retrieve the example,
    and ``metadata`` is a string containing
    metadata: the source of the data,
    and any relevant citation.

    See Also
    --------
    vocalpy.example

    Notes
    -----
    To retrieve an example, call :func:`vocalpy.example`.
    """
    print("Examples built into VocalPy")
    print("=" * 72)
    for example in EXAMPLES:
        print(f"name: {example.name}\n" "metadata:\n" f"{example.metadata}\n")
