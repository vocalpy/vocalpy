"""Registry for example data."""
from __future__ import annotations

import importlib.resources
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pooch

if TYPE_CHECKING:
    import vocalpy


@dataclass
class Example:
    name: str
    metadata: str
    type: str
    fname: str | None = None
    ext: str = "wav"
    requires_download: bool = False


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
    Example(
        name="bfsongrepo",
        metadata="""Sample of song from Bengalese Finch Song Repository.    
Nicholson, David; Queen, Jonah E.; J. Sober, Samuel (2017). Bengalese Finch song repository. figshare. Dataset. https://doi.org/10.6084/m9.figshare.4805749.v9
https://nickledave.github.io/bfsongrepo
Files are approximately 20 songs from bird with ID "gy6or6", from the day "032312"
""",
        type="audio",
        ext="cbin",
        fname="bfsongrepo.tar.gz",
        requires_download=True,
    ),
    Example(
        name="jourjine-et-al-2023",
        metadata="""Sample of deer mouse vocalizations from:
Jourjine, Nicholas et al. (2023). Data from: 
Two pup vocalization types are genetically and functionally separable in deer mice [Dataset].
Dryad. https://doi.org/10.5061/dryad.g79cnp5ts
Audio files are 20-second clips from approximately 10 files in the developmentLL data,
(https://datadryad.org/stash/downloads/file_stream/2143657), generated with the script:
tests/scripts/generate_ava_segment_test_data/generate_test_audio_for_ava_segment_from_jourjine_etal_2023.py
""",
        type="audio",
        fname="jourjine-et-al-2023.tar.gz",
        requires_download=True,
    )
]


REGISTRY = {example.name: example for example in EXAMPLES}


POOCH = pooch.create(
    path=pooch.os_cache("vocalpy"),
    base_url="doi:10.5281/zenodo.10685782",
    registry=None,
)
POOCH.load_registry_from_doi()


def example(name: str) -> vocalpy.Sound | list[vocalpy.Sound]:
    """Get an example from :mod:`vocalpy.examples`.

    Returns a single example, e.g., a :class:`vocalpy.Sound`.

    Parameters
    ----------
    name : str
        Name of the example.
        To see names of examples and the associated metadata,
        call :func:`vocalpy.examples.list`.

    Returns
    -------
    example : vocalpy.Sound
        A :class:`vocalpy.Sound` instance with the example data
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
    import vocalpy  # avoid circular import

    example_: Example = REGISTRY[name]
    if example_.requires_download:
        if example_.fname.endswith('.tar.gz'):
            paths = POOCH.fetch(example_.fname, processor=pooch.Untar())
            if example_.ext.endswith("cbin"):
                sounds = [
                    vocalpy.Sound.read(path) for path in paths
                    # don't try to load .rec files
                    if path.endswith("cbin")
                ]
            else:
                sounds = [
                    vocalpy.Sound.read(path) for path in paths
                ]
            return sounds
        else:
            path = POOCH.fetch(example_.fname)
            sound = vocalpy.Sound.read(path)
            return sound
    else:
        path = importlib.resources.files("vocalpy.examples").joinpath(name)
        if example_.type == "audio":
            return vocalpy.Sound.read(path)
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
