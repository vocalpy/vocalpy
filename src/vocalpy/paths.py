"""Functions for working with paths.
"""

from __future__ import annotations

import pathlib


def from_dir(dir: str | pathlib.Path, ext: str, recurse: bool = False):
    """Get all paths with the specified extension from a directory.

    Searches the directory with the :meth:`pathlib.Path.glob` method.
    If ``recurse`` is True, a recursive glob is used
    (by prefixing the search string with double asterisks:
    ``sorted(dir.glob(f'**/*ext')``.

    Parameters
    ----------
    dir : str, pathlib.Path
        The path to the directory.
    ext : str
        The file extension to find, e.g. 'wav' or '.wav'.
    recurse : bool
        If True, search recursively in sub-directories of ``dir``.

    Returns
    -------
    paths : list
        A :class:`list` of :class:`pathlib.Path` instances.
        Will be empty if no files with the extension were found.
    """
    dir = pathlib.Path(dir)
    if not dir.is_dir():
        raise NotADirectoryError(
            f"`dir` argument not recognized as a directory: {dir}"
        )

    if recurse:
        paths = sorted(dir.glob(f"**/*{ext}"))
    else:
        paths = sorted(dir.glob(f"*{ext}"))

    return paths
