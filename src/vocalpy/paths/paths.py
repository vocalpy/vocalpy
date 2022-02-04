from pathlib import Path


def from_dir(cls, dir_path, ext):
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(
            f'not recognized as a directory: {dir}'
        )
    paths = sorted(dir_path.glob(f'*{ext}'))
    return paths
