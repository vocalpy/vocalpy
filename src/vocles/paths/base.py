from pathlib import Path


class BasePaths:
    def __init__(self, paths):
        self._paths = paths

    @classmethod
    def from_dir(cls, dir, globstr='*'):
        dir_path = Path(dir)
        if not dir_path.is_dir():
            raise NotADirectoryError(
                f'not recognized as a directory: {dir}'
            )
        paths = sorted(dir_path.glob(globstr))
        return cls(paths)
