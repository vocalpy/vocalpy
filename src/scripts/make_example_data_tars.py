"""Make .tar.gz archives of example data

For now we use a subset of data that is in ./tests/data-for-tests/source.
"""
import pathlib
import tarfile

import vocalpy as voc


def tar_source_data_subidr(dataset_root,
                           tar_dst,
                           archive_name,
                           ext='wav',
                           dry_run=False,
                           skip_exists=False):
    dataset_root = pathlib.Path(dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise NotADirectoryError(
            f'Dataset root not found: {dataset_root}'
        )
    tar_dst = pathlib.Path(tar_dst).expanduser().resolve()
    if not tar_dst.exists():
        raise NotADirectoryError(
            f'.tar destination root not found: {tar_dst}'
        )

    archive_path = tar_dst / f"{archive_name}.tar.gz"
    print(
        f'will create archive: {archive_path}'
    )

    if not dry_run:
        if skip_exists:
            if archive_path.exists():
                print('Archive exists already, skipping.')
                return

        paths = voc.paths.from_dir(dataset_root, ext)
        if ext == '.cbin':
            # we also need to get .rec ("record") files,
            # in addition to .cbin audio files
            # since the .rec files have the sampling rate
            # and are used by `evfuncs.load_cbin` to load .cbin audio
            paths.extend(
                voc.paths.from_dir(dataset_root, '.rec')
            )

        tar = tarfile.open(archive_path, 'w:gz')
        try:
            print("Adding files to archive.")
            for path in paths:
                print(
                    f'Adding: {path.name}'
                )
                arcname = str(path).replace(str(dataset_root) + '/', '')
                tar.add(name=path, arcname=arcname)
        finally:
            tar.close()


HERE = pathlib.Path(__file__).parent
REPO_ROOT = HERE / '..' / '..'
DATASET_ROOT = REPO_ROOT / 'tests/data-for-tests/source'
DIRS_TO_TAR_EXT_ARCHIVE_NAME = [
    (DATASET_ROOT / 'audio_cbin_annot_notmat/gy6or6/032312', '.cbin', 'bfsongrepo'),
    (DATASET_ROOT / 'jourjine-et-al-2023/developmentLL', '.wav', 'jourjine-et-al-2023'),
]
TAR_DST = HERE / 'example_data'


def main():
    for dataset_dir, ext, archive_name in DIRS_TO_TAR_EXT_ARCHIVE_NAME:
        tar_source_data_subidr(dataset_dir, TAR_DST, archive_name, ext)


main()
