"""This is the script used to generate the audio clips in tests/data-for-tests/source/jourjine-et-al-2023.

The source data is from:
Jourjine, Nicholas et al. (2023). Data from:
Two pup vocalization types are genetically and functionally separable in deer mice [Dataset].
Dryad. https://doi.org/10.5061/dryad.g79cnp5ts

Specifically the developmentLL tar:
https://datadryad.org/stash/downloads/file_stream/2143657
"""
import pathlib

import vocalpy as voc

data_dir = pathlib.Path('~/Documents/data/vocal/jourjine-et-al-2023/developmentLL/').expanduser()
wav_paths = voc.paths.from_dir(data_dir, 'wav')

dst = pathlib.Path('./tests/data-for-tests/source/jourjine-et-al-2023/developmentLL')

dst.mkdir(exist_ok=True)

CLIP_LEN = 20.0  # seconds
N_CLIPS = 5

WAV_PATH_INDS = [
    1, 3, 4, 5, 6, 7, 8, 9, 10
]

for ind in WAV_PATH_INDS:
    wav_path = wav_paths[ind]
    sound = voc.Sound.read(wav_path)
    clip_end_ind = int(CLIP_LEN * sound.samplerate)
    new_sound = voc.Sound(
        data=sound.data[..., :clip_end_ind],
        samplerate=sound.samplerate,
    )
    new_sound.write(dst / (wav_path.stem + '-clip.wav'))
