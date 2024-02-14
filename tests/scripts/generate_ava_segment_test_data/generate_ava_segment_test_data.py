"""Adapted from autoencoded-vocal-analysis/examples/mouse_sylls_mwe.py
and https://github.com/nickjourjine/peromyscus-pup-vocal-evolution/tree/main

This should be run in the environment file created with
vocalpy/tests/scripts/generate_ava_test_data/environment.yml
"""
from itertools import repeat

from joblib import Parallel, delayed
import os

from ava.data.data_container import DataContainer
from ava.segmenting.segment import segment
from ava.segmenting.amplitude_segmentation import get_onsets_offsets

# from: https://github.com/nickjourjine/peromyscus-pup-vocal-evolution/blob/main/scripts/Segmenting%20and%20UMAP.ipynb
seg_params = {
    'min_freq': 20e3, # minimum frequency
    'max_freq': 125e3, # maximum frequency
    'nperseg': 1024, # FFT
    'noverlap': 1024//2, # FFT
    'spec_min_val': .8, # minimum log-spectrogram value
    'spec_max_val': 6, # maximum log-spectrogram value
    'fs': 250000, # audio samplerate
    'th_1':.3, # segmenting threshold 1
    'th_2':.3, # segmenting threshold 2
    'th_3':.35, # segmenting threshold 3
    'min_dur':0.015, # minimum syllable duration
    'max_dur': 1, # maximum syllable duration
    'min_intersyllable': .004,
    'smoothing_timescale': 0.00025, # amplitude
    'softmax': False, # apply softmax to the frequency bins to calculate
                      # amplitude
    'temperature':0.01, # softmax temperature parameter
    'thresholds_path': None,
    'algorithm': get_onsets_offsets
}

root = '/home/pimienta/Documents/data/vocal/goffinet'
audio_dirs = [os.path.join(root, 'BM003')]
seg_dirs = [os.path.join(root, 'segs')]
proj_dirs = [os.path.join(root, 'projections')]
spec_dirs = [os.path.join(root, 'specs')]
model_filename = os.path.join(root, 'checkpoint_150.tar')
plots_dir = root
dc = DataContainer(projection_dirs=proj_dirs, spec_dirs=spec_dirs,
                   plots_dir=plots_dir, model_filename=model_filename)

n_jobs = min(len(audio_dirs), os.cpu_count()-1)
gen = zip(audio_dirs, seg_dirs, repeat(params['segment']))
Parallel(n_jobs=n_jobs)(delayed(segment)(*args) for args in gen)
