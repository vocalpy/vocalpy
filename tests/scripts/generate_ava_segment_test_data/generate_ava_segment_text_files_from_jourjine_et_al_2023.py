"""This should be run in the environment file created with
vocalpy/tests/scripts/generate_ava_test_data/environment.yml
"""
########################################################################################################################
#                                 start of vendored code from autoencoded-vocal-analysis                               #
########################################################################################################################
import os
import warnings
from itertools import repeat

from joblib import Parallel, delayed
import numpy as np
from scipy.signal import stft
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
from scipy.ndimage import gaussian_filter


EPSILON = 1e-9


def softmax(arr, t=0.5):
	"""Softmax along first array dimension. Not numerically stable."""
	temp = np.exp(arr/t)
	temp /= np.sum(temp, axis=0) + EPSILON
	return np.sum(np.multiply(arr, temp), axis=0)


def get_spec(audio, p):
	"""
	Get a spectrogram.

	Much simpler than ``ava.preprocessing.utils.get_spec``.

	Raises
	------
	- ``AssertionError`` if ``len(audio) < p['nperseg']``.

	Parameters
	----------
	audio : numpy array of floats
		Audio
	p : dict
		Spectrogram parameters. Should the following keys: `'fs'`, `'nperseg'`,
		`'noverlap'`, `'min_freq'`, `'max_freq'`, `'spec_min_val'`,
		`'spec_max_val'`

	Returns
	-------
	spec : numpy array of floats
		Spectrogram of shape [freq_bins x time_bins]
	dt : float
		Time step between time bins.
	f : numpy.ndarray
		Array of frequencies.
	"""
	assert len(audio) >= p['nperseg'], \
			"len(audio): " + str(len(audio)) + ", nperseg: " + str(p['nperseg'])
	f, t, spec = stft(audio, fs=p['fs'], nperseg=p['nperseg'], \
			noverlap=p['noverlap'])
	i1 = np.searchsorted(f, p['min_freq'])
	i2 = np.searchsorted(f, p['max_freq'])
	f, spec = f[i1:i2], spec[i1:i2]
	spec = np.log(np.abs(spec) + EPSILON)
	spec -= p['spec_min_val']
	spec /= p['spec_max_val'] - p['spec_min_val']
	spec = np.clip(spec, 0.0, 1.0)
	return spec, t[1]-t[0], f


def get_onsets_offsets(audio, p, return_traces=False):
	"""
	Segment the spectrogram using thresholds on its amplitude.

	A syllable is detected if the amplitude trace exceeds `p['th_3']`. An offset
	is then detected if there is a subsequent local minimum in the amplitude
	trace with amplitude less than `p['th_2']`, or when the amplitude drops
	below `p['th_1']`, whichever comes first. Syllable onset is determined
	analogously.

	Note
	----
	`p['th_1'] <= p['th_2'] <= p['th_3']`

	Parameters
	----------
	audio : numpy.ndarray
		Raw audio samples.
	p : dict
		Parameters.
	return_traces : bool, optional
		Whether to return traces. Defaults to `False`.

	Returns
	-------
	onsets : numpy array
		Onset times, in seconds
	offsets : numpy array
		Offset times, in seconds
	traces : list of a single numpy array
		The amplitude trace used in segmenting decisions. Returned if
		`return_traces` is `True`.
	"""
	if len(audio) < p['nperseg']:
		if return_traces:
			return [], [], None
		return [], []
	spec, dt, _ = get_spec(audio, p)
	min_syll_len = int(np.floor(p['min_dur'] / dt))
	max_syll_len = int(np.ceil(p['max_dur'] / dt))
	th_1, th_2, th_3 = p['th_1'], p['th_2'], p['th_3'] # tresholds
	onsets, offsets = [], []
	too_short, too_long = 0, 0

	# Calculate amplitude and smooth.
	if p['softmax']:
		amps = softmax(spec, t=p['temperature'])
	else:
		amps = np.sum(spec, axis=0)
	amps = gaussian_filter(amps, p['smoothing_timescale']/dt)

	# Find local maxima greater than th_3.
	local_maxima = []
	for i in range(1,len(amps)-1,1):
		if amps[i] > th_3 and amps[i] == np.max(amps[i-1:i+2]):
			local_maxima.append(i)

	# Then search to the left and right for onsets and offsets.
	for local_max in local_maxima:
		if len(offsets) > 1 and local_max < offsets[-1]:
			continue
		i = local_max - 1
		while i > 0:
			if amps[i] < th_1:
				onsets.append(i)
				break
			elif amps[i] < th_2 and amps[i] == np.min(amps[i-1:i+2]):
				onsets.append(i)
				break
			i -= 1
		if len(onsets) != len(offsets) + 1:
			onsets = onsets[:len(offsets)]
			continue
		i = local_max + 1
		while i < len(amps):
			if amps[i] < th_1:
				offsets.append(i)
				break
			elif amps[i] < th_2 and amps[i] == np.min(amps[i-1:i+2]):
				offsets.append(i)
				break
			i += 1
		if len(onsets) != len(offsets):
			onsets = onsets[:len(offsets)]
			continue

	# Throw away syllables that are too long or too short.
	new_onsets = []
	new_offsets = []
	for i in range(len(offsets)):
		t1, t2 = onsets[i], offsets[i]
		if t2 - t1 + 1 <= max_syll_len and t2 - t1 + 1 >= min_syll_len:
			new_onsets.append(t1 * dt)
			new_offsets.append(t2 * dt)
		elif t2 - t1 + 1 > max_syll_len:
			too_long += 1
		else:
			too_short += 1

	# Return decisions.
	if return_traces:
		return new_onsets, new_offsets, [amps]
	return new_onsets, new_offsets


def _is_audio_file(fn):
	return len(fn) >= 4 and fn[-4:] == '.wav'


def get_audio_seg_filenames(audio_dir, segment_dir, p=None):
	"""
	Return lists of sorted filenames.

	Warning
	-------
	- `p` is unused. This will be removed in a future version!

	Parameters
	----------
	audio_dir : str
		Audio directory.
	segment_dir : str
		Segments directory.
	p : dict, optional
		Unused! Defaults to ``None``.
	"""
	temp_filenames = [i for i in sorted(os.listdir(audio_dir)) if \
			_is_audio_file(i)]
	audio_filenames = [os.path.join(audio_dir, i) for i in temp_filenames]
	temp_filenames = [i[:-4] + '.txt' for i in temp_filenames]
	seg_filenames = [os.path.join(segment_dir, i) for i in temp_filenames]
	return audio_filenames, seg_filenames


def segment(audio_dir, seg_dir, p, verbose=True):
	"""
	Segment audio files in `audio_dir` and write decisions to `seg_dir`.

	Parameters
	----------
	audio_dir : str
		Directory containing audio files.
	seg_dir : str
		Directory containing segmenting decisions.
	p : dict
		Segmenting parameters. Must map the key `'algorithm'` to a segmenting
		algorithm, for example
		`ava.segmenting.amplitude_segmentation.get_onsets_offsets`. Must
		additionally contain keys requested by the segmenting algorithm.
	verbose : bool, optional
		Defaults to ``True``.
	"""
	if verbose:
		print("Segmenting audio in", audio_dir)
	if not os.path.exists(seg_dir):
		os.makedirs(seg_dir)
	num_sylls = 0
	audio_fns, seg_fns = get_audio_seg_filenames(audio_dir, seg_dir, None)
	for audio_fn, seg_fn in zip(audio_fns, seg_fns):
		# Collect audio.
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=WavFileWarning)
			fs, audio = wavfile.read(audio_fn)
		# Segment.
		onsets, offsets = p['algorithm'](audio, p)
		combined = np.stack([onsets, offsets]).T
		num_sylls += len(combined)
		# Write.
		header = "Onsets/offsets for " + audio_fn
		np.savetxt(seg_fn, combined, fmt='%.5f', header=header)
	if verbose:
		print("\tFound", num_sylls, "segments in", audio_dir)
########################################################################################################################
#                                 end of vendored code from autoencoded-vocal-analysis                                 #
########################################################################################################################

########################################################################################################################
# Below, adapted from autoencoded-vocal-analysis/examples/mouse_sylls_mwe.py                                           #
# and https://github.com/nickjourjine/peromyscus-pup-vocal-evolution/tree/main                                         #
########################################################################################################################

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
    #  'min_intersyllable': .004,  this is for functionality in Jourjine et al. code, not in autoencoded-vocal-analysis
    'smoothing_timescale': 0.00025, # amplitude
    'softmax': False, # apply softmax to the frequency bins to calculate
                      # amplitude
    'temperature':0.01, # softmax temperature parameter
    'thresholds_path': None,
    'algorithm': get_onsets_offsets
}


# from autoencoded-vocal-analysis/examples/mouse_sylls_mwe.py
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent / '..' / '..' / '..'
root = REPO_ROOT / 'tests/data-for-tests/source/jourjine-et-al-2023'
audio_dirs = [os.path.join(root, 'developmentLL')]
SEG_DIR_ROOT = REPO_ROOT / 'tests/data-for-tests/generated/segment'
seg_dirs = [os.path.join(SEG_DIR_ROOT, 'ava-segment-txt')]


def main():
	n_jobs = min(len(audio_dirs), os.cpu_count()-1)
	gen = zip(audio_dirs, seg_dirs, repeat(seg_params))
	Parallel(n_jobs=n_jobs)(delayed(segment)(*args) for args in gen)


main()
