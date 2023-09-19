import functools

import numpy as np
import vocalpy as voc


SPECT_PARAMS = {
    'min_freq': 30000.0,
    'max_freq': 110000.0,
    'nperseg': 1024,
    'noverlap': 512,
    'spect_min_val': -10.0,
    'spect_max_val': 2.0,
    'transform': 'log_magnitude',
}

SEGMENT_PARAMS = {
    'thresh_max': 0.305,
    'thresh_min': 0.3,
    'thresh_lowest': 0.295,
    'min_dur': 0.03,
    'max_dur': 0.2,
    'smoothing_timescale': 0.007,
    'temperature': 0.5,
}


def test_ava(goffinet_etal_2021_wav_seg_tuple):
    """Test that :func:`vocalpy.segment.ava.segment` replicates segmenting results
    obtained with the ``ava`` package."""
    wav_path, seg_txt_path = goffinet_etal_2021_wav_seg_tuple

    spect_callable = functools.partial(voc.segment.ava.get_spectrogram, **SPECT_PARAMS)

    segs = np.loadtxt(seg_txt_path)
    segs = segs.reshape(-1,2)
    onsets_txt, offsets_txt = segs[:,0], segs[:,1]

    audio = voc.Audio.read(wav_path)
    onsets, offsets = voc.segment.ava.segment(audio.data, audio.samplerate,
                                          spect_callback=spect_callable,
                                          **SEGMENT_PARAMS)
    assert isinstance(onsets, np.ndarray)
    assert isinstance(offsets, np.ndarray)
    np.allclose(onsets_txt, onsets)
    np.allclose(offsets_txt, offsets)
