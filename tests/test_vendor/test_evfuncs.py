"""
test evfuncs module
"""
import numpy as np

import vocalpy._vendor.evfuncs


def test_readrecf(all_rec_paths):
    rec_dict = vocalpy._vendor.evfuncs.readrecf(all_rec_paths)
    assert 'header' in rec_dict
    assert isinstance(rec_dict['header'], list)
    assert 'sample_freq' in rec_dict
    assert isinstance(rec_dict['sample_freq'], (int, float))
    assert 'num_channels' in rec_dict
    assert isinstance(rec_dict['num_channels'], int)
    assert 'num_samples' in rec_dict
    assert isinstance(rec_dict['num_samples'], int)
    assert 'iscatch' in rec_dict
    assert 'outfile' in rec_dict
    assert 'time_before' in rec_dict
    assert isinstance(rec_dict['time_before'], float)
    assert 'time_after' in rec_dict
    assert isinstance(rec_dict['time_after'], float)
    assert 'thresholds' in rec_dict
    assert isinstance(rec_dict['thresholds'], list)
    assert all(
        [isinstance(thresh, float) for thresh in rec_dict['thresholds']]
    )
    assert 'feedback_info' in rec_dict
    assert isinstance(rec_dict['feedback_info'], dict)


def test_load_cbin(all_cbin_paths):
    dat, fs = vocalpy._vendor.evfuncs.load_cbin(all_cbin_paths)
    assert isinstance(dat, np.ndarray)
    assert dat.dtype == '>i2'  # should be big-endian 16 bit
    assert isinstance(fs, int)
