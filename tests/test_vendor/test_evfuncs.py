"""
test evfuncs module
"""
import numpy as np

import evfuncs


import pytest


HERE = pathlib.Path(__file__).parent
DATA_FOR_TESTS_ROOT = HERE / '..' / 'data_for_tests'


@pytest.fixture
def data_for_tests_root():
    return DATA_FOR_TESTS_ROOT


@pytest.fixture
def gy6or6_032312_subset_root(data_for_tests_root):
    return data_for_tests_root / 'gy6or6_032312_subset'


@pytest.fixture
def cbins(gy6or6_032312_subset_root):
    return sorted(gy6or6_032312_subset_root.glob('*.cbin'))


@pytest.fixture
def notmats(gy6or6_032312_subset_root):
    return sorted(gy6or6_032312_subset_root.glob('*.not.mat'))


@pytest.fixture
def rec_files(gy6or6_032312_subset_root):
    return sorted(gy6or6_032312_subset_root.glob('*.rec'))


@pytest.fixture
def filtsong_mat_files(gy6or6_032312_subset_root):
    return sorted(gy6or6_032312_subset_root.glob('*filtsong*.mat'))


@pytest.fixture
def smooth_data_mat_files(gy6or6_032312_subset_root):
    return sorted(gy6or6_032312_subset_root.glob('*smooth_data*.mat'))


@pytest.fixture
def segment_mats(gy6or6_032312_subset_root):
    return sorted(gy6or6_032312_subset_root.glob('*unedited_SegmentNotes_output.mat'))


@pytest.fixture
def notmat_with_single_annotated_segment(data_for_tests_root):
    return data_for_tests_root / 'or60yw70-song-edited-to-have-single-segment' / 'or60yw70_300912_0725.437.cbin.not.mat'



def test_readrecf(rec_files):
    for rec_file in rec_files:
        rec_dict = evfuncs.readrecf(rec_file)
        assert 'header' in rec_dict
        assert type(rec_dict['header']) == list
        assert 'sample_freq' in rec_dict
        assert type(rec_dict['sample_freq']) == int or type(rec_dict['sample_freq']) == float
        assert 'num_channels' in rec_dict
        assert type(rec_dict['num_channels']) == int
        assert 'num_samples' in rec_dict
        assert type(rec_dict['num_samples']) == int
        assert 'iscatch' in rec_dict
        assert 'outfile' in rec_dict
        assert 'time_before' in rec_dict
        assert type(rec_dict['time_before']) == float
        assert 'time_after' in rec_dict
        assert type(rec_dict['time_after']) == float
        assert 'thresholds' in rec_dict
        assert type(rec_dict['thresholds']) == list
        assert all(
            [type(thresh) == float for thresh in rec_dict['thresholds']]
        )
        assert 'feedback_info' in rec_dict
        assert type(rec_dict['feedback_info']) == dict


def test_load_cbin(cbins):
    for cbin in cbins:
        dat, fs = evfuncs.load_cbin(cbin)
        assert type(dat) == np.ndarray
        assert dat.dtype == '>i2'  # should be big-endian 16 bit
        assert type(fs) == int


def test_load_notmat(notmats):
    for notmat in notmats:
        notmat_dict = evfuncs.load_notmat(notmat)
        assert type(notmat_dict) is dict
        assert 'onsets' in notmat_dict
        assert type(notmat_dict['onsets']) == np.ndarray
        assert notmat_dict['onsets'].dtype == float
        assert 'offsets' in notmat_dict
        assert type(notmat_dict['offsets']) == np.ndarray
        assert notmat_dict['offsets'].dtype == float
        assert 'labels' in notmat_dict
        assert type(notmat_dict['labels']) == str
        assert 'Fs' in notmat_dict
        assert type(notmat_dict['Fs']) == int
        assert 'fname' in notmat_dict
        assert type(notmat_dict['fname']) == str
        assert 'min_int' in notmat_dict
        assert type(notmat_dict['min_int']) == int
        assert 'min_dur' in notmat_dict
        assert type(notmat_dict['min_dur']) == int
        assert 'threshold' in notmat_dict
        assert type(notmat_dict['threshold']) == int
        assert 'sm_win' in notmat_dict
        assert type(notmat_dict['sm_win']) == int


def test_load_notmat_single_annotated_segment(notmat_with_single_annotated_segment):
    notmat_dict = evfuncs.load_notmat(notmat_with_single_annotated_segment)
    assert type(notmat_dict) is dict
    assert 'onsets' in notmat_dict
    assert type(notmat_dict['onsets']) == np.ndarray
    assert notmat_dict['onsets'].dtype == float
    assert 'offsets' in notmat_dict
    assert type(notmat_dict['offsets']) == np.ndarray
    assert notmat_dict['offsets'].dtype == float
    assert 'labels' in notmat_dict
    assert type(notmat_dict['labels']) == str
    assert 'Fs' in notmat_dict
    assert type(notmat_dict['Fs']) == int
    assert 'fname' in notmat_dict
    assert type(notmat_dict['fname']) == str
    assert 'min_int' in notmat_dict
    assert type(notmat_dict['min_int']) == int
    assert 'min_dur' in notmat_dict
    assert type(notmat_dict['min_dur']) == int
    assert 'threshold' in notmat_dict
    assert type(notmat_dict['threshold']) == int
    assert 'sm_win' in notmat_dict
    assert type(notmat_dict['sm_win']) == int
