import math

import numpy as np

import vocalpy as voc


MIN_FS_MUPET = 90000  # minimum fs that MUPET requires


def mupet(
    sound: voc.Sound,
    target_sampfreq: int = 250000,
    # MUPET calls `frame_length` the `frame_win`; default is 
    # 250k * 0.0020 ms = 50
    frame_length: int = 50,  
    # MUPET call `hop_length` the `frame_shift`; default is
    # 250k * 0.0016 ms = 40
    hop_length: int = 40,
    nfft: int = 512,
    fmin: int = 35000,  # min freq *for USVs*!
    fmax: int = 110000,  # max freq *for USVs*!
) -> voc.Segments:
    """MUPET segmentation algorithm"""
    if sound.sampfreq < MIN_FS_MUPET:
        raise ValueError(
            "Minimum sampling frequency required by MUPET segmentation algorithm is "
            f"{MIN_FS_MUPET} Hz, but `sound` has sampling frequency {sound.sampfreq}"
        )

    if sound.sampfreq != target_sampfreq:
        # FIXME: looks like they window to resample if needed? with `resample_normal` flag
        data = np.resample(sound.data, target_sampfreq, sound.sampfreq)
    else:
        data = sound.data

    # energy
    logE_thr = 0.2
    smooth_fac = math.floor(5 *(0.0016 / frame_shift_ms))
    smooth_fac_low = 10
    grow_fac = math.floor(3 * (0.0016 / frame_shift_ms))

    Nmin = math.floor(nfft / (fs / 2) * fmin)
    Nmax = math.floor(nfft / (fs / 2) * fmax)
    (
        gt_sonogram,
        sonogram,
        E_low,
        E_usv,
        T
    ) = FE_GT_spectra(
        handles, data, fs, frame_length, hop_length, Nfft, Nmin, Nmax
    )