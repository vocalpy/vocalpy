import pathlib

import numpy as np
import vocalpy as voc


test_data_dir = pathlib.Path(
    '/home/pimienta/Documents/data/vocal/goffinet'
)
wav_dir = test_data_dir / 'BM003'



wav_paths = voc.paths.from_dir(wav_dir, ext='wav')



segs_dir = test_data_dir / 'segs'



seg_txt_files = sorted(segs_dir.glob('*.txt'))


spect_params = {
    'min_freq': 30000.0,
    'max_freq': 110000.0,
    'nperseg': 1024,
    'noverlap': 512,
    'spect_min_val': -10.0,
    'spect_max_val': 2.0,
    'transform': 'log_magnitude',
}


# In[7]:


import functools

spect_callable = functools.partial(voc.segment.ava.get_spectrogram, **spect_params)


# In[8]:


segment_params = {
    'thresh_max': 0.305,
    'thresh_min': 0.3,
    'thresh_lowest': 0.295,
    'min_dur': 0.03,
    'max_dur': 0.2,
    'smoothing_timescale': 0.007,
    'temperature': 0.5,
}


# In[20]:


for seg_txt_path in seg_txt_files:
    segs = np.loadtxt(seg_txt_path)
    segs = segs.reshape(-1,2)
    onsets_txt, offsets_txt = segs[:,0], segs[:,1]
    wav_path = (wav_dir / seg_txt_path.name.replace('.txt', '.wav'))
    audio = voc.Audio.read(wav_path)
    onsets, offsets = voc.segment.ava.segment(audio.data, audio.samplerate,
                                          spect_callback=spect_callable,
                                          **segment_params)
    np.allclose(onsets_txt, onsets)
    np.allclose(offsets_txt, offsets)

