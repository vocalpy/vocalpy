(what-is-vocalpy)=
# What is VocalPy?

You read on the landing page of the docuemntation 
that VocalPy is the core package of the [VocalPy community](http://www.vocalpy.org/).
You also learned that it is a Python library,
meant for researchers studying how animals communicate with sound.
This page explains the goals and features of VocalPy in more detail.

## Goals

The goals of VocalPy are to:
- make it easy to work with a wide array of data formats: audio, array (spectrograms, features), annotation
- provide classes that represent commonly-used data types: audio, spectograms, features, annotations
- provide classes that represent common processes and steps in pipelines: segmenting audio, computing spectrograms, 
  extracting features
- make it easier for scientist-coders to flexibly and iteratively build datasets, 
  without needing to deal directly with a database if they don't want to
- make it possible to re-use code you have already written for your own research group
- and finally: 
  - make code easier to share and read across research groups, by providing these classes, and *idiomiatic* ways 
    of coding with them; think of VocalPy as an interoperability layer and a common language
  - facilitate collaboration between scientist-coders writing imperative analysis scripts and 
    [research software engineers](https://www.nature.com/articles/d41586-022-01516-2) 
    developing libraries and applications

For more on acoustic communication research in general, please see the {ref}`background` page.

## Features

###  Data types for acoustic communication data: audio, spectrograms, annotations

#### The `vocalpy.Sound` data type

- Works with a wide array of audio formats, thanks to [soundfile](https://github.com/bastibe/python-soundfile).
- Also works with the cbin audio format saved by the LabView app EvTAF used by many neuroscience labs studying birdsong,
  thanks to [evfuncs](https://github.com/NickleDave/evfuncs).

```python
>>> import vocalpy as voc
>>> data_dir = ('tests/data-for-tests/source/audio_wav_annot_birdsongrec/Bird0/Wave/')
>>> wav_paths = voc.paths.from_dir(data_dir, 'wav')
>>> sounds = [voc.Sound.read(wav_path) for wav_path in wav_paths]
>>> print(sounds[0])
vocalpy.Sound(data=array([3.0517...66210938e-04]), samplerate=32000, path ="tests/data-for-tests/source/audio_wav_annot_birdsongrec/Bird0/Wave/0.wav")
```

#### The `vocalpy.Spectrogram` data type

- Save expensive-to-compute spectrograms to array files, so you don't regenerate them over and over again

```python
>>> import vocalpy as voc
>>> data_dir = ('tests/data-for-tests/generated/spect_npz/')
>>> spect_paths = voc.paths.from_dir(data_dir, 'wav.npz')
>>> spects = [voc.Spectrogram.read(spect_path) for spect_path in spect_paths]
>>> print(spects[0])
vocalpy.Spectrogram(data=array([[3.463...7970774e-14]]), frequencies=array([0....7.5, 16000. ]), times=array([0.008,...7.648, 7.65 ]), 
path=PosixPath('tests/data-for-tests/generated/spect_npz/0.wav.npz'), audio_path=None)
```

#### The `vocalpy.Annotation` data type

- Load many different annotation formats using the pyOpenSci package [crowsetta](https://crowsetta.readthedocs.io/en/latest/)

```python
>>> import vocalpy as voc
>>> data_dir = ('tests/data-for-tests/source/audio_cbin_annot_notmat/gy6or6/032312/')
>>> notmat_paths = voc.paths.from_dir(data_dir, '.not.mat')
>>> annots = [voc.Annotation.read(notmat_path, format='notmat') for notmat_path in notmat_paths]
>>> print(annots[1])
Annotation(data=Annotation(annot_path=PosixPath('tests/data-for-tests/source/audio_cbin_annot_notmat/gy6or6/032312/gy6or6_baseline_230312_0809.141.cbin.not.mat'), 
notated_path=PosixPath('tests/data-for-tests/source/audio_cbin_annot_notmat/gy6or6/032312/gy6or6_baseline_230312_0809.141.cbin'), 
seq=<Sequence with 57 segments>), path=PosixPath('tests/data-for-tests/source/audio_cbin_annot_notmat/gy6or6/032312/gy6or6_baseline_230312_0809.141.cbin.not.mat'))
```

### Classes for common steps in your pipelines and workflows

#### A `Segmenter` for segmentation into sequences of units

```python
>>> import evfuncs
>>> import vocalpy as voc
>>> data_dir = ('tests/data-for-tests/source/audio_cbin_annot_notmat/gy6or6/032312/')
>>> cbin_paths = voc.paths.from_dir(data_dir, 'cbin')
>>> sounds = [voc.Sound.read(cbin_path) for cbin_path in cbin_paths]
>>> segment_params = {'threshold': 1500, 'min_syl_dur': 0.01, 'min_silent_dur': 0.006}
>>> segmenter = voc.Segmenter(callback=evfuncs.segment_song, segment_params=segment_params)
>>> seqs = segmenter.segment(sounds, parallelize=True)
[  ########################################] | 100% Completed | 122.91 ms
>>> print(seqs[1])
Sequence(units=[Unit(onset=2.19075, offset=2.20428125, label='-', audio=None, spectrogram=None),
                Unit(onset=2.35478125, offset=2.38815625, label='-', audio=None, spectrogram=None),
                Unit(onset=2.8410625, offset=2.86715625, label='-', audio=None, spectrogram=None),
                Unit(onset=3.48234375, offset=3.49371875, label='-', audio=None, spectrogram=None),
                Unit(onset=3.57021875, offset=3.60296875, label='-', audio=None, spectrogram=None),
                Unit(onset=3.64403125, offset=3.67721875, label='-', audio=None, spectrogram=None),
                Unit(onset=3.72228125, offset=3.74478125, label='-', audio=None, spectrogram=None),
                Unit(onset=3.8036875, offset=3.8158125, label='-', audio=None, spectrogram=None),
                Unit(onset=3.82328125, offset=3.83646875, label='-', audio=None, spectrogram=None),
                Unit(onset=4.13759375, offset=4.16346875, label='-', audio=None, spectrogram=None),
                Unit(onset=4.80278125, offset=4.814, label='-', audio=None, spectrogram=None),
                Unit(onset=4.908125, offset=4.922875, label='-', audio=None, spectrogram=None),
                Unit(onset=4.9643125, offset=4.992625, label='-', audio=None, spectrogram=None),
                Unit(onset=5.039625, offset=5.0506875, label='-', audio=None, spectrogram=None),
                Unit(onset=5.10165625, offset=5.1385, label='-', audio=None, spectrogram=None),
                Unit(onset=5.146875, offset=5.16203125, label='-', audio=None, spectrogram=None),
                Unit(onset=5.46390625, offset=5.49409375, label='-', audio=None, spectrogram=None),
                Unit(onset=6.14503125, offset=6.1565625, label='-', audio=None, spectrogram=None),
                Unit(onset=6.31003125, offset=6.346125, label='-', audio=None, spectrogram=None),
                Unit(onset=6.38996875, offset=6.4018125, label='-', audio=None, spectrogram=None),
                Unit(onset=6.46053125, offset=6.4796875, label='-', audio=None, spectrogram=None),
                Unit(onset=6.83525, offset=6.8643125, label='-', audio=None, spectrogram=None)], method='segment_song',
         segment_params={'threshold': 1500, 'min_syl_dur': 0.01, 'min_silent_dur': 0.006},
         audio=vocalpy.Sound(data=None, samplerate=None, path="tests/data-for-tests/source/audio_cbin_annot_notmat / gy6or6 / 032312 / gy6or6_baseline_230312_0809.141.cbin"), spectrogram=None)
```
  
#### A `SpectrogramMaker` for computing spectrograms

```python
>>> import vocalpy as voc
>>> wav_paths = voc.paths.from_dir('wav')
>>> sounds = [voc.Sound(wav_path) for wav_path in wav_paths]
>>> spect_params = {'fft_size': 512, 'step_size': 64}
>>> spect_maker = voc.SpectrogramMaker(spect_params=spect_params)
>>> spects = spect_maker.make(sounds, parallelize=True)
```

### `Dataset`s you flexibly build from pipelines and convert to databases

- The `vocalpy.dataset` module contains classes that represent common types of datasets 
- You make these classes with outputs of your pipelines, e.g. a `list` of `vocalpy.Sequence`s 
  or `vocalpy.Spectrogram`s
- Because of the design of `vocalpy`, these datasets capture key metadata from your pipeline: 
  - parameters and data provenance details; 
    e.g., what parameters did you use to segment? What audio file did this sequence come from?
- Then you can save the dataset along with metadata to databases, or later load from databases
  - `vocalpy` comes with built-in support for persisting to [SQLite](https://www.sqlite.org/index.html), 
    a lightweight, efficient single-file database format.
    It is the only database file format 
    [recommended by the US Library of Congress for archival data](https://www.sqlite.org/locrsf.html),
    and it's [built into Python](https://docs.python.org/3/library/sqlite3.html) 
    -- no need to install separate database software like MySQL

#### A `SequenceDataset` for common analyses of sequences of units

```python
>>> import evfuncs
>>> import vocalpy as voc
>>> data_dir = 'tests/data-for-tests/source/audio_cbin_annot_notmat/gy6or6/032312/'
>>> cbin_paths = voc.paths.from_dir(data_dir, 'cbin')
>>> sounds = [voc.Sound.read(cbin_path) for cbin_path in cbin_paths]
>>> segment_params = {
    'threshold': 1500,
    'min_syl_dur': 0.01,
    'min_silent_dur': 0.006,
}
>>> segmenter = voc.Segmenter(
    callback=evfuncs.segment_song,
    segment_params=segment_params
)
>>> seqs = segmenter.segment(sounds)
>>> seq_dataset = voc.dataset.SequenceDataset(sequences=seqs)
>>> seq_dataset.to_sqlite(db_name='gy6or6-032312.db', replace=True)
>>> print(seq_dataset)
SequenceDataset(sequences=[Sequence(units=[Unit(onset=2.18934375, offset=2.21, label='-', audio=None, spectrogram=None),
                                           Unit(onset=2.346125, offset=2.373125, label='-', audio=None, spectrogram=None), 
                                           Unit(onset=2.50471875, offset=2.51546875, label='-', audio=None, spectrogram=None),
                                           Unit(onset=2.81909375, offset=2.84740625, label='-', audio=None, spectrogram=None),
                                           ...
>>> # test that we can load the dataset
>>> seq_dataset_loaded = voc.dataset.SequenceDataset.from_sqlite(db_name='gy6or6-032312.db')
>>> seq_dataset_loaded == seq_dataset
True
```