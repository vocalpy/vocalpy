---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

::::{grid}
:reverse:
:gutter: 2 1 1 1
:margin: 4 4 1 1

:::{grid-item}
:columns: 8
:class: sd-fs-3

A core package for acoustic communication research in Python.
:::

:::{grid-item}
:columns: 4

```{image} ./_static/vocalpy-secondary.png
:width: 150px
```
:::

::::

There are many great software tools for researchers studying acoustic communication in animals[^1].
But our research groups work with a wide range of different data formats: for audio, for array data, for annotations. 
This means we write a lot of low-level code to deal with those formats, 
and then our code for analyses is *tightly coupled* to those formats.
In turn, this makes it hard for other groups to read our code, 
and it takes a real investment to understand our analyses, workflows and pipelines.
It also means that it requires significant work to translate an 
analysis worked out by a scientist-coder in a Jupyter notebook 
into a generalized, robust service provided by an application 
developed by a research software engineer.

In particular, acoustic communication researchers working with the Python programming language face these problems. 
How can our scripts and libraries talk to each other?
Luckily, Python is a great glue language! Let's use it to solve these problems.

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

[^1]: For a curated collection, see <https://github.com/rhine3/bioacoustics-software>.

## Features

###  Data types for acoustic communication data: audio, spectrograms, annotations

#### The `vocalpy.Audio` data type

- Works with a wide array of audio formats, thanks to [soundfile](https://github.com/bastibe/python-soundfile).
- Also works with the cbin audio format saved by the LabView app EvTAF used by many neuroscience labs studying birdsong,
  thanks to [evfuncs](https://github.com/NickleDave/evfuncs).

```python
>>> import vocalpy as voc
>>> data_dir = ('tests/data-for-tests/source/audio_wav_annot_birdsongrec/Bird0/Wave/')
>>> wav_paths = voc.paths.from_dir(data_dir, 'wav')
>>> audios = [voc.Audio.read(wav_path) for wav_path in wav_paths]
>>> print(audios[0])
vocalpy.Audio(data=array([3.0517...66210938e-04]), samplerate=32000, channels=1), 
path=tests/data-for-tests/source/audio_wav_annot_birdsongrec/Bird0/Wave/0.wav)
```

#### The `vocalpy.Spectrogram` data type

- Save expensive-to-compute spectrograms to array files, so you don't regenerate them over and over again

```python
>>> import vocalpy as voc
>>> data_dir = ('tests/data-for-tests/generated/spect_npz/')
>>> spect_paths = voc.paths.from_dir(data_dir, 'wav.npz')
>>> spects = [voc.Spectrogram.read(spect_path) for spect_path in spect_paths]
>>> print(spects[0])
vocalpy.Spectrogram(data=array([[3.463...7970774e-14]]), frequencies=array([    0....7.5, 16000. ]), times=array([0.008,...7.648, 7.65 ]), 
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
>>> audios = [voc.Audio.read(cbin_path) for cbin_path in cbin_paths]
>>> segment_params = {'threshold': 1500, 'min_syl_dur': 0.01, 'min_silent_dur': 0.006}
>>> segmenter = voc.Segmenter(callback=evfuncs.segment_song, segment_params=segment_params)
>>> seqs = segmenter.segment(audios, parallelize=True)
[########################################] | 100% Completed | 122.91 ms
>>> print(seqs[1])
Sequence(units=[Unit(onset=2.19075, offset=2.20428125, label='-', audio=None, spectrogram=None), 
Unit(onset=2.35478125, offset=2.38815625, label='-', audio=None, spectrogram=None), Unit(onset=2.8410625, offset=2.86715625, label='-', audio=None, spectrogram=None), 
Unit(onset=3.48234375, offset=3.49371875, label='-', audio=None, spectrogram=None), Unit(onset=3.57021875, offset=3.60296875, label='-', audio=None, spectrogram=None), 
Unit(onset=3.64403125, offset=3.67721875, label='-', audio=None, spectrogram=None), Unit(onset=3.72228125, offset=3.74478125, label='-', audio=None, spectrogram=None), 
Unit(onset=3.8036875, offset=3.8158125, label='-', audio=None, spectrogram=None), Unit(onset=3.82328125, offset=3.83646875, label='-', audio=None, spectrogram=None), 
Unit(onset=4.13759375, offset=4.16346875, label='-', audio=None, spectrogram=None), Unit(onset=4.80278125, offset=4.814, label='-', audio=None, spectrogram=None), 
Unit(onset=4.908125, offset=4.922875, label='-', audio=None, spectrogram=None), Unit(onset=4.9643125, offset=4.992625, label='-', audio=None, spectrogram=None), 
Unit(onset=5.039625, offset=5.0506875, label='-', audio=None, spectrogram=None), Unit(onset=5.10165625, offset=5.1385, label='-', audio=None, spectrogram=None), 
Unit(onset=5.146875, offset=5.16203125, label='-', audio=None, spectrogram=None), Unit(onset=5.46390625, offset=5.49409375, label='-', audio=None, spectrogram=None), 
Unit(onset=6.14503125, offset=6.1565625, label='-', audio=None, spectrogram=None), Unit(onset=6.31003125, offset=6.346125, label='-', audio=None, spectrogram=None), 
Unit(onset=6.38996875, offset=6.4018125, label='-', audio=None, spectrogram=None), Unit(onset=6.46053125, offset=6.4796875, label='-', audio=None, spectrogram=None), 
Unit(onset=6.83525, offset=6.8643125, label='-', audio=None, spectrogram=None)], method='segment_song', 
segment_params={'threshold': 1500, 'min_syl_dur': 0.01, 'min_silent_dur': 0.006}, 
audio=vocalpy.Audio(data=None, samplerate=None, channels=None), path=tests/data-for-tests/source/audio_cbin_annot_notmat/gy6or6/032312/gy6or6_baseline_230312_0809.141.cbin), spectrogram=None)
```
 
#### A `SpectrogramMaker` for computing spectrograms

```python
import vocalpy as voc

wav_paths = voc.paths.from_dir('wav')
audios = [voc.Audio(wav_path) for wav_path in wav_paths]
spect_params = {'fft_size': 512, 'step_size': 64}
spect_maker = voc.SpectrogramMaker(spect_params=spect_params)
spects = spect_maker.make(audios, parallelize=True)
```

# Getting Started

If you are new to the library, start with {ref}`tutorial`.

```{toctree}
:hidden: true
:maxdepth: 2

tutorial
howto
api/index
development/index
reference/index
```

# Support

To report a bug or request a feature (such as a new annotation format), 
please use the issue tracker on GitHub:  
<https://github.com/vocalpy/vocalpy/issues>

To ask a question about vocalpy, discuss its development, 
or share how you are using it, 
please start a new topic on the VocalPy forum 
with the vocalpy tag:  
<https://forum.vocalpy.org/>

# Contribute

- Issue Tracker: <https://github.com/vocalpy/vocalpy/issues>
- Source Code: <https://github.com/vocalpy/vocalpy>

# License

The project is licensed under the
[BSD license](https://github.com/vocalpy/vocalpy/blob/master/LICENSE).

# CHANGELOG

You can see project history and work in progress in the
[CHANGELOG](https://github.com/vocalpy/vocalpy/blob/main/doc/CHANGELOG.md).

# Citation

If you use vocalpy, please cite the DOI:

```{image} https://zenodo.org/badge/DOI/10.5281/zenodo.7905426.svg
:target: https://doi.org/10.5281/zenodo.7905426
```