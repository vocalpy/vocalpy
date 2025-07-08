<br>
<div align="center">
<img src="https://github.com/vocalpy/vocalpy/blob/main/docs/images/vocalpy-primary.png?raw=True" width="400">
</div>
<hr>

## A core package for acoustic communication research in Python

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Build Status](https://github.com/vocalpy/vocalpy/actions/workflows/ci.yml/badge.svg)](https://github.com/vocalpy/vocalpy/actions)
[![Documentation Status](https://readthedocs.org/projects/vocalpy/badge/?version=latest)](https://vocalpy.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14237724.svg)](https://doi.org/10.5281/zenodo.14237724)
[![PyPI version](https://badge.fury.io/py/vocalpy.svg)](https://badge.fury.io/py/vocalpy)
[![PyPI Python versions](https://img.shields.io/pypi/pyversions/vocalpy)](https://img.shields.io/pypi/pyversions/vocalpy)
[![codecov](https://codecov.io/gh/vocalpy/vocalpy/branch/main/graph/badge.svg?token=TXtNTxXKmb)](https://codecov.io/gh/vocalpy/vocalpy)
[![All Contributors](https://img.shields.io/github/all-contributors/vocalpy/vocalpy?color=ee8449)](#contributors-)

There are many great software tools for researchers studying acoustic communication in animals[^1].
But our research groups work with a wide range of different data formats: for audio, for array data, for annotations. 
This means we write a lot of low-level code to deal with those formats, 
and then our code for analyses is *tightly coupled* to those formats.
In turn, this makes it hard for other groups to read our code, 
and it takes a real investment to understand our analyses, workflows and pipelines.
It also means that it requires significant work to translate from a 
pipeline or analysis worked out by a scientist-coder in a Jupyter notebook 
into a generalized, robust service provided by an application.

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

A [paper introducing VocalPy and its design](docs/fa2023/Introducing_VocalPy__a_core_Python_package_for_researchers_studying_animal_acoustic_communication.pdf) 
has been accepted at [Forum Acusticum 2023](https://www.fa2023.org/) 
as part of the session "Open-source software and cutting-edge applications in bio-acoustics",
and will be published in the proceedings.

[^1]: For a curated collection, see <https://github.com/rhine3/bioacoustics-software>.

## Features

###  Data types for acoustic communication data: audio, spectrogram, annotations, features

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
vocalpy.Sound(data=array([3.0517...66210938e-04]), samplerate=32000, channels=1))
```

#### The `vocalpy.Spectrogram` data type

- Save expensive-to-compute spectrograms to array files, so you don't regenerate them over and over again

```python
>>> import vocalpy as voc
>>> data_dir = ('tests/data-for-tests/generated/spect_npz/')
>>> spect_paths = voc.paths.from_dir(data_dir, 'wav.npz')
>>> spects = [voc.Spectrogram.read(spect_path) for spect_path in spect_paths]
>>> print(spects[0])
vocalpy.Spectrogram(data=array([[3.463...7970774e-14]]), frequencies=array([    0....7.5, 16000. ]), times=array([0.008,...7.648, 7.65 ]))
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
>>> audios = [voc.Sound.read(cbin_path) for cbin_path in cbin_paths]
>>> segment_params = {'threshold': 1500, 'min_syl_dur': 0.01, 'min_silent_dur': 0.006}
>>> segmenter = voc.Segmenter(callback=evfuncs.segment_song, segment_params=segment_params)
>>> seqs = segmenter.segment(audios, parallelize=True)
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
         audio=vocalpy.Sound(data=None, samplerate=None, channels=None), path=tests / data -
for -tests / source / audio_cbin_annot_notmat / gy6or6 / 032312 / gy6or6_baseline_230312_0809.141.cbin), spectrogram=None)
```
 
#### A `SpectrogramMaker` for computing spectrograms

```python
>>> import vocalpy as voc
>>> wav_paths = voc.paths.from_dir('wav')
>>> audios = [voc.Sound(wav_path) for wav_path in wav_paths]
>>> spect_params = {'fft_size': 512, 'step_size': 64}
>>> spect_maker = voc.SpectrogramMaker(spect_params=spect_params)
>>> spects = spect_maker.make(audios, parallelize=True)
```

### And more!

For a crash course in VocalPy, please see the [quickstart](https://vocalpy.readthedocs.io/en/latest/getting_started/quickstart.html) 
in the [documentation](https://vocalpy.readthedocs.io/en/latest/index.html). 
And for walkthroughs on how to use VocalPy for common tasks, please see the [How-Tos](https://vocalpy.readthedocs.io/en/latest/user/howto/index.html) 
section of the [user guide](https://vocalpy.readthedocs.io/en/latest/user/index.html).

## Installation
#### With `pip`
```
pip install vocalpy
```
#### With `conda`
```
conda install vocalpy -c conda-forge
```
For more detail see [Getting Started - Installation](https://vocalpy.readthedocs.io/en/latest/getting_started/installation.html)

### Support

To report a bug or request a feature, 
please use the issue tracker on GitHub:  
<https://github.com/vocalpy/vocalpy/issues>

To ask a question about vocalpy, discuss its development, 
or share how you are using it, 
please start a new topic on the VocalPy forum 
with the vocalpy tag:  
<https://forum.vocalpy.org/>

### Contribute

#### Code of conduct

Please note that this project is released with a [Contributor Code of Conduct](./CODE_OF_CONDUCT.md).
By participating in this project you agree to abide by its terms.

#### Contributing Guidelines

Below we provide some quick links, 
but you can learn more about how you can help and give feedback  
by reading our [Contributing Guide](./CONTRIBUTING.md).

To ask a question about vocalpy, discuss its development, 
or share how you are using it, 
please start a new "Q&A" topic on the VocalPy forum 
with the vocalpy tag:  
<https://forum.vocalpy.org/>

To report a bug, or to request a feature, 
please use the issue tracker on GitHub:  
<https://github.com/vocalpy/vocalpy/issues>

### CHANGELOG
You can see project history and work in progress in the [CHANGELOG](./docs/CHANGELOG.md)

### License

The project is licensed under the [BSD license](./LICENSE).

### Citation
If you use vocalpy, please cite the DOI:  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7905426.svg)](https://doi.org/10.5281/zenodo.7905426)

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ralphpeterson"><img src="https://avatars.githubusercontent.com/u/4031329?v=4?s=100" width="100px;" alt="Ralph Emilio Peterson"/><br /><sub><b>Ralph Emilio Peterson</b></sub></a><br /><a href="#ideas-ralphpeterson" title="Ideas, Planning, & Feedback">🤔</a> <a href="#userTesting-ralphpeterson" title="User Testing">📓</a> <a href="https://github.com/vocalpy/vocalpy/commits?author=ralphpeterson" title="Documentation">📖</a> <a href="https://github.com/vocalpy/vocalpy/issues?q=author%3Aralphpeterson" title="Bug reports">🐛</a> <a href="https://github.com/vocalpy/vocalpy/commits?author=ralphpeterson" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/tkoyama010"><img src="https://avatars.githubusercontent.com/u/7513610?v=4?s=100" width="100px;" alt="Tetsuo Koyama"/><br /><sub><b>Tetsuo Koyama</b></sub></a><br /><a href="https://github.com/vocalpy/vocalpy/commits?author=tkoyama010" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/avanikop"><img src="https://avatars.githubusercontent.com/u/39831515?v=4?s=100" width="100px;" alt="avanikop"/><br /><sub><b>avanikop</b></sub></a><br /><a href="#ideas-avanikop" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
