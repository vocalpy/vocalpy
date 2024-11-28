# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.0]
### Added
- Add `clip` method to `Sound`, to clip a sound at a given stop and start time
  [#189](https://github.com/vocalpy/vocalpy/pull/189).
  Fixes [#149](https://github.com/vocalpy/vocalpy/issues/149).
- Add `segment` method to `Sound`, that returns a list of `Sound`s given a set of `Segments`  
  [#189](https://github.com/vocalpy/vocalpy/pull/189).
  This makes it possible to actually *do something* with `Segments` once we have them.
  Fixes [#150](https://github.com/vocalpy/vocalpy/issues/150).
- Add single example files of Bengalese finch song and deer mouse pup calls
  [#189](https://github.com/vocalpy/vocalpy/pull/189).
  Fixes [#182](https://github.com/vocalpy/vocalpy/issues/182).
- Add `Segments.from_csv` method that makes it possible to load segmentation
  from an existing csv, since people will typically save segmentation in 
  a csv file, e.g. via the `pandas.DataFrame.to_csv` method; 
  we want to make it easy to work with those existing segmentations 
  [#189](https://github.com/vocalpy/vocalpy/pull/189).
  Fixes [#182](https://github.com/vocalpy/vocalpy/issues/182).
- Add mouse pup call data and zebra finch call data used in 
  [ACAB bootcamp](https://github.com/vocalpy/acoustic-communication-and-bioacoustics-bootcamp)
  [#189](https://github.com/vocalpy/vocalpy/pull/189) 
  so that we can use in example vignettes.
  Fixes [#183](https://github.com/vocalpy/vocalpy/issues/183).
- Add `to_mono` method to `Sound`, to speed up feature extraction in the case where we don't need per-channel features
  [#189](https://github.com/vocalpy/vocalpy/pull/189).
  Fixes [#192](https://github.com/vocalpy/vocalpy/issues/192).
- Add vignette on how to use VocalPy with scikit-learn
  [#189](https://github.com/vocalpy/vocalpy/pull/189).
  Fixes [#165](https://github.com/vocalpy/vocalpy/issues/165).
- Add vignette on how to use VocalPy with UMAP and HDBSCAN
  [#189](https://github.com/vocalpy/vocalpy/pull/189).
  Fixes [#166](https://github.com/vocalpy/vocalpy/issues/166).
- Add tests for `FeatureExtractor`
  [#189](https://github.com/vocalpy/vocalpy/pull/189).
  Fixes [#161](https://github.com/vocalpy/vocalpy/issues/161).

### Changed
- Have `FeatureExtractor` only take `Sound` or list of `Sound`s
  [#189](https://github.com/vocalpy/vocalpy/pull/189).
  Fixes [#163](https://github.com/vocalpy/vocalpy/issues/163).
- Decouple `Segments` from `Sound`,
  so that you can work with `Segments` without needing to have the source `Sound`
  [#189](https://github.com/vocalpy/vocalpy/pull/189).
  Fixes [#154](https://github.com/vocalpy/vocalpy/issues/154).
- Remove `path` attribute from `Sound`; the goal is to make it as easy as possible 
  to treat `Sound`s as functional data types, instead of treating them as if they are 
  tied to specific files
  [#189](https://github.com/vocalpy/vocalpy/pull/189).
  Fixes [#162](https://github.com/vocalpy/vocalpy/issues/162).
- Refactor `examples` API
    - instead of having to specify a `return_type`, 
    we default to returning VocalPy's domain-specific data types, e.g. a `Sound`,
    but we allow specifying `return_path=True` to get the path when that is needed,
    e.g. to demonstrate functionality like the `vocalpy.Sound.read` method
    - add an `ExampleData` class, a `dict`-like where values can be accessed with 
    dot notation (adapted from scikit-learn `Bunch`); we return this for any example
    data that consists of multiple files
  [#189](https://github.com/vocalpy/vocalpy/pull/189).
  Fixes [#185](https://github.com/vocalpy/vocalpy/issues/185).
- Rename feature extraction functions to keep the API terse:
  `vocalpy.feature.sat.similarity_features` -> `vocalpy.feature.sat` and 
  `vocalpy.features.soundsig.predefined_acoustic_features` -> `vocalpy.feature.biosound`  
  [#189](https://github.com/vocalpy/vocalpy/pull/189).
  Fixes [#190](https://github.com/vocalpy/vocalpy/issues/190).
- Make nox session "dev" download data for tests if needed
  [#189](https://github.com/vocalpy/vocalpy/pull/189).
  Fixes [#196](https://github.com/vocalpy/vocalpy/issues/196).

### Fixed
- `FeatureExtractor` does not make `vocalpy.Features` instances
  [#189](https://github.com/vocalpy/vocalpy/pull/189).
  Fixes [#146](https://github.com/vocalpy/vocalpy/issues/146).
- `feature.soundsig.predefined_acoustic_features` returns `vocalpy.Features`
  [#189](https://github.com/vocalpy/vocalpy/pull/189).
  Fixes [#145](https://github.com/vocalpy/vocalpy/issues/145).

### Removed
- Remove `Segment` class -- our abstractions don't require representing a single `Segment`
  since we just use `Segments` to convert a `Sound` into multiple `Sound`s
  [#189](https://github.com/vocalpy/vocalpy/pull/189).
  Fixes [#154](https://github.com/vocalpy/vocalpy/issues/154).

## [0.9.5]
### Fixed
- Remove upper bound on numpy version, and 
  raise lower bound on librosa version to 0.10.2.post1
  (newest version that works with numpy 2.0)
  [#177](https://github.com/vocalpy/vocalpy/pull/177).

## [0.9.4]
### Changed
- Change range of Pythons that vocalpy works with to 3.10-3.12
  [#108](https://github.com/vocalpy/vocalpy/pull/108).
  Fixes [#102](https://github.com/vocalpy/vocalpy/issues/102).

### Fixed
- Put temporary upper bound on numpy version
  [#108](https://github.com/vocalpy/vocalpy/pull/108).
  Fixes [#173](https://github.com/vocalpy/vocalpy/issues/173).

## [0.9.3]
### Fixed
- Handle edge case in `vocalpy.metrics.segmentation.ir.precision_recall_fscore`, 
  where we were returning a score of 0.0 if both `reference` and `hypothesis` had no
  boundaries [#171](https://github.com/vocalpy/vocalpy/pull/171).
  We now return a score of 1.0, but we do not report any "hits"
  (since there are no boundaries to correctly detect).
  This avoids punishing a correct hypothesis of "no boundaries".
  Fixes [#170](https://github.com/vocalpy/vocalpy/issues/170).

## [0.9.2]
### Fixed
- Handle edge case in `vocalpy.segment.ava`, that could sometimes 
  set last segment > length of sound in samples
  [#169](https://github.com/vocalpy/vocalpy/pull/169).
  Fixes [#167](https://github.com/vocalpy/vocalpy/issues/167).

## [0.9.1]
### Fixed
- Fix `ConnectionError` when importing VocalPy without internet connection
  [#153](https://github.com/vocalpy/vocalpy/pull/153).
  Fixes [#152](https://github.com/vocalpy/vocalpy/issues/152).

## [0.9.0]
### Added
- Add attributes `channels`, `samples`, and `duration` to `vocalpy.Sound`
  [#124](https://github.com/vocalpy/vocalpy/pull/124).
  Fixes [#90](https://github.com/vocalpy/vocalpy/issues/90).
- Add functionality to `vocalpy.segment.ava.segment` to replicate segmenting in
  papers that use this method
  [#130](https://github.com/vocalpy/vocalpy/pull/130).
  Fixes [#126](https://github.com/vocalpy/vocalpy/issues/126).
- Add larger example datasets, that are downloaded with `pooch`
  [#130](https://github.com/vocalpy/vocalpy/pull/130).
  Fixes [#33](https://github.com/vocalpy/vocalpy/issues/33).
- Add `dtype` argument to `Sound.read`, that defaults to `numpy.float64`.
  This means that all audio loads with this numpy DType by default, 
  including cbins. This makes the behavior of `vocalpy.Sound`
  consistent with `soundfile`. 
  [#132](https://github.com/vocalpy/vocalpy/pull/132).
  Fixes [#131](https://github.com/vocalpy/vocalpy/issues/131).
- Add `Segments` class, that represents a set of line segments returned by a segmenting algorithm
  [#133](https://github.com/vocalpy/vocalpy/pull/133).
  As discussed in [#127](https://github.com/vocalpy/vocalpy/issues/127).
- Add how-to on using segmentation metrics from information retrieval
  [#133](https://github.com/vocalpy/vocalpy/pull/133).
- Add `Params` class, an abstract class that represents parameters used with a method.
  Parameters for specific methods are represented by sub-classes of `Params`.
  This makes it possible to annotate the expected type of the `params` argument for classes 
  representing a workflow step as `Params` (usually, `Params | dict`).
  The class also has the required methods that make it possible to unpack it 
  with the `**` operator.
  This means any workflow-step class boils down to doing the following: 
  `self.callback(input, **self.params)`
  [#133](https://github.com/vocalpy/vocalpy/pull/133).
- Add `samplerate` attribute to `Segment` class, so feature extraction
  functions can compute time-frequency representations from a `Segment`
  if they need to
  [#138](https://github.com/vocalpy/vocalpy/pull/138).
  Fixes [#137](https://github.com/vocalpy/vocalpy/issues/137).
- Add `FeatureExtractor` class, that represents feature 
  extraction step in a workflow
  [#141](https://github.com/vocalpy/vocalpy/pull/141).
  Fixes [#94](https://github.com/vocalpy/vocalpy/issues/94).
- Add implementation of acoustic features from [soundsig](https://github.com/theunissenlab/soundsig)
  [#141](https://github.com/vocalpy/vocalpy/pull/141).
  Fixes [#27](https://github.com/vocalpy/vocalpy/issues/27).

### Changed
- Rename `vocalpy.Audio` to `vocalpy.Sound` 
  [#124](https://github.com/vocalpy/vocalpy/pull/124).
  Fixes [#90](https://github.com/vocalpy/vocalpy/issues/90).
- Change how we handle the number of channels in audio to be consistent 
  across single and multi-channel. `vocalpy.Sound.data` always has 
  dimensions (channel, sample), and likewise `vocalpy.Spectrogram.data` 
  has dimensions (channel, frequency, time), and feature extraction 
  functions that extract one feature value per frame return arrays with
  dimensions (shape, time)
  [#124](https://github.com/vocalpy/vocalpy/pull/124).
  Fixes [#90](https://github.com/vocalpy/vocalpy/issues/90).
- Rename `vocalpy.Segmenter.segment_params` and `vocalpy.SpectrogramMaker.spect_params`
  to just `params`
  [#133](https://github.com/vocalpy/vocalpy/pull/133).
- Rename `vocalpy.segment.ava.segment` -> `vocalpy.segment.ava`
  [#133](https://github.com/vocalpy/vocalpy/pull/133).

### Fixed
- Fix `vocalpy.segment.ava.segment` to better replicate original function
  [#130](https://github.com/vocalpy/vocalpy/pull/130).
  Fixes [#126](https://github.com/vocalpy/vocalpy/issues/126).
- Fix `vocalpy.segment.meansquared` so it replicates behavior of original
  function `segment_song` in evsonganaly. As with `vocalpy.segment.ava.segment`,
  we now assume the input is by default `numpy.float64` with range [-1.0, 1.0],
  and we rescale to int16 range that the original function expected.
  We now test that we replicate the original behavior using oracle data from the 
  [evfuncs](https://github.com/NickleDave/evfuncs) package.
  [#132](https://github.com/vocalpy/vocalpy/pull/132).
  Fixes [#129](https://github.com/vocalpy/vocalpy/issues/129).

### Removed
- Remove `dataset` module and `SequenceDataset` class for now
  [#133](https://github.com/vocalpy/vocalpy/pull/133).

## 0.8.2
### Fixed
- Fix how we compute cepstrum in `vocalpy.spectral.sat`
  [#123](https://github.com/vocalpy/vocalpy/pull/123).
  Fixes [#122](https://github.com/vocalpy/vocalpy/issues/122).

## 0.8.1
### Added
- Add `trough_threshold` parameter to `feature.sat.pitch` and `feature.sat.similarity_features`
  [#117](https://github.com/vocalpy/vocalpy/pull/117).

### Fixed
- Fix how `find_hits` matches hits
  [#120](https://github.com/vocalpy/vocalpy/pull/120).
  Fixes [#119](https://github.com/vocalpy/vocalpy/issues/119).

### Changed
- Vendor code from evfuncs, reduce number of dependencies
  [#121](https://github.com/vocalpy/vocalpy/pull/121).
  Fixes [#118](https://github.com/vocalpy/vocalpy/issues/118).
- Bump minimum required version of crowsetta to 5.0.2 (that also vendors code)
  [#121](https://github.com/vocalpy/vocalpy/pull/121).

## 0.8.0
### Added
- Add convenience function `vocalpy.spectrogram`, that computes a spectrogram from 
  a `vocalpy.Audio` using a specified `method`, and then returns it as a `vocalpy.Spectrogram`
  [#109](https://github.com/vocalpy/vocalpy/pull/109).
  Fixes [#88](https://github.com/vocalpy/vocalpy/issues/88).

### Fixed
- Fix hard-coded value in `vocalpy.spectral.sat` that should have been `n_fft` parameter
  [#111](https://github.com/vocalpy/vocalpy/pull/111).
  Fixes [#100](https://github.com/vocalpy/vocalpy/issues/100).

## 0.7.0
### Added
- Add `feature` module with functions to extract 
  [Sound Analysis Toolbox](http://soundanalysispro.com/matlab-sat) 
  features [#85](https://github.com/vocalpy/vocalpy/pull/85).
  Fixes [#3](https://github.com/vocalpy/vocalpy/issues/3).
  Also adds a `spectral` module with a `sat` function that computes 
  the spectral representations used by the functions in 
  `vocalpy.features.sat`.

## 0.6.1
### Fixed
- Fix return values for `vocalpy.metrics.segmentation.ir.precision_recall_fscore`
  [#78](https://github.com/vocalpy/vocalpy/pull/78).

## 0.6.0
### Changed
- Rename `vocalpy.segment.energy` to `vocalpy.segment.meansquared`,
  and rename `vocalpy.signal.audio.energy` to `vocalpy.signal.audio.meansquared`
  [#76](https://github.com/vocalpy/vocalpy/pull/76).
  Fixes [#75](https://github.com/vocalpy/vocalpy/issues/75).

### Fixed
- Add back `vocalpy.signal.audio.bandpass_filtfilt`; 
  have `vocalpy.signal.audio.meansquared` use this function
  (and by extension `vocalpy.segment.meansquared`)
  [#76](https://github.com/vocalpy/vocalpy/pull/76).
  Fixes [#74](https://github.com/vocalpy/vocalpy/issues/74).

## 0.5.0
### Changed
- Rename `vocalpy.segment.audio_amplitude` to `vocalpy.segment.energy`
  [#63](https://github.com/vocalpy/vocalpy/pull/63).
  Fixes [#62](https://github.com/vocalpy/vocalpy/issues/62).

## 0.4.0
### Added
- Add `vocalpy.segment` module with algorithms for segmenting, 
  and add the algorithm from the `ava` package to that module,
  along with the existing method that thresholds audio amplitude 
  [#53](https://github.com/vocalpy/vocalpy/pull/53).
  Fixes [#40](https://github.com/vocalpy/vocalpy/issues/40)
  and [#46](https://github.com/vocalpy/vocalpy/issues/46).
- Add `vocalpy.metrics.segmentation.ir` with metrics 
  from information retrieval: precision, recall, F-score
  [#57](https://github.com/vocalpy/vocalpy/pull/57).
  Fixes [#54](https://github.com/vocalpy/vocalpy/issues/54)

## 0.3.0
### Added
- Add initial plot module
  [#35](https://github.com/vocalpy/vocalpy/pull/35).
- Add MVP of SequenceDataset saving to / loading from SQLite
  [#36](https://github.com/vocalpy/vocalpy/pull/36).

## 0.2.0
### Added
- Develop based on domain model 
  [#29](https://github.com/vocalpy/vocalpy/pull/29).

### Changed
- Rework core classes
  [#31](https://github.com/vocalpy/vocalpy/pull/31).

### Fixed
- Fixup audio class
  [#21](https://github.com/vocalpy/vocalpy/pull/21).
