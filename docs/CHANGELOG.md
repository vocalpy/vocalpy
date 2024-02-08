# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
