# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
