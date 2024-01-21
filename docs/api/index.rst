.. _api:

API Reference
=============

.. automodule:: vocalpy

This section documents the vocalpy `API <https://en.wikipedia.org/wiki/API>`_.

.. currentmodule:: vocalpy

Data types
----------

Data types for acoustic communication data.

.. autosummary::
   :toctree: generated
   :template: class.rst

   Audio
   Spectrogram
   Annotation


Classes for Pipelines
---------------------

Classes for common steps in your pipelines and workflows

.. autosummary::
   :toctree: generated
   :template: class.rst

   Segmenter
   SpectrogramMaker
   SpectrogramParameters

Signal processing
-----------------

.. autosummary::
   :toctree: generated
   :template: module.rst
   :recursive:

   signal
   segment
   spectral

Feature Extraction
------------------

.. autosummary::
   :toctree: generated
   :template: module.rst
   :recursive:

   feature


Metrics
-------

.. autosummary::
   :toctree: generated
   :template: module.rst
   :recursive:

   metrics
   metrics.segmentation
   metrics.segmentation.ir

Datasets
--------

.. autosummary::
   :template: class
   :toctree: generated

   dataset.SequenceDataset
   Sequence
   Unit

Visualization
-------------
.. autosummary::
   :toctree: generated
   :template: module.rst
   :recursive:

   plot

Other
-------

.. autosummary::
   :toctree: generated
   :template: module.rst
   :recursive:

   constants
   validators
   paths
