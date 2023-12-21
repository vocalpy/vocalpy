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

   Audio
   Spectrogram
   Annotation


Classes for Pipelines
---------------------

Classes for common steps in your pipelines and workflows

.. autosummary::
   :toctree: generated

   Segmenter
   SpectrogramMaker
   SpectrogramParameters

Signal processing
-----------------

.. autosummary::
   :toctree: generated
   :recursive:

   signal
   segment
   spectral

Feature Extraction
------------------

.. autosummary::
   :toctree: generated
   :recursive:

   feature


Metrics
-------

.. autosummary::
   :toctree: generated
   :recursive:

   metrics
   metrics.segmentation
   metrics.segmentation.ir

Datasets
--------

.. autosummary::
   :toctree: generated

   dataset.SequenceDataset
   Sequence
   Unit

Visualization
-------------
.. autosummary::
   :toctree: generated
   :recursive:

   plot

Other
-------

.. autosummary::
   :toctree: generated
   :recursive:

   constants
   validators
   paths
