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
   :template: module.rst

   Audio
   Spectrogram
   Annotation


Classes for Pipelines
---------------------

Classes for common steps in your pipelines and workflows

.. autosummary::
   :toctree: generated
   :template: module.rst

   Segmenter
   SpectrogramMaker
   SpectrogramParameters

Signal processing
-----------------

.. autosummary::
   :toctree: generated
   :recursive:
   :template: module.rst

   signal
   segment

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
   :toctree: generated
   :template: module.rst

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
