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

   Sound
   Spectrogram
   Annotation
   Segments
   Features


Classes for Pipelines
---------------------

Classes for common steps in pipelines.

.. autosummary::
   :toctree: generated
   :template: class.rst

   Segmenter
   SpectrogramMaker
   FeatureExtractor
   Params

Signal processing
-----------------

.. autosummary::
   :toctree: generated
   :template: module.rst
   :recursive:

   signal
   segment
   spectral

.. autosummary::
   :toctree: generated

   spectrogram

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

Visualization
-------------
.. autosummary::
   :toctree: generated
   :template: module.rst
   :recursive:

   plot

Example Data
------------

.. autosummary::
   :toctree: generated

   example

.. autosummary::
   :toctree: generated
   :template: module.rst
   :recursive:

   examples

.. autosummary::
   :toctree: generated
   :template: class.rst

   examples.ExampleData


Other
-------

.. autosummary::
   :toctree: generated
   :template: module.rst
   :recursive:

   constants
   validators
   paths
