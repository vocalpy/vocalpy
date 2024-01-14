(installation)=

# Installation

VocalPy can be installed with either one of two package managers, pip and conda.
To install VocalPy, copy and paste one of the commands below in the terminal.

```{eval-rst}

.. tabs::

   .. code-tab:: pure Python environment

         @NickleDave to add python virtual env instructions

   .. code-tab:: conda environment

         conda create --n vocalpy python=3.10
         conda activate vocalpy
         conda install vocalpy -c conda-forge
```

If you are starting a new project, we recommend creating a new virtual environment, 
and installing VocalPy into that environment [^1][^2][^3].
If you are using conda, you should prefer installing VocalPy with conda, not pip [^4].

[^1]: https://realpython.com/python-virtual-environments-a-primer/#why-do-you-need-virtual-environments
[^2]: https://towardsdatascience.com/conda-essential-concepts-and-tricks-e478ed53b5b#073c
[^3]: https://carpentries-incubator.github.io/introduction-to-conda-for-data-scientists/02-working-with-environments/index.html#avoid-installing-packages-into-your-base-conda-environment ↩
[^4]: https://www.anaconda.com/blog/using-pip-in-a-conda-environment
