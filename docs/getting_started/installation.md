(installation)=

# Installation

VocalPy can be installed with either one of two package managers: pip or conda.
The [next section](#install-package-manager) shows commands for installing with either.

It is recommended to always use a virtual environment [^1][^2][^3][^4].
For convenience, we also provide commands to create a new environment 
and install VocalPy into it [below](#install-virtual-env).

:::{admonition} Which version of Python do I need?
:class: tip
:name: install-python-version

Roughly speaking, you can use any of the latest 3 micro versions.
As of January 2024, that would be Python 3.10-3.12. 
That's because VocalPy depends on the core scientific Python 
packages, that have adopted a policy of dropping support for 
Python versions 3 years after their initial release.
For more detail, see [SPEC0](https://scientific-python.org/specs/spec-0000/).

:::


(install-package-manager)=
## Install vocalpy with pip or conda

To install VocalPy, copy and paste one of the commands below in the terminal.

```{eval-rst}

.. tabs::

   .. code-tab:: shell pip

      python -m pip install vocalpy

   .. code-tab:: shell conda

      conda install vocalpy -c conda-forge

```

(install-virtual-env)=
## Create a new virtual environment and install VocalPy

:::{tip}
If you are using conda to create the virtual environment, 
then you should install VocalPy with conda.
(If you need to reproduce an environment, for example on different computers, 
then pip should only be used after conda [^5].)
:::

### With venv and pip

```{eval-rst}
.. tabs::

   .. code-tab:: shell macOS + Linux

      python3 -m venv .venv
      source .venv/bin/activate
      python3 -m pip install vocalpy

   .. code-tab:: shell Windows PowerShell

      python -m venv .venv
      .venv\Scripts\activate
      python -m pip install vocalpy
```

### With conda


```console
conda create -n vocalpy python=3.11
conda activate vocalpy
conda install vocalpy -c conda-forge
```


[^1]: https://realpython.com/python-virtual-environments-a-primer/#why-do-you-need-virtual-environments
[^2]: https://snarky.ca/how-virtual-environments-work/
[^3]: https://the-turing-way.netlify.app/reproducible-research/renv/renv-package#making-and-using-environments
[^4]: https://carpentries-incubator.github.io/introduction-to-conda-for-data-scientists/02-working-with-environments/index.html#avoid-installing-packages-into-your-base-conda-environment ↩
[^5]: https://www.anaconda.com/blog/using-pip-in-a-conda-environment
