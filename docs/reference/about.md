## About VocalPy

There are many great software tools for researchers studying acoustic communication in animals[^1].
But our research groups work with a wide range of different data formats: for audio, for array data, for annotations. 
This means we write a lot of low-level code to deal with those formats, 
and then our code for analyses is *tightly coupled* to those formats.
In turn, this makes it hard for other groups to read our code, 
and it takes a real investment to understand our analyses, workflows and pipelines.
It also means that it requires significant work to translate an 
analysis worked out by a scientist-coder in a Jupyter notebook 
into a generalized, robust service provided by an application 
developed by a research software engineer.

In particular, acoustic communication researchers working with the Python programming language face these problems. 
How can our scripts and libraries talk to each other?
Luckily, Python is a great glue language! Let's use it to solve these problems.

A [paper introducing VocalPy and its design](docs/fa2023/Introducing_VocalPy__a_core_Python_package_for_researchers_studying_animal_acoustic_communication.pdf) 
was presented at [Forum Acusticum 2023](https://www.fa2023.org/) 
as part of the session "Open-source software and cutting-edge applications in bio-acoustics".

[^1]: For a curated collection, see <https://github.com/rhine3/bioacoustics-software>.