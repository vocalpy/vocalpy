---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(quickstart)=
![VocalPy logo](./images/vocalpy-primary.png)
# Quickstart: VocalPy 🐍 💬 in 15 minutes ⏲️

+++

This tutorial will introduce you to VocalPy, a core Python package for acoustic communication research.

+++

## Set up

+++

First we download some example data, from the [Bengalese Finch song repository](https://nickledave.github.io/bfsongrepo/).

```{code-cell} ipython3
:tags: [hide-output]
!curl -sSL https://raw.githubusercontent.com/vocalpy/vak/main/src/scripts/download_autoannotate_data.py | python3 -
```

And then we'll move that data into a `./data` directory.

```{code-cell} ipython3
import pathlib
```

```{code-cell} ipython3
pathlib.Path('./data').mkdir(exist_ok=True,parents=True)
```

```{code-cell} ipython3
pathlib.Path('./bfsongrepo').rename('./data/bfsongrepo')
```

Now that we've got some data to work with, we can import `vocalpy`

```{code-cell} ipython3
import vocalpy as voc
```

## Data types for acoustic communication

+++

Let's look at the data types that VocalPy provides for acoustic comunication.

+++

We load all the wav files from a directory using a convenience function that VocalPy gives us in its `paths`, `vocalpy.paths.from_dir`:

```{code-cell} ipython3
data_dir = ('data/bfsongrepo/gy6or6/032312/')

wav_paths = voc.paths.from_dir(data_dir, 'wav')
```

### Data type for audio: `vocalpy.Audio`

+++

Next we load all the wav files into the data type that VocalPy provides for audio, `vocalpy.Audio`, using the method `vocalpy.Audio.read`:

```{code-cell} ipython3
audios = [
    voc.Audio.read(wav_path) for wav_path in wav_paths
]
```

Let's inspect one of the `vocalpy.Audio` instances

```{code-cell} ipython3
an_audio = audios[0]
print(an_audio)
```

We can see that it has four attributes:

1. `data`, the audio signal itself

```{code-cell} ipython3
print(an_audio.data)
```

2. `samplerate`, the sampling rate for the audio

```{code-cell} ipython3
print(an_audio.samplerate)
```

3. `channels`, the number of channels

```{code-cell} ipython3
print(an_audio.channels)
```

and finally,  

4. `path`, the path to the file that the audio was read from

```{code-cell} ipython3
print(an_audio.path)
```

One of the reasons VocalPy provides this data type, and the others we're about to show you here, is that it helps you write more succinct code that's easier to read: for you, when you come back to your code months from now, and for others that want to read the code you share.

+++

## Classes for steps in pipelines for processing data in acoustic communication

+++

In addition to data types for acoustic communication, VocalPy provides you with classes that represent steps in pipelines for processing that data. These classes are also written with readability and reproducibility in mind.

+++

Let's use one of those classes, `SpectrogramMaker`, to make a spectrogram from each one of the wav files that we loaded above.

We'll write a brief snippet to do so, and then we'll explain what we did.

```{code-cell} ipython3
spect_params = {'fft_size': 512, 'step_size': 64}
callback = voc.signal.spectrogram.spectrogram
spect_maker = voc.SpectrogramMaker(callback=callback, spect_params=spect_params)
spects = spect_maker.make(audios, parallelize=True)
```

Notice a couple of things about this snippet:
- In line 1, you declare the parameters that you use to generate spectrograms explicitly, as a dictionary. This helps with reproducibility by encouraging you to document those parameters
- In line 2, you also decide what code you will use to generate the spectrograms, by using what's called a "callback", because the `SpectrogramMaker` will call this function for you.
- In line 3, you create an instance of the `SpectrogramMaker` class with the function you want to use to generate spectrograms, and the parameters to use with that function.
- In line 4, you make the spectrograms, with a single call to the method `vocalpy.SpectrogramMaker.make`. You pass in the audio we loaded earlier, and you tell VocalPy that you want to parallelize the generation of the spectrograms. This is done for you, using the library `dask`.

+++

### Data type: `vocalpy.Spectrogram`

+++

As you might have guessed, when we call `SpectrogramMaker.make`, we get back a list of spectrograms.

This is the next data type we'll look at. 

+++

We inspect the first spectrogram we loaded.

```{code-cell} ipython3
a_spect = spects[0]
print(a_spect)
```

As before, we'll walk through the attributes of this class.
But since the whole point of a spectrogram is to let us see sound, let's actually look at the spectrogram, instead of staring at arrays of numbers.

We'll make a new spectrogram where we log transform the data so it's easier to visualize.

+++

We import NumPy so we can do a quick-and-dirty transform.

```{code-cell} ipython3
import numpy as np
```

```{code-cell} ipython3
a_spect_log = voc.Spectrogram(data=np.log(a_spect.data),
                              frequencies=a_spect.frequencies,
                              times=a_spect.times)
```

Then we'll plot the log-transformed spectrogram using a function built into the `vocalpy.plot` module.

```{code-cell} ipython3
voc.plot.spectrogram(
    a_spect_log,
    tlim = [2.6, 4],
    flim=[500,12500],
    pcolormesh_kwargs={'vmin':-25, 'vmax': -10}
)
```

We see that we have a spectrogram of Bengalese finch song.

Now that we know what we're working with, let's actually inspect the attributes of the `vocalpy.Spectrogram` instance.

+++

There are five attributes we care about here.

1. `data`: this is the spectrogram itself -- as with the other data types,like `vocalpy.Audio`, the attribute name `data` indiciates this main data we care about

```{code-cell} ipython3
print(a_spect.data)
```

Let's look at the shape of `data`. It's really just a NumPy array, so we inspect the array's `shape` attribute.

```{code-cell} ipython3
print(a_spect.data.shape)
```

We see that we have a matrix with some number of rows and columns. These correspond to the next two attributes we will look at.

+++

2. `frequencies`, a vector of the number of frequency bins

```{code-cell} ipython3
print(a_spect.frequencies[:10])
```

```{code-cell} ipython3
print(a_spect.frequencies.shape)
```

(We see it is equal to the number of rows.)

3. `times`, a vector of time bin centers

```{code-cell} ipython3
print(a_spect.times[:10])
```

```{code-cell} ipython3
print(a_spect.times.shape)
```

Just like with the `Audio` class, VocalPy gives us the ability to conveniently read and write spectrograms from files. This saves us from generating spectrograms over and over. Computing spectrograms can be computionally expensive, if your audio has a high sampling rate or you are using methods like multi-taper spectrograms. Saving spectrograms from files also makes it easier for you to share your data in the exact form you used it, so that it's easier to replicate your analyses.

+++

To see this in action, let's write our spectrograms to files.

```{code-cell} ipython3
import pathlib

for spect in spects:
    spect.write(
        spect.audio_path.parent / (spect.audio_path.name + '.spect.npz')
    )
```

Notice that the extension is `'npz'`; this is a file format that NumPy uses to save mulitple arrays in a single file. By convention we include the file extension of the source audio, and another "extension" that incidicates this is a spectrogram, so that the file name ends with `'.wav.spect.npz'`.

+++

We can confirm that reading and writing spectrograms to disk works as we expect using the method `vocalpy.Spectrogram.read`

```{code-cell} ipython3
spect_paths = voc.paths.from_dir(data_dir, '.wav.spect.npz')
```

```{code-cell} ipython3
spects_loaded = [
    voc.Spectrogram.read(spect_path)
    for spect_path in spect_paths
]
```

We compare with the equality operator to confirm we loaded what we saved.

```{code-cell} ipython3
# this happens to work 
# because VocalPy always gives us back `sorted` lists,
# but it wouldn't work in the more general case--
# we'd need to pair by filename first or something
for spect, spect_loaded in zip(spects, spects_loaded):
    assert spect == spect_loaded
```

### Data type: `vocalpy.Annotation`

+++

The last data type we'll look at is for annotations. Such annotations are important for analysis of aocustic communication and behavior. Under the hood, VocalPy uses the pyOpenSci package [crowsetta](https://github.com/vocalpy/crowsetta).

```{code-cell} ipython3
import vocalpy as voc

csv_paths = voc.paths.from_dir(data_dir, '.wav.csv')
```

```{code-cell} ipython3
annots = [voc.Annotation.read(notmat_path, format='simple-seq') 
          for notmat_path in csv_paths]
```

We inspect one of the annotations. Again as with other data types, we can see there is a `data` attribute. In this case it contains the `crowsetta.Annotation`.

```{code-cell} ipython3
print(annots[1])
```

We plot the spectrogram along with the annotations.

```{code-cell} ipython3
voc.plot.annotated_spectrogram(
    spect=voc.Spectrogram(data=np.log(spects[1].data),
                    frequencies=spects[1].frequencies,
                    times=spects[1].times),
    annot=annots[1],
    tlim = [3.2, 3.9],
    flim=[500,12500],
    pcolormesh_kwargs={'vmin':-25, 'vmax': -10}
);
```

This crash course in VocalPy has introduced you to the key features and goals of the library. To learn more, please check out [the documentation](https://vocalpy.readthedocs.io/en/latest/) and read our Forum Acusticum 2023 Proceedings Paper, ["Introducing VocalPy"](https://github.com/vocalpy/vocalpy/blob/main/docs/fa2023/Introducing_VocalPy__a_core_Python_package_for_researchers_studying_animal_acoustic_communication.pdf). We are actively developing the library to meet your needs and would love to hear your feedback in [our forum](https://forum.vocalpy.org/)
