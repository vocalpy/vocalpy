---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
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

First we import `vocalpy`.

```{code-cell} ipython3
import vocalpy as voc
```

Then we get some example data, from the [Bengalese Finch song repository](https://nickledave.github.io/bfsongrepo/).

```{code-cell} ipython3
:tags: [hide-output]

sounds = voc.example('bfsongrepo', return_type='sound')
```

## Data types for acoustic communication

+++

Let's look at the data types that VocalPy provides for acoustic comunication.

+++

### Data type for sound: `vocalpy.Sound`

+++

Calling `vocalpy.example('bfsongrepo')` gave us back a list of of `vocalpy.Sound` instances.
Let's inspect one of them.

```{code-cell} ipython3
a_sound = sounds[0]
print(a_sound)
```

We can see that it has three attributes:

1. `data`, the audio signal itself, with two dimensions: (channels, samples)

```{code-cell} ipython3
print(a_sound.data)
```

2. `samplerate`, the sampling rate for the audio

```{code-cell} ipython3
print(a_sound.samplerate)
```

and finally,  

3. `path`, the path to the file that the sound was read from

```{code-cell} ipython3
print(a_sound.path)
```

A `Sound` also has three properties, derived from its data:
1. `channels`, the number of channels
2. `samples`, the number of samples, and
3. `duration`, the number of samples divided by the sampling rate.

```{code-cell} ipython3
print(
    f"This sound comes from an audio file with {a_sound.channels} channel, "
    f"{a_sound.samples} samples, and a duration of {a_sound.duration:.3f} seconds"
)
```

One of the reasons VocalPy provides this data type, and the others we're about to show you here, is that it helps you write more succinct code that's easier to read: for you, when you come back to your code months from now, and for others that want to read the code you share.

+++

When you are working with your own data, instead of example data built into VocalPy, you will do something like:  

1. Load all the sound files from a directory using a convenience function that VocalPy gives us in its `paths` module, `vocalpy.paths.from_dir`
2. Load all the wav files into the data type that VocalPy provides for sound, `vocalpy.Sound`, using the method `vocalpy.Sound.read`:

This is shown in the snippet below.
```python
data_dir = ('data/bfsongrepo/gy6or6/032312/')
wav_paths = voc.paths.from_dir(data_dir, 'wav')
sounds = [
    voc.Sound.read(wav_path) for wav_path in wav_paths
]
```

+++

## Classes for steps in pipelines for processing data in acoustic communication

+++

In addition to data types for acoustic communication, VocalPy provides you with classes that represent steps in pipelines for processing that data. These classes are also written with readability and reproducibility in mind.

+++

Let's use one of those classes, `SpectrogramMaker`, to make a spectrogram from each one of the wav files that we loaded above.

We'll write a brief snippet to do so, and then we'll explain what we did.

```{code-cell} ipython3
:tags: [hide-output]
params = {'n_fft': 512, 'hop_length': 64}
callback = voc.spectrogram
spect_maker = voc.SpectrogramMaker(callback=callback, params=params)
spects = spect_maker.make(sounds, parallelize=True)
```

Notice a couple of things about this snippet:
- In line 1, you declare the parameters that you use to generate spectrograms explicitly, as a dictionary. This helps with reproducibility by encouraging you to document those parameters
- In line 2, you also decide what function you will use to generate the spectrograms. Here we use the helper function `vocalpy.spectrogram`.
- In line 3, you create an instance of the `SpectrogramMaker` class with the function you want to use to generate spectrograms, and the parameters to use with that function. We refer to the function we pass in as a `callback`, because the `SpectrogramMaker` will "call back" to this function when it makes a spectrogram.
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

We do so by calling `vocalpy.plot.spectrogram`.

```{code-cell} ipython3
voc.plot.spectrogram(
    a_spect,
    tlim = [2.6, 4],
    flim=[500,12500],
)
```

We see that we have a spectrogram of Bengalese finch song.

Now that we know what we're working with, let's actually inspect the attributes of the `vocalpy.Spectrogram` instance.

+++

There are five attributes we care about here.

1. `data`: this is the spectrogram itself -- as with the other data types,like `vocalpy.Sound`, the attribute name `data` indiciates this main data we care about

```{code-cell} ipython3
print(a_spect.data)
```

Let's look at the shape of `data`. It's really just a NumPy array, so we inspect the array's `shape` attribute.

```{code-cell} ipython3
print(a_spect.data.shape)
```

We see that we have an array with dimensions (channels, frequencies, times). The last two dimensions correspond to the next two attributes we will look at.

+++

2. `frequencies`, a vector of the frequency for each row of the spectrogram.

```{code-cell} ipython3
print(a_spect.frequencies[:10])
```

```{code-cell} ipython3
print(a_spect.frequencies.shape)
```

(We see it is equal to the number of rows.)

3. `times`, a vector of the time for each column in the spectrogram.

```{code-cell} ipython3
print(a_spect.times[:10])
```

```{code-cell} ipython3
print(a_spect.times.shape)
```

Just like with the `Sound` class, VocalPy gives us the ability to conveniently read and write spectrograms from files. This saves us from generating spectrograms over and over. Computing spectrograms can be computionally expensive, if your audio has a high sampling rate or you are using methods like multi-taper spectrograms. Saving spectrograms from files also makes it easier for you to share your data in the exact form you used it, so that it's easier to replicate your analyses.

+++

To see this in action, let's write our spectrograms to files.

```{code-cell} ipython3
import pathlib

DATA_DIR = pathlib.Path('./data')
DATA_DIR.mkdir(exist_ok=True)

for spect in spects:
    spect.write(
        DATA_DIR / (spect.audio_path.name + '.spect.npz')
    )
```

Notice that the extension is `'npz'`; this is a file format that NumPy uses to save mulitple arrays in a single file. By convention we include the file extension of the source audio, and another "extension" that incidicates this is a spectrogram, so that the file name ends with `'.wav.spect.npz'`.

+++

We can confirm that reading and writing spectrograms to disk works as we expect using the method `vocalpy.Spectrogram.read`

```{code-cell} ipython3
spect_paths = voc.paths.from_dir(DATA_DIR, '.spect.npz')
```

```{code-cell} ipython3
spects_loaded = [
    voc.Spectrogram.read(spect_path)
    for spect_path in spect_paths
]
```

We compare with the equality operator to confirm we loaded what we saved.

Before doing so, we sort the original ``spects`` by the ``audio_path`` of the sound they were generated from, and the ``spects_loaded`` by the path they were loaded from. In this case, doing so puts both lists in the same order, because we used the audio file's filename as part of the spectrogram file's filename. (It might not work more generally, if you name your files differently.)

```{code-cell} ipython3
spects = sorted(spects, key=lambda spect: spect.audio_path)
spects_loaded = sorted(spects_loaded, key=lambda spect: spect.path)
for spect, spect_loaded in zip(spects, spects_loaded):
    assert spect == spect_loaded
```

### Data type: `vocalpy.Annotation`

+++

The last data type we'll look at is for annotations. Such annotations are important for analysis of aocustic communication and behavior. Under the hood, VocalPy uses the pyOpenSci package [crowsetta](https://github.com/vocalpy/crowsetta).

```{code-cell} ipython3
import vocalpy as voc

# We get back the paths to all the files in this example dataset, 
# but only keep the ones that are csv files, because those are the annotations.
# This filters out the wav files.
csv_paths = [path for path in voc.example('bfsongrepo') if path.name.endswith('csv')]
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
    spect=spects[1],
    annot=annots[1],
    tlim = [3.2, 3.9],
    flim=[500,12500],
);
```

This crash course in VocalPy has introduced you to the key features and goals of the library. To learn more, please check out [the documentation](https://vocalpy.readthedocs.io/en/latest/) and read our Forum Acusticum 2023 Proceedings Paper, ["Introducing VocalPy"](https://github.com/vocalpy/vocalpy/blob/main/docs/fa2023/Introducing_VocalPy__a_core_Python_package_for_researchers_studying_animal_acoustic_communication.pdf). We are actively developing the library to meet your needs and would love to hear your feedback in [our forum](https://forum.vocalpy.org/).
