---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

## How to use VocalPy with UMAP and HDBScan for dimensionality reduction and clustering

It is becoming more and more common for researchers studying acoustic communication to apply dimensionality reduction methods to their data, and then cluster the data once it is embedded in that lower dimensional space. Many researchers use the UMAP method for dimensionality reduction, via the Python library that implements it, and the HDBSCAN method to cluster, as implemented in the HDBSCAN library. You can install these by running `pip install umap-learn` and `pip install hdbscan`, respectively. Note that an implementation of HDBSCAN is now provided by [scikit-learn](https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.HDBSCAN.html), but here we show using the HDBSCAN package. Our understanding is that as of this writing, there are cases where using the original package may be more computationally efficient (see [this issue](https://github.com/scikit-learn-contrib/hdbscan/issues/633), but that is beyond the scope of this how-to.

Here we provide a brief walkthrough of how you would use VocalPy to work with your data, and prepare it for dimensionality reduction with UMAP and clustering with HDBSCAN. We will use a (very tiny!) sub-set of the [dataset](https://datadryad.org/stash/dataset/doi:10.5061/dryad.g79cnp5ts) that accompanied the paper ["Two pup vocalization types are genetically and functionally separable in deer mice"](https://www.cell.com/current-biology/fulltext/S0960-9822(23)00185-9?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0960982223001859%3Fshowall%3Dtrue). Some code is also adapted from the [repository that reproduces the paper results](https://github.com/nickjourjine/peromyscus-pup-vocal-evolution). Thank you to [Nick Jourjine](https://nickjourjine.github.io/) for helping us pick a subset of their data that is a good size for a tutorial. The original material for this how-to comes from a [bootcamp](https://github.com/vocalpy/acoustic-communication-and-bioacoustics-bootcamp) at the [Neural Mechanism of Acoustic Communciation 2024 Graduate Research Seminar](https://www.grc.org/neural-mechanisms-of-acoustic-communication-grs-conference/2024/) organized by Nick and [Diana Liao](https://scholar.google.com/citations?user=QeqBfDMAAAAJ&hl=en).

For a more detailed tutorial on applying UMAP to animal sounds, please see [this paper](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/1365-2656.13754) from [Mara Thomas](https://www.ab.mpg.de/person/109360/2736) and co-authors. Material here is adapted in part from [this project](https://github.com/marathomas/tutorial_repo) shared under CC-BY-4.0 license, that accompanied the paper from Thomas, et al. We encourage you to also read [Tim Sainburg](https://timsainburg.com/)'s [earlier work](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008228) demonstrating how UMAP can be used with animal sounds.

```{code-cell} ipython3
import pathlib

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
import hdbscan
import vocalpy as voc
```

## Load mouse pup call data

+++

We start by loading the example data, that is given to us as {py:class}`vocalpy.Sound` and {py:class}`vocalpy.Segments` instances.

```{code-cell} ipython3
twopup = voc.example("twopup")
```

We can see that {py:func}`vocalpy.example` gives us back a {py:class}`vocalpy.examples.ExampleData` instance with two attributes, `sound` and `segments`.

```{code-cell} ipython3
twopup
```

We use the :meth:`~vocalpy.Sound.segment` method of the {py:class}`~vocalpy.Sound` class with the {py:class}`vocalpy.Segments` to get a list of {py:class}`~vocalpy.Sound` instances, one for each segment in the sound.

```{code-cell} ipython3
sound, segments = twopup.sound, twopup.segments
all_sounds = sound.segment(segments)
```

Typically in research code you would have done that by looping through start and stop times of segments, e.g., in a csv file, that is the output of some function that segments for you. E.g., this audio was segmented with the {py:func}`vocalpy.segment.ava` function -- you can replicate the results of the paper with .

What we wrote above is just concise "syntactice sugar" that does the same thing under the hood. Now we have a list of sounds.

```{code-cell} ipython3
len(all_sounds)
```

We write a function to give us back Mel spectrograms, so we can parallelize processing with VocalPy.

```{code-cell} ipython3
def melspectrogram(
    sound: voc.Sound, n_mels: int=50, window: str = "hann", 
    n_fft: int = 512, hop_length: int = 128, fmin=5000, fmax=125000
) -> voc.Spectrogram:
    S = librosa.feature.melspectrogram(y=sound.data,
                                       sr=sound.samplerate, 
                                       n_mels=n_mels , 
                                       fmax=fmax, 
                                       fmin=fmin,
                                       n_fft=n_fft,
                                       hop_length=hop_length, 
                                       window=window, 
                                       win_length=n_fft)
    S = librosa.power_to_db(S, ref=np.max)
    t = librosa.frames_to_time(frames=np.arange(S.shape[-1]), sr=sound.samplerate, hop_length=hop_length)
    f = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    return voc.Spectrogram(data=S[0, ...], frequencies=f, times=t)
```

```{code-cell} ipython3
callback = melspectrogram

spect_maker = voc.SpectrogramMaker(callback)
```

```{code-cell} ipython3
all_spects = spect_maker.make(all_sounds, parallelize=True)
```

Now we get the numpy arrays directly, and we throw away the channel dimension.

```{code-cell} ipython3
all_spects = [
    spect.data[0, ...]
    for spect in all_spects
]
```

This should give us a two-dimensional array, with dimensions (frequency, time)

```{code-cell} ipython3
all_spects[0].shape
```

Let's visualize a random subset, just to inspect our data.

```{code-cell} ipython3
import random

fig, ax_arr = plt.subplots(4, 6)
ax_arr = ax_arr.ravel()
for ax, spect in zip(ax_arr, random.sample(all_spects, len(all_spects))):
    ax.pcolormesh(spect)
    ax.set_axis_off()
```

Now we need to know the maximum number of time bins in any of the spectrograms, so we can pad all the spectrograms to the same size.

```{code-cell} ipython3
max_width = np.max([
    spect.data.shape[1] for spect in all_spects   
])
```

We also use the minimum value to pad, so we don't add some very large value in the padding, that could impact the UMAP calculation.

```{code-cell} ipython3
min_val = np.min([spect.min() for spect in all_spects])
```

```{code-cell} ipython3
def pad_spect(spect, max_width=max_width, constant_values=min_val):
    pad_width = max_width - spect.shape[1]
    # pad with half the width needed on both sides
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
    return np.pad(
        spect, ((0, 0), (left_pad, right_pad)), mode='constant', constant_values=min_val,
    )
```

```{code-cell} ipython3
all_spects = [
    pad_spect(spect, max_width)
    for spect in all_spects
]
```

Let's make sure that worked

```{code-cell} ipython3
all(
    [spect.shape[1] == max_width for spect in all_spects]
)
```

Again we visualize a random sample just to double check things are working as expected.

```{code-cell} ipython3
import random

fig, ax_arr = plt.subplots(4, 6)
ax_arr = ax_arr.ravel()
for ax, spect in zip(ax_arr, random.sample(all_spects, len(all_spects))):
    ax.pcolormesh(spect)
    ax.set_axis_off()
```

```{code-cell} ipython3
data = np.array([
    spect.flatten() for spect in all_spects
])
```

## Dimensionality reduction with UMAP

```{code-cell} ipython3
reducer = umap.UMAP(
    n_components=2, min_dist=0.25, n_neighbors=15, verbose=True
)
```

```{code-cell} ipython3
embedding = reducer.fit_transform(data)
```

```{code-cell} ipython3
plt.scatter(embedding[:, 0], embedding[:, 1])
```

## Clustering with HDBSCAN

```{code-cell} ipython3
clusterer = hdbscan.HDBSCAN(min_cluster_size=100, allow_single_cluster=True)
```

```{code-cell} ipython3
clusterer.fit(embedding)
```

We plot using the "labels", that is, the integer class representing the cluster that each data point has been assigned to.

```{code-cell} ipython3
y = clusterer.labels_
import matplotlib as mpl
cmap = mpl.colormaps['tab20'].resampled(np.unique(y).shape[0])
c = cmap.colors[y]
```

```{code-cell} ipython3
plt.scatter(embedding[:, 0], embedding[:, 1], c=c);
```
