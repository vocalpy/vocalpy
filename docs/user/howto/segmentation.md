---
jupytext:
  formats: md:myst
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

# How do I evaluate segmentation methods with information retrieval metrics?

This how-to walks through using metrics from information retrieval to evaluate segmentation methods.

We will compare two algorithms that take as input a :class:`vocalpy.Sound`, and return as output :class:`~vocalpy.Segments`. The :class:`vocalpy.Segments` class represents a set of line segments, with each segment having a starting index and length (or, equivalently, a start time and a stop time). One of the goals of this how-to is to show you how using the :class:`vocalpy.Segments` class makes it easier for you to evaluate segmentation algorithms.

The goal of our evaluation is to get a measure of how close the segments are to a ground truth segmentation, produced by a human annotator. In this case, the human annotator used a Graphical User Interface (GUI) to clean up segments that were returned by one of the algorithms we will look at. So it shouldn't be too surprising if that algorithm in particular does a pretty good job of getting close to the ground truth segmentation. The algorithm in question is :func:`vocalpy.segment.meansquared`. This algorithm is used by a Matlab GUI `evsonganaly` originally developed by Evren Tumer in the Brainard Lab, as used in [Tumer Brainard 2012](link). The version of the algorithm built into VocalPy is adapted from the Python implementation in the `evfuncs` package.

* talk about energy here -- plot energy and show threshold?


* define "information retrieval" and the metrics? reference the papers that we reference in the docstrings?

What we want to understand is the role that different parameters play in the algorithm. To understand 

talk @ clean-up

To understand the role of these parameters, we will evaluate the output of :func:`vocalpy.segment.meansquared` with and without the clean-up parameters. We will also compare with another algorithm that simply sets the threshold to the average of the signal.

There are three conditions we want to compare:
1. The 
2. The
3. A baseline algorithm that 

This how-to replicates in part the analysis from Ghaffari Devos 2023 [^1].

[Ghaffari Devos 2023](https://dael.euracoustics.org/confs/fa2023/data/articles/000897.pdf).

+++

## Baseline algorithm for segmentation

Before walking through the analysis, we need to implement the baseline we will use.

We include the function here to give a brief example of writing a segmenting algorithm.

Notice that the first parameter of the function is a :class:`~vocalpy.Sound` and that it returns a :class:`vocalpy.Segments`. This is the key thing you will need to write a segmenting algorithm. You can then pass this function to the :class:`vocalpy.Segmenter` class, to be used as a "callback", the function that the class calls when you tell it to `segment` some `Sound`s.

```{code-cell} ipython3
import numpy as np
import scipy.signal

import vocalpy as voc


def average_envelope_threshold(sound: voc.Sound, cutoff=500, order=40) -> voc.Segments:
    """Segment audio by threshold with the average of the envelope.

    This function (1) high-pass filters the audio to reduce noise,
    (2) extracts the Hilbert envelope, (3) smooths the envelope with a Hann window,
    and then (4) thresholds the smooothed envelope to segment, 
    setting the threshold to average of the envelope.
    
    Adapted from https://github.com/houtan-ghaffari/bird_syllable_segmentation
    See https://dael.euracoustics.org/confs/fa2023/data/articles/000897.pdf for detail.
    """
    if sound.data.shape[0] > 1:
        raise ValueError(
            f"The ``sound`` has {sound.data.shape[0]} channels, but segmentation is not implemented "
            "for sounds with multiple channels. This is because there can be a different number of segments "
            "per channel, which cannot be represented as a rectangular array. To segment each channel, "
            "first split the channels into separate ``vocalpy.Sound`` instances, then pass each to this function."
            "For example,\n"
            ">>> sound_channels = [sound_ for sound_ in sound]  # split with a list comprehension\n"
            ">>> channel_segments = [vocalpy.segment.meansquared(sound_) for sound_ in sound_channels]\n"
        )

    x = sound.data.squeeze(axis=0)

    sos = scipy.signal.butter(
        order, cutoff, btype="highpass", analog=False, output="sos", fs=sound.samplerate
    )
    x = scipy.signal.sosfiltfilt(sos, x)
    x = np.abs(scipy.signal.hilbert(x))
    win = scipy.signal.windows.hann(512)
    x = scipy.signal.convolve(x, win, mode='same') / sum(win)
    threshold = np.mean(x)
    above_threshold = x > threshold

    # convolving with h causes:
    # +1 whenever above_th changes from 0 to 1
    # and -1 whenever above_th changes from 1 to 0
    h = np.array([1, -1])
    above_convolved = np.convolve(h, above_threshold)
    onsets_sample = np.where(above_convolved > 0)[0]
    offsets_sample = np.where(above_convolved < 0)[0]
    lengths = offsets_sample - onsets_sample

    return voc.Segments(        
        start_inds=onsets_sample,
        lengths=lengths,
        sound=sound
    )    
```

## Compare segmentation algorithms

### Get data

We will use a subset of data from the [Bengalese Finch Song Repository](https://nickledave.github.io/bfsongrepo/), that is built into VocalPy as an example.

```{code-cell} ipython3
sounds = voc.example('bfsongrepo', return_type='sound')
```

### Segment audio

To set ourselves up to analyze the results below, we will make a Python dictionary that maps a string `'name'` (for the algorithm we are testing) to the :class:`list` of :class:`vocalpy.Segments` returned by the algorithm. Since we test the same algorithm twice with different parameters, we will think of this name as a "condition" (the three conditions we outlined above).

To actually get the results, we will write a loop.
Inside the loop, we will use the :class:`vocalpy.Segmenter` class.  
This class takes has two parameters: a `callback` function, and the `params` (parameters) we will pass to that function.
Each time through the loop, we will make an instance of the `Segmenter` class by passing in a specific `callback` and set of `params` as the two arguments. Then when we call the `segment` method on that instance, the class will call the `callback` function.
So, at the top of the loop, we define a tuple of 3-element tuples that we iterate through. Each 3-element tuple has the condition `name`, the `callback` and the `params` we will use with it.

```{code-cell} ipython3
algo_segments_map = {}

for name, callback, params in (
    ('meansquared', voc.segment.meansquared, voc.segment.MeanSquaredParams(threshold=1500, min_dur=0.01, min_silent_dur=0.006)),
    # here we set the parameters for minimum durations to zero, meaning "don't filter"
    ('meansquared-no-cleanup', voc.segment.meansquared, voc.segment.MeanSquaredParams(threshold=1500, min_dur=0., min_silent_dur=0.)),
    # our baseline. We set the params to None. We write it here so we have a value for "params" when we loop through these
    # even though the default is None
    ('average_envelope_threshold', average_envelope_threshold, None)
):
    print(f"Segmenting with algorithm/condition: '{name}'")
    segmenter = voc.Segmenter(callback, params)
    algo_segments_map[name] = segmenter.segment(sounds)
```

### Get ground truth data

Now we need our reference segmentation to compare to.

We will need list of :class:`vocalpy.Segments` for this too, but here we make them using :class:`~vocalpy.Annotation` that we load from the example datasets. These annotations contain the ground truth segmentation that we want to compare with the results from the algorithms.

```{code-cell} ipython3
annots = voc.example('bfsongrepo', return_type='annotation')

ref_segments_list = []
for annot, sound in zip(annots, sounds):
    start_inds = (annot.data.seq.onsets_s * sound.samplerate).astype(int)
    stop_inds = (annot.data.seq.offsets_s * sound.samplerate).astype(int)
    lengths = stop_inds - start_inds
    ref_segments_list.append(
        voc.Segments(
            start_inds,
            lengths,
            sound=sound,
        )
    )
```

### Compute metrics

Finally we use the functions in the :mod:`vocalpy.metrics.segmentation.ir` module to compute metrics that we use to compare the segmentation algorithms.

The functions in this module expect two arguments: a `reference` segmentation, and a `hypothesis`. In other words, the ground truth and the output of some algorithm that we want to compare with that ground truth.

Notice that we use a `tolerance` of 10 milliseconds. This is a standard value used in previous work--you may find it instructive to vary this value and examine the results.

We will make a :class:`pandas.DataFrame` with the results, that we will then plot with :module:`seaborn`.

```{code-cell} ipython3
import pandas as pd

results_records = []  # will become a DataFrame


TOLERANCE = 0.01  # milliseconds


for name, hypothesis_segments_list in algo_segments_map.items():
    for reference, hypothesis in zip(ref_segments_list, hypothesis_segments_list):
        prec, _, _ = voc.metrics.segmentation.ir.precision(
            reference=reference.all_times,
            hypothesis=hypothesis.all_times,
            tolerance=0.01  # 10 milliseconds
        )
        rec, _, _ = voc.metrics.segmentation.ir.recall(
            reference=reference.all_times,
            hypothesis=hypothesis.all_times,
            tolerance=0.01  # 10 milliseconds
        )
        fscore, _, _ = voc.metrics.segmentation.ir.fscore(
            reference=reference.all_times,
            hypothesis=hypothesis.all_times,
            tolerance=TOLERANCE
        )
        for metric_name, metric_val in zip(
            ('precision', 'recall', 'fscore'),
            (prec, rec, fscore)
        ):
            results_records.append(
                {
                    'condition': name,
                    'metric': metric_name,
                    'value': metric_val,
                }
            )

results_df = pd.DataFrame.from_records(results_records)
```

We inspect the dataframe to check that it looks like what we expect.  

```{code-cell} ipython3
results_df.head()
```

Note this is in "long" form where we have a "variable" column -- in this case, the different metrics -- and the value that each "variable" takes on.

We could alternatively have the `DataFrame` in wide form, with columns for `'precision'`, `'recall'`, and `'fscore`, but as we'll see, the long form makes it easier to plot below.

+++

### Visualize results

Now let's do some final clean-up of the dataframe, for plotting.

```{code-cell} ipython3
results_df['value'] = results_df['value'] * 100.

results_df['metric'] = results_df['metric'].map(
    {
        'fscore': '$F$-score (%)',
        'precision': 'Precision (%)',
        'recall': 'Recall (%)',
    },
    
)
```

```{code-cell} ipython3
import seaborn as sns

sns.set_context('notebook')
sns.set_style('darkgrid')

sns.catplot(
    results_df,
    y='value',
    hue='condition',
    col='metric',
    kind='bar',
)
```

We can see that the `'meansquared'` algorithm *with* the clean-up steps has the highest precision and recall. Perhaps surprisingly, the `'meansquared'` algorithm *without* clean-up has a *lower* precision than the `average_envelope_threshold`, and even more surprisingly, it has the highest recall of all. We can understand this as follows: the `'meansquared'` algorithm finds "more" segments than the `'average_envelope_threshold'` algorithm--giving it a higher recall--but that also means it returns more false positives. More generally, a result like this would suggest that the clean-up steps have a bigger impact on performance than the methods used to compute the energy and the exact threshold used. Keep in mind that we've shown here is just a demo. To really draw this conclusion we'd need to do an extensive analysis across datasets, and be very clear about our intended use cases for the algorithm.
