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

# How to use VocalPy with scikit-learn to fit supervised learning models to acoustic features

Many analyses in bioacoustics and communication rely on machine learning models. For example, it is common to fit a classifier to acoustic features extracted from a set of sounds so that the classifier predicts the individual that emitted the sound. Many papers argue based on the accuracy of the fit classifier that information is present in the sound that tells other conspecifics the identity of the individual.

scikit-learn is one of the most widely used library in the Python programming language for fitting supervised machine learning models. How can you extract acoustic features from sounds and then fit a model to those features with scikit-learn? 
In this notebook we walk through an example of classifying individual zebra finches using acoustic parameters extracted from their calls.

Please note that material here is adapted in part from https://github.com/theunissenlab/BioSoundTutorial

```{code-cell} ipython3
import numpy as np
import pandas as pd
import sklearn
import vocalpy as voc
```

For this how-to, we use a subset of data from [this dataset](https://figshare.com/articles/dataset/Vocal_repertoires_from_adult_and_chick_male_and_female_zebra_finches_Taeniopygia_guttata_/11905533). 
To get this subset, we can call the `vocalpy.example` function (that, under the hood, "fetches" the data using the excellent library [`pooch`](https://www.fatiando.org/pooch/latest/index.html)).

By default, the function gives us back vocalpy data types like {py:class}`vocalpy.Sound`, but in this case we want the paths to the files. That's because the filenames contain the ID of the zebra finch that made the sound, and below we are going to train a model to classify IDs. To get back the paths instead of {py:class}`vocalpy.Sound` instances, we set the argument `return_path` to `True`.

```{code-cell} ipython3
zblib = voc.example("zblib", return_path=True)
```

```{code-cell} ipython3
print(zblib.sound[0])
```

We make a helper function to get the bird IDs from the filenames.  

We will use this below when we want to predict the bird ID from the extracted features.

```{code-cell} ipython3
def bird_id_from_path(wav_path):
    """Helper functoin that gets a bird ID from a path"""
    return wav_path.name.split('_')[0]
```

```{code-cell} ipython3
bird_id_from_path(zblib.sound[0])
```

We use a list comprehension to get the ID from all 91 files.

```{code-cell} ipython3
bird_ids = [
    bird_id_from_path(wav_path)
    for wav_path in zblib.sound
]
```

## Feature extraction

Now we extract the acoustic features we will use to classify.  

For this example we use the temporal and spectral features from `soundsig`, since those are relatively quick to extract. For an example that uses fundamental frequency estimation, see https://github.com/theunissenlab/BioSoundTutorial/blob/master/BioSound4.ipynb

```{code-cell} ipython3
callback = voc.feature.biosound
params = dict(ftr_groups=("temporal", "spectral"))
extractor = voc.FeatureExtractor(callback, params)
```

Notice that we are going to only use channel of the audio to extract features. The function we will use to extract features, {py:func}`vocalpy.feature.biosound`, will work on audio with multiple channels, but for demonstration purposes we just need one.

```{code-cell} ipython3
sounds = []
for wav_path in zblib.sound:
    sounds.append(
        voc.Sound.read(wav_path)[0]  # indexing with `[0]` gives us the first channel
    )
```

```{code-cell} ipython3
features_list = extractor.extract(sounds, parallelize=True)
```

## Data preparation

Now what we want to get from our extracted features is two NumPy arrays, `X` and `y`.  

These represent the samples $X_i$ in our dataset with their features $x$, and the labels for those samples $y_i$. In this case we have a total of $m=$91 samples (where $i \in 1, 2, ... m$).

We get these arrays as follows (noting there are always multiple ways to do things when you're programming):
- Take the `data` attribute of the {py:class}`~vocalpy.Features` we got back from the {py:class}`~vocalpy.FeatureExtractor` and convert it to a {py:class}`pandas.DataFrame` with one row: the scalar set of features for exactly one sound
- Use {py:mod}`pandas` to concatenate all those {py:class}`~pandas.DataFrame`s, so we end up with 91 rows
- Add a column to this `DataFrame` with the IDs of the birds -- we then have $X$ and $y$ in a single table we could save to a csv file, to do further analysis on later
- We get $X$ by using the `values` attribute of the `DataFrame`, which is a numpy array
- We get $y$ using {py:func}`pandas.factorize`, that converts the unique set of strings in the `"id"` column into integer class labels: i.e., since there are 4 birds, for every row we get a value from $\{0, 1, 2, 3\}$

```{code-cell} ipython3
df = pd.concat(
    [features.data.to_pandas()
    for features in features_list]
)
```

```{code-cell} ipython3
df.head()
```

```{code-cell} ipython3
df["id"] = pd.array(bird_ids, dtype="str")
y, _ = df["id"].factorize()
X = df.values[:, :-1]  # -1 because we don't want 'id' column
```

## Fitting a Random Forest classifier

Finally we will train a classifer from `scikit-learn` to classify these individuals.

```{code-cell} ipython3
import sklearn.model_selection
```

```{code-cell} ipython3
X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
    X, y, stratify=y, train_size=0.8
)
```

```{code-cell} ipython3
from sklearn.ensemble import RandomForestClassifier
```

```{code-cell} ipython3
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)
```

```{code-cell} ipython3
print(
    f"Accuracy: {clf.score(X_val, y_val) * 100:0.2f}%"
)
```
