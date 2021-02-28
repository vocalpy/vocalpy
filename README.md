# vocles
## `voc`alization tab`les`
`pandas` and `numpy` extensions for working with animal vocalizations 
and other types of bioacoustics data.

### why?
Researchers studying animal vocalizations and other types  
of bioacoustics data typically need to work with large, 
heterogeneous datasets: audio files, array files 
that contain spectrograms made from the audio, and 
annotations of the vocalizations or other animal sounds 
that appear in the audio and array files. 
All of these files may be in one 
of any number of formats!

Once researchers have created 
such a dataset, they now have to "munge" the 
files together so that they can perform 
any kind of analysis. For example, they need 
to associate each spectrogram with its annotation, 
and then select only the spectrograms 
that contain a certain label in their annotations.

The primary goal of `vocles` is to make this process 
of "munging" files together as easy as possible. 
A second goal is to make datasets as **portable** 
as possible, so that researchers can move files around, 
combine datasets, and importantly **share** their 
datasets with others, without needing to 
run through the entire processing pipeline from scratch.
Lastly, we provide the ability to `apply` functions 
to your files in a parallelized fashion, so that you 
can rapidly create spectrograms from audio files, 
map your audio files to annotations from files, etc.

### how?

We provide a `pandas`-based `PathArray` that lets you 
build a `DataFrame` from directories of audio files, 
array files containing spectrograms, etc.
