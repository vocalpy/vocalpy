# vocles
## `voc`alization tab`les`
a tool for building and processing datasets of animal vocalizations
and other types of bioacoustics data.

### why?
Researchers studying animal vocalizations and other types  
of bioacoustics data typically need to work with large, 
heterogeneous datasets: audio files, array files 
that contain spectrograms made from the audio,  
annotations of the vocalizations or other animal sounds 
that appear in the audio and array files, and other 
associated files, such as features extracted from audio 
and spectrograms. 
All of these files may be in one 
of any number of formats!

Once researchers studying animal vocalizations 
have collected and annotated data, they now need a way  
work with it as a single dataset, 
so that they can perform any kind of analysis.
In data science, this is often referred to 
as "munging" the data.
For example, researchers need 
to associate each spectrogram with its annotation, 
and then select only the spectrograms 
that contain a certain label in their annotations.

Many tools have been developed to analyze data, 
but they often require researchers to organize their 
files in a specific data structure. As the size 
of datasets grows, this becomes more and more 
unweildy.

The primary goal of `vocles` is to make the process 
of working with a collection of files as a unified 
dataset as easy as possible. 
A second goal is to make datasets as **portable** 
as possible, so that researchers can move files around, 
combine datasets, and importantly **share** their 
datasets with others, without needing to 
run through the entire processing pipeline from scratch.
Lastly, we provide the ability to `apply` functions 
to your files in a parallelized fashion, so that you 
can rapidly process them: e.g., segment audio of 
vocalizations into individual units such as calls, 
create spectrograms from audio files, 
map your audio files to annotations from files, etc.

### how?
Basically, `vocles` builds a database for you. 
The power of a database is that it lets you ignore 
where your files live. `vocles` uses a very 
lightweight text format for databases built into Python,
that you can easily save, share, add to a repository of code, 
etc. It does this in a way that lets you avoid   
the details of working with databases as much as possible, 
although you can always work with the dataset this way if you 
really want to.
