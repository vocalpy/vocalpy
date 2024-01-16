(background)=

### Background

Are humans unique among animals? 
We speak languages, but is speech somehow like other animal behaviors, such as birdsong? 
Questions like these are answered by studying how animals communicate with sound. 
This research requires cutting edge computational methods and big team science across a wide range of disciplines, 
including ecology, ethology, bioacoustics, psychology, neuroscience, linguistics, and 
genomics [^cite_wir2019][^cite_SainburgGentner2020][^cite_Stowell2022][^cite_Cohenetal2022a]. 

Although the VocalPy library can be used more generally for bioacoustics, 
our focus is on animal acoustic communication [^cite_SainburgGentner2020][^cite_Beecher2020] 
and related research areas like *vocal learning* [^cite_wikipedia]
and *cultural evolution* [^cite_YoungbloodLahti2018].

### Goals

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

### References

[^cite_SainburgGentner2020]: Sainburg, Tim, and Timothy Q. Gentner. 
   “Toward a Computational Neuroethology of Vocal Communication: 
   From Bioacoustics to Neurophysiology, Emerging Tools and Future Directions.” 
   Frontiers in Behavioral Neuroscience 15 (December 20, 2021): 811737. https://doi.org/10.3389/fnbeh.2021.811737.
   <https://www.frontiersin.org/articles/10.3389/fnbeh.2021.811737/full>

[^cite_Stowell2022]: Stowell, Dan. 
   “Computational Bioacoustics with Deep Learning: A Review and Roadmap,” 2022, 46.
   <https://peerj.com/articles/13152/>

[^cite_Cohenetal2022a]: Cohen, Yarden, et al. 
   "Recent Advances at the Interface of Neuroscience and Artificial Neural Networks." 
   Journal of Neuroscience 42.45 (2022): 8514-8523.
   <https://www.jneurosci.org/content/42/45/8514>

[^cite_Beecher2020]: Beecher, Michael D. 
   “Animal Communication.” 
   In Oxford Research Encyclopedia of Psychology. 
   Oxford University Press, 2020. <https://doi.org/10.1093/acrefore/9780190236557.013.646>.  
   <https://oxfordre.com/psychology/view/10.1093/acrefore/9780190236557.001.0001/acrefore-9780190236557-e-646>.

[^cite_YoungbloodLahti2018]: Youngblood, Mason, and David Lahti. 
   “A Bibliometric Analysis of the Interdisciplinary Field of Cultural Evolution.” 
   Palgrave Communications 4, no. 1 (December 2018): 120. <https://doi.org/10.1057/s41599-018-0175-8>.
   <https://www.nature.com/articles/s41599-018-0175-8>.

[^cite_wikipedia]: <https://en.wikipedia.org/wiki/Vocal_learning>

[^cite_wir2019]: Wirthlin M, Chang EF, Knörnschild M, Krubitzer LA, Mello CV, Miller CT,
    Pfenning AR, Vernes SC, Tchernichovski O, Yartsev MM.
    "A modular approach to vocal learning: disentangling the diversity of
    a complex behavioral trait." Neuron. 2019 Oct 9;104(1):87-99.
    <https://www.sciencedirect.com/science/article/pii/S0896627319308396>

[^1]: For a curated collection, see <https://github.com/rhine3/bioacoustics-software>.