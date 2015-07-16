# Reproducible Experiment Platform (REP)

__REP__ is ipython-based environment for conducting data-driven research in a consistent and reproducible way.

Main features:

  * unified python wrapper for different ML libraries (wrappers follow extended __scikit-learn__ interface)
    * TMVA
    * Sklearn
    * XGBoost
    * uBoost
    * Theanets
    * Pybrain
    * Neurolab
  * parallel training of classifiers on cluster 
  * classification/regression reports with plots
  * interactive plots supported
  * smart grid-search algorithms with parallelized execution
  * versioning of research using git
  * pluggable quality metrics for classification
  * meta-algorithm design (aka 'rep-lego')


### Howto examples
To get started, look at the notebooks in /howto/  <br />
Notebooks can be viewed (not executed) online at [nbviewer](http://nbviewer.ipython.org/github/yandex/rep/tree/master/howto/)  <br />
There are basic introductory notebooks (about python, IPython) and more advanced ones (about the REP itself)

### Running using docker
We provide the docker container with __REP__ and all it's dependencies <br />
https://github.com/yandex/rep/wiki/Running-REP-using-Docker/

### Installation
However, if you want to install __REP__ on your machine, follow this manual:  <br />
https://github.com/yandex/rep/wiki/Installing-manually <br />
and https://github.com/yandex/rep/wiki/Running-manually

### License
Apache 2.0, library is open-source.

### Links
* [documentation](http://yandex.github.io/rep/)
* [bugtracker](https://github.com/yandex/rep/issues)
* [howto](http://nbviewer.ipython.org/github/yandex/rep/tree/master/howto/)
* [contributing new estimator](https://github.com/yandex/rep/wiki/Contributing-new-estimator)
* [contributing new metric](https://github.com/yandex/rep/wiki/Contributing-new-metrics)


