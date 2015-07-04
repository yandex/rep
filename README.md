# Reproducible Experiment Platform (REP)

REP is environment for conducting data-driven research in a consistent and reproducible way.

Main REP features include:

  * unified classifiers wrapper for variety of implementations
    * TMVA
    * Sklearn
    * XGBoost
    * uBoost
    * Theanets
    * Pybrain
    * Neurolab
  * parallel training of classifiers on cluster 
  * classification/regression reports with plots
  * support for interactive plots
  * grid-search algorithms with parallelized execution
  * versioning of research using git
  * pluggable quality metrics for classification
  * meta-algorithm design (aka 'rep-lego')


### Running using docker
We provide the docker container with `REP` and all it's dependencies <br />
https://github.com/yandex/rep/wiki/Running-REP-using-Docker/

### Installation
However, if you want to install `REP` on your machine, follow this manual:  <br />
https://github.com/yandex/rep/wiki/Installing-manually <br />
and https://github.com/yandex/rep/wiki/Running-manually

### First steps

To get started with the framework, look at the notebooks in /howto/  <br />
Notebooks in repository can be viewed (not executed) online at nbviewer: http://nbviewer.ipython.org/github/yandex/rep/tree/master/howto/  <br />
There are basic introductory notebooks (about python, IPython) and more advanced ones (about the REP itself)
