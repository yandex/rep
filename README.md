# Reproducible Experiment Platform (REP)

[![Join the chat at https://gitter.im/yandex/rep](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/yandex/rep?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

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
To get started, look at the notebooks in [/howto/](https://github.com/yandex/rep/tree/master/howto)

Notebooks can be viewed (not executed) online at [nbviewer](http://nbviewer.ipython.org/github/yandex/rep/tree/master/howto/)  <br />
There are basic introductory notebooks (about python, IPython) and more advanced ones (about the **REP** itself)

### Installation with Docker

We provide the [docker image](https://registry.hub.docker.com/u/anaderi/rep/) with `REP` and all it's dependencies 
* [install with Docker on Mac](https://github.com/yandex/rep/wiki/Instal-REP-with-Docker-(Mac-OS-X))
* [install with Docker on Linux](https://github.com/yandex/rep/wiki/Install-REP-with-Docker-(Linux))
* [install with docker-compose on Linux](https://github.com/yandex/rep/wiki/Install-REP-with-docker-compose-(Linux))


### Installation with bare hands
However, if you want to install `REP` and all of its dependencies on your machine yourself, follow this manual: 
[installing manually](https://github.com/yandex/rep/wiki/Installing-manually) and 
[running manually](https://github.com/yandex/rep/wiki/Running-manually).
 
### Links
* [documentation](http://yandex.github.io/rep/)
* [howto](http://nbviewer.ipython.org/github/yandex/rep/tree/master/howto/)
* [bugtracker](https://github.com/yandex/rep/issues)
* [gitter chat, troubleshooting](https://gitter.im/yandex/rep)
* [API, contributing new estimator](https://github.com/yandex/rep/wiki/Contributing-new-estimator)
* [API, contributing new metric](https://github.com/yandex/rep/wiki/Contributing-new-metrics)

### License
Apache 2.0, library is open-source.


