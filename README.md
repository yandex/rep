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

We provide the [docker image](https://registry.hub.docker.com/u/yandex/rep/) with `REP` and all it's dependencies 
* [install with Docker on Linux](https://github.com/yandex/rep/wiki/Install-REP-with-Docker-(Linux))
* [install with Docker on Mac and Windows](https://github.com/yandex/rep/wiki/Instal-REP-with-Docker-(Mac-OS-X))


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

### Minimal examples

__REP__ wrappers are python compatible:

```python
from rep.estimators import XGBoostClassifier, SklearnClassifier, TheanetsClassifier
clf = XGBoostClassifier(n_estimators=300, eta=0.1).fit(trainX, trainY)
probabilities = clf.predict_proba(testX)
```

Beloved trick of kagglers is to run bagging over complex algorithms. This is how it is done in __REP__:

```python
from sklearn.ensemble import BaggingClassifier
clf = BaggingClassifier(base_estimator=XGBoostClassifier(), n_estimators=10)
# wrapping sklearn to REP wrapper
clf = SklearnClassifier(clf)
```

Another useful trick is to use folding instead of splitting data into train/test. 
This is specially useful when you're using some kind of complex stacking

```python
from rep.metaml import FoldingClassifier
clf = FoldingClassifier(TheanetsClassifier(), n_folds=3)
probabilities = clf.fit(X, y).predict_proba(X)
```
In example above all data are splitted into 3 folds, 
and each fold is predicted by classifier which was trained on other 2 folds.  

Also __REP__ classifiers provide report:

```python
report = clf.test_on(testX, testY)
report.roc().plot() # plot ROC curve
from rep.report.metrics import RocAuc
# learning curves are useful when training GBDT!
report.learning_curve(RocAuc(), steps=10)  
```

You can read about other __REP__ tools (like distributed grid search) in documentation and howto examples.


