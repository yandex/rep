.. REP documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to REP's documentation!
===============================

REP (Reproducible Experiment Platform) library provides functionality for all your basic needs to deal with machine learning.

It includes:


  * :doc:`data` provides operations with data
  * :doc:`estimators` is sklearn-like wrappers for variety of machine learning libraries:

    * TMVA

    * Sklearn

    * XGBoost

    * Pybrain

    * Neurolab

    * Theanets.

    These can be used as base estimators in sklearn.

  * :doc:`metaml` contains factory (the set of estimators), grid search, folding algorithm. Also parallel execution on a cluster is supported
  * :doc:`report` contains helpful classes to get model result information on any dataset
  * :doc:`plotting` is  wrapper for different plotting libraries including interactive plots

    * matplotlib

    * bokeh

    * tmva

    * plotly

  * :doc:`utils`  contains additional functions
  * `Howto notebooks <http://nbviewer.ipython.org/github/yandex/rep/tree/master/howto/>`_ contains examples

Main repository: http://github.com/yandex/REP

Contents:
=========
.. toctree::
    :maxdepth: 2

    data
    estimators
    metaml
    report
    plotting
    utils
    Howto notebooks <http://nbviewer.ipython.org/github/yandex/rep/tree/master/howto/>
