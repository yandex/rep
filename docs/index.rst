.. REP documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to REP's documentation!
===============================

REP (Reproducible Experiment Platform) library provides functionality for all your basic needs to deal with machine learning.

It includes:


  * :doc:`data` support different data transformations, including operations in memory and on the disk
  * :doc:`estimators` is sklearn-like wrappers for variety of machine learning libraries implementations (**Sklearn, Uboost, XGBoost, TMVA**). You can use them as base estimators in sklearn
  * :doc:`metaml` contains factory (the set of estimators), grid search, folding algorithm. Also parallel execution on a cluster is supported
  * :doc:`report` contains helpful classes to get model result information on any dataset
  * :doc:`plotting` is  wrapper for different plotting libraries including interactive plots (**matplotlib, bokeh, tmva, plotly**)
  * :doc:`utils`  contains additional functions

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



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

