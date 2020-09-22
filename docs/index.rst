.. REP documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to REP's documentation!
===============================

REP (Reproducible Experiment Platform) library provides functionality for all basic needs to deal with machine learning.

.. raw:: html

    <div style='height: 36px;'>
    <iframe src="https://ghbtns.com/github-btn.html?user=yandex&repo=rep&type=star&count=true" frameborder="0" scrolling="0" width="120px" height="23px"></iframe>
    <iframe src="https://ghbtns.com/github-btn.html?user=yandex&repo=rep&type=watch&count=true&v=2" frameborder="0" scrolling="0" width="120px" height="23px"></iframe>
    <div style='clear: both' ></div>
    </div>

It includes:


  * :doc:`estimators` is sklearn-like wrappers for variety of machine learning libraries:

    * TMVA
    * Sklearn
    * XGBoost
    * Pybrain
    * Neurolab
    * Theanets.
    * MatrixNet service (**available to CERN**)

    These can be used as base estimators in sklearn.

  * :doc:`metaml` contains factory (the set of estimators), grid search, folding algorithm. Also parallel execution on a cluster is supported.
  * :doc:`metrics` implement some basis for metrics used in reports and during grid search.
  * :doc:`report` contains helpful classes to get model result information on any dataset.
  * :doc:`plotting` is a wrapper for different plotting libraries including interactive plots.

    * matplotlib
    * bokeh
    * tmva

  * :doc:`parallel` describes REP way to parallelize tasks.
  * :doc:`data` defines LabeledDataStorage - a custom way to store training data in a single object.
  * :doc:`utils`  contains additional functions.
  * :doc:`reproducibility` is a recipe to make research reliable.
  * `Howto notebooks <http://nbviewer.ipython.org/github/yandex/rep/tree/master/howto/>`_ contains examples.

Main repository: http://github.com/yandex/rep

Installation
============

REP provides several installation ways.

Please find instructions at the `repository <https://github.com/yandex/REP>`_ main page.

Documentation index:
====================
.. toctree::
    :maxdepth: 2

    estimators
    metaml
    report
    metrics
    plotting
    parallel
    data
    utils
    reproducibility
    Howto notebooks <http://nbviewer.ipython.org/github/yandex/rep/tree/master/howto/>
