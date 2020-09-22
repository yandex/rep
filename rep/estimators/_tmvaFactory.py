"""
Supplementary script to train a TMVA estimator.
"""

from __future__ import division, print_function, absolute_import
import sys
import os

import numpy
import pandas
from root_numpy.tmva import add_classification_events, add_regression_events

import ROOT
from . import tmva
import six
from six.moves import cPickle as pickle


__author__ = 'Tatiana Likhomanenko'


def tmva_process(estimator, info, data, target, sample_weight):
    """
    Create a TMVA classification/regression factory; training, testing and evaluating.

    :param estimator: classifier/regressor which should be trained
    :type estimator: rep.estimators.tmva.TMVAClassifier or rep.estimators.tmva.TMVARegressor
    :param rep.estimators.tmva._AdditionalInformation info: additional information
    :param pandas.DataFrame data: training data
    :param target: array-like targets
    :param sample_weight: array-like samples weights
    """

    ROOT.TMVA.Tools.Instance()

    file_out = ROOT.TFile(os.path.join(info.directory, info.tmva_root), "RECREATE")
    factory = ROOT.TMVA.Factory(info.tmva_job, file_out, estimator.factory_options)

    for var in data.columns:
        factory.AddVariable(var)

    # Set data
    if info.model_type == 'classification':
        if estimator.method == 'kCuts':
            # signal must be the first added to the tree, because method *rectangular cut optimization* doesn't work in another way
            inds = numpy.argsort(target)[::-1]
            data = data.ix[inds, :]
            target = target[inds]
            sample_weight = sample_weight[inds]
        add_classification_events(factory, numpy.array(data), target, weights=sample_weight)
        add_classification_events(factory, numpy.array(data), target, weights=sample_weight, test=True)
    elif info.model_type == 'regression':
        factory.AddTarget('target')
        add_regression_events(factory, numpy.array(data), target, weights=sample_weight)
        add_regression_events(factory, numpy.array(data), target, weights=sample_weight, test=True)
    else:
        raise NotImplementedError("Doesn't support type {}".format(info.model_type))

    factory.PrepareTrainingAndTestTree(ROOT.TCut('1'), "")
    # Set method
    parameters = ":".join(
        ["{key}={value}".format(key=key, value=value) for key, value in estimator.method_parameters.items()])
    factory.BookMethod(ROOT.TMVA.Types.__getattribute__(ROOT.TMVA.Types, estimator.method), estimator._method_name,
                       parameters)

    factory.TrainAllMethods()
    file_out.Close()


def main():
    # Python 2 dumps in text mode. Python 3 in binary.
    if six.PY2:
        stdin = sys.stdin
    else:
        stdin = sys.stdin.buffer

    # Reading the configuration from the stdin
    classifier = pickle.load(stdin)
    info = pickle.load(stdin)
    data = pickle.load(stdin)
    labels = numpy.array(pickle.load(stdin))
    sample_weight = numpy.array(pickle.load(stdin))
    assert isinstance(classifier, tmva.TMVAClassifier) or isinstance(classifier, tmva.TMVARegressor)
    assert isinstance(info, tmva._AdditionalInformation)
    assert isinstance(data, pandas.DataFrame)
    tmva_process(classifier, info, data, labels, sample_weight)
