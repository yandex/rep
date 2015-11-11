"""
    TMVA factory runs with classifier and additional information
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


def tmva_process(classifier, info, data, labels, sample_weight):
    """
    Create TMVA classification factory, train, test and evaluate all methods

    :param classifier: classifier to train
    :type classifier: rep.estimators.tmva.TMVAClassifier or rep.estimators.tmva.TMVARegressor
    :param rep.estimators.tmva._AdditionalInformation info: additional information
    :param pandas.DataFrame data: train data
    :param labels: array-like - targets
    :param sample_weight: array-like - weights
    """

    ROOT.TMVA.Tools.Instance()

    file_out = ROOT.TFile(os.path.join(info.directory, info.tmva_root), "RECREATE")
    print(classifier.factory_options)
    factory = ROOT.TMVA.Factory(info.tmva_job, file_out, classifier.factory_options)

    for var in data.columns:
        factory.AddVariable(var)

    # Set data
    if info.model_type == 'classification':
        if classifier.method == 'kCuts':
            # signal must the first added tree, because rectangular cut optimization in another way doesn't work
            inds = numpy.argsort(labels)[::-1]
            data = data.ix[inds, :]
            labels = labels[inds]
            sample_weight = sample_weight[inds]
        add_classification_events(factory, numpy.array(data), labels, weights=sample_weight)
        add_classification_events(factory, numpy.array(data), labels, weights=sample_weight, test=True)
    elif info.model_type == 'regression':
        factory.AddTarget('target')
        add_regression_events(factory, numpy.array(data), labels, weights=sample_weight)
        add_regression_events(factory, numpy.array(data), labels, weights=sample_weight, test=True)
    else:
        raise NotImplementedError("Doesn't support type {}".format(info.model_type))

    factory.PrepareTrainingAndTestTree(ROOT.TCut('1'), "")
    # Set method
    parameters = ":".join(
        ["{key}={value}".format(key=key, value=value) for key, value in classifier.method_parameters.items()])
    factory.BookMethod(ROOT.TMVA.Types.__getattribute__(ROOT.TMVA.Types, classifier.method), classifier._method_name,
                       parameters)

    factory.TrainAllMethods()
    file_out.Close()


def main():
    # Python 2 dumps in text mode. Python 3 in binary.
    if six.PY2:
        stdin = sys.stdin
    else:
        stdin = sys.stdin.buffer

    # Reading the configuration from stdin
    classifier = pickle.load(stdin)
    info = pickle.load(stdin)
    data = pickle.load(stdin)
    labels = numpy.array(pickle.load(stdin))
    sample_weight = numpy.array(pickle.load(stdin))
    assert isinstance(classifier, tmva.TMVAClassifier) or isinstance(classifier, tmva.TMVARegressor)
    assert isinstance(info, tmva._AdditionalInformation)
    assert isinstance(data, pandas.DataFrame)
    tmva_process(classifier, info, data, labels, sample_weight)
