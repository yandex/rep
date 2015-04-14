"""
    TMVA factory runs with classifier and additional information
"""

from __future__ import division, print_function, absolute_import
import sys
import os

from rootpy.io import root_open

import ROOT
from . import tmva
from six.moves import cPickle as pickle


__author__ = 'Tatiana Likhomanenko'


def tmva_process(classifier, info):
    """
    Create TMVA classification factory, train, test and evaluate all methods

    :param rep.estimators.tmva.TMVAClassifier | rep.estimators.tmva.TMVARegressor classifier: classifier to train
    :param rep.estimators.tmva._AdditionalInformation info: additional information

    """

    ROOT.TMVA.Tools.Instance()

    file_out = ROOT.TFile(os.path.join(info.directory, info.tmva_root), "RECREATE")
    factory = ROOT.TMVA.Factory(info.tmva_job, file_out, classifier.factory_options)

    for var in info.features:
        factory.AddVariable(var)

    # Set data
    file_root = root_open(info.filename, mode='update')
    if info.model_type == 'classification':
        # signal must the first added tree, because rectangular cut optimization in another wat doesn't work
        factory.AddTree(file_root[info.treename], 'Signal', 1.,
                        ROOT.TCut("{column} == {label}".format(column=info.target_column, label=1)),
                        'Training')
        factory.AddTree(file_root[info.treename], 'Signal', 1.,
                        ROOT.TCut("{column} == {label}".format(column=info.target_column, label=1)),
                        'Testing')
        factory.AddTree(file_root[info.treename], 'Background', 1.,
                        ROOT.TCut("{column} == {label}".format(column=info.target_column, label=0)),
                        'Training')
        factory.AddTree(file_root[info.treename], 'Background', 1.,
                        ROOT.TCut("{column} == {label}".format(column=info.target_column, label=0)),
                        'Testing')
        factory.SetWeightExpression(info.weight_column)
    elif info.model_type == 'regression':
        factory.AddTarget(info.target_column)
        factory.AddTree(file_root[info.treename], 'Regression', 1., ROOT.TCut(""), "Training")
        factory.AddTree(file_root[info.treename], 'Regression', 1., ROOT.TCut(""), 'Testing')
        factory.SetWeightExpression(info.weight_column, "Regression")
    else:
        raise NotImplementedError("Doesn't support type {}".format(info.model_type))

    # Set method
    parameters = ":".join(
        ["{key}={value}".format(key=key, value=value) for key, value in classifier.method_parameters.items()])
    factory.BookMethod(ROOT.TMVA.Types.__getattribute__(ROOT.TMVA.Types, classifier.method), classifier._method_name,
                       parameters)

    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()

    file_out.Close()
    file_root.Close()


def main():
    # Reading the configuration from stdin
    classifier = pickle.load(sys.stdin)
    info = pickle.load(sys.stdin)
    assert isinstance(classifier, tmva.TMVAClassifier) or isinstance(classifier, tmva.TMVARegressor)
    assert isinstance(info, tmva._AdditionalInformation)
    tmva_process(classifier, info)
