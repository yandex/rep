"""
Supplementary script to predict using a TMVA model.
"""

from __future__ import division, print_function, absolute_import
import sys
import array

import pandas
from root_numpy.tmva import evaluate_reader

from . import tmva
import six
from six.moves import cPickle as pickle


__author__ = 'Tatiana Likhomanenko'


def tmva_process(info, data):
    """
    Create a TMVA reader and predict data.

    :param rep.estimators.tmva._AdditionalInformationPredict info: additional information
    :param pandas.DataFrame data: data to predict

    """
    import ROOT

    reader = ROOT.TMVA.Reader()

    for feature in data.columns:
        reader.AddVariable(feature, array.array('f', [0.]))

    model_type, sigmoid_function = info.model_type
    reader.BookMVA(info.method_name, info.xml_file)

    signal_efficiency = None
    if model_type == 'classification' and sigmoid_function is not None and 'sig_eff' in sigmoid_function:
        signal_efficiency = float(sigmoid_function.strip().split('=')[1])
        assert 0.0 <= signal_efficiency <= 1., 'signal efficiency must be in [0, 1], not {}'.format(
            signal_efficiency)

    if signal_efficiency is not None:
        predictions = evaluate_reader(reader, info.method_name, data, aux=signal_efficiency)
    else:
        predictions = evaluate_reader(reader, info.method_name, data)
    return predictions


def main():
    # Python 2 dumps in text mode. Python 3 in binary.
    if six.PY2:
        stdin = sys.stdin
    else:
        stdin = sys.stdin.buffer

    # Reading the configuration from the stdin
    info = pickle.load(stdin)
    data = pickle.load(stdin)
    assert isinstance(info, tmva._AdditionalInformationPredict)
    assert isinstance(data, pandas.DataFrame)
    predictions = tmva_process(info, data)
    with open(info.result_filename, 'wb') as predictions_file:
        pickle.dump(predictions, predictions_file)
