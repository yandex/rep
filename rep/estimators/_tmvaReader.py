"""
    TMVA reader runs with additional information
"""

from __future__ import division, print_function, absolute_import
import sys
import array

from rootpy.io import root_open

from . import tmva
from six.moves import cPickle as pickle

__author__ = 'Tatiana Likhomanenko'


def tmva_process(info):
    """
    Create TMVA classification factory, train, test and evaluate all methods

    :param rep.estimators.tmva._AdditionalInformationPredict info: additional information

    """
    import ROOT

    reader = ROOT.TMVA.Reader()

    features_pointers = []
    for feature in info.features:
        features_pointers.append(array.array('f', [0.]))
        reader.AddVariable(feature, features_pointers[-1])

    model_type, sigmoid_function = info.model_type
    reader.BookMVA(info.method_name, info.xml_file)

    file_root = root_open(info.filename, mode='update')
    tree = file_root[info.treename]

    for ind, feature in enumerate(info.features):
        tree.SetBranchAddress(feature, features_pointers[ind])

    tree.create_branches({info.method_name: 'F'})
    branch = tree.get_branch(info.method_name)

    signal_efficiency = None
    if model_type == 'classification' and sigmoid_function is not None and 'sig_eff' in sigmoid_function:
        signal_efficiency = float(sigmoid_function.strip().split('=')[1])
        assert 0.0 <= signal_efficiency <= 1., 'signal efficiency must be in [0, 1], not {}'.format(
            signal_efficiency)

    for event in range(tree.GetEntries()):
        tree.GetEntry(event)
        if model_type == 'classification':
            if signal_efficiency is not None:
                prediction = reader.EvaluateMVA(info.method_name, signal_efficiency)
            else:
                prediction = reader.EvaluateMVA(info.method_name)
        else:
            prediction = reader.EvaluateRegression(info.method_name)[0]
        tree.__setattr__(info.method_name, prediction)
        branch.Fill()
    tree.Write()
    file_root.Close()


def main():
    # Reading the configuration from stdin
    info = pickle.load(sys.stdin)
    assert isinstance(info, tmva._AdditionalInformationPredict)
    tmva_process(info)
