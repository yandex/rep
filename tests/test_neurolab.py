from __future__ import division, print_function, absolute_import
from ._test_classifier import check_classifier
from sklearn.ensemble import AdaBoostClassifier
from rep.estimators.sklearn import SklearnClassifier
from rep.estimators.neurolab import NeurolabClassifier
import neurolab as nl

__author__ = 'Sterzhanov Vladislav'


def test_neurolab_single_classification():
    f = nl.trans.LogSig()
    f2 = nl.trans.SoftMax()
    check_classifier(NeurolabClassifier(show=0, layers=[], epochs=400, trainf=nl.train.train_rprop),
                     supports_weight=False, has_staged_pp=False, has_importances=False)


#def test_neurolab_multiple_classification():
#    check_classifier(NeurolabClassifier(show=0, layers=[], epochs=400, trainf=nl.train.train_rprop),
#                     supports_weight=False, has_staged_pp=False, has_importances=False)


#def test_neurolab_stacking():
#    base_nlab = NeurolabClassifier(show=0, layers=[], epochs=400, trainf=nl.train.train_rprop)
#    check_classifier(SklearnClassifier(clf=AdaBoostClassifier(base_estimator=base_nlab, n_estimators=3)),
#                     has_staged_pp=False)