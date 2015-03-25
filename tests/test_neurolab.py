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
    check_classifier(NeurolabClassifier(size=[25]*201, transf=[f]*202,
                                        epochs=10, trainf=nl.train.train_rprop, show=1),
                     supports_weight=False, has_staged_pp=False, has_importances=False)
    check_classifier(NeurolabClassifier(size=[150, 150],
                                        transf=[nl.trans.Competitive(), nl.trans.Competitive(), nl.trans.Competitive()],
                                        epochs=3, trainf=nl.train.train_rprop, initf=nl.init.initnw, show=0),

                      supports_weight=False, has_staged_pp=False, has_importances=False)


def test_neurolab_multiple_classification():
    check_classifier(NeurolabClassifier(size=[150], transf=[nl.trans.LogSig(), nl.trans.LogSig()],
                                        epochs=10, trainf=nl.train.train_rprop, show=1),
                     supports_weight=False, has_staged_pp=False, has_importances=False)


def test_neurolab_stacking_theanets():
    base_nlab = NeurolabClassifier(size=[150], transf=[nl.trans.LogSig(), nl.trans.LogSig()],
                                   epochs=10, trainf=nl.train.train_rprop, show=0)
    check_classifier(SklearnClassifier(clf=AdaBoostClassifier(base_estimator=base_nlab, n_estimators=3)),
                     has_staged_pp=False)