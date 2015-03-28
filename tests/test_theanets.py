from __future__ import division, print_function, absolute_import
from ._test_classifier import check_classifier
from sklearn.ensemble import BaggingClassifier
from rep.estimators.sklearn import SklearnClassifier
from rep.estimators.theanets import TheanetsClassifier

__author__ = 'Lisa Ignatyeva'


def test_theanets_single_classification():
    check_classifier(TheanetsClassifier(),
                     supports_weight=False, has_staged_pp=False, has_importances=False)
    check_classifier(TheanetsClassifier(layers=[20], trainers=[{'optimize': 'sgd', 'learning_rate': 0.3}]),
                     supports_weight=False, has_staged_pp=False, has_importances=False)


def test_theanets_multiple_classification():
    check_classifier(TheanetsClassifier(trainers=[{'optimize': 'pretrain'}, {'optimize': 'nag'}]),
                     supports_weight=False, has_staged_pp=False, has_importances=False)


def test_simple_stacking_theanets():
    base_tnt = TheanetsClassifier()
    check_classifier(SklearnClassifier(clf=BaggingClassifier(base_estimator=base_tnt, n_estimators=3)),
                     supports_weight=False, has_staged_pp=False, has_importances=False)

