from __future__ import division, print_function, absolute_import
from ._test_classifier import check_classifier
from sklearn.ensemble import AdaBoostClassifier
from rep.estimators.sklearn import SklearnClassifier
from rep.estimators.theanets import TheanetsClassifier

__author__ = 'Lisa Ignatyeva'


def test_theanets_single_classification():
    check_classifier(TheanetsClassifier(layers=[-1, 10, -1]),
                     supports_weight=False, has_staged_pp=False, has_importances=False)
    check_classifier(TheanetsClassifier(layers=[-1, 10, -1], [{'optimize': 'sgd', 'learning_rate': 0.3}]),
                     supports_weight=False, has_staged_pp=False, has_importances=False)


def test_theanets_multiple_classification():
    check_classifier(TheanetsClassifier(layers=[-1, 10, -1], [{'optimize': 'pretrain'}, {'optimize': 'nag'}]),
                     supports_weight=False, has_staged_pp=False, has_importances=False)


def test_simple_stacking_theanets():
    base_tnt = TheanetsClassifier(layers=[-1, 10, -1])
    check_classifier(SklearnClassifier(clf=AdaBoostClassifier(base_estimator=base_tnt, n_estimators=3)),
                     has_staged_pp=False)



