from __future__ import division, print_function, absolute_import
from ._test_classifier import check_classifier, check_regression
from rep.estimators import PyBrainClassifier
from rep.estimators.pybrain import PyBrainRegressor
from sklearn.ensemble import BaggingClassifier
from rep.estimators.sklearn import SklearnClassifier
from sklearn.preprocessing import StandardScaler


__author__ = 'Artem Zhirokhov'


def test_pybrain_classification():
    check_classifier(PyBrainClassifier(), has_staged_pp=False, has_importances=False, supports_weight=False)

def test_pybrain_rprop():
    check_classifier(PyBrainClassifier(use_rprop=True), has_staged_pp=False, has_importances=False, supports_weight=False)

def test_pybrain_multiclassification():
    check_classifier(PyBrainClassifier(), has_staged_pp=False, has_importances=False, supports_weight=False, n_classes=4)

def test_pybrain_regression():
    check_regression(PyBrainRegressor(), has_staged_predictions=False, has_importances=False, supports_weight=False)

def test_simple_stacking_pybrain():
    base_pybrain = PyBrainClassifier()
    check_classifier(SklearnClassifier(clf=BaggingClassifier(base_estimator=base_pybrain, n_estimators=3)),
                     has_staged_pp=False, has_importances=False, supports_weight=False)