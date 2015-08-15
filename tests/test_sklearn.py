from __future__ import division, print_function, absolute_import

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, AdaBoostRegressor, \
    GradientBoostingRegressor

from rep.test.test_estimators import check_classifier, check_regression
from rep.estimators import SklearnClassifier, SklearnRegressor


__author__ = 'Alex Rogozhnikov'


def test_sklearn_classification():
    # supports weights
    check_classifier(SklearnClassifier(clf=AdaBoostClassifier(n_estimators=10)))
    check_classifier(SklearnClassifier(clf=AdaBoostClassifier(n_estimators=10)), n_classes=3)
    # doesn't support weights
    check_classifier(SklearnClassifier(clf=GradientBoostingClassifier(n_estimators=10)), supports_weight=False)


def test_sklearn_regression():
    # supports weights
    check_regression(SklearnRegressor(clf=AdaBoostRegressor(n_estimators=50)))
    # doesn't support weights
    check_regression(SklearnRegressor(clf=GradientBoostingRegressor(n_estimators=50)),
                     supports_weight=False)
