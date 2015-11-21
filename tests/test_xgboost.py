from __future__ import division, print_function, absolute_import

import numpy
from rep.estimators import XGBoostClassifier, XGBoostRegressor
from rep.test.test_estimators import check_classifier, check_regression, generate_classification_data

__author__ = 'Alex Rogozhnikov'


def test_basic_xgboost():
    X, y, w = generate_classification_data(n_classes=2)
    clf = XGBoostClassifier(n_estimators=10).fit(X, y)
    clf.predict(X)
    clf.predict_proba(X)
    # testing that returned features in importances are correct and in the same order
    assert numpy.all(clf.features == clf.get_feature_importances().index)


def test_xgboost():
    check_classifier(XGBoostClassifier(n_estimators=20), n_classes=2)
    check_classifier(XGBoostClassifier(n_estimators=20), n_classes=4)
    check_regression(XGBoostRegressor(n_estimators=20))

