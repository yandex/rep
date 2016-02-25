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


def test_xgboost_works_with_different_dtypes():
    dtypes = ['float32', 'float64', 'int32', 'int64', 'uint32']
    for dtype in dtypes:
        X, y, weights = generate_classification_data(n_classes=2, distance=5)
        clf = XGBoostClassifier(n_estimators=10)
        clf.fit(X.astype(dtype=dtype), y.astype(dtype=dtype), sample_weight=weights.astype(dtype))
        probabilities = clf.predict_proba(X.astype(dtype))

    # testing single pandas.DataFrame with different dtypes
    X, y, weights = generate_classification_data(n_classes=2, distance=5)
    import pandas
    X = pandas.DataFrame()
    for dtype in dtypes:
        X[dtype] = numpy.random.normal(0, 10, size=len(y)).astype(dtype)
    clf = XGBoostClassifier(n_estimators=10)
    clf.fit(X, y, sample_weight=weights)
    probabilities = clf.predict_proba(X)

