from __future__ import division, print_function, absolute_import

import numpy
from rep.estimators import XGBoostClassifier, XGBoostRegressor
from rep.test.test_estimators import check_classifier, check_regression, generate_classification_data
from sklearn.utils import check_random_state

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


def test_xgboost_feature_importance():
    X, y, weights = generate_classification_data(n_classes=2, distance=5)
    clf = XGBoostClassifier(n_estimators=1, max_depth=1)
    clf.fit(X, y)
    importances = clf.get_feature_importances()
    original_features = set(X.columns)
    importances_features = set(importances.index)
    print(original_features, importances_features)
    assert original_features == importances_features, 'feature_importances_ return something wrong'

    assert len(original_features) == len(clf.feature_importances_)


def test_xgboost_random_states():
    X, y, weights = generate_classification_data(n_classes=2, distance=5)
    for random_state in [145, None, check_random_state(None), check_random_state(145)]:
        clf1 = XGBoostClassifier(n_estimators=5, max_depth=1, subsample=0.1, random_state=random_state)
        clf1.fit(X, y)
        clf2 = XGBoostClassifier(n_estimators=5, max_depth=1, subsample=0.1, random_state=random_state)
        clf2.fit(X, y)
        if isinstance(random_state, numpy.random.RandomState):
            assert not numpy.allclose(clf1.predict_proba(X), clf2.predict_proba(X)), 'seed: {}'.format(random_state)
        else:
            assert numpy.allclose(clf1.predict_proba(X), clf2.predict_proba(X)), 'seed: {}'.format(random_state)
