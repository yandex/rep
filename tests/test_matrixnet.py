from __future__ import division, print_function, absolute_import

from sklearn.ensemble import AdaBoostClassifier
from rep.estimators import SklearnClassifier
from rep.metaml import FoldingClassifier
from rep.test.test_estimators import check_regression, generate_classification_data, check_classifier, \
    check_classification_reproducibility

from rep.estimators import MatrixNetClassifier, MatrixNetRegressor


__author__ = 'Tatiana Likhomanenko, Alex Rogozhnikov'


def test_mn_classification():
    clf = MatrixNetClassifier(iterations=20, auto_stop=1e-3)
    check_classifier(clf, n_classes=2)
    assert {'effect', 'information', 'efficiency'} == set(clf.get_feature_importances().columns)


def test_mn_regression():
    clf = MatrixNetRegressor()
    check_regression(clf)
    assert {'effect', 'information', 'efficiency'} == set(clf.get_feature_importances().columns)


def test_mn_baseline():
    clf = MatrixNetClassifier(iterations=20, baseline_feature='column0')
    check_classifier(clf, n_classes=2)


def test_simple_stacking_mn():
    base_mn = MatrixNetClassifier(iterations=10)
    check_classifier(SklearnClassifier(clf=AdaBoostClassifier(base_estimator=base_mn, n_estimators=2)),
                     has_staged_pp=True)


def test_mn_reproducibility():
    clf = MatrixNetClassifier(iterations=10)
    X, y, _ = generate_classification_data()
    check_classification_reproducibility(clf, X, y)


def test_complex_stacking_mn():
    # Ada over kFold over MatrixNet
    base_kfold = FoldingClassifier(base_estimator=MatrixNetClassifier(iterations=30))
    check_classifier(SklearnClassifier(clf=AdaBoostClassifier(base_estimator=base_kfold, n_estimators=3)),
                     has_staged_pp=False, has_importances=False)

