from __future__ import division, print_function, absolute_import
from rep.test.test_estimators import check_classifier, generate_classification_data
from rep.estimators import NolearnClassifier

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from rep.estimators.sklearn import SklearnClassifier
from sklearn.base import clone


def test_nolearn_reproducibility():
    X, y, sample_weight = generate_classification_data()
    cl = NolearnClassifier()
    y_predicted_1 = cl.fit(X, y).predict(X)
    y_predicted_2 = cl.fit(X, y).predict(X)
    assert (y_predicted_1 == y_predicted_2).all(), 'fitting the classifier twice gives different predictions'
    y_predicted_3 = clone(cl).fit(X, y).predict(X)
    assert (y_predicted_1 == y_predicted_3).all(), 'cloned classifier gives different prediction'


def test_nolearn_classification():
    cl = NolearnClassifier()
    check_classifier(cl, check_instance=True, has_staged_pp=False, has_importances=False, supports_weight=False)

    cl = NolearnClassifier(layers=[])
    check_classifier(cl, check_instance=True, has_staged_pp=False, has_importances=False, supports_weight=False)

    cl = NolearnClassifier(layers=[5, 5])
    check_classifier(cl, check_instance=True, has_staged_pp=False, has_importances=False, supports_weight=False)


def test_nolearn_multiple_classification():
    cl = NolearnClassifier()
    check_classifier(cl, check_instance=True, has_staged_pp=False, has_importances=False, supports_weight=False,
                     n_classes=4)


def test_simple_stacking_nolearn():
    # AdaBoostClassifier fails because sample_weight is not supported in nolearn
    base_nl = NolearnClassifier()
    check_classifier(SklearnClassifier(clf=BaggingClassifier(base_estimator=base_nl, n_estimators=3)),
                     has_staged_pp=False, has_importances=False, supports_weight=False)
