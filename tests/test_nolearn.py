from __future__ import division, print_function, absolute_import
from ._test_classifier import check_classifier, check_regression
from rep.estimators import NolearnClassifier

from sklearn.ensemble import AdaBoostClassifier
from rep.estimators.sklearn import SklearnClassifier


def test_nolearn_classification():
    cl = NolearnClassifier()
    check_classifier(cl, check_instance=True, has_staged_pp=False, has_importances=False,
                     supports_weight=False)

    cl = NolearnClassifier(layer_sizes=[-1, 10, -1])
    check_classifier(cl, check_instance=True, has_staged_pp=False, has_importances=False, supports_weight=False)

def test_simple_stacking_nolearn():
    base_nl = NolearnClassifier()
    check_classifier(SklearnClassifier(clf=AdaBoostClassifier(base_estimator=base_nl, n_estimators=3)),
                     has_staged_pp=False, has_importances=False, supports_weight=False)
