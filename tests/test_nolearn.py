from __future__ import division, print_function, absolute_import
from rep.test.test_estimators import check_classifier, check_regression
from rep.estimators import NolearnClassifier

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from rep.estimators.sklearn import SklearnClassifier


def test_nolearn_classification():
    cl = NolearnClassifier()
    check_classifier(cl, check_instance=True, has_staged_pp=False, has_importances=False, supports_weight=False)

    cl = NolearnClassifier(layers=[])
    check_classifier(cl, check_instance=True, has_staged_pp=False, has_importances=False, supports_weight=False)


def test_simple_stacking_nolearn():
    # AdaBoostClassifier fails because sample_weight is not supported in nolearn
    base_nl = NolearnClassifier()
    check_classifier(SklearnClassifier(clf=BaggingClassifier(base_estimator=base_nl, n_estimators=3)),
                     has_staged_pp=False, has_importances=False, supports_weight=False)
