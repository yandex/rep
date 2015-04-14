from __future__ import division, print_function, absolute_import

from rep.estimators import XGBoostClassifier, XGBoostRegressor
from rep.test.test_estimators import check_classifier, check_regression, generate_classification_data

__author__ = 'Alex Rogozhnikov'


def test_xgboost():
    check_classifier(XGBoostClassifier(), n_classes=2)
    check_classifier(XGBoostClassifier(), n_classes=4)
    check_regression(XGBoostRegressor())


def test_feature_importances():
    clf = XGBoostClassifier()
    X, y, sample_weight = generate_classification_data()
    clf.fit(X, y, sample_weight=sample_weight)
    # checking feature importance (three ways)

    res_default = clf.xgboost_classifier.get_fscore()
    res2 = clf._get_fscore()
    res3 = clf.feature_importances_

    assert res_default == res2, res_default
    for i, val in enumerate(res3):
        if val > 0.0:
            assert val == res_default['f' + str(i)]
