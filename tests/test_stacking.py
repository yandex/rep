from __future__ import division, print_function, absolute_import

import numpy
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier

from rep.test.test_estimators import check_classifier, generate_classification_data
from rep.estimators import XGBoostClassifier, TMVAClassifier
from rep.metaml import FoldingClassifier
from rep.estimators.sklearn import SklearnClassifier


__author__ = 'Alex Rogozhnikov'


def test_feature_splitter():
    # testing splitter
    from rep.metaml import FeatureSplitter

    X, y, sample_weight = generate_classification_data(n_classes=3)
    split_column = X.columns[0]
    splitters = numpy.random.randint(0, 3, size=len(X))
    X[split_column] = splitters
    X.ix[splitters == 1, :] += 4
    X.ix[splitters == 2, :] -= 4
    fs = FeatureSplitter(base_estimator=XGBoostClassifier(n_estimators=10, max_depth=3),
                         split_feature=split_column, train_features=list(X.columns[1:]))
    fs.fit(X, y, sample_weight=sample_weight)
    assert fs.score(X, y) > 0.9
    p_final = fs.predict_proba(X)
    for p in fs.staged_predict_proba(X):
        pass
    assert numpy.allclose(p_final, p), 'end of iterations differs from expected'


def test_simple_stacking_xgboost():
    base_xgboost = XGBoostClassifier()
    classifier = SklearnClassifier(clf=AdaBoostClassifier(base_estimator=base_xgboost, n_estimators=3))
    check_classifier(classifier,
                     has_staged_pp=False)


def test_simple_stacking_sklearn():
    base_sk = AdaBoostClassifier(n_estimators=30)
    check_classifier(SklearnClassifier(clf=AdaBoostClassifier(base_estimator=base_sk, n_estimators=3)))


def test_simple_stacking_tmva():
    base_tmva = TMVAClassifier()
    check_classifier(SklearnClassifier(clf=BaggingClassifier(base_estimator=base_tmva, n_estimators=3, random_state=13)),
                     has_staged_pp=False, has_importances=False)


def test_complex_stacking_sk():
    # Ada over kFold over Ada over Trees
    base_kfold = FoldingClassifier(base_estimator=AdaBoostClassifier())
    check_classifier(SklearnClassifier(clf=AdaBoostClassifier(base_estimator=base_kfold, n_estimators=3)),
                     has_staged_pp=False, has_importances=False)


def test_complex_stacking_tmva():
    # Ada over kFold over TMVA
    base_kfold = FoldingClassifier(base_estimator=TMVAClassifier(), random_state=13)
    check_classifier(SklearnClassifier(clf=AdaBoostClassifier(base_estimator=base_kfold, n_estimators=3)),
                     has_staged_pp=False, has_importances=False)


def test_complex_stacking_xgboost():
    # Ada over kFold over xgboost
    base_kfold = FoldingClassifier(base_estimator=XGBoostClassifier())
    check_classifier(SklearnClassifier(clf=AdaBoostClassifier(base_estimator=base_kfold, n_estimators=3)),
                     has_staged_pp=False, has_importances=False)

