from __future__ import division, print_function, absolute_import

import numpy
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score

from rep.estimators import SklearnClassifier, SklearnRegressor
from rep.metaml import FoldingRegressor, FoldingClassifier
from rep.test.test_estimators import generate_classification_data, check_classification_model


__author__ = 'antares'


def check_folding(classifier, check_instance=True, has_staged_pp=True, has_importances=True):
    X, y, sample_weight = generate_classification_data(distance=0.6)

    assert classifier == classifier.fit(X, y, sample_weight=sample_weight)
    assert list(classifier.features) == list(X.columns)

    check_classification_model(classifier, X, y, check_instance=check_instance, has_staged_pp=has_staged_pp,
                               has_importances=has_importances)

    def mean_vote(x):
        return numpy.mean(x, axis=0)

    labels = classifier.predict(X, mean_vote)
    proba = classifier.predict_proba(X, mean_vote)
    assert numpy.all(proba == classifier.predict_proba(X, mean_vote))

    score = accuracy_score(y, labels)
    print(score)
    assert score > 0.7
    assert numpy.allclose(proba.sum(axis=1), 1), 'probabilities do not sum to 1'
    assert numpy.all(proba >= 0.), 'negative probabilities'

    auc_score = roc_auc_score(y, proba[:, 1])
    print(auc_score)
    assert auc_score > 0.8
    if has_staged_pp:
        for p in classifier.staged_predict_proba(X, mean_vote):
            assert p.shape == (len(X), 2)
        # checking that last iteration coincides with previous
        assert numpy.all(p == proba)


def test_folding_regressor(n_samples=100, n_features=3):
    """
    checking mostly different things with quality of predictions and not using train data during predictions.
    """
    from sklearn.metrics import mean_squared_error

    X = numpy.random.normal(size=[n_samples, n_features])
    y = numpy.random.normal(size=n_samples)
    kfolder = FoldingRegressor(SklearnRegressor(GradientBoostingRegressor()), n_folds=2)
    kfolder.fit(X, y)
    preds = kfolder.predict(X)
    # checking that we fitted fine
    assert mean_squared_error(y, preds) > mean_squared_error(y * 0., preds) * 0.5

    # shuffled preds
    p = numpy.random.permutation(n_samples)
    preds2 = kfolder.predict(X[p])[numpy.argsort(p)]

    # Now let's compare this with shuffled kFolding:
    assert mean_squared_error(y, preds) > mean_squared_error(y, preds2) * 0.5

    preds_mean = kfolder.predict(X, vote_function=lambda x: numpy.mean(x, axis=0))
    # Now let's compare this with mean prediction:
    assert mean_squared_error(y, preds) > mean_squared_error(y, preds_mean)


def test_folding_classifier():
    base_ada = SklearnClassifier(AdaBoostClassifier())
    folding_str = FoldingClassifier(base_ada, n_folds=2)
    check_folding(folding_str, True, True, True)

    base_svm = SklearnClassifier(SVC())
    folding_str = FoldingClassifier(base_svm, n_folds=4)
    check_folding(folding_str, True, False, False)