from __future__ import division, print_function, absolute_import
import numpy
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

from rep.estimators import SklearnClassifier, SklearnRegressor
from rep.metaml import FoldingRegressor, FoldingClassifier
from rep.test.test_estimators import generate_classification_data, \
    check_classification_model, check_regression
from nose.tools import raises

__author__ = 'Tatiana Likhomanenko, Alex Rogozhnikov'


def check_folding(classifier, check_instance=True, has_staged_pp=True, has_importances=True, use_weights=True):
    X, y, sample_weight = generate_classification_data(distance=0.6)

    if not use_weights:
        sample_weight = None
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


def test_splitting_correctness(n_samples=100, n_features=3):
    """ Check that train and test are different for each fold estimator."""
    X = numpy.random.normal(size=[n_samples, n_features])
    y_ = numpy.random.normal(size=n_samples)
    for y, folder in [
        (y_, FoldingRegressor(SklearnRegressor(KNeighborsRegressor(n_neighbors=1)), n_folds=2)),
        ((y_ > 0) * 1, FoldingClassifier(SklearnClassifier(KNeighborsClassifier(n_neighbors=1)), n_folds=2)),
        ((y_ > 0) * 1, FoldingClassifier(SklearnClassifier(KNeighborsClassifier(n_neighbors=1)), stratified=True)),
    ]:
        folder.set_params(verbose=False)
        folder.fit(X, y)
        try:
            preds = folder.predict_proba(X)[:, 1]
        except:
            preds = folder.predict(X)
        # checking that we split well
        assert mean_squared_error(y, preds) > mean_squared_error(y.mean() + y * 0, preds) * 0.9

        # passing in wrong order
        p = numpy.random.permutation(n_samples)
        preds_shuffled = folder.predict(X[p])[numpy.argsort(p)]

        # Now let's compare this with shuffled kFolding:
        assert mean_squared_error(y, preds) > mean_squared_error(y, preds_shuffled)

        preds_mean = folder.predict(X, vote_function=lambda x: numpy.mean(x, axis=0))
        # Now let's compare this with mean prediction:
        assert mean_squared_error(y, preds) > mean_squared_error(y, preds_mean)


@raises(ValueError)
def test_regressor_fails_with_startified():
    FoldingRegressor(SklearnRegressor(GradientBoostingRegressor(n_estimators=5)),
                     n_folds=2, stratified=True)


def test_stratification(n_samples=100, n_features=2):
    """ensure that splitting is equal among classes. Leaving n_samples = n_folds for class 1."""
    X = numpy.random.normal(size=[n_samples, n_features])
    y = numpy.zeros(n_samples, dtype=int)

    for n_folds in range(2, 10):
        y = numpy.zeros(n_samples, dtype=int)
        y[numpy.random.choice(n_samples, replace=False, size=n_folds)] = 1
        assert sum(y) == n_folds
        folder = FoldingClassifier(SklearnClassifier(GradientBoostingRegressor(n_estimators=5)),
                                   n_folds=n_folds, stratified=True)
        folder.fit(X, y)
        folds = folder._stratified_folds_saved_column.copy()
        for fold in range(n_folds):
            assert y[folds == fold].sum() == 1


def test_folding_regressor_functions():
    """Testing folding functions """
    data, y, sample_weight = generate_classification_data()

    for X in [data, numpy.array(data)]:
        kfolder = FoldingRegressor(SklearnRegressor(GradientBoostingRegressor(n_estimators=5)), n_folds=2)
        kfolder.fit(X, y, sample_weight=sample_weight)
        preds = kfolder.predict(X)
        for p in kfolder.staged_predict(X):
            pass
        assert numpy.allclose(p, preds)

        importances = kfolder.feature_importances_
        other_importances = kfolder.get_feature_importances()


def test_folding_classifier():
    for stratified in [True, False]:
        base_ada = SklearnClassifier(AdaBoostClassifier())
        folding_str = FoldingClassifier(base_ada, n_folds=2, stratified=stratified)
        check_folding(folding_str, True, True, True)

        base_log_reg = SklearnClassifier(LogisticRegression())
        folding_str = FoldingClassifier(base_log_reg, n_folds=4, stratified=stratified)
        check_folding(folding_str, True, False, False, False)


def test_folding_regressor_with_check_model():
    base_clf = SklearnRegressor(GradientBoostingRegressor(n_estimators=4))
    folding_str = FoldingRegressor(base_clf, n_folds=2)
    check_regression(folding_str, True, True, True)
