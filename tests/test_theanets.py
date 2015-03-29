from __future__ import division, print_function, absolute_import
from ._test_classifier import check_classifier
from ._test_classifier import generate_classification_data
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import BaggingClassifier
from rep.estimators.sklearn import SklearnClassifier
from rep.estimators.theanets import TheanetsClassifier

__author__ = 'Lisa Ignatyeva'


def test_theanets_single_classification():
    check_classifier(TheanetsClassifier(),
                     supports_weight=False, has_staged_pp=False, has_importances=False)
    check_classifier(TheanetsClassifier(layers=[20], trainers=[{'optimize': 'sgd', 'learning_rate': 0.3}]),
                     supports_weight=False, has_staged_pp=False, has_importances=False)


def test_theanets_multiple_classification():
    check_classifier(TheanetsClassifier(trainers=[{'optimize': 'adadelta'}, {'optimize': 'nag'}]),
                     supports_weight=False, has_staged_pp=False, has_importances=False)


def test_theanets_partial_fit():
    clf_complete = TheanetsClassifier(trainers=[{'optimize': 'rmsprop'}, {'optimize': 'rprop'}])
    clf_partial = TheanetsClassifier(trainers=[{'optimize': 'rmsprop'}])
    X, y, sample_weight = generate_classification_data()
    clf_complete.fit(X, y)
    clf_partial.fit(X, y)
    clf_partial.partial_fit(X, y, optimize='rprop')

    assert clf_complete.trainers == clf_partial.trainers, 'trainers not saved in partial fit'

    auc_complete = roc_auc_score(y, clf_complete.predict_proba(X)[:, 1])
    auc_partial = roc_auc_score(y, clf_partial.predict_proba(X)[:, 1])

    assert auc_complete == auc_partial, 'same networks return different results'


def test_theanets_reproducibility():
    clf = TheanetsClassifier()
    X, y, sample_weight = generate_classification_data()
    clf.fit(X, y)
    auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    for i in range(2):
        clf.fit(X, y)
        curr_auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
        assert auc == curr_auc, 'running a network twice produces different results'

    cloned_clf = clone(clf)
    cloned_clf.fit(X, y)
    cloned_auc = roc_auc_score(y, cloned_clf.predict_proba(X)[:, 1])
    assert cloned_auc == auc, 'cloned network produces different result'


def test_theanets_simple_stacking():
    base_tnt = TheanetsClassifier()
    check_classifier(SklearnClassifier(clf=BaggingClassifier(base_estimator=base_tnt, n_estimators=3)),
                     supports_weight=False, has_staged_pp=False, has_importances=False)