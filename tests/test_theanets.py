from __future__ import division, print_function, absolute_import
from ._test_classifier import check_classifier
from ._test_classifier import generate_classification_data
from sklearn.base import clone
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
    # according to reproducibility failures in theanets, it's impossible to compare predictions' quality,
    # so we have to compare what we do compare now
    clf = TheanetsClassifier(trainers=[{'optimize': 'rmsprop'}])
    X, y, sample_weight = generate_classification_data()
    clf.fit(X, y)
    clf.partial_fit(X, y, optimize='rprop')
    cloned_clf = clone(clf)
    assert len(cloned_clf.trainers) == 2, 'wrong amount of trainers'
    assert cloned_clf.trainers[0]['optimize'] == 'rmsprop', 'wrong 1st trainer'
    assert cloned_clf.trainers[1]['optimize'] == 'rprop', 'wrong 2nd trainer'


def test_theanets_simple_stacking():
    base_tnt = TheanetsClassifier()
    check_classifier(SklearnClassifier(clf=BaggingClassifier(base_estimator=base_tnt, n_estimators=3)),
                     supports_weight=False, has_staged_pp=False, has_importances=False)