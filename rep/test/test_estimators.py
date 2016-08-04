from __future__ import division, print_function, absolute_import

from copy import deepcopy

from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import numpy
import pandas
from scipy.special import expit

from six.moves import cPickle
from ..estimators import Classifier, Regressor
from ..report.metrics import OptimalMetric

__author__ = 'Tatiana Likhomanenko, Alex Rogozhnikov'

"""
Abstract code to test any classifier or regressor
"""


# TODO test of features parameters

def generate_classification_sample(n_samples, n_features, distance=1.5, n_classes=2):
    """Generates some test distribution,
    distributions are gaussian with centers at (x, x, x, ...  x), where x = class_id * distance
    """
    from sklearn.datasets import make_blobs

    centers = numpy.zeros((n_classes, n_features))
    centers += numpy.arange(n_classes)[:, numpy.newaxis] * distance

    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers)
    columns = ["column" + str(x) for x in range(n_features)]
    X = pandas.DataFrame(X, columns=columns)
    return X, y


def generate_regression_sample(n_samples, n_features):
    """
    Generates dataset for regression,  fratures are drawn from multivariate gaussian,
    target is logistic function of features' sum + small noise
    """
    X = numpy.random.normal(size=[n_samples, n_features])
    columns = ["column" + str(x) for x in range(n_features)]
    X = pandas.DataFrame(X, columns=columns)

    y = expit(numpy.sum(X, axis=1)) + numpy.random.normal(size=n_samples) * 0.05
    return X, y


def generate_classification_data(n_classes=2, distance=1.5):
    """ Generates random number of samples and features. """
    n_samples = 1000 + numpy.random.poisson(1000)
    n_features = numpy.random.randint(10, 16)
    sample_weight = numpy.ones(n_samples, dtype=float)
    X, y = generate_classification_sample(n_features=n_features, n_samples=n_samples, n_classes=n_classes,
                                          distance=distance)
    return X, y, sample_weight


def generate_regression_data(n_targets=1):
    """ Generates random number of samples and features."""
    n_samples = 1000 + numpy.random.poisson(1000)
    n_features = numpy.random.randint(10, 16)
    sample_weight = numpy.ones(n_samples, dtype=float)
    X, y = generate_regression_sample(n_features=n_features, n_samples=n_samples)
    if n_targets > 1:
        y = numpy.vstack([y * numpy.random.random() for _ in range(n_targets)]).T
    assert len(X) == len(y)
    return X, y, sample_weight


def check_picklability_and_predictions(estimator):
    # testing picklability
    dump_string = cPickle.dumps(estimator)
    loaded_estimator = cPickle.loads(dump_string)
    assert type(estimator) == type(loaded_estimator)
    # testing clone-ability
    classifier_clone = clone(estimator)
    assert type(estimator) == type(classifier_clone)
    assert set(estimator.get_params().keys()) == set(classifier_clone.get_params().keys()), \
        'something strange was loaded'
    # testing get_params, set_params
    params = estimator.get_params(deep=False)
    params = estimator.get_params(deep=True)
    classifier_clone.set_params(**params)
    params = classifier_clone.get_params()
    return loaded_estimator


def check_classification_model(classifier, X, y, check_instance=True, has_staged_pp=True, has_importances=True):
    n_classes = len(numpy.unique(y))
    if check_instance:
        assert isinstance(classifier, Classifier)

    labels = classifier.predict(X)
    proba = classifier.predict_proba(X)
    print('PROBABILITIES:', proba)

    score = accuracy_score(y, labels)
    print('ROC AUC:', score)
    assert score > 0.7

    assert numpy.allclose(proba.sum(axis=1), 1), 'probabilities do not sum to 1'
    assert numpy.all(proba >= 0.), 'negative probabilities'

    if n_classes == 2:
        # only for binary classification
        auc_score = roc_auc_score(y == numpy.unique(y)[1], proba[:, 1])
        print(auc_score)
        assert auc_score > 0.8

    if has_staged_pp:
        for p in classifier.staged_predict_proba(X):
            assert p.shape == (len(X), n_classes)
            # checking that last iteration coincides with previous
        assert numpy.all(p == proba), "staged_pp and pp predictions are different"

    if has_importances:
        importances = classifier.feature_importances_
        assert numpy.array(importances).shape == (len(classifier.features),)

    loaded_classifier = check_picklability_and_predictions(classifier)
    assert numpy.all(classifier.predict_proba(X) == loaded_classifier.predict_proba(X)), 'something strange was loaded'


def check_regression_model(regressor, X, y, check_instance=True, has_stages=True, has_importances=True):
    if check_instance:
        assert isinstance(regressor, Regressor)

    predictions = regressor.predict(X)
    score = mean_squared_error(y, predictions)
    std = numpy.std(y)
    assert score < std * 0.5, 'Too big error: ' + str(score)

    if has_stages:
        for p in regressor.staged_predict(X):
            assert p.shape == (len(X),)
        # checking that last iteration coincides with previous
        assert numpy.all(p == predictions)

    if has_importances:
        importances = regressor.feature_importances_
        assert numpy.array(importances).shape == (len(regressor.features),)

    loaded_regressor = check_picklability_and_predictions(regressor)
    assert numpy.all(regressor.predict(X) == loaded_regressor.predict(X)), 'something strange was loaded'


def fit_on_data(estimator, X, y, sample_weight, supports_weight):
    if supports_weight:
        learned = estimator.fit(X, y, sample_weight=sample_weight)
    else:
        learned = estimator.fit(X, y)
    # checking that fit returns the classifier
    assert learned == estimator, "fitting doesn't return initial classifier"

    return estimator


def check_classifier(classifier, check_instance=True, has_staged_pp=True, has_importances=True, supports_weight=True,
                     n_classes=2):
    X, y, sample_weight = generate_classification_data(n_classes=n_classes)
    check_deepcopy(classifier)
    fit_on_data(classifier, X, y, sample_weight, supports_weight=supports_weight)
    assert list(classifier.features) == list(X.columns)

    check_classification_model(classifier, X, y, check_instance=check_instance, has_staged_pp=has_staged_pp,
                               has_importances=has_importances)


def check_regression(regressor, check_instance=True, has_staged_predictions=True, has_importances=True,
                     supports_weight=True, n_targets=1):
    X, y, sample_weight = generate_regression_data(n_targets=n_targets)
    check_deepcopy(regressor)
    fit_on_data(regressor, X, y, sample_weight, supports_weight=supports_weight)
    assert list(regressor.features) == list(X.columns)

    check_regression_model(regressor, X, y, check_instance=check_instance, has_stages=has_staged_predictions,
                           has_importances=has_importances)


def check_params(estimator_type, n_attempts=4, **params):
    """
    Checking that init, get, set are working normally
    :param estimator_type: i.e. sklearn.ensemble.AdaBoostRegressor
    :param n_attempts: how many times to check
    :param params: parameters that are acceptable for estimator
    """
    import numpy

    for _ in range(n_attempts):
        subparams = {k: v for k, v in params.items() if numpy.random.random() > 0.5}
        classifier = estimator_type(**subparams)
        for clf in [classifier, clone(classifier), deepcopy(classifier)]:
            saved_params = clf.get_params()
            for name, value in subparams.items():
                assert saved_params[name] == value, \
                    'Problem with init/get_params {} {} {}'.format(name, value, saved_params[name])


def check_classification_reproducibility(classifier, X, y):
    """
    Check if given estimator after refitting / cloning gives same parameters.
    """
    classifier.fit(X, y)
    auc = roc_auc_score(y, classifier.predict_proba(X)[:, 1])

    cloned_clf = clone(classifier)
    cloned_clf.fit(X, y)
    cloned_auc = roc_auc_score(y, cloned_clf.predict_proba(X)[:, 1])
    assert auc == cloned_auc, 'cloned network produces different result, {} {}'.format(auc, cloned_auc)

    classifier.fit(X, y)
    refitted_auc = roc_auc_score(y, classifier.predict_proba(X)[:, 1])
    assert auc == refitted_auc, 'running a network twice produces different results, {} {}'.format(auc, refitted_auc)


def check_deepcopy(classifier):
    """
    Checks that simple deepcopy works (it uses the mechanism as pickle/unpickle)
    """
    classifier_copy = deepcopy(classifier)
    assert type(classifier) == type(classifier_copy)
    assert set(classifier.get_params().keys()) == set(classifier_copy.get_params().keys())
