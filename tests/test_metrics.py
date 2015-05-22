from __future__ import division, print_function, absolute_import

import numpy
from rep.report import metrics

__author__ = 'Alex Rogozhnikov'


def test_optimal_metrics(size=1000):
    prediction = numpy.random.random(size=size)
    random_labels = numpy.random.choice(2, size=size)

    def ams_like(s, b):
        return s / (b + 1. / 100. / size)

    # setting 'the best event' to be signal
    random_labels[numpy.argmax(prediction)] = 1
    optimal_ams = metrics.OptimalMetric(ams_like)
    proba = numpy.ndarray((len(prediction), 2))
    proba[:, 0] = 1 - prediction
    proba[:, 1] = prediction
    score = optimal_ams(random_labels, proba)

    assert score >= 100


def test_logloss(size=1000):
    from sklearn.metrics import log_loss
    prediction = numpy.random.random(size=size)
    random_labels = numpy.random.choice(2, size=size)

    proba = numpy.ndarray((len(prediction), 2))
    proba[:, 0] = 1 - prediction
    proba[:, 1] = prediction

    loss = metrics.LogLoss().fit(proba, y=random_labels, sample_weight=None)
    value = log_loss(random_labels, prediction)
    value2 = loss(random_labels, proba)

    print(value, value2)

    assert numpy.allclose(value, value2)


def test_roc_auc(size=1000):
    from sklearn.metrics import roc_auc_score
    prediction = numpy.random.random(size=size)
    random_labels = numpy.random.choice(2, size=size)

    proba = numpy.ndarray((len(prediction), 2))
    proba[:, 0] = 1 - prediction
    proba[:, 1] = prediction

    roc_auc_metric = metrics.RocAuc().fit(proba, y=random_labels, sample_weight=None)
    value = roc_auc_score(random_labels, prediction)
    value2 = roc_auc_metric(random_labels, proba)

    print(value, value2)

    assert numpy.allclose(value, value2)


def fpr_tpr(size, prediction):
    from sklearn.metrics import roc_curve
    random_labels = numpy.random.choice(2, size=size)

    proba = numpy.ndarray((len(prediction), 2))
    proba[:, 0] = 1 - prediction
    proba[:, 1] = prediction
    sample_weight = numpy.random.random(size=size)

    threshold = 0.75
    loss_fpr, loss_tpr = metrics.FPRatTPR(threshold), metrics.TPRatFPR(threshold)
    fprs, tprs, _ = roc_curve(random_labels, prediction, sample_weight=sample_weight)
    value_fpr = loss_fpr(random_labels, proba, sample_weight=sample_weight)
    value_tpr = loss_tpr(random_labels, proba, sample_weight=sample_weight)
    value_fpr2 = fprs[numpy.searchsorted(tprs, threshold)]
    value_tpr2 = tprs[numpy.searchsorted(fprs, threshold) - 1]
    print(value_fpr, value_fpr2)
    print(value_tpr, value_tpr2)

    assert numpy.allclose(value_fpr, value_fpr2, atol=1e-3)
    assert numpy.allclose(value_tpr, value_tpr2, atol=1e-3)


def test_fpr_tpr(size=10000):
    prediction = numpy.random.permutation(size)
    fpr_tpr(size, prediction)
    prediction = numpy.ones(size)
    fpr_tpr(size, prediction)
