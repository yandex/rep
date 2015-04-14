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