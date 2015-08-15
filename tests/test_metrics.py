from __future__ import division, print_function, absolute_import

import numpy
from rep.report import metrics
from rep.report.metrics import OptimalAMS, OptimalSignificance, significance, ams

__author__ = 'Alex Rogozhnikov'


def test_optimal_metrics_ndim(size=1000):
    prediction = numpy.random.random(size=size)
    pid = numpy.random.random(size=size)
    random_labels = numpy.random.choice(2, size=size)

    def ams_like(s, b):
        return s / (b + 1. / 100. / size)

    # setting 'the best event' to be signal
    random_labels[numpy.argmax(prediction)] = 1
    optimal_ams = metrics.OptimalMetricNdim(ams_like)
    proba = numpy.ndarray((len(prediction), 2))
    proba[:, 0] = 1 - prediction
    proba[:, 1] = prediction
    pid_2d = numpy.ndarray((len(prediction), 2))
    pid_2d[:, 0] = 1 - pid
    pid_2d[:, 1] = pid
    score = optimal_ams(random_labels, None, proba, pid_2d)

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


def test_optimal_metric_function(size=10000):
    labels = numpy.random.randint(0, 2, size=size)
    predictions = numpy.random.random(size=[size, 2])
    predictions /= predictions.sum(axis=1, keepdims=True)
    sample_weight = numpy.random.random(size=size)

    for metric, optimal_metric in [(significance, OptimalSignificance()),
                                   (ams, OptimalAMS())]:
        optimal_metric.fit(None, labels, sample_weight=sample_weight)
        value = optimal_metric(labels, predictions, sample_weight=sample_weight)
        thresholds, values = optimal_metric.compute(labels, predictions, sample_weight=sample_weight)
        assert numpy.max(values) == value, "maximal value doesn't coincide"
        index = numpy.random.randint(0, len(thresholds))
        threshold = thresholds[index]
        passed = numpy.array(predictions[:, 1] >= threshold)

        s = optimal_metric.expected_s * numpy.average(passed, weights=sample_weight * (labels == 1))
        b = optimal_metric.expected_b * numpy.average(passed, weights=sample_weight * (labels == 0))
        assert numpy.allclose(metric(s, b), values[index]), 'no coincidence {} {} {}'.format(type(optimal_metric),
                                                                                             metric(s, b),
                                                                                             values[index])
