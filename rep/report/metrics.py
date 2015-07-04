"""
This file contains definitions for useful metrics in specific REP format.
In general case, metrics follows standard sklearn convention for **estimators**, provides

    * constructor (you should create instance of metric!):

    >>> metric = RocAuc(parameter=1)

    * fitting, where checks and heavy computations performed
      (this step is needed for ranking metrics, uniformity metrics):

    >>> metric.fit(X, y, sample_weight=None)

    * computation of metrics by probabilities:

    >>> proba = classifier.predict_proba(X)
    >>> metrics(proba)


This way metrics can be used in learning curves, for instance. Once fitted, then for every stage
computation will be very fast.


Correspondence between physical terms and ML terms
**************************************************

Some notation used below:

    * IsSignal (IsS) --- is really signal
    * AsSignal (AsS) --- classified as signal
    * IsBackgroundAsSignal - background, but classified as signal

... and so on. Cute, right?

There are many ways to denote this things:

    * tpr = s = isSasS / isS
    * fpr = b = isBasS / isB

Here we used normalized s and b, while physicists usually normalize
them to particular values of expected amount of s and b.

    * signal efficiency = tpr = s

    the following line used only in HEP

    * background efficiency = fpr = b
"""


from __future__ import division, print_function, absolute_import
import numpy
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, roc_curve
from ..utils import check_arrays
from ..utils import check_sample_weight, weighted_percentile


__author__ = 'Alex Rogozhnikov'


class MetricMixin(object):
    """Class with helpful methods for metrics,
     metrics are expected (but not obliged) to be derived from it."""
    def _prepare(self, X, y, sample_weight):
        """
        Preparation

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: labels of events - array-like of shape [n_samples]
        :param sample_weight: weight of events,
               array-like of shape [n_samples] or None if all weights are equal

        :return: X, y, sample_weight, indices
        """
        assert len(X) == len(y), 'Lengths are different!'
        sample_weight = check_sample_weight(y, sample_weight=sample_weight)
        self.classes_, indices = numpy.unique(y, return_inverse=True)
        self.probabilities_shape = (len(y), len(self.classes_))
        return X, y, sample_weight, indices

    def fit(self, X, y, sample_weight=None):
        """
        Prepare metrics for usage, preprocessing is done in this function.

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: labels of events - array-like of shape [n_samples]
        :param sample_weight: weight of events,
               array-like of shape [n_samples] or None if all weights are equal
        :return: self
        """
        return self


class RocAuc(BaseEstimator, MetricMixin):
    """
    Computes area under the ROC curve.

    :param int positive_label: label of class, in case of more then two classes,
     will compute ROC AUC for this specific class vs others
    """
    def __init__(self, positive_label=1):
        self.positive_label = positive_label

    def fit(self, X, y, sample_weight=None):
        """
        Prepare metrics for usage, preprocessing is done in this function.

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: labels of events - array-like of shape [n_samples]
        :param sample_weight: weight of events,
               array-like of shape [n_samples] or None if all weights are equal
        :return: self
        """
        X, y, self.sample_weight, _ = self._prepare(X, y, sample_weight=sample_weight)
        # computing index of positive label
        self.positive_index = self.classes_.tolist().index(self.positive_label)
        self.true_class = (numpy.array(y) == self.positive_label)
        return self

    def __call__(self, y, proba, sample_weight=None):
        assert numpy.all(self.classes_ < proba.shape[1])
        return roc_auc_score(self.true_class, proba[:, self.positive_index],
                             sample_weight=self.sample_weight)


class LogLoss(BaseEstimator, MetricMixin):
    """
    Log loss,
    which is the same as minus log-likelihood,
    and the same as logistic loss,
    and the same as cross-entropy loss.
    """
    def __init__(self, regularization=1e-15):
        self.regularization = regularization

    def fit(self, X, y, sample_weight=None):
        """
        Prepare metrics for usage, preprocessing is done in this function.

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: labels of events - array-like of shape [n_samples]
        :param sample_weight: weight of events,
               array-like of shape [n_samples] or None if all weights are equal
        :return: self
        """
        X, y, sample_weight, self.class_indices = self._prepare(X, y, sample_weight=sample_weight)
        self.sample_weight = sample_weight / sample_weight.sum()
        self.samples_indices = numpy.arange(len(X))
        return self

    def __call__(self, y, proba, sample_weight=None):
        # assert proba.shape == self.probabilities_shape, 'Wrong shape of probabilities'
        assert numpy.all(self.classes_ < proba.shape[1])
        correct_probabilities = proba[self.samples_indices, self.class_indices]
        return - (numpy.log(correct_probabilities + self.regularization) * self.sample_weight).sum()


class OptimalMetric(BaseEstimator, MetricMixin):
    """
    Class to calculate optimal threshold on predictions using some metric

    :param function metric: metrics(s, b) -> float
    :param expected_s: float, total weight of signal
    :param expected_b: float, total weight of background
    """
    def __init__(self, metric, expected_s=1., expected_b=1., signal_label=1):
        self.metric = metric
        self.expected_s = expected_s
        self.expected_b = expected_b
        self.signal_label = signal_label

    def compute(self, y_true, proba, sample_weight=None):
        """
        Compute metric for each possible prediction threshold

        :param y_true: array-like true labels
        :param proba: array-like of shape [n_samples, 2] with predicted probabilities
        :param sample_weight: array-like weight

        :rtype: tuple(array, array)
        :return: thresholds and corresponding metric values
        """
        y_true, proba, sample_weight = check_arrays(y_true, proba, sample_weight)
        pred = proba[:, self.signal_label]
        b, s, thresholds = roc_curve(y_true == self.signal_label, pred,
                                     sample_weight=sample_weight)

        metric_values = self.metric(s * self.expected_s, b * self.expected_b)
        thresholds = numpy.clip(thresholds, pred.min() - 1e-6, pred.max() + 1e-6)
        return thresholds, metric_values

    def plot_vs_cut(self, y_true, proba, sample_weight=None):
        """
        Compute metric for each possible prediction threshold

        :param y_true: array-like true labels
        :param proba: array-like of shape [n_samples, 2] with predicted probabilities
        :param sample_weight: array-like weight

        :rtype: plotting.FunctionsPlot
        """
        from .. import plotting

        y_true, proba, sample_weight = check_arrays(y_true, proba, sample_weight)
        ordered_proba, metrics_val = self.compute(y_true, proba, sample_weight)
        ind = numpy.argmax(metrics_val)

        print('Optimal cut=%1.4f, quality=%1.4f' % (ordered_proba[ind], metrics_val[ind]))

        plot_fig = plotting.FunctionsPlot({self.metric.__name__: (ordered_proba, metrics_val)})
        plot_fig.xlabel = 'cut'
        plot_fig.ylabel = 'metrics ' + self.metric.__name__
        return plot_fig

    def __call__(self, y_true, proba, sample_weight=None):
        """ proba is predicted probabilities of shape [n_samples, 2] """
        thresholds, metrics_val = self.compute(y_true, proba, sample_weight)
        return numpy.max(metrics_val)


def significance(s, b):
    """
    Approximate significance of discovery:
     s / sqrt(b).
    Here we use normalization, so maximal s and b are equal to 1.
    """
    return s / numpy.sqrt(b + 1e-6)


class OptimalSignificance(OptimalMetric):
    """
    Optimal values of significance:
     s / sqrt(b)

    :param float expected_s: expected amount of signal
    :param float expected_b: expected amount of background
    """
    def __init__(self, expected_s=1., expected_b=1.):
        OptimalMetric.__init__(self, metric=significance,
                               expected_s=expected_s,
                               expected_b=expected_b)


def ams(s, b, br=10.):
    """
    Regularized approximate median significance

    :param s: amount of signal passed
    :param b: amount of background passed
    :param br: regularization
    """
    radicand = 2 * ((s + b + br) * numpy.log(1.0 + s / (b + br)) - s)
    return numpy.sqrt(radicand)


class OptimalAMS(OptimalMetric):
    """
    Optimal values of AMS (average median significance)

    default values of expected_s and expected_b are from HiggsML challenge.

    :param float expected_s: expected amount of signal
    :param float expected_b: expected amount of background
    """
    def __init__(self, expected_s=691.988607712, expected_b=410999.847):
        OptimalMetric.__init__(self, metric=ams,
                               expected_s=expected_s,
                               expected_b=expected_b)


class FPRatTPR(BaseEstimator, MetricMixin):
    """
    Fix TPR value on roc curve and return FPR value.
    """
    def __init__(self, tpr):
        self.tpr = tpr

    def __call__(self, y, proba, sample_weight=None):
        if sample_weight is None:
            sample_weight = numpy.ones(len(proba))
        y, proba, sample_weight = check_arrays(y, proba, sample_weight)
        threshold = weighted_percentile(proba[y == 1, 1], (1. - self.tpr), sample_weight=sample_weight[y == 1])
        return numpy.sum(sample_weight[(y == 0) & (proba[:, 1] >= threshold)]) / sum(sample_weight[y == 0])


class TPRatFPR(BaseEstimator, MetricMixin):
    """
    Fix FPR value on roc curve and return TPR value.
    """
    def __init__(self, fpr):
        self.fpr = fpr

    def __call__(self, y, proba, sample_weight=None):
        if sample_weight is None:
            sample_weight = numpy.ones(len(proba))
        y, proba, sample_weight = check_arrays(y, proba, sample_weight)
        threshold = weighted_percentile(proba[y == 0, 1], (1 - self.fpr), sample_weight=sample_weight[y == 0])
        return numpy.sum(sample_weight[(y == 1) & (proba[:, 1] > threshold)]) / sum(sample_weight[y == 1])
