"""

Metric-object API
-----------------

**REP** introduces several metric functions in specific format.
Metric functions following this format can be used in grid search and reports.

In general case, metrics follow standard sklearn convention for **estimators**, provides

    * constructor (you should create an instance of metric!), all fine-tuning should be done at this step:

    >>> metric = RocAuc(positive_label=2)

    * fitting, where checks and heavy computations performed
      (this step is needed for ranking metrics, uniformity metrics):

    >>> metric.fit(X, y, sample_weight=None)

    * computation of metrics by probabilities (important: metric should be computed on exactly same dataset as was used on previous step):

    >>> # in case of classification
    >>> proba = classifier.predict_proba(X)
    >>> metric(y, proba, sample_weight=None)
    >>> # in case of regression
    >>> prediction = regressor.predict(X)
    >>> metric(y, prediction, sample_weight=None)

This way metrics can be used in learning curves, for instance.
Once fitted (and heavy computations done in fitting), then for every stage computation is fast.


Metric-function (convenience) API
---------------------------------

Many metric functions do not require complex settings and different precomputing,
 so **REP** also works with functions having following API:

    >>> # for classification
    >>> metric(y, probabilities, sample_weight=None)
    >>> # for regression
    >>> metric(y, predictions, sample_weight=None)

As an example, `mean_squared_error` and `mean_absolute_error` from sklearn can be used in **REP**.


.. seealso::
    `API of metrics <https://github.com/yandex/rep/wiki/Contributing-new-metrics>`_ for details and explanations on API.


Correspondence between physics terms and ML terms
-------------------------------------------------

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

    * signal efficiency = s = tpr

the following line used only in HEP

    * background efficiency = b = fpr



Available Metric functions
--------------------------

.. autoclass:: RocAuc
    :show-inheritance:

.. autoclass:: LogLoss
    :show-inheritance:

.. autoclass:: OptimalAccuracy
    :show-inheritance:

.. autoclass:: OptimalAMS
    :show-inheritance:

.. autoclass:: OptimalSignificance
    :show-inheritance:

.. autoclass:: TPRatFPR
    :show-inheritance:

.. autoclass:: FPRatTPR
    :show-inheritance:


Supplementary functions
-----------------------

Building blocks that should be useful to create new metrics.

.. autoclass:: MetricMixin
    :show-inheritance:
    :members:

.. autoclass:: OptimalMetric
    :show-inheritance:

.. autofunction:: ams

.. autofunction:: significance

.. autoclass:: OptimalMetricNdim
    :show-inheritance:
"""

from __future__ import division, print_function, absolute_import
import numpy
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, roc_curve
from ..utils import check_arrays
from ..utils import check_sample_weight, weighted_quantile
from itertools import product

__author__ = 'Alex Rogozhnikov'


class MetricMixin(object):
    """Class with helpful methods for metrics,
     metrics are expected (but not obliged) to be derived from this mixin. """

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
    def __init__(self, positive_label=1):
        """
        Computes area under the ROC curve.
        General-purpose quality measure for binary classification

        :param int positive_label: label of class, in case of more then two classes,
         will compute ROC AUC for this specific class vs others
        """
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
        assert numpy.all(self.classes_ < proba.shape[1]), 'passed probabilities not for all classes'
        return roc_auc_score(self.true_class, proba[:, self.positive_index],
                             sample_weight=self.sample_weight)


class LogLoss(BaseEstimator, MetricMixin):
    def __init__(self, regularization=1e-15):
        """
        Log loss,
        which is the same as minus log-likelihood,
        and the same as logistic loss,
        and the same as cross-entropy loss.

        Appropriate metric if algorithm is optimizing log-likelihood.

        :param regularization: minimal value for probability,
            to avoid high (or infinite) penalty for zero probabilities.

        """
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
        assert numpy.all(self.classes_ < proba.shape[1]), 'number of classes in predictions is greater than expected'
        correct_probabilities = proba[self.samples_indices, self.class_indices]
        return - (numpy.log(correct_probabilities + self.regularization) * self.sample_weight).sum()


class OptimalAccuracy(BaseEstimator, MetricMixin):
    def __init__(self, sb_ratio=None):
        """
        Estimation of binary classification accuracy for

        :param sb_ratio: ratio of signal (class 1) and background (class 0).
            If none, the parameter is estimated from test data.
        """
        self.sb_ratio = sb_ratio

    def compute(self, y_true, proba, sample_weight=None):
        """
        Compute metric for each possible prediction threshold

        :param y_true: array-like true labels
        :param proba: array-like of shape [n_samples, 2] with predicted probabilities
        :param sample_weight: array-like weight

        :rtype: tuple(array, array)
        :return: thresholds and corresponding metric values
        """
        sample_weight = check_sample_weight(y_true, sample_weight=sample_weight)
        sample_weight = sample_weight.copy()
        assert numpy.in1d(y_true, [0, 1]).all(), 'labels passed should be 0 and 1'

        if self.sb_ratio is not None:
            sample_weight_s = sample_weight[y_true == 1].sum()
            sample_weight_b = sample_weight[y_true == 0].sum()
            sample_weight[y_true == 1] *= self.sb_ratio * sample_weight_b / sample_weight_s

            assert numpy.allclose(self.sb_ratio, sample_weight[y_true == 1].sum() / sample_weight[y_true == 0].sum())

        sample_weight /= sample_weight.sum()
        signal_weight = sample_weight[y_true == 1].sum()
        bck_weight = sample_weight[y_true == 0].sum()

        fpr, tpr, thresholds = roc_curve(y_true == 1, proba[:, 1], sample_weight=sample_weight)
        accuracy_values = tpr * signal_weight + (1. - fpr) * bck_weight
        return thresholds, accuracy_values

    def __call__(self, y_true, proba, sample_weight=None):
        """
        Compute maximal value of accuracy by checking all possible thresholds.

        :param y_true: array-like true labels
        :param proba: array-like of shape [n_samples, 2] with predicted probabilities
        :param sample_weight: array-like weight
        """
        thresholds, accuracy_values = self.compute(y_true, proba, sample_weight)
        return numpy.max(accuracy_values)


class OptimalMetric(BaseEstimator, MetricMixin):
    def __init__(self, metric, expected_s=1., expected_b=1., signal_label=1):
        """
        Class to calculate optimal threshold on predictions for some binary metric.

        :param function metric: metrics(s, b) -> float
        :param expected_s: float, total weight of signal
        :param expected_b: float, total weight of background
        """
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
    Approximate significance of discovery: s / sqrt(b).
    Here we use normalization, so maximal s and b are equal to 1.

    :param s: amount of signal passed
    :param b: amount of background passed
    """
    return s / numpy.sqrt(b + 1e-6)


class OptimalSignificance(OptimalMetric):
    """
    Optimal values of significance: s / sqrt(b)

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

    Default values of expected_s and expected_b are from HiggsML challenge.

    :param float expected_s: expected amount of signal
    :param float expected_b: expected amount of background
    """

    def __init__(self, expected_s=691.988607712, expected_b=410999.847):
        OptimalMetric.__init__(self, metric=ams,
                               expected_s=expected_s,
                               expected_b=expected_b)


class FPRatTPR(BaseEstimator, MetricMixin):
    def __init__(self, tpr):
        """Fix TPR value on ROC curve and return corresponding FPR value.

        :param float tpr: target value true positive rate, from range (0, 1)
        """
        self.tpr = tpr

    def __call__(self, y, proba, sample_weight=None):
        sample_weight = check_sample_weight(y, sample_weight=sample_weight)
        y, proba, sample_weight = check_arrays(y, proba, sample_weight)
        threshold = weighted_quantile(proba[y == 1, 1], (1. - self.tpr), sample_weight=sample_weight[y == 1])
        return numpy.sum(sample_weight[(y == 0) & (proba[:, 1] >= threshold)]) / sum(sample_weight[y == 0])


class TPRatFPR(BaseEstimator, MetricMixin):
    def __init__(self, fpr):
        """Fix FPR value on ROC curve and return corresponding TPR value.

        :param float fpr: target value false positive rate, from range (0, 1)
        """
        self.fpr = fpr

    def __call__(self, y, proba, sample_weight=None):
        sample_weight = check_sample_weight(y, sample_weight=sample_weight)
        y, proba, sample_weight = check_arrays(y, proba, sample_weight)
        threshold = weighted_quantile(proba[y == 0, 1], (1 - self.fpr), sample_weight=sample_weight[y == 0])
        return numpy.sum(sample_weight[(y == 1) & (proba[:, 1] > threshold)]) / sum(sample_weight[y == 1])


class OptimalMetricNdim(BaseEstimator):
    """
    Class to calculate optimal thresholds on predictions of several classifier
    (prediction_1, prediction_2, .. prediction_n) simultaneously to maximize some binary metric.

    This metric differs from :class:`OptimalMetric`

    :param function metric: metrics(s, b) -> float, binary metric
    :param expected_s: float, total weight of signal
    :param expected_b: float, total weight of background
    :param int step: step in sorted array of predictions for each dimension to choose thresholds

    >>> proba1 = classifier1.predict_proba(X)[:, 1]
    >>> proba2 = classifier2.predict_proba(X)[:, 1]
    >>> optimal_ndim = OptimalMetricNdim(ams)
    >>> optimal_ndim(y, sample_weight, proba1, proba2)
    >>> # returns optimal AUC and thresholds for proba1 and proba2
    >>> 0.99, (0.88, 0.45)
    """

    def __init__(self, metric, expected_s=1., expected_b=1., step=10):
        self.metric = metric
        self.expected_s = expected_s
        self.expected_b = expected_b
        self.step = step

    def __call__(self, y_true, sample_weight, *arrays):
        """
        Compute metric for each possible predictions thresholds

        :param y_true: array-like true labels
        :param sample_weight: array-like weight
        :param arrays: sequence of different predictions of shape [n_samples]
        :rtype: tuple(array, array)
        :return: optimal metric value and corresponding thresholds for each dimension
        """
        all_data = check_arrays(y_true, sample_weight, *arrays)
        y_true, sample_weight, variables = all_data[0], all_data[1], all_data[2:]
        if sample_weight is None:
            sample_weight = numpy.ones(len(y_true))

        sample_weight = numpy.copy(sample_weight)
        sample_weight[y_true == 0] /= numpy.sum(sample_weight[y_true == 0]) / self.expected_b
        sample_weight[y_true == 1] /= numpy.sum(sample_weight[y_true == 1]) / self.expected_s

        thresholds = []
        for array in variables[:-1]:
            thr = numpy.sort(array)
            thresholds.append(thr[::self.step])
        optimal_metric_value = None
        optimal_threshold = None

        dim_last_pred = variables[-1]

        indices = numpy.argsort(dim_last_pred)[::-1]
        sorted_last_pred = dim_last_pred[indices]
        sorted_y = y_true[indices]
        sorted_weights = sample_weight[indices]
        sorted_pred = numpy.array(variables)[:, indices]

        for threshold in product(*thresholds):
            mask = numpy.ones(len(y_true), dtype=bool)
            for t, arr in zip(threshold, sorted_pred):
                mask *= arr >= t

            s = numpy.cumsum(sorted_y * sorted_weights * mask)
            b = numpy.cumsum((1 - sorted_y) * sorted_weights * mask)

            metric_values = self.metric(s, b)
            ind_optimal = numpy.argmax(metric_values)
            if (optimal_metric_value is None) or (optimal_metric_value < metric_values[ind_optimal]):
                optimal_metric_value = metric_values[ind_optimal]
                optimal_threshold = list(threshold) + [sorted_last_pred[ind_optimal]]
        return optimal_metric_value, optimal_threshold