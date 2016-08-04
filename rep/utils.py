"""
Different helpful functions, objects, methods are collected here.
"""

from __future__ import division, print_function, absolute_import
from collections import OrderedDict

import time
import numpy
import pandas
from sklearn.utils.validation import column_or_1d
from sklearn.metrics import roc_curve


def weighted_quantile(array, quantiles, sample_weight=None, array_sorted=False, old_style=False):
    """Computing quantiles of array. Unlike the numpy.percentile, this function supports weights,
    but it is inefficient and performs complete sorting.

    :param array: distribution, array of shape [n_samples]
    :param quantiles: floats from range [0, 1] with quantiles of shape [n_quantiles]
    :param sample_weight: optional weights of samples, array of shape [n_samples]
    :param array_sorted: if True, the sorting step will be skipped
    :param old_style: if True, will correct output to be consistent with numpy.percentile.
    :return: array of shape [n_quantiles]

    Example:

    >>> weighted_quantile([1, 2, 3, 4, 5], [0.5])
    Out: array([ 3.])
    >>> weighted_quantile([1, 2, 3, 4, 5], [0.5], sample_weight=[3, 1, 1, 1, 1])
    Out: array([ 2.])
    """
    array = numpy.array(array)
    quantiles = numpy.array(quantiles)
    sample_weight = check_sample_weight(array, sample_weight)
    assert numpy.all(quantiles >= 0) and numpy.all(quantiles <= 1), 'Percentiles should be in [0, 1]'

    if not array_sorted:
        array, sample_weight = reorder_by_first(array, sample_weight)

    weighted_quantiles = numpy.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= numpy.sum(sample_weight)
    return numpy.interp(quantiles, weighted_quantiles, array)


def reorder_by_first(*arrays):
    """
    Applies the same permutation to all passed arrays,
    permutation sorts the first passed array
    """
    arrays = check_arrays(*arrays)
    order = numpy.argsort(arrays[0])
    return [arr[order] for arr in arrays]


def check_sample_weight(y_true, sample_weight):
    """Checks the weights, if None, returns array.

    :param y_true: labels (or any array of length [n_samples])
    :param sample_weight: None or array of length [n_samples]
    :return: numpy.array of shape [n_samples]
    """
    if sample_weight is None:
        return numpy.ones(len(y_true), dtype=numpy.float)
    else:
        sample_weight = numpy.array(sample_weight, dtype=numpy.float)
        assert len(y_true) == len(sample_weight), \
            "The length of weights is different: not {0}, but {1}".format(len(y_true), len(sample_weight))
        return sample_weight


class Flattener(object):
    def __init__(self, data, sample_weight=None):
        """
        Prepares normalization function for some set of values
        transforms it to uniform distribution from [0, 1].

        :param data: predictions
        :type data: list or numpy.array
        :param sample_weight: weights
        :type sample_weight: None or list or numpy.array
        :return func: normalization function

        Example of usage:

        >>> normalizer = Flattener(signal)
        >>> hist(normalizer(background))
        >>> hist(normalizer(signal))

        """
        sample_weight = check_sample_weight(data, sample_weight=sample_weight)
        data = column_or_1d(data)
        assert numpy.all(sample_weight >= 0.), 'sample weight must be non-negative'
        self.data, sample_weight = reorder_by_first(data, sample_weight)
        self.predictions = numpy.cumsum(sample_weight) / numpy.sum(sample_weight)

    def __call__(self, data):
        return numpy.interp(data, self.data, self.predictions)


class Binner(object):
    def __init__(self, values, bins_number):
        """
        Binner is a class that helps to split the values into several bins.
        Initially an array of values is given, which is then splitted into 'bins_number' equal parts,
        and thus we are computing limits (boundaries of bins).
        """
        percentiles = [i * 100.0 / bins_number for i in range(1, bins_number)]
        self.limits = numpy.percentile(values, percentiles)

    def get_bins(self, values):
        """Given the values of feature, compute the index of bin

        :param values: array of shape [n_samples]
        :return: array of shape [n_samples]
        """
        return numpy.searchsorted(self.limits, values)

    def set_limits(self, limits):
        """Change the thresholds inside bins."""
        self.limits = limits

    @property
    def bins_number(self):
        """:return: number of bins"""
        return len(self.limits) + 1

    def split_into_bins(self, *arrays):
        """
        :param arrays: data to be splitted, the first array corresponds
        :return: sequence of length [n_bins] with values corresponding to each bin.
        """
        values = arrays[0]
        for array in arrays:
            assert len(array) == len(values), "passed arrays have different length"
        bins = self.get_bins(values)
        result = []
        for bin in range(len(self.limits) + 1):
            indices = bins == bin
            result.append([numpy.array(array)[indices] for array in arrays])
        return result


def calc_ROC(prediction, signal, sample_weight=None, max_points=10000):
    """
    Calculate roc curve, returns limited number of points.
    This is needed for interactive plots, which suffer

    :param prediction: predictions
    :type prediction: numpy.ndarray or list
    :param signal: true labels
    :type signal: array or list
    :param sample_weight: weights
    :type sample_weight: None or array or list
    :param int max_points: maximum of used points on roc curve

    :return: (tpr, tnr), (err_tnr, err_tpr), thresholds
    """
    sample_weight = numpy.ones(len(signal)) if sample_weight is None else sample_weight
    prediction, signal, sample_weight = check_arrays(prediction, signal, sample_weight)

    assert set(signal) == {0, 1}, "the labels should be 0 and 1, labels are " + str(set(signal))
    fpr, tpr, thresholds = roc_curve(signal, prediction, sample_weight=sample_weight)
    tpr = numpy.insert(tpr, 0, [0.])
    fpr = numpy.insert(fpr, 0, [0.])
    thresholds = numpy.insert(thresholds, 0, [thresholds[0] + 1.])
    tnr = 1 - fpr

    weight_bck = sample_weight[signal == 0]
    weight_sig = sample_weight[signal == 1]
    err_tnr = numpy.sqrt(tnr * (1 - tnr) * numpy.sum(weight_bck ** 2)) / numpy.sum(weight_bck)
    err_tpr = numpy.sqrt(tpr * (1 - tpr) * numpy.sum(weight_sig ** 2)) / numpy.sum(weight_sig)

    if len(prediction) > max_points:
        sum_weights = numpy.cumsum((fpr + tpr) / 2.)
        sum_weights /= sum_weights[-1]
        positions = numpy.searchsorted(sum_weights, numpy.linspace(0, 1, max_points))
        tpr, tnr = tpr[positions], tnr[positions]
        err_tnr, err_tpr = err_tnr[positions], err_tpr[positions]
        thresholds = thresholds[positions]

    return (tpr, tnr), (err_tnr, err_tpr), thresholds


def calc_feature_correlation_matrix(df, weights=None):
    """
    Calculate correlation matrix

    :param pandas.DataFrame df: data of shape [n_samples, n_features]
    :param weights: weights of shape [n_samples] (optional)
    :return: correlation matrix for dataFrame of shape [n_features, n_features]
    :rtype: numpy.ndarray
    """
    values = numpy.array(df)
    weights = check_sample_weight(df, sample_weight=weights)
    means = numpy.average(values, weights=weights, axis=0)
    values -= means
    covariation = values.T.dot(values * weights[:, None])
    diag = covariation.diagonal()
    return covariation / numpy.sqrt(diag)[:, None] / numpy.sqrt(diag)[None, :]


def calc_hist_with_errors(x, weight=None, bins=60, normed=True, x_range=None, ignored_sideband=0.0):
    """
    Calculate data for error bar (for plot pdf with errors)

    :param x: data
    :type x: list or numpy.array
    :param weight: weights
    :type weight: None or list or numpy.array

    :return: tuple (x-points (list), y-points (list), y points errors (list), x points errors (list))
    """
    weight = numpy.ones(len(x)) if weight is None else weight
    x, weight = check_arrays(x, weight)

    if x_range is None:
        x_range = numpy.percentile(x, [100 * ignored_sideband, 100 * (1 - ignored_sideband)])

    ans, bins = numpy.histogram(x, bins=bins, normed=normed, weights=weight, range=x_range)
    yerr = []
    normalization = 1.0
    if normed:
        normalization = float(len(bins) - 1) / float(sum(weight)) / (x_range[1] - x_range[0])
    for i in range(len(bins) - 1):
        weight_bin = weight[(x > bins[i]) * (x <= bins[i + 1])]
        yerr.append(numpy.sqrt(sum(weight_bin * weight_bin)) * normalization)
    bins_mean = [0.5 * (bins[i] + bins[i + 1]) for i in range(len(ans))]
    xerr = [0.5 * (bins[i + 1] - bins[i]) for i in range(len(ans))]
    return bins_mean, ans, yerr, xerr


def get_efficiencies(prediction, spectator, sample_weight=None, bins_number=20,
                     thresholds=None, errors=False, ignored_sideband=0.0):
    """
    Construct efficiency function dependent on spectator for each threshold

    Different score functions available: Efficiency, Precision, Recall, F1Score,
    and other things from sklearn.metrics

    :param prediction: list of probabilities
    :param spectator: list of spectator's values
    :param bins_number: int, count of bins for plot

    :param thresholds: list of prediction's threshold

        (default=prediction's cuts for which efficiency will be [0.2, 0.4, 0.5, 0.6, 0.8])

    :return:
        if errors=False
        OrderedDict threshold -> (x_values, y_values)

        if errors=True
        OrderedDict threshold -> (x_values, y_values, y_err, x_err)

        All the parts: x_values, y_values, y_err, x_err are numpy.arrays of the same length.
    """
    prediction, spectator, sample_weight = \
        check_arrays(prediction, spectator, sample_weight)

    spectator_min, spectator_max = weighted_quantile(spectator, [ignored_sideband, (1. - ignored_sideband)])
    mask = (spectator >= spectator_min) & (spectator <= spectator_max)
    spectator = spectator[mask]
    prediction = prediction[mask]
    bins_number = min(bins_number, len(prediction))
    sample_weight = sample_weight if sample_weight is None else numpy.array(sample_weight)[mask]

    if thresholds is None:
        thresholds = [weighted_quantile(prediction, quantiles=1 - eff, sample_weight=sample_weight)
                      for eff in [0.2, 0.4, 0.5, 0.6, 0.8]]

    binner = Binner(spectator, bins_number=bins_number)
    if sample_weight is None:
        sample_weight = numpy.ones(len(prediction))
    bins_data = binner.split_into_bins(spectator, prediction, sample_weight)

    bin_edges = numpy.array([spectator_min] + list(binner.limits) + [spectator_max])
    xerr = numpy.diff(bin_edges) / 2.
    result = OrderedDict()
    for threshold in thresholds:
        x_values = []
        y_values = []
        N_in_bin = []
        for num, (masses, probabilities, weights) in enumerate(bins_data):
            y_values.append(numpy.average(probabilities > threshold, weights=weights))
            N_in_bin.append(numpy.sum(weights))
            if errors:
                x_values.append((bin_edges[num + 1] + bin_edges[num]) / 2.)
            else:
                x_values.append(numpy.mean(masses))

        x_values, y_values, N_in_bin = check_arrays(x_values, y_values, N_in_bin)
        if errors:
            result[threshold] = (x_values, y_values, numpy.sqrt(y_values * (1 - y_values) / N_in_bin), xerr)
        else:
            result[threshold] = (x_values, y_values)
    return result


def train_test_split(*arrays, **kw_args):
    """
    Does the same thing as sklearn.cross_validation.train_test_split.
    Additionally has 'allow_none' parameter.

    :param arrays: arrays to split with same first dimension
    :type arrays: list[numpy.array] or list[pandas.DataFrame]
    :param bool allow_none: default False, is set to True, allows
        non-first arguments to be None (in this case, both resulting train and test parts are None).

    """
    from sklearn import cross_validation
    allow_none = kw_args.pop('allow_none', False)

    assert len(arrays) > 0, "at least one array should be passed"
    length = len(arrays[0])
    for array in arrays:
        assert len(array) == length, "different size"
    train_indices, test_indices = cross_validation.train_test_split(range(length), **kw_args)
    result = []
    for array in arrays:
        if isinstance(array, pandas.DataFrame):
            result.append(array.iloc[train_indices, :])
            result.append(array.iloc[test_indices, :])
        elif (array is None) and allow_none:
            # specially for checking weights
            result.append(None)
            result.append(None)
        else:
            result.append(numpy.array(array)[train_indices])
            result.append(numpy.array(array)[test_indices])
    return result


def train_test_split_group(group_column, *arrays, **kw_args):
    """
    Modification of :class:`train_test_split` which alters splitting rule.

    :param group_column: array-like of shape [n_samples] with indices of groups,
        events from one group will be kept together (all events in train or all events in test).
        If `group_column` is used, train_size and test_size will refer to number of groups, not events
    :param arrays: arrays to split
    :type arrays: list[numpy.array] or list[pandas.DataFrame]
    :param bool allow_none: default False
        (useful for sample_weight - after splitting train and test of `None` are again `None`)
    """
    from sklearn import cross_validation
    allow_none = kw_args.pop('allow_none', None)

    assert len(arrays) > 0, "at least one array should be passed"
    length = len(arrays[0])
    for array in arrays:
        assert len(array) == length, "different size"

    initial_data = numpy.array(group_column)
    assert len(initial_data) == length, "group column must have the same length"
    group_ids = numpy.unique(initial_data)

    train_indices, test_indices = cross_validation.train_test_split(group_ids, **kw_args)
    train_indices = numpy.in1d(initial_data, train_indices)
    test_indices = numpy.in1d(initial_data, test_indices)

    result = []
    for array in arrays:
        if isinstance(array, pandas.DataFrame):
            result.append(array.iloc[train_indices, :])
            result.append(array.iloc[test_indices, :])
        elif (array is None) and allow_none:
            # specially for checking weights
            result.append(None)
            result.append(None)
        else:
            result.append(numpy.array(array)[train_indices])
            result.append(numpy.array(array)[test_indices])
    return result


def get_columns_dict(columns):
    """
    Get (new column: old column) dict expressions.
    This function is used to process names of features, which can contain expressions.

    :param list[str] columns: columns names
    :rtype: dict
    """
    result = OrderedDict()
    for column in columns:
        column_split = column.split(':')
        assert len(column_split) < 3, 'Error in parsing feature expression {}'.format(column)
        if len(column_split) == 2:
            result[column_split[0].strip()] = column_split[1].strip()
        else:
            result[column] = column
    return result


def get_columns_in_df(df, columns):
    """
    Get columns in data frame using *numexpr* evaluation

    :param pandas.DataFrame df: data
    :param columns: necessary columns
    :param columns: None or list[str]
    :return: data frame with pointed columns
    """
    if columns is None:
        return df
    columns_dict = get_columns_dict(columns)
    df_new = OrderedDict()
    for column_new, column in columns_dict.items():
        if column in df.columns:
            df_new[column_new] = df[column]
        else:
            # warning - this thing is known to fail in threads
            # numexpr.evaluate(column, local_dict=df)
            # thus we are using python engine, which is slow :(
            df_new[column_new] = df.eval(column, engine='python')
    return pandas.DataFrame(df_new)


def check_arrays(*arrays):
    """
    Left for consistency, version of `sklearn.validation.check_arrays`

    :param list[iterable] arrays: arrays with same length of first dimension.
    """
    assert len(arrays) > 0, 'The number of array must be greater than zero'
    checked_arrays = []
    shapes = []
    for arr in arrays:
        if arr is not None:
            checked_arrays.append(numpy.array(arr))
            shapes.append(checked_arrays[-1].shape[0])
        else:
            checked_arrays.append(None)
    assert numpy.sum(numpy.array(shapes) == shapes[0]) == len(shapes), 'Different shapes of the arrays {}'.format(
        shapes)
    return checked_arrays


def fit_metric(metric, *args, **kargs):
    """
    Metric can implement one of two interfaces (function or object).
    This function fits metrics, if it is required (by simply checking presence of fit method).

    :param metric: metric function, following REP conventions
    """
    if hasattr(metric, 'fit'):
        metric.fit(*args, **kargs)


class Stopwatch(object):
    """
    Simple tool to measure time.
    If your internet connection is reliable, use %time magic.

    >>> with Stopwatch() as timer:
    >>>     # do something here
    >>>     classifier.fit(X, y)
    >>> # print how much time was spent
    >>> print(timer)
    """

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, err_type, err_value, err_traceback):
        self.stop = time.time()
        self.err_type = err_type
        self.err_value = err_value
        self.err_traceback = err_traceback

    @property
    def elapsed(self):
        return self.stop - self.start

    def __repr__(self):
        result = "interval: {:.2f} sec".format(self.elapsed)
        if self.err_type is not None:
            message = "\nError {error} of type {error_type} was raised"
            result += message.format(error=repr(self.err_value), error_type=self.err_type)
        return result
