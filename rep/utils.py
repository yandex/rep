"""
Different helpful functions, objects, methods are collected here.
"""

from __future__ import division, print_function, absolute_import
from collections import OrderedDict
import numexpr

import numpy
import pandas
from sklearn.utils.validation import check_arrays, column_or_1d
from sklearn.metrics import roc_curve


def weighted_percentile(array, percentiles, sample_weight=None, array_sorted=False, old_style=False):
    array = numpy.array(array)
    percentiles = numpy.array(percentiles)
    sample_weight = check_sample_weight(array, sample_weight)
    assert numpy.all(percentiles >= 0) and numpy.all(percentiles <= 1), 'Percentiles should be in [0, 1]'

    if not array_sorted:
        array, sample_weight = reorder_by_first(array, sample_weight)

    weighted_quantiles = numpy.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= numpy.sum(sample_weight)
    return numpy.interp(percentiles, weighted_quantiles, array)


def reorder_by_first(*arrays):
    """
    Applies the same permutation to all passed arrays,
    permutation sorts the first passed array
    """
    arrays = check_arrays(*arrays)
    order = numpy.argsort(arrays[0])
    return [arr[order] for arr in arrays]


def check_sample_weight(y_true, sample_weight):
    """
    Checks the weights, returns normalized version
    """
    if sample_weight is None:
        return numpy.ones(len(y_true), dtype=numpy.float)
    else:
        sample_weight = numpy.array(sample_weight, dtype=numpy.float)
        assert len(y_true) == len(sample_weight), \
            "The length of weights is different: not {0}, but {1}".format(len(y_true), len(sample_weight))
        return sample_weight


class Flattener(object):
    """
        Prepares normalization function for some set of values
        transforms it to uniform distribution from [0, 1]. Example of usage:

        Parameters:
        -----------
        :param data: predictions
        :type data: list or numpy.array
        :param sample_weight: weights
        :type sample_weight: None or list or numpy.array

        Example:
        --------
        >>> normalizer = Flattener(signal)
        >>> hist(normalizer(background))
        >>> hist(normalizer(signal))

        :return func: normalization function
        """
    def __init__(self, data, sample_weight=None):
        sample_weight = check_sample_weight(data, sample_weight=sample_weight)
        data = column_or_1d(data)
        assert numpy.all(sample_weight >= 0.), 'sample weight must be non-negative'
        self.data, sample_weight = reorder_by_first(data, sample_weight)
        self.predictions = numpy.cumsum(sample_weight) / numpy.sum(sample_weight)

    def __call__(self, data):
        return numpy.interp(data, self.data, self.predictions)


class Binner:
    def __init__(self, values, bins_number):
        """
        Binner is a class that helps to split the values into several bins.
        Initially an array of values is given, which is then splitted into 'bins_number' equal parts,
        and thus we are computing limits (boundaries of bins)."""
        percentiles = [i * 100.0 / bins_number for i in range(1, bins_number)]
        self.limits = numpy.percentile(values, percentiles)

    def get_bins(self, values):
        return numpy.searchsorted(self.limits, values)

    def get_bins_dumb(self, values):
        """This is the sane as previous function, but a bit slower and naive"""
        result = numpy.zeros(len(values))
        for limit in self.limits:
            result += values > limit
        return result

    def set_limits(self, limits):
        self.limits = limits

    def bins_number(self):
        return len(self.limits) + 1

    def split_into_bins(self, *arrays):
        """
        Splits the data of parallel arrays into bins, the first array is binning variable
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

    :param prediction: predictions
    :type prediction: array or list
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
    tpr = numpy.insert(tpr, 0, 0.)
    fpr = numpy.insert(fpr, 0, 0.)
    thresholds = numpy.insert(thresholds, 0, thresholds[0] + 1.)
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


def calc_feature_correlation_matrix(df):
    """
    Calculate correlation matrix

    :param pandas.DataFrame df: data
    :return: correlation matrix for dataFrame
    :rtype: numpy.ndarray
    """
    # TODO use weights
    return numpy.corrcoef(df.values.T)


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

    Parameters:
    -----------
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
    prediction, spectator = \
        check_arrays(prediction, spectator)

    spectator_min, spectator_max = numpy.percentile(spectator, [100 * ignored_sideband, 100 * (1. - ignored_sideband)])
    mask = (spectator >= spectator_min) & (spectator <= spectator_max)
    spectator = spectator[mask]
    prediction = prediction[mask]
    bins_number = min(bins_number, len(prediction))
    sample_weight = sample_weight if sample_weight is None else numpy.array(sample_weight)[mask]

    if thresholds is None:
        thresholds = [weighted_percentile(prediction, percentiles=1 - eff, sample_weight=sample_weight)
                      for eff in [0.2, 0.4, 0.5, 0.6, 0.8]]

    binner = Binner(spectator, bins_number=bins_number)
    bins_data = binner.split_into_bins(spectator, prediction)

    bin_edges = numpy.array([spectator_min] + list(binner.limits) + [spectator_max])
    xerr = numpy.diff(bin_edges) / 2.
    result = OrderedDict()
    for threshold in thresholds:
        x_values = []
        y_values = []
        for num, (masses, probabilities) in enumerate(bins_data):
            y_values.append(numpy.mean(probabilities > threshold))
            if errors:
                x_values.append((bin_edges[num + 1] + bin_edges[num]) / 2.)
            else:
                x_values.append(numpy.mean(masses))

        x_values, y_values = check_arrays(x_values, y_values)
        if errors:
            result[threshold] = (x_values, y_values, numpy.sqrt(y_values * (1 - y_values) / len(y_values)), xerr)
        else:
            result[threshold] = (x_values, y_values)
    return result


def train_test_split(*arrays, **kw_args):
    """Does the same thing as train_test_split, but preserves names of columns in DataFrames.
    Uses the same parameters: test_size, train_size, random_state, and has almost the same interface

    :param arrays: arrays to split
    :type arrays: list[numpy.array] or list[pandas.DataFrame]

    :param group_column: array-like of shape [n_samples] with indices of groups,
    events from one group will be kept together (all events in train or all events in test).
    If `group_column` is used, train_size and test_size will refer to number of groups, not events
    :param bool allow_none: default False
    (specially for sample_weight - after splitting train and test of `None` are `None` too)
    """
    from sklearn import cross_validation
    allow_none = kw_args.pop('allow_none', None)

    assert len(arrays) > 0, "at least one array should be passed"
    length = len(arrays[0])
    for array in arrays:
        assert len(array) == length, "different size"

    if 'group_column' in kw_args:
        initial_data = numpy.array(kw_args.pop('group_column'))
        assert len(initial_data) == length, "group column must have the same length"
        group_ids = numpy.unique(initial_data)
    else:
        initial_data = numpy.arange(length)
        group_ids = numpy.arange(length)

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
            result.append(array[train_indices])
            result.append(array[test_indices])
    return result


def get_columns_dict(columns):
    """
    Get (new column: old column) dict expressions

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
        df_new[column_new] = numexpr.evaluate(column, local_dict=df)
    return pandas.DataFrame(df_new)
