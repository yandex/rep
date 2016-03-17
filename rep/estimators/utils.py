from __future__ import division, print_function, absolute_import

import numpy
import pandas
import warnings
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import column_or_1d

from ..utils import check_sample_weight, get_columns_in_df


__author__ = 'Alex Rogozhnikov'


def check_inputs(X, y, sample_weight, allow_none_weights=True, allow_multiple_targets=False):
    if allow_multiple_targets:
        y = numpy.array(y)
    else:
        y = column_or_1d(y)
    if allow_none_weights and sample_weight is None:
        # checking only X, y
        if len(X) != len(y):
            raise ValueError('Different size of X: {} and y: {}'.format(X.shape, y.shape))
        return X, y, None

    if sample_weight is None:
        sample_weight = numpy.ones(len(y), dtype=float)

    sample_weight = column_or_1d(sample_weight)
    assert sum(numpy.isnan(sample_weight)) == 0, "Weight contains nan, this format isn't supported"
    if not (len(X) == len(y) == len(sample_weight)):
        message = 'Different sizes of X: {}, y: {} and sample_weight: {}'
        raise ValueError(message.format(X.shape, y.shape, sample_weight.shape))

    return X, y, sample_weight


def score_to_proba(score):
    proba = numpy.zeros([len(score), 2])
    proba[:, 1] = expit(score)
    proba[:, 0] = 1 - proba[:, 1]
    return proba


def proba_to_two_dimensions(probability):
    proba = numpy.zeros([len(probability), 2])
    proba[:, 1] = probability
    proba[:, 0] = 1 - proba[:, 1]
    return proba


def proba_to_score(proba):
    assert proba.shape[1] == 2, 'Converting proba to score is possible only for two-class classification'
    proba = proba / proba.sum(axis=1, keepdims=True)
    score = logit(proba[:, 1])
    return score


def normalize_weights(y, sample_weight, per_class=True):
    """Returns normalized weights with average = 1.

    :param y: answers
    :param sample_weight: original weights (can not be None)
    :param per_class: if True
    """
    sample_weight = check_sample_weight(y, sample_weight=sample_weight)
    if per_class:
        sample_weight = sample_weight.copy()
        for label in numpy.unique(y):
            sample_weight[y == label] /= numpy.mean(sample_weight[y == label])
        return sample_weight
    else:
        return sample_weight / numpy.mean(sample_weight)


def _get_features(features, X, allow_nans=False):
    """
    Get data with necessary features

    :param list[str] features: features
    :param pandas.DataFrame X: train dataset

    :return: pandas.DataFrame with used features, features
    """
    new_features = features
    if isinstance(X, numpy.ndarray):
        X = pandas.DataFrame(X, columns=['Feature_%d' % index for index in range(X.shape[1])])
    else:
        assert isinstance(X, pandas.DataFrame), 'Support only numpy.ndarray and pandas.DataFrame'
    if features is None:
        new_features = list(X.columns)
        X_features = X
    elif list(X.columns) == list(features):
        X_features = X
    else:
        # assert set(self.features).issubset(set(X.columns)), "Data doesn't contain all training features"
        # X_features = X.ix[:, self.features]
        X_features = get_columns_in_df(X, features)

    if not allow_nans:
        # check column-by-column in order not to create copy of whole DataFrame
        for column in X_features.columns:
            assert numpy.all(numpy.isfinite(X_features[column])), "Does not support NaN: " + str(column)
    return X_features, new_features


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """
    Identity transformer is a very neat technology:
    it in a constant, reproducible manner makes nothing with input,
    though may convert it to some provided dtype
    """
    def __init__(self, dtype='float32'):
        self.dtype = dtype

    def fit(self, X, y, **kwargs):
        return self

    def transform(self, X):
        if self.dtype is None:
            return X
        else:
            return numpy.array(X, dtype=self.dtype)


def check_scaler(scaler):
    """
    Used in neural networks. To unify usage in different neural networks.

    :param scaler: scaler
    :type scaler: str or False or TransformerMixin
    :return: TransformerMixin, scaler
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    transformers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'identity': IdentityTransformer(),
        False: IdentityTransformer()
    }

    if scaler in transformers.keys():
        return transformers[scaler]
    else:
        if not isinstance(scaler, TransformerMixin):
            warnings.warn("Passed scaler wasn't derived from TransformerMixin.")
        return clone(scaler)


def one_hot_transform(y, n_classes=None, dtype='float32'):
    """
    For neural networks, this function needed only in training.
    Classes in 'y' should be [0, 1, 2, .. n_classes -1]
    """
    if n_classes is None:
        n_classes = numpy.max(y) + 1
    target = numpy.zeros([len(y), n_classes], dtype=dtype)
    target[numpy.arange(len(y)), y] = 1
    return target


def remove_first_line(string):
    """
    Returns the copy of string without first line (needed for descriptions which differ in one line)

    :param string: initial string
    :return: copy of string without first line
    """
    return '\n'.join(string.split('\n')[1:])
