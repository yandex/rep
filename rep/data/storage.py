"""
This is a wrapper for `pandas.DataFrame`, which allows you to define dataset (data, labels/values, sample weights) for an estimator in a simple way.
"""
from __future__ import division, print_function, absolute_import
import numbers

from numpy.random.mtrand import RandomState
import pandas
import numpy
from sklearn.utils import check_random_state

from ..utils import get_columns_dict, get_columns_in_df


# generating random seeds in the interval [0, RANDINT)
RANDINT = 10000000


class LabeledDataStorage(object):
    def __init__(self, data, target=None, sample_weight=None, random_state=None, shuffle=False):
        """
        This class implements an interface of data for estimators training. It contains data, labels/values and weights -
        all information to train a model.

        :param pandas.DataFrame data: features, array-like of shape [n_samples, n_features]
        :param target: labels/values for classification/regression (set None for the predictive methods)
        :type target: None or numbers.Number or array-like
        :param sample_weight: weight (set None for predictive methods)
        :type sample_weight: None or numbers.Number or array-like
        :param random_state: state for a pseudo random generator
        :type random_state: None or int or RandomState
        :param bool shuffle: shuffle data or not

        """
        self.data = data
        self.target = self._get_key(self.data, target)
        self.sample_weight = self._get_key(self.data, sample_weight, allow_nones=True)
        assert len(self.data) == len(self.target), 'ERROR: Lengths are different for data and target'
        if self.sample_weight is not None:
            assert len(self.data) == len(self.sample_weight), 'ERROR: Lengths are different for data and sample_weight'
        self._random_state = check_random_state(random_state).randint(RANDINT)
        self.shuffle = shuffle
        self._indices = None

    def _get_key(self, ds, key, allow_nones=False):
        """
        Get data from the storage by key.

        :param pandas.DataFrame ds: data
        :param key: key, which describe data in the storage
        :type key: None or numbers.Number or array-like

        :return: data corresponding to the key
        """
        if isinstance(key, str) and ds is not None:
            # assert key in set(ds.columns), self._print_err('ERROR:', '%s is absent in data storage' % key)
            name = list(get_columns_dict([key]).keys())[0]
            return numpy.array(get_columns_in_df(self.data, [key])[name])
        elif isinstance(key, numbers.Number):
            return numpy.array([key] * len(ds))
        else:
            if not allow_nones:
                return numpy.array(key) if key is not None else numpy.ones(len(ds))
            else:
                return numpy.array(key) if key is not None else key

    def __len__(self):
        """
        Return number of samples.

        :return: count of rows in the storage
        :rtype: int
        """
        return len(self.data)

    def get_data(self, features=None):
        """
        Return data.

        :param features: set of feature names (if None then use all features in data storage)
        :type features: None or list[str]

        :rtype: pandas.DataFrame
        """
        df = get_columns_in_df(self.data, features)

        if self.shuffle:
            return df.irow(self.get_indices())
        return df

    def get_targets(self):
        """
        Return sample target, labels or values.

        :rtype: numpy.array
        """
        if self.shuffle:
            return self.target[self.get_indices()]
        return self.target

    def get_weights(self, allow_nones=False):
        """
        Return sample weights.

        :rtype: numpy.array
        """
        if self.sample_weight is None:
            if allow_nones:
                return self.sample_weight
            else:
                return numpy.ones(len(self.data))
        else:
            if self.shuffle:
                return self.sample_weight[self.get_indices()]
            return self.sample_weight

    def get_indices(self):
        """
        Return data indices.

        :rtype: numpy.array
        """
        if self._indices is None:
            rs = RandomState(seed=self._random_state)
            self._indices = rs.permutation(len(self))
        return self._indices

    def col(self, index):
        """
        Return column from the data.

        :param index: names
        :type index: None or str or list(str)

        :rtype: pandas.Series or pandas.DataFrame
        """
        if isinstance(index, str):
            name = list(get_columns_dict([index]).keys())[0]
            return self.get_data([index])[name]
        return self.get_data(index)

    def eval_column(self, expression):
        """
        Evaluate some expression to obtain necessary columns for the data

        :type expression: numbers.Number or array-like or str or function(pandas.DataFrame)
        :rtype: numpy.array or str or
        """
        if isinstance(expression, numbers.Number):
            return numpy.zeros(len(self), dtype=type(expression)) + expression
        elif isinstance(expression, str):
            return numpy.array(self.col(expression))
        elif hasattr(expression, '__call__'):
            return numpy.array(expression(self.get_data()))
        else:
            assert len(expression) == len(self), 'Different length'
            return numpy.array(expression)
