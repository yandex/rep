"""
There are interfaces for **classification** and **regression** wrappers.
"""
from __future__ import division, print_function, absolute_import
from abc import ABCMeta, abstractmethod
import logging

import numpy
import pandas

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from .utils import _get_features

__author__ = 'Tatiana Likhomanenko, Alex Rogozhnikov'

logger = logging.getLogger(__name__)


class Classifier(BaseEstimator, ClassifierMixin):
    """
    Interface to train different **classification** model from different
    machine learning libraries, like **Sklearn, TMVA, XGBoost**...

    :param features: features used to train model
    :type features: list[str] or None

    .. note::
        * Classes must be from 0 to n_classes-1!!!

        * if `features` aren't set (**None**), then all features in training dataset will be used

        * Datasets should be `pandas.DataFrame`, `not numpy.array`.
          Provided this, you'll be able to choose features used in training by setting e.g.
          `features=['FlightTime', 'p']` in constructor.

        * It works fine with `numpy.array` as well, but in this case all the features will be used.
    """
    __metaclass__ = ABCMeta

    def __init__(self, features=None):
        self.features = list(features) if features is not None else features

    def _get_features(self, X, allow_nans=False):
        """
        :param pandas.DataFrame X: train dataset

        :return: pandas.DataFrame with used features
        """
        X_prepared, self.features = _get_features(self.features, X, allow_nans=allow_nans)
        return X_prepared

    def _set_classes(self, y):
        self.classes_, indices = numpy.unique(y, return_index=True)
        self.n_classes_ = len(self.classes_)
        assert self.n_classes_ >= 2, "Number of labels must be >= 2 (data contain {})".format(self.n_classes_)
        assert numpy.all(self.classes_ == numpy.arange(self.n_classes_)), \
            'Labels must be from 0..n_classes-1, instead of {}'.format(self.classes_)
        return indices

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """
        Train the classifier model on dataset

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :param y: labels of events - array-like of shape [n_samples]
        :param sample_weight: weight of events,
               array-like of shape [n_samples] or None if all weights are equal

        :return: self
        """
        pass

    def predict(self, X):
        """
        Predict labels for all events in dataset

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :rtype: numpy.array of shape [n_samples] with integer labels
        """
        proba = self.predict_proba(X)
        return self.classes_.take(numpy.argmax(proba, axis=1), axis=0)

    @abstractmethod
    def predict_proba(self, X):
        """
        Predict probabilities for each class label on dataset

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :rtype: numpy.array of shape [n_samples, n_classes] with probabilities
        """
        pass

    @abstractmethod
    def staged_predict_proba(self, X):
        """
        Predicts probabilities on each stage

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :rtype: iterator
        """
        pass

    def get_feature_importances(self):
        """
        Get features importance

        :rtype: pandas.DataFrame with `index=self.features`
        """
        try:
            return pandas.DataFrame({"effect": self.feature_importances_}, index=self.features)
        except AttributeError:
            raise AttributeError("Haven't feature_importances_ property")

    def fit_lds(self, lds):
        """
        Train the classifier on specific type dataset

        :param LabeledDataStorage lds: data

        :return: self
        """
        X, y, sample_weight = lds.get_data(self.features), lds.get_targets(), lds.get_weights(allow_nones=True)
        return self.fit(X, y, sample_weight=sample_weight)

    def test_on_lds(self, lds):
        """
        Prepare classification report for a single classifier

        :param LabeledDataStorage lds: data
        :return: ClassificationReport
        """
        from ..report import ClassificationReport
        return ClassificationReport(classifiers={'clf': self}, lds=lds)

    def test_on(self, X, y, sample_weight=None):
        """
        Prepare classification report for a single classifier

        :param X: data, pandas.DataFrame
        :param y: target
        :param sample_weight: weights, optional.
        :return: ClassificationReport
        """
        from ..data import LabeledDataStorage
        lds = LabeledDataStorage(data=X, target=y, sample_weight=sample_weight)
        return self.test_on_lds(lds=lds)


class Regressor(BaseEstimator, RegressorMixin):
    """
    Interface to train different **regression** model from different
     machine learning libraries, like **TMVA, Sklearn, XGBoost**...

    :param features: features used to train model
    :type features: list[str] or None

    .. note::
        * if `features` aren't set (**None**), then all features in training dataset will be used

        * Datasets should be `pandas.DataFrame`, `not numpy.array`.
          Provided this, you'll be able to choose features used in training by setting e.g.
          `features=['FlightTime', 'p']` in constructor.

        * It works fine with `numpy.array` as well, but in this case all the features will be used.
    """
    __metaclass__ = ABCMeta

    def __init__(self, features=None):
        self.features = list(features) if features is not None else features

    def _get_features(self, X, allow_nans=False):
        """
        :param pandas.DataFrame X: train dataset

        :return: pandas.DataFrame with used features
        """
        X_prepared, self.features = _get_features(self.features, X, allow_nans=allow_nans)
        return X_prepared

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """
        Train the regressor model

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :param y: values - array-like of shape [n_samples]
        :param sample_weight: weight of events,
               array-like of shape [n_samples] or None if all weights are equal

        :return: self
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict values for all events in dataset.

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :rtype: numpy.array of shape [n_samples] with predicted values
        """
        pass

    @abstractmethod
    def staged_predict(self, X):
        """
        Predicts values on each stage

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :rtype: iterator
        """
        pass

    def fit_lds(self, lds):
        """
        Train the regressor model on specific dataset

        :param LabeledDataStorage lds: data

        :return: self
        """
        X, y, sample_weight = lds.get_data(self.features), lds.get_targets(), lds.get_weights()
        if sample_weight is None:
            return self.fit(X, y)
        else:
            return self.fit(X, y, sample_weight=sample_weight)

    def get_feature_importances(self):
        """
        Get features importances

        :rtype: pandas.DataFrame with `index=self.features`
        """
        try:
            return pandas.DataFrame({"effect": self.feature_importances_}, index=self.features)
        except AttributeError:
            raise AttributeError("Classifier doesn't provide feature_importances_ property")

    def test_on_lds(self, lds):
        """
        Prepare regression report for a single classifier

        :param LabeledDataStorage lds: data
        :return: RegressionReport
        """
        from ..report import RegressionReport
        return RegressionReport(regressors={'clf': self}, lds=lds)

    def test_on(self, X, y, sample_weight=None):
        """
        Prepare regression report for a single classifier

        :param X: data, pandas.DataFrame
        :param y: target
        :param sample_weight: weights, optional.
        :return: RegressionReport
        """
        from ..data import LabeledDataStorage
        lds = LabeledDataStorage(data=X, target=y, sample_weight=sample_weight)
        return self.test_on_lds(lds=lds)

