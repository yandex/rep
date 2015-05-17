"""
This module contains stacking strategies (meta-algorithms of machine learning).
"""
from __future__ import division, print_function, absolute_import

import numpy
from sklearn.base import clone

from ..estimators import Classifier
from ..estimators.utils import check_inputs


__author__ = 'Alex Rogozhnikov'


class FeatureSplitter(Classifier):
    """
    Dataset is split by values of `split_feature`,
     for each value of feature, new classifier is trained.

    When building predictions, classifier predicts the events with
     the same value of `split_feature` it was trained on.

    :param str split_feature: the name of key feature,
    :param base_estimator: the classifier, its' copies are trained on parts of dataset
    :param list[str] features: list of columns classifier uses
    """
    def __init__(self, split_feature, base_estimator, features=None):
        self.base_estimator = base_estimator
        self.split_feature = split_feature
        Classifier.__init__(self, features=features)

    def fit(self, X, y, sample_weight=None):
        """
        Fit dataset.
        :param X: pandas.DataFrame of shape [n_samples, n_features] with features
        :param y: array-like of shape [n_samples] with targets
        :param sample_weight: array-like of shape [n_samples] with events weights or None.
        :return: self, trained classifier.
        """
        # Not taking training features - they should be passed to children classifier
        X, y, sample_weight = check_inputs(X, y, sample_weight=sample_weight, allow_none_weights=False)
        X = self._get_train_features(X)
        self._set_classes(y)
        # TODO cover the case of missing labels in subsets.
        split_column_values = numpy.array(X[self.split_feature])
        self.base_estimators = {}
        for value in numpy.unique(split_column_values):
            rows = numpy.array(split_column_values) == value
            base_classifier = clone(self.base_estimator)
            if sample_weight is None:
                base_classifier.fit(X.loc[rows, :], y[rows])
            else:
                base_classifier.fit(X.loc[rows, :], y[rows], sample_weight=sample_weight[rows])
            self.base_estimators[value] = base_classifier
        return self

    def predict_proba(self, X):
        """
        Predict probabilities. Each event will be predicted by the classifier
         with trained on corresponding value of `split_feature`

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :return: probabilities of shape [n_samples, n_classes]
        """
        result = numpy.zeros([len(X), self.n_classes_])
        for value, estimator in self.base_estimators.items():
            mask = numpy.array(X[self.split_feature]) == value
            result[mask, :] = estimator.predict_proba(X.loc[mask, :])
        return result

    def staged_predict_proba(self, X):
        """
        Predict probabilities after each stage of base classifier.
        Each event will be predicted by the classifier
         with trained on corresponding value of `split_feature`

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :return: iterable sequence of numpy.arrays of shape [n_samples, n_classes]
        """
        result = numpy.zeros([len(X), self.n_classes_])
        masks_iterators = []
        for value, estimator in self.base_estimators.items():
            mask = numpy.array(X[self.split_feature]) == value
            prediction_iterator = estimator.staged_predict_proba(X.loc[mask, :])
            masks_iterators.append([mask, prediction_iterator])
        try:
            while True:
                for mask, prediction_iterator in masks_iterators:
                    result[mask, :] = next(prediction_iterator)
                yield result
        except StopIteration:
            pass
