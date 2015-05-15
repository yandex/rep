"""
This module allows to train estimators on datasets splitting by special feature.
"""
from __future__ import division, print_function, absolute_import
import numpy
import pandas
from sklearn.base import clone
from ..estimators import Classifier
from ..estimators.utils import check_inputs


__author__ = 'Alex Rogozhnikov'


class FeatureSplitter(Classifier):
    """
    Dataset is splitted by values of split_feature

    :param str split_feature: the name of key feature
    :param base_estimator: the classifier, its' copies are trained on parts of dataset
    :param list[str] features: list of columns classifier uses
    """
    def __init__(self, split_feature, base_estimator):
        self.base_estimator = base_estimator
        self.split_feature = split_feature
        Classifier.__init__(self, features=None)

    def fit(self, X, y, sample_weight=None):
        # Not taking training features - they should be passed to children classifier
        X, y, sample_weight = check_inputs(X, y, sample_weight=sample_weight, allow_none_weights=False)
        self._set_classes(y)

        assert isinstance(X, pandas.DataFrame), 'passed object was not pandas.DataFrame'
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
        result = numpy.zeros([len(X), self.n_classes_])
        for value, estimator in self.base_estimators.items():
            mask = numpy.array(X[self.split_feature]) == value
            result[mask, :] = estimator.predict_proba(X.loc[mask, :])
        return result

    def staged_predict_proba(self, X):
        raise NotImplementedError('TODO implement')