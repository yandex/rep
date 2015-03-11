#  File encoding: utf-8

from __future__ import division, print_function, absolute_import
from abc import ABCMeta

from .interface import Classifier, Regressor
from .utils import check_inputs

__author__ = 'Alexey Berdnikov'

class NolearnClassifier(Classifier):
    def __init__(self, clf, features=None):
        # Что, если clf.layer_sizes[0] не равно len(features)?
        self.clf = clf
        Classifier.__init__(self, features=features)

    def fit(self, X, y, sample_weight=None):
        X, y, sample_weight = check_inputs(X, y, sample_weight=sample_weight, allow_none_weights=True)
        if sample_weight is not None:
            raise ValueError("The sample_weight parameter is not supported for nolearn")       
        return self.clf.fit(self._get_train_features(X).values, y)

    def predict(self, X):
        return self.clf.predict(self._get_train_features(X).values)

    def predict_proba(self, X):
        return self.clf.predict_proba(self._get_train_features(X).values)

    def staged_predict_proba(self, X):
        raise AttributeError("Not supported for nolearn")

    def score(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise ValueError("The sample_weight parameter is not supported for nolearn")
        return self.clf.score(self._get_train_features(X).values, y)

