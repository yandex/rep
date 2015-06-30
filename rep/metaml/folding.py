"""
This is specific meta-algorithm based on the idea of cross-validation.
"""
from __future__ import division, print_function, absolute_import

import numpy
from sklearn import clone

from six.moves import zip
from . import utils
from sklearn.cross_validation import KFold
from sklearn.utils.validation import check_random_state
from .factory import train_estimator
from ..estimators.interface import Classifier
from ..estimators.utils import check_inputs

__author__ = 'Tatiana Likhomanenko'


class FoldingClassifier(Classifier):
    """
    This meta-classifier implements folding algorithm:

    * training data is splitted into n equal parts;

    * then n times union of n-1 parts is used to train classifier;

    * at the end we have n-estimators, which are used to classify new events


    To build unbiased predictions for data, pass the **same** dataset (with same order of events)
    as in training to `predict`, `predict_proba` or `staged_predict_proba`, in which case
    classifier will use to predict each event that base classifier which didn't use that event during training.

    To use information from not one, but several classifiers during predictions,
    provide appropriate voting function.

    Parameters:
    -----------
    :param sklearn.BaseEstimator base_estimator: base classifier, which will be used for training
    :param int n_folds: count of folds
    :param features: features used in training
    :type features: None or list[str]
    :param ipc_profile: profile for IPython cluster, None to compute locally.
    :type ipc_profile: None or str
    :param random_state: random state for reproducibility
    :type random_state: None or int or RandomState
    """

    def __init__(self,
                 base_estimator,
                 n_folds=2,
                 random_state=None,
                 features=None,
                 ipc_profile=None):
        super(FoldingClassifier, self).__init__(features=features)

        self.estimators = []
        self.ipc_profile = ipc_profile
        self.n_folds = n_folds
        self.base_estimator = base_estimator
        self._folds_indices = None
        self.random_state = random_state
        self._random_number = None

    def _get_folds_column(self, length):
        if self._random_number is None:
            self._random_number = check_random_state(self.random_state).randint(0, 100000)
        folds_column = numpy.zeros(length)
        for fold_number, (_, folds_indices) in enumerate(
                KFold(length, self.n_folds, shuffle=True, random_state=self._random_number)):
            folds_column[folds_indices] = fold_number
        return folds_column

    def fit(self, X, y, sample_weight=None):
        """
        Train the classifier, will train several base classifiers on overlapping
        subsets of training dataset.

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :param y: labels of events - array-like of shape [n_samples]
        :param sample_weight: weight of events,
               array-like of shape [n_samples] or None if all weights are equal
        """
        X, y, sample_weight = check_inputs(X, y, sample_weight=sample_weight, allow_none_weights=True)
        X = self._get_features(X)
        self._set_classes(y)
        folds_column = self._get_folds_column(len(X))

        for _ in range(self.n_folds):
            self.estimators.append(clone(self.base_estimator))
            try:
                self.estimators[-1].set_params(features=self.features)
            except ValueError:
                pass

        if sample_weight is None:
            weights_iterator = (None for _ in range(self.n_folds))
        else:
            weights_iterator = (sample_weight[folds_column != index] for index in range(self.n_folds))

        result = utils.map_on_cluster(self.ipc_profile, train_estimator,
                                      range(len(self.estimators)),
                                      self.estimators,
                                      (X.iloc[folds_column != index, :].copy() for index in range(self.n_folds)),
                                      (y[folds_column != index] for index in range(self.n_folds)),
                                      weights_iterator)
        for status, data in result:
            if status == 'success':
                name, classifier, spent_time = data
                self.estimators[name] = classifier
            else:
                print('Problem while training on the node, report:\n', data)
        return self

    def _get_estimators_proba(self, estimator, data):
        try:
            return estimator.predict_proba(data)
        except AttributeError:
            probabilities = numpy.zeros(shape=(len(data), self.n_classes_))
            labels = estimator.predict(data)
            probabilities[numpy.arange(len(labels)), labels] = 1
            return probabilities

    def predict(self, X, vote_function=None):
        """
        Predict labels. To get unbiased predictions, you can pass training dataset
        (with same order of events) and vote_function=None.

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :param vote_function: function to combine prediction of folds' estimators.
            If None then folding scheme is used. Parameters: numpy.ndarray [n_classifiers, n_samples]
        :type vote_function: None or function, if None, will use folding scheme.
        :rtype: numpy.array of shape [n_samples, n_classes] with labels
        """
        proba = self.predict_proba(X, vote_function=vote_function)
        return self.classes_.take(numpy.argmax(proba, axis=1), axis=0)

    def predict_proba(self, X, vote_function=None):
        """
        Predict probabilities. To get unbiased predictions, you can pass training dataset
        (with same order of events) and vote_function=None.

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :param vote_function: function to combine prediction of folds' estimators.
            If None then self.vote_function is used. Parameters: numpy.ndarray [n_classifiers, n_samples, n_classes]
        :type vote_function: None or function

        :rtype: numpy.array of shape [n_samples, n_classes] with probabilities
        """
        if vote_function is not None:
            print('Using voting KFold prediction')
            X = self._get_features(X)
            probabilities = []
            for classifier in self.estimators:
                probabilities.append(self._get_estimators_proba(classifier, X))
            # probabilities: [n_classifiers, n_samples, n_classes], reduction over 0th axis
            probabilities = numpy.array(probabilities)
            return vote_function(probabilities)
        else:
            print('KFold prediction using folds column')
            X = self._get_features(X)
            folds_column = self._get_folds_column(len(X))
            probabilities = numpy.zeros(shape=(len(X), self.n_classes_))
            for fold in range(self.n_folds):
                prob = self._get_estimators_proba(self.estimators[fold], X.iloc[folds_column == fold, :])
                probabilities[folds_column == fold] = prob
            return probabilities

    def staged_predict_proba(self, X, vote_function=None):
        """
        Predict probabilities on each stage. To get unbiased predictions, you can pass training dataset
        (with same order of events) and vote_function=None.

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :param vote_function: function to combine prediction of folds' estimators.
            If None then self.vote_function is used.
        :type vote_function: None or function

        :return: iterator for numpy.array of shape [n_samples, n_classes] with probabilities
        """
        if vote_function is not None:
            print('Using voting KFold prediction')
            X = self._get_features(X)
            iterators = [estimator.staged_predict_proba(X) for estimator in self.estimators]
            for fold_prob in zip(*iterators):
                probabilities = numpy.array(fold_prob)
                yield vote_function(probabilities)
        else:
            print('Default prediction')
            X = self._get_features(X)
            folds_column = self._get_folds_column(len(X))
            iterators = [self.estimators[fold].staged_predict_proba(X.iloc[folds_column == fold, :])
                         for fold in range(self.n_folds)]
            for fold_prob in zip(*iterators):
                probabilities = numpy.zeros(shape=(len(X), 2))
                for fold in range(self.n_folds):
                    probabilities[folds_column == fold] = fold_prob[fold]
                yield probabilities
