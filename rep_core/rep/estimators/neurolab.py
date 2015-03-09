from __future__ import division, print_function, absolute_import
from abc import ABCMeta

from .interface import Classifier, Regressor
from .utils import check_inputs

import neurolab as nl
import numpy as np


__author__ = 'Sterzhanov Vladislav'


class NeurolabClassifier(Classifier):
    """
    NeurolabClassifier is wrapper on Neurolab network-like **classifiers**

    Parameters:
    -----------
    :param neurolab.Net clf: your neural network, which will be trained and used for classification
    :param features: features used in training
    :type features: list[str] or None
    :param initf: layer initializers
    :type initf: nl.init or #TODO list[nl.init] of shape [n_layers]
    :param trainf: net train function
    :param epochs: number of epochs to train
    :param show: verbose flag
    """
    def __init__(self, clf, features=None,
                 initf=nl.init.init_zeros,
                 trainf=nl.train.train_rprop,
                 epochs=10,
                 show=0):
        if isinstance(clf, Classifier):
            raise ValueError('Base class should be simply sklearn classifier, not descendant of Classifier')
        Classifier.__init__(self, features=features)
        for l in clf.layers:
            l.initf = initf
            clf.init()
        clf.trainf = trainf
        self.epochs = epochs
        self.show = show
        self.clf = clf

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise ValueError('sample_weight not supported')
        self.clf.train(self._get_train_features(X), y, epochs=self.epochs, show=self.show)
        self.classes_ = np.unique(y)

    def predict_proba(self, X):
        """
        Predict probabilities for each class label on dataset

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :rtype: numpy.array of shape [n_samples, n_classes] with probabilities
        """
        return self.clf.sim(self._get_train_features(X))

    def staged_predict_proba(self, X):
        """
        Predicts probabilities on each stage

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :return: iterator

        .. warning:: Doesn't support for Neurolab (**AttributeError** will be thrown)
        """
        raise AttributeError("Not supported for Neurolab networks?")