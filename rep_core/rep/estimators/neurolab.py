from __future__ import division, print_function, absolute_import
from abc import ABCMeta

from .interface import Classifier, Regressor
from .utils import check_inputs

import neurolab as nl
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, Imputer


__author__ = 'Sterzhanov Vladislav'


class NeurolabClassifier(Classifier):
    """
    NeurolabClassifier is wrapper on Neurolab network-like classifiers

    Parameters:
    -----------
    :param string net_type: type of network
    One of {'feed-forward', 'single-layer', 'competing-layer', 'learning-vector',
            'elman-recurrent', 'hopfield-recurrent', 'hemming-recurrent'}
    :param features: features used in training
    :type features: list[str] or None
    :param initf: layer initializers
    :type initf: nl.init or #TODO list[nl.init] of shape [n_layers]
    :param trainf: net train function
    :param epochs: number of epochs to train
    :param show: verbose step
    :param dict kwargs: additional arguments to net __init__
    #TODO add documentation for altered parameters
    """
    def __init__(self, net_type='feed-forward',
                 features=None,
                 initf=nl.init.init_zeros,
                 trainf=nl.train.train_rprop,
                 epochs=10,
                 show=0,
                 **kwargs):
        Classifier.__init__(self, features=features)
        # TODO: add init functions for all possible networks
        self.NET_TYPES = {'feed-forward': (_init_ff, _min_max_transform, _one_hot_transform), 'single-layer': nl.net.newp,
                          'competing-layer': nl.net.newc, 'learning-vector': nl.net.newlvq,
                          'elman-recurrent': nl.net.newelm, 'hopfield-recurrent': nl.net.newhop,
                          'hemming-recurrent': nl.net.newhem}
        self.train_params = {'initf': initf, 'trainf': trainf, 'epochs': epochs, 'show': show}
        self.net_params = kwargs
        self._prepare_clf, self._transform_features, self._transform_labels = self._get_initializers(net_type)
        self.clf = None
        self.classes_ = None

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            # TODO: support?
            raise ValueError('sample_weight not supported')

        self.classes_ = np.unique(y)

        x_train = self._transform_features(self._get_train_features(X))
        y_train = self._transform_labels(y)
        clf = self._prepare_clf(x_train, y_train, **self.net_params)

        for l in clf.layers:
            # TODO allow for multiple initf functions
            l.initf = self.train_params['initf']
            clf.init()
        clf.trainf = self.train_params['trainf']

        clf.train(x_train, y_train,
                  epochs=self.train_params['epochs'], show=self.train_params['show'])
        self.clf = clf

    def predict_proba(self, X):
        """
        Predict probabilities for each class label on dataset

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :rtype: numpy.array of shape [n_samples, n_classes] with probabilities
        """
        return self.clf.sim(self._transform_features(self._get_train_features(X)))

    def staged_predict_proba(self, X):
        """
        Predicts probabilities on each stage

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :return: iterator

        .. warning:: Doesn't support for Neurolab (**AttributeError** will be thrown)
        """
        # TODO: find a way to implement.
        raise AttributeError("Not supported for Neurolab networks?")

    def set_params(self, **params):
        """
        #TODO: Allow alter _prepare_clf, _transform_features and _transform_labels?
        Set the parameters of this estimator.

        :param dict params: parameters to set in model
        """
        for name, value in params.items():
            if name in self.train_params:
                self.train_params[name] = value
            elif name == 'net_type':
                self._prepare_clf, self._transform_features, self._transform_labels = self._get_initializers(value)
            else:
                self.net_params[name] = value

    def _get_initializers(self, net_type):
        if net_type not in self.NET_TYPES:
            raise AttributeError('Got unexpected network type: \'{}\''.format(net_type))
        return self.NET_TYPES.get(net_type)


def _one_hot_transform(y):
    return np.array(OneHotEncoder(n_values=2).fit_transform(y.reshape((len(y), 1))).todense())


def _min_max_transform(X):
    return MinMaxScaler().fit_transform(Imputer().fit_transform(X))


def _init_ff(X, y, layers_size=[300], transf=None):
        init = []
        for _ in range(X.shape[1]):
            init.append([0, 1])
        clf = nl.net.newff(init, [X.shape[1]] + layers_size + [len(np.unique(y))], transf)
        return clf