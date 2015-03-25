from __future__ import division, print_function, absolute_import
from abc import ABCMeta

from .interface import Classifier, Regressor
from .utils import check_inputs

import neurolab as nl
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, Imputer


__author__ = 'Sterzhanov Vladislav'


def _one_hot_transform(y):
    return np.array(OneHotEncoder(n_values=2).fit_transform(y.reshape((len(y), 1))).todense())


def _min_max_transform(X):
    return MinMaxScaler().fit_transform(Imputer().fit_transform(X))


# TODO: add init functions for all possible networks
# TODO: restructure set: get rid of transforms
NET_TYPES = {'feed-forward':       (nl.net.newff, _min_max_transform, _one_hot_transform),
             'single-layer':       (nl.net.newp, _min_max_transform, _one_hot_transform),
             'competing-layer':    (nl.net.newc, _min_max_transform, _one_hot_transform),
             'learning-vector':    (nl.net.newlvq, _min_max_transform, _one_hot_transform),
             'elman-recurrent':    (nl.net.newelm, _min_max_transform, _one_hot_transform),
             'hemming-recurrent':  (nl.net.newhem),
             'hopfield-recurrent': (nl.net.newhop)}

NET_PARAMS_NAMES = {'minmax', 'cn', 'size', 'transf', 'target',
                    'max_init', 'max_iter', 'delta', 'cn0', 'pc'}


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
    :type initf: nl.init or list[nl.init] of shape [n_layers]
    :param trainf: net train function
    :param dict kwargs: additional arguments to net __init__
    """
    def __init__(self, net_type='feed-forward',
                 features=None,
                 initf=nl.init.init_zeros,
                 trainf=None,
                 **kwargs):
        Classifier.__init__(self, features=features)
        self.train_params = {}
        self.net_params = {}
        self.trainf=trainf
        self.initf=initf
        self._prepare_clf, self._transform_features, self._transform_labels = self._get_initializers(net_type)
        self.set_params(**kwargs)
        self.clf = None
        self.classes_ = None

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise ValueError('sample_weight not supported')

        X, y, sample_weight = check_inputs(X, y, sample_weight)
        self.classes_ = np.unique(y)
        x_train = self._transform_features(self._get_train_features(X))
        y_train = self._transform_labels(y)

        net_params = dict(self.net_params)
        # Hemming and Hopfield networks do not transform input
        if 'net_type' not in {'hemming-recurrent', 'hopfield-recurrent'}:
            net_params['minmax'] = [[0, 1]]*(x_train.shape[1])
        # Output layers for classifiers contain exactly nclasses output neurons
        if 'size' in net_params:
            net_params['size'] = net_params['size'] + [y_train.shape[1]]
        clf = self._prepare_clf(**net_params)

        # To allow similar initf function on all layers
        initf_iterable = self.initf if hasattr(self.initf, '__iter__') else [self.initf]*len(clf.layers)
        for l, f in zip(clf.layers, initf_iterable):
            l.initf = f
            clf.init()

        if self.trainf is not None:
            clf.trainf = self.trainf

        clf.train(x_train, y_train, **self.train_params)

        self.clf = clf
        return self

    def predict_proba(self, X):
        """
        Predict probabilities for each class label on dataset

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :rtype: numpy.array of shape [n_samples, n_classes] with probabilities
        """
        assert self.clf is not None
        return self.clf.sim(self._transform_features(self._get_train_features(X)))

    def staged_predict_proba(self, X):
        """
        Predicts probabilities on each stage

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :return: iterator

        .. warning:: Doesn't support for Neurolab (**AttributeError** will be thrown)
        """
        raise AttributeError("Not supported by Neurolab networks")

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        Additionaly, _prepare_clf, _transform_features and _transform_labels could be set here.
        :param dict params: parameters to set in model
        """
        for name, value in params.items():
            if name in NET_PARAMS_NAMES:
                self.net_params[name] = value
            elif name == 'net_type':
                self._prepare_clf, self._transform_features, self._transform_labels = self._get_initializers(value)
            elif name in {"trainf", "initf", "_prepare_clf", '_transform_features', '_transform_labels'}:
                setattr(self, name, value)
            else:
                self.train_params[name] = value

    def _get_initializers(self, net_type):
        if net_type not in NET_TYPES:
            raise AttributeError('Got unexpected network type: \'{}\''.format(net_type))
        return NET_TYPES.get(net_type)