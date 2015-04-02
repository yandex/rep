from __future__ import division, print_function, absolute_import
from abc import ABCMeta

from .interface import Classifier, Regressor
from .utils import check_inputs

from copy import deepcopy

import neurolab as nl
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, Imputer


__author__ = 'Sterzhanov Vladislav'


def _one_hot_transform(y):
    return np.array(OneHotEncoder().fit_transform(y.reshape((len(y), 1))).todense())


NET_TYPES = {'feed-forward':       nl.net.newff,
             'single-layer':       nl.net.newp,
             'competing-layer':    nl.net.newc,
             'learning-vector':    nl.net.newlvq,
             'elman-recurrent':    nl.net.newelm,
             'hemming-recurrent':  nl.net.newhem,
             'hopfield-recurrent': nl.net.newhop}

NET_PARAMS = ('minmax', 'cn', 'layers', 'transf', 'target',
                    'max_init', 'max_iter', 'delta', 'cn0', 'pc')

BASIC_PARAMS = ('net_type', 'trainf', 'initf', 'scaler')

CANT_CLASSIFY = ('learning-vector', 'hopfield-recurrent', 'competing-layer', 'hemming-recurrent')


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
    :param scaler: transformer to apply to the input objects
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
        self.net_type = net_type
        self.clf = None
        self.classes_ = None
        self.scaler = MinMaxScaler()
        self.set_params(**kwargs)

    def fit(self, X, y):
        X, y, _ = check_inputs(X, y, None)

        _prepare_clf = self._get_initializers(self.net_type)

        self.classes_ = np.unique(y)
        x_train = self._transform_input(self._get_train_features(X), fit=1)
        y_train = _one_hot_transform(y)

        # Some networks do not support classification
        assert self.net_type not in CANT_CLASSIFY, 'Network type does not support classification'

        net_params = self._prepare_parameters_for_classification(self.net_params, x_train, y_train)

        clf = _prepare_clf(**net_params)

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
        return self.clf.sim(self._transform_input(self._get_train_features(X), fit=0))

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
        Set the parameters of this estimator
        :param dict params: parameters to set in model
        """
        for name, value in params.items():
            if name in {'random_state'}:
                continue
            if name in NET_PARAMS:
                self.net_params[name] = value
            elif name in BASIC_PARAMS:
                setattr(self, name, value)
            else:
                self.train_params[name] = value

    def get_params(self, deep=True):
        """
        Get parameters of this estimator
        :return dict
        """
        parameters = deepcopy(self.net_params)
        parameters.update(self.train_params)
        for name in BASIC_PARAMS:
            parameters[name] = getattr(self, name)
        return parameters

    def _transform_input(self, X, fit=1):
        if fit:
            self.scaler.fit(X)
        return self.scaler.transform(X)

    @staticmethod
    def _get_initializers(net_type):
        if net_type not in NET_TYPES:
            raise AttributeError('Got unexpected network type: \'{}\''.format(net_type))
        return NET_TYPES.get(net_type)

    def _prepare_parameters_for_classification(self, params, x_train, y_train):
        net_params = deepcopy(params)

        # Network expects features to be [0, 1]-scaled
        net_params['minmax'] = [[0, 1]]*(x_train.shape[1])

        # To unify the layer-description argument with other supported networks
        if 'layers' in net_params:
            net_params['size'] = net_params['layers']
            net_params.pop('layers')

        # For some reason Neurolab asks for a separate cn parameter instead of accessing size[-1]
        # (e.g. In case of Single-Layer Perceptron)
        if 'cn' in net_params:
            net_params['cn'] = len(self.classes_)

        # Output layers of classifiers contain exactly nclasses output neurons
        if 'size' in net_params:
            net_params['size'] += [y_train.shape[1]]

        # Classification networks should have SoftMax as the transfer function on output layer
        if 'transf' not in net_params:
            net_params['transf'] = \
                [nl.trans.TanSig()] * len(net_params['size']) if 'size' in net_params else nl.trans.SoftMax()

        if hasattr(net_params['transf'], '__iter__'):
            net_params['transf'][-1] = nl.trans.SoftMax()
        else:
            net_params['transf'] = nl.trans.SoftMax()

        return net_params