from __future__ import division, print_function, absolute_import

from abc import ABCMeta

from .interface import Classifier, Regressor
from .utils import check_inputs

from copy import deepcopy

import neurolab as nl
import numpy as np
import scipy

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone


__author__ = 'Sterzhanov Vladislav'


def _one_hot_transform(y):
    return np.array(OneHotEncoder(n_values=len(np.unique(y))).fit_transform(y.reshape((len(y), 1))).todense())


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


class NeurolabBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, net_type, initf,
                 trainf, **kwargs):
        self.train_params = {}
        self.net_params = {}
        self.trainf = trainf
        self.initf = initf
        self.net_type = net_type
        self.net = None
        self.scaler = StandardScaler()
        self.set_params(**kwargs)

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

    def _fit(self, x_train, y_train, **net_params):
        net = self._prepare_net(**net_params)

        # To allow similar initf function on all layers
        initf_iterable = self.initf if hasattr(self.initf, '__iter__') else [self.initf]*len(net.layers)
        for l, f in zip(net.layers, initf_iterable):
            l.initf = f
            net.init()

        if self.trainf is not None:
            net.trainf = self.trainf

        net.train(x_train, y_train, **self.train_params)
        return net

    def _sim(self, X):
        assert self.net is not None, 'Classifier not fitted, predict denied'
        transformed_x = self._transform_input(X, fit=False)
        assert np.all((transformed_x <= 1) & (transformed_x >= 0)), \
            'Transformer returned illegal features (out of [0,1] range)'
        return self.net.sim(transformed_x)

    def _transform_input(self, X, y=None, fit=True):
        # FIXME: Need this while using sklearn < 0.16
        X = np.copy(X)
        if fit:
            self.scaler = clone(self.scaler)
            self.scaler.fit(X, y)
        return scipy.special.expit(self.scaler.transform(X) / 3)

    def _prepare_net(self, **net_params):
        init = self._get_initializers(self.net_type)
        return init(**net_params)

    @staticmethod
    def _get_initializers(net_type):
        if net_type not in NET_TYPES:
            raise AttributeError('Got unexpected network type: \'{}\''.format(net_type))
        return NET_TYPES.get(net_type)


class NeurolabRegressor(NeurolabBase, Regressor):
    """
    NeurolabRegressor is a wrapper on Neurolab network-like regressors

    Parameters:
    -----------
    :param string net_type: type of network
    One of {'feed-forward', 'single-layer', 'competing-layer', 'learning-vector',
            'elman-recurrent', 'hopfield-recurrent', 'hemming-recurrent'}
    :param features: features used in training
    :type features: list[str] or None
    :param initf: layer initializers
    :type initf: anything implementing call(layers). e.g. nl.init.* or list[nl.init.*] of shape [n_layers]
    :param trainf: net train function
    :param scaler: transformer to apply to the input objects
    :param dict kwargs: additional arguments to net __init__
    """

    def __init__(self, net_type='feed-forward',
                 features=None,
                 initf=nl.init.init_zeros,
                 trainf=None,
                 **kwargs):
        Regressor.__init__(self, features=features)
        NeurolabBase.__init__(self, net_type=net_type, initf=initf, trainf=trainf, **kwargs)

    def fit(self, X, y):
        # TODO Some networks do not support regression?

        X, y, _ = check_inputs(X, y, None)

        x_train = self._transform_input(self._get_train_features(X), y)
        y_train = y.reshape(len(y), 1)

        net_params = self._prepare_parameters_for_regression(self.net_params, x_train, y_train)

        net = self._fit(x_train, y_train, **net_params)

        self.net = net
        return self

    def predict(self, X):
        """
        Predict model

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :return: numpy.array of shape n_samples with values
        """
        # FIXME: Dirty hack 'cause there is no base object defining _get_train_features that
        # Classisifier and Regressor would inherit
        return self._sim(self._get_train_features(X))

    def staged_predict(self, X, step=10):
        """
        Predicts probabilities on each stage

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param int step: step for returned iterations
        :return: iterator
        .. warning:: Doesn't have support in Neurolab (**AttributeError** will be thrown)
        """
        raise AttributeError("Not supported by Neurolab networks")

    def _prepare_parameters_for_regression(self, params, x_train, y_train):
        # TODO
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
            net_params['cn'] = 1

        # Output layers of regressors contain exactly 1 output neuron
        if 'size' in net_params:
            net_params['size'] += [1]

        return net_params


class NeurolabClassifier(NeurolabBase, Classifier):
    """
    NeurolabClassifier is a wrapper on Neurolab network-like classifiers

    Parameters:
    -----------
    :param string net_type: type of network
    One of {'feed-forward', 'single-layer', 'competing-layer', 'learning-vector',
            'elman-recurrent', 'hopfield-recurrent', 'hemming-recurrent'}
    :param features: features used in training
    :type features: list[str] or None
    :param initf: layer initializers
    :type initf: anything implementing call(layers). e.g. nl.init.* or list[nl.init.*] of shape [n_layers]
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
        NeurolabBase.__init__(self, net_type=net_type, initf=initf, trainf=trainf, **kwargs)
        self.classes_ = None

    def fit(self, X, y):
        # Some networks do not support classification
        assert self.net_type not in CANT_CLASSIFY, 'Network type does not support classification'

        X, y, _ = check_inputs(X, y, None)

        x_train = self._transform_input(self._get_train_features(X), y)
        y_train = _one_hot_transform(y)

        self.classes_ = self._get_classes(y, y_train)
        net_params = self._prepare_parameters_for_classification(self.net_params, x_train, y_train)

        net = self._fit(x_train, y_train, **net_params)

        self.net = net
        return self

    def predict_proba(self, X):
        """
        Predict probabilities for each class label on dataset

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :rtype: numpy.array of shape [n_samples, n_classes] with probabilities
        """

        # FIXME: Dirty hack 'cause there is no base object defining _get_train_features that
        # Classisifier and Regressor would inherit
        return self._sim(self._get_train_features(X))

    def staged_predict_proba(self, X):
        """
        Predicts probabilities on each stage

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :return: iterator

        .. warning:: Doesn't have support in Neurolab (**AttributeError** will be thrown)
        """
        raise AttributeError("Not supported by Neurolab networks")

    def _get_classes(self, y, y_transformed):
        # I found no proved evidence that OHE is guaranteed to encode values in accordance to their order
        # So this hack appeared
        cl = np.zeros(len(np.unique(y)))
        for true_cl, enc in zip(y, y_transformed):
            cl[true_cl] = np.argmax(enc)
        return cl

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