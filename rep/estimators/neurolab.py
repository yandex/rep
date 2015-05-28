# Copyright 2014-2015 Yandex LLC and contributors <https://yandex.com/>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# <http://www.apache.org/licenses/LICENSE-2.0>
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division, print_function, absolute_import
from abc import ABCMeta

from .interface import Classifier, Regressor
from .utils import check_inputs

from copy import deepcopy, copy

import neurolab as nl
import numpy as np
import scipy

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone

__author__ = 'Vlad Sterzhanov'


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

    def __init__(self, net_type, initf, trainf, **kwargs):
        self.train_params = {}
        self.net_params = {}
        self.trainf = trainf
        self.initf = initf
        self.net_type = net_type
        self.net = None
        self.scaler = None
        self.set_params(**kwargs)

    def set_params(self, **params):
        """
        Set the parameters of this estimator
        :param dict params: parameters to set in model
        """
        if 'scaler' in params:
            scaler = params['scaler']
            self.scaler = (StandardScaler() if scaler is None else scaler)
            params.pop('scaler')

        for name, value in params.items():
            if name in {'random_state'}:
                continue
            if name.startswith("scaler__"):
                assert hasattr(self.scaler, 'set_params'), \
                    "Trying to set {} without scaler".format(name)
                self.scaler.set_params({name[len("scaler__"):]: value})
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
        parameters.update(deepcopy(self.train_params))
        for name in BASIC_PARAMS:
            parameters[name] = getattr(self, name)
        return parameters

    def _fit(self, X, y, y_train):
        x_train = self._transform_input(self._get_train_features(X), y)

        # Prepare parameters depending on network purpose (classification \ regression)
        net_params = self._prepare_params(self.net_params, x_train, y_train)

        init = self._get_initializers(self.net_type)
        net = init(**net_params)

        # To allow similar initf function on all layers
        initf_iterable = self.initf if hasattr(self.initf, '__iter__') else [self.initf]*len(net.layers)
        for l, f in zip(net.layers, initf_iterable):
            l.initf = f
            net.init()

        if self.trainf is not None:
            net.trainf = self.trainf

        net.train(x_train, y_train, **self.train_params)

        self.net = net
        return self

    def _sim(self, X):
        assert self.net is not None, 'Classifier not fitted, predict denied'
        transformed_x = self._transform_input(X, fit=False)
        return self.net.sim(transformed_x)

    def _transform_input(self, X, y=None, fit=True):
        if self.scaler is False:
            return X
        # FIXME: Need this while using sklearn < 0.16
        X = np.copy(X)
        if fit:
            self.scaler = clone(self.scaler)
            self.scaler.fit(X, y)
        # HACK: neurolab requires all features (even those of predicted objects) to be in [min, max]
        # so this dark magic appeared, seems to work ok for most reasonable usecases
        return scipy.special.expit(self.scaler.transform(X) / 3)

    def _prepare_params(self, net_params, x_train, y_train):
        params = deepcopy(net_params)
        # Network expects features to be [0, 1]-scaled
        params['minmax'] = [[0, 1]]*(x_train.shape[1])

        # To unify the layer-description argument with other supported networks
        if 'layers' in params:
            params['size'] = params['layers']
            params.pop('layers')

        # For some reason Neurolab asks for a separate cn parameter instead of accessing size[-1]
        # (e.g. In case of Single-Layer Perceptron)
        if 'cn' in params:
            params['cn'] = y_train.shape[1]

        # Set output layer size
        if 'size' in params:
            params['size'] += [y_train.shape[1]]

        return params

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
    :type scaler: sklearn-like scaler or False (do not scale features -- use with care and keep track of minmax param)
    :param list layers: list of numbers denoting size of each hidden layer
    :param dict kwargs: additional arguments to net __init__, varies with different net_types
                        See: https://pythonhosted.org/neurolab/lib.html
    """

    def __init__(self, net_type='feed-forward',
                 features=None,
                 initf=nl.init.init_zeros,
                 trainf=None,
                 scaler=None,
                 **kwargs):
        Regressor.__init__(self, features=features)
        NeurolabBase.__init__(self, net_type=net_type, initf=initf, trainf=trainf, scaler=scaler, **kwargs)

    def fit(self, X, y):
        """
        Fit model on data

        :param X: pandas.DataFrame
        :param y: iterable denoting corresponding value in object
        :return: self
        """
        # TODO Some networks do not support regression?
        X, y, _ = check_inputs(X, y, None)
        y_train = y.reshape(len(y), 1 if len(y.shape) == 1 else y.shape[1])
        return self._fit(X, y, y_train)

    def predict(self, X):
        """
        Predict model

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :return: numpy.array of shape n_samples with values
        """
        modeled = self._sim(self._get_train_features(X))
        return modeled if modeled.shape[1] != 1 else np.ravel(modeled)

    def staged_predict(self, X, step=10):
        """
        Predicts probabilities on each stage

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param int step: step for returned iterations
        :return: iterator
        .. warning:: Doesn't have support in Neurolab (**AttributeError** will be thrown)
        """
        raise AttributeError("Not supported by Neurolab networks")


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
    :type scaler: sklearn-like scaler or False (do not scale features -- use with care and keep track of minmax param)
    :param list layers: list of numbers denoting size of each hidden layer
    :param dict kwargs: additional arguments to net __init__, varies with different net_types
                        See: https://pythonhosted.org/neurolab/lib.html
    """
    def __init__(self, net_type='feed-forward',
                 features=None,
                 initf=nl.init.init_zeros,
                 trainf=None,
                 scaler=None,
                 **kwargs):
        Classifier.__init__(self, features=features)
        NeurolabBase.__init__(self, net_type=net_type, initf=initf, trainf=trainf, scaler=scaler, **kwargs)
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit model on data

        :param X: pandas.DataFrame
        :param y: iterable denoting corresponding object classes
        :return: self
        """
        # Some networks do not support classification
        assert self.net_type not in CANT_CLASSIFY, 'Network type does not support classification'
        X, y, _ = check_inputs(X, y, None)
        self._set_classes(y)
        y_train = NeurolabClassifier._one_hot_transform(y)
        return self._fit(X, y, y_train)

    def predict_proba(self, X):
        """
        Predict probabilities for each class label on dataset

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :rtype: numpy.array of shape [n_samples, n_classes] with probabilities
        """
        return self._sim(self._get_train_features(X))

    def staged_predict_proba(self, X):
        """
        Predicts probabilities on each stage

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :return: iterator

        .. warning:: Doesn't have support in Neurolab (**AttributeError** will be thrown)
        """
        raise AttributeError("Not supported by Neurolab networks")

    @staticmethod
    def _one_hot_transform(y):
        return np.array(OneHotEncoder(n_values=len(np.unique(y))).fit_transform(y.reshape((len(y), 1))).todense())

    def _prepare_params(self, params, x_train, y_train):
        net_params = super(NeurolabClassifier, self)._prepare_params(params, x_train, y_train)

        # Default parameters for transfer functions in classifier networks
        if 'transf' not in net_params:
            net_params['transf'] = \
                [nl.trans.TanSig()] * len(net_params['size']) if 'size' in net_params else nl.trans.SoftMax()

        # Classification networks should have SoftMax as the transfer function on output layer
        if hasattr(net_params['transf'], '__iter__'):
            net_params['transf'][-1] = nl.trans.SoftMax()
        else:
            net_params['transf'] = nl.trans.SoftMax()

        return net_params
