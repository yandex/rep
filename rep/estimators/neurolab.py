"""
These classes are wrappers for `neurolab <https://pythonhosted.org/neurolab/lib.html>`_ - neural network python library

.. warning:: To make neurolab reproducible we change global random seed

    ::

        numpy.random.seed(42)
"""
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
from copy import deepcopy

import neurolab as nl
import numpy
import scipy

from .interface import Classifier, Regressor
from .utils import check_inputs, check_scaler, one_hot_transform, remove_first_line


__author__ = 'Vlad Sterzhanov, Alex Rogozhnikov, Tatiana Likhomanenko'
__all__ = ['NeurolabBase', 'NeurolabClassifier', 'NeurolabRegressor']

NET_TYPES = {'feed-forward': nl.net.newff,
             'single-layer': nl.net.newp,
             'competing-layer': nl.net.newc,
             'learning-vector': nl.net.newlvq,
             'elman-recurrent': nl.net.newelm,
             'hemming-recurrent': nl.net.newhem,
             'hopfield-recurrent': nl.net.newhop}

NET_PARAMS = ('minmax', 'cn', 'layers', 'transf', 'target',
              'max_init', 'max_iter', 'delta', 'cn0', 'pc')

BASIC_PARAMS = ('layers', 'net_type', 'trainf', 'initf', 'scaler', 'random_state')

# Instead of single layer use feed-forward.
CANT_CLASSIFY = ('hopfield-recurrent', 'competing-layer', 'hemming-recurrent', 'single-layer')


class NeurolabBase(object):
    """Base class for estimators from Neurolab library.

    Parameters:
    -----------
    :param features: features used in training
    :type features: list[str] or None
    :param list[int] layers: sequence, number of units inside each **hidden** layer.
    :param string net_type: type of network
        One of 'feed-forward', 'single-layer', 'competing-layer', 'learning-vector',
        'elman-recurrent', 'hopfield-recurrent', 'hemming-recurrent'
    :param initf: layer initializers
    :type initf: anything implementing call(layer), e.g. nl.init.* or list[nl.init.*] of shape [n_layers]
    :param trainf: net train function, default value depends on type of network
    :param scaler: transformer to apply to the input objects
    :type scaler: str or sklearn-like transformer or False (do not scale features)
    :param random_state: ignored, added for uniformity.
    :param dict kwargs: additional arguments to net __init__, varies with different net_types

    .. seealso:: https://pythonhosted.org/neurolab/lib.html for supported train functions and their parameters.
    """

    __metaclass__ = ABCMeta

    def __init__(self,
                 features=None,
                 layers=(10,),
                 net_type='feed-forward',
                 initf=nl.init.init_rand,
                 trainf=None,
                 scaler='standard',
                 random_state=None,
                 **other_params):
        self.features = list(features) if features is not None else features
        self.layers = list(layers)
        self.trainf = trainf
        self.initf = initf
        self.net_type = net_type
        self.scaler = scaler
        self.random_state = random_state

        self.net = None
        self.train_params = {}
        self.net_params = {}
        self.set_params(**other_params)

    def is_fitted(self):
        """
        Check if net is fitted

        :return: If estimator was fitted
        :rtype: bool
        """
        return self.net is not None

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        :param dict params: parameters to set in model
        """
        for name, value in params.items():
            if name.startswith("scaler__"):
                assert hasattr(self.scaler, 'set_params'), \
                    "Trying to set {} without scaler".format(name)
                self.scaler.set_params(**{name[len("scaler__"):]: value})
            elif name.startswith('layers__'):
                index = int(name[len('layers__'):])
                self.layers[index] = value
            elif name.startswith('initf__'):
                index = int(name[len('initf__'):])
                self.initf[index] = value
            elif name in NET_PARAMS:
                self.net_params[name] = value
            elif name in BASIC_PARAMS:
                setattr(self, name, value)
            else:
                self.train_params[name] = value

    def get_params(self, deep=True):
        """
        Get parameters of this estimator

        :rtype: dict
        """
        parameters = deepcopy(self.net_params)
        parameters.update(deepcopy(self.train_params))
        for name in BASIC_PARAMS:
            parameters[name] = getattr(self, name)
        return parameters

    def _partial_fit(self, X, y_original, y_train):
        """
        y_train is always 2-dimensional (one-hot for classification)
        y_original is what originally was passed to `fit`.
        """
        # magic reproducibilizer
        numpy.random.seed(42)

        if self.is_fitted():
            x_train = self._transform_input(X, y_original, fit=False)
        else:
            x_train = self._transform_input(X, y_original, fit=True)

            # Prepare parameters depending on network purpose (classification / regression)
            net_params = self._prepare_params(self.net_params, x_train, y_train)

            initializer = self._get_initializer(self.net_type)
            net = initializer(**net_params)

            # To allow similar initf function on all layers
            initf_iterable = self.initf if hasattr(self.initf, '__iter__') else [self.initf] * len(net.layers)
            for layer, init_function in zip(net.layers, initf_iterable):
                layer.initf = init_function
                net.init()

            if self.trainf is not None:
                net.trainf = self.trainf

            self.net = net

        self.net.train(x_train, y_train, **self.train_params)
        return self

    def _sim(self, X):
        assert self.net is not None, 'Classifier not fitted, prediction denied'
        transformed_x = self._transform_input(X, fit=False)
        return self.net.sim(transformed_x)

    def _transform_input(self, X, y=None, fit=True):
        X = self._get_features(X)
        # The following line fights the bug in sklearn < 0.16,
        # most of transformers there modify X if it is pandas.DataFrame.
        X = numpy.copy(X)
        if fit:
            self.scaler = check_scaler(self.scaler)
            self.scaler.fit(X, y)
        X = self.scaler.transform(X)

        # HACK: neurolab requires all features (even those of predicted objects) to be in [min, max]
        # so this dark magic appeared, seems to work ok for most reasonable use-cases,
        # while allowing arbitrary inputs.
        return scipy.special.expit(X / 3)

    def _prepare_params(self, net_params, x_train, y_train):
        net_params = deepcopy(net_params)
        # Network expects features to be [0, 1]-scaled
        net_params['minmax'] = [[0, 1]] * (x_train.shape[1])

        # To unify the layer-description argument with other supported networks
        if 'size' not in net_params:
            net_params['size'] = self.layers
        else:
            if self.layers != (10, ):
                raise ValueError('For neurolab please use either `layers` or `sizes`, not both')

        # Set output layer size
        net_params['size'] = list(net_params['size']) + [y_train.shape[1]]

        # Default parameters for transfer functions in classifier networks
        if 'transf' not in net_params:
            net_params['transf'] = [nl.trans.TanSig()] * len(net_params['size'])
        if not hasattr(net_params['transf'], '__iter__'):
            net_params['transf'] = [net_params['transf']] * len(net_params['size'])
        net_params['transf'] = list(net_params['transf'])

        return net_params

    @staticmethod
    def _get_initializer(net_type):
        if net_type not in NET_TYPES:
            raise AttributeError("Got unexpected network type: '{}'".format(net_type))
        return NET_TYPES.get(net_type)


class NeurolabClassifier(NeurolabBase, Classifier):
    __doc__ = "Classifier from neurolab library. \n" + remove_first_line(NeurolabBase.__doc__)

    def fit(self, X, y):
        """
        Train the classifier

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: labels of events - array-like of shape [n_samples]
        :param sample_weight: weight of events,
               array-like of shape [n_samples] or None if all weights are equal

        :return: self
        """
        # erasing results of previous training
        self.net = None
        return self.partial_fit(X, y)

    def partial_fit(self, X , y):
        """
        Additional training of the classifier

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: labels of events - array-like of shape [n_samples]

        :return: self
        """
        assert self.net_type not in CANT_CLASSIFY, 'Network type does not support classification'
        X, y, _ = check_inputs(X, y, None)
        if not self.is_fitted():
            self._set_classes(y)
        y_train = one_hot_transform(y, n_classes=len(self.classes_)) * 0.98 + 0.01
        return self._partial_fit(X, y, y_train)

    def predict_proba(self, X):
        """
        Predict labels for all events in dataset

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :rtype: numpy.array of shape [n_samples] with integer labels
        """
        return self._sim(X)

    def staged_predict_proba(self, X):
        """
        .. warning:: not supported in Neurolab (**AttributeError** will be thrown)
        """
        raise AttributeError("staged_predict_proba is not supported by Neurolab networks")

    def _prepare_params(self, params, x_train, y_train):
        net_params = super(NeurolabClassifier, self)._prepare_params(params, x_train, y_train)
        # Classification networks should have SoftMax as the transfer function on output layer
        net_params['transf'][-1] = nl.trans.SoftMax()
        return net_params


class NeurolabRegressor(NeurolabBase, Regressor):
    __doc__ = "Regressor from neurolab library. \n" + remove_first_line(NeurolabBase.__doc__)

    def fit(self, X, y):
        """
        Train the classifier

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: labels of events - array-like of shape [n_samples]
        :param sample_weight: weight of events,
               array-like of shape [n_samples] or None if all weights are equal

        :return: self
        """
        # erasing results of previous training
        self.net = None
        return self.partial_fit(X, y)

    def partial_fit(self, X , y):
        """
        Additional training of the classifier

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: labels of events - array-like of shape [n_samples]

        :return: self
        """
        assert self.net_type not in CANT_CLASSIFY, 'Network type does not support regression'
        X, y, _ = check_inputs(X, y, None, allow_multiple_targets=True)
        y_train = y.reshape(len(y), 1 if len(y.shape) == 1 else y.shape[1])
        return self._partial_fit(X, y, y_train)

    def predict(self, X):
        """
        Predict values for all events in dataset.

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :rtype: numpy.array of shape [n_samples] with predicted values
        """
        modeled = self._sim(X)
        return modeled if modeled.shape[1] != 1 else numpy.ravel(modeled)

    def staged_predict(self, X, step=10):
        """
        .. warning:: not supported in Neurolab (**AttributeError** will be thrown)
        """
        raise AttributeError("Staged predict is not supported by Neurolab networks")

    def _prepare_params(self, params, x_train, y_train):
        net_params = super(NeurolabRegressor, self)._prepare_params(params, x_train, y_train)
        net_params['transf'][-1] = nl.trans.PureLin()
        return net_params
