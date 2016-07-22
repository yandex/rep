"""
These classes are wrappers for the `Neurolab library <https://pythonhosted.org/neurolab/lib.html>`_ --- a neural network python library.

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
__all__ = ['NeurolabClassifier', 'NeurolabRegressor']

NET_TYPES = {'feed-forward': nl.net.newff,
             'competing-layer': nl.net.newc,
             'learning-vector': nl.net.newlvq,
             'elman-recurrent': nl.net.newelm,
             'hemming-recurrent': nl.net.newhem,
             'hopfield-recurrent': nl.net.newhop
             }

NET_PARAMS = ('minmax', 'cn', 'layers', 'transf', 'target',
              'max_init', 'max_iter', 'delta', 'cn0', 'pc')

BASIC_PARAMS = ('layers', 'net_type', 'trainf', 'initf', 'scaler', 'random_state')

# Instead of a single layer use feed-forward.
CANT_CLASSIFY = ('hopfield-recurrent', 'competing-layer', 'hemming-recurrent')
CANT_DO_REGRESSION = ('hopfield-recurrent', )


class NeurolabBase(object):
    """ A base class for estimators from the Neurolab library.

    :param features: features used in training
    :type features: list[str] or None
    :param list[int] layers: sequence, number of units inside each **hidden** layer.
    :param string net_type: type of the network; possible values are:

        * `feed-forward`
        * `competing-layer`
        * `learning-vector`
        * `elman-recurrent`
        * `hemming-recurrent`

    :param initf: layer initializers
    :type initf: anything implementing call(layer), e.g. neurolab.init.* or list[neurolab.init.*] of shape [n_layers]
    :param trainf: net training function; default value depends on the type of a network
    :param scaler: transformer which is applied to the input samples. If it is False, scaling will not be used
    :type scaler: str or sklearn-like transformer or False
    :param random_state: this parameter is ignored and is added for uniformity.
    :param dict kwargs: additional arguments to net `__init__`, varies with different `net_types`

    .. seealso:: `Supported training functions and their parameters <https://pythonhosted.org/neurolab/lib.html>`_
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

    def _is_fitted(self):
        """
        Check if the estimator is fitted or not.

        :rtype: bool
        """
        return self.net is not None

    def set_params(self, **params):
        """
        Set the parameters of the estimator.

        :param dict params: parameters to be set in the model
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
        Get parameters of the estimator.

        :rtype: dict
        """
        parameters = deepcopy(self.net_params)
        parameters.update(deepcopy(self.train_params))
        for name in BASIC_PARAMS:
            parameters[name] = getattr(self, name)
        return parameters

    def _partial_fit(self, X, y_original, y_train):
        """
        Train the estimator by training the existing estimator again.

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y_train: array-like target, which is always 2-dimensional (one-hot for classification)
        :param y_original: array-like target, which originally was passed to `fit`.
        :return: self
        """
        # magic reproducibilizer
        numpy.random.seed(42)

        if self._is_fitted():
            x_train = self._transform_data(X, y_original, fit=False)
        else:
            x_train = self._transform_data(X, y_original, fit=True)

            # Prepare parameters depending on the network purpose (classification / regression)
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

    def _activate_on_dataset(self, X):
        """
        Predict data.

        :param pandas.DataFrame X: data to be predicted
        :return: array-like predictions [n_samples, n_targets]
        """
        assert self.net is not None, 'Model is not fitted, prediction is denied'
        transformed_x = self._transform_data(X, fit=False)
        return self.net.sim(transformed_x)

    def _transform_data(self, X, y=None, fit=True):
        """
        Transform input samples by the scaler.

        :param pandas.DataFrame X: input data
        :param y: array-like target
        :param bool fit: true if scaler is not trained yet
        :return: array-like transformed data
        """
        X = self._get_features(X)
        # The following line fights the bug in sklearn < 0.16,
        # most of the transformers there modify X if it is pandas.DataFrame.
        X = numpy.copy(X)
        if fit:
            self.scaler = check_scaler(self.scaler)
            self.scaler.fit(X, y)
        X = self.scaler.transform(X)

        # HACK: neurolab requires all features (even those of predicted objects) to be in [min, max]
        # so this dark magic appeared, seems to work ok for the most reasonable use-cases,
        # while allowing arbitrary inputs.
        return scipy.special.expit(X / 3)

    def _prepare_params(self, net_params, x_train, y_train):
        """
        Set parameters for the neurolab net.

        :param dict net_params: parameters
        :param x_train: array-like training data
        :param y_train: array-like training target
        :return: prepared parameters in the neurolab interface
        """
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

        # Default parameters for the transfer functions in the networks
        if self.net_type != 'learning-vector':
            if 'transf' not in net_params:
                net_params['transf'] = [nl.trans.TanSig()] * len(net_params['size'])
            if not hasattr(net_params['transf'], '__iter__'):
                net_params['transf'] = [net_params['transf']] * len(net_params['size'])
            net_params['transf'] = list(net_params['transf'])

        return net_params

    @staticmethod
    def _get_initializer(net_type):
        """
        Return a neurolab net type object.

        :param str net_type: net type
        :return: a neurolab object corresponding to the net type
        """
        if net_type not in NET_TYPES:
            raise AttributeError("Got unexpected network type: '{}'".format(net_type))
        return NET_TYPES.get(net_type)


class NeurolabClassifier(NeurolabBase, Classifier):
    __doc__ = "Implements a classification model from the Neurolab library. \n" + remove_first_line(NeurolabBase.__doc__)

    def fit(self, X, y):
        """
        Train a classification model on the data.

        :param pandas.DataFrame X: data of shape [n_samples, n_features]
        :param y: labels of samples --- array-like of shape [n_samples]
        :return: self
        """
        # erasing results of the previous training
        self.net = None
        return self.partial_fit(X, y)

    def partial_fit(self, X , y):
        """
        Additional training of the classifier.

        :param pandas.DataFrame X: data of shape [n_samples, n_features]
        :param y: labels of samples, array-like of shape [n_samples]
        :return: self
        """
        assert self.net_type not in CANT_CLASSIFY, 'Network type does not support classification'
        X, y, _ = check_inputs(X, y, None)
        if not self._is_fitted():
            self._set_classes(y)
        y_train = one_hot_transform(y, n_classes=len(self.classes_)) * 0.98 + 0.01
        return self._partial_fit(X, y, y_train)

    def predict_proba(self, X):
        return self._activate_on_dataset(X)

    predict_proba.__doc__ = Classifier.predict_proba.__doc__

    def staged_predict_proba(self, X):
        """
        .. warning:: This is not supported in the Neurolab (**AttributeError** will be thrown)
        """
        raise AttributeError("'staged_predict_proba' is not supported by the Neurolab networks")

    def _prepare_params(self, params, x_train, y_train):
        net_params = super(NeurolabClassifier, self)._prepare_params(params, x_train, y_train)
        # Classification networks should have SoftMax as the transfer function on output layer
        net_params['transf'][-1] = nl.trans.SoftMax()
        return net_params

    _prepare_params.__doc__ = NeurolabBase._prepare_params.__doc__


class NeurolabRegressor(NeurolabBase, Regressor):
    __doc__ = "Implements a regression model from the Neurolab library. \n" + remove_first_line(NeurolabBase.__doc__)

    def fit(self, X, y):
        """
        Train a regression model on the data.

        :param pandas.DataFrame X: data of shape [n_samples, n_features]
        :param y: values for samples --- array-like of shape [n_samples]
        :return: self
        """
        # erasing results of previous training
        self.net = None
        return self.partial_fit(X, y)

    def partial_fit(self, X , y):
        """
        Additional training of the regressor.

        :param pandas.DataFrame X: data of shape [n_samples, n_features]
        :param y: values for samples, array-like of shape [n_samples]
        :return: self
        """
        if self.net_type in CANT_DO_REGRESSION:
            raise RuntimeError('Network type does not support regression')
        X, y, _ = check_inputs(X, y, None, allow_multiple_targets=True)
        y_train = y.reshape(len(y), 1 if len(y.shape) == 1 else y.shape[1])
        return self._partial_fit(X, y, y_train)

    def predict(self, X):
        modeled = self._activate_on_dataset(X)
        return modeled if modeled.shape[1] != 1 else numpy.ravel(modeled)

    predict.__doc__ = Regressor.predict.__doc__

    def staged_predict(self, X):
        """
        .. warning:: This is not supported in the Neurolab (**AttributeError** will be thrown)
        """
        raise AttributeError("'staged_predict' is not supported by the Neurolab networks")

    def _prepare_params(self, params, x_train, y_train):
        net_params = super(NeurolabRegressor, self)._prepare_params(params, x_train, y_train)
        net_params['transf'][-1] = nl.trans.PureLin()
        return net_params

    _prepare_params.__doc__ = NeurolabBase._prepare_params.__doc__
