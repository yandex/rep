"""
These classes are wrappers for `theanets library <http://theanets.readthedocs.org/>`_ --- a neural network python library.

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
from abc import abstractmethod, ABCMeta

import numpy
import theanets as tnt

from .interface import Classifier, Regressor
from .utils import check_inputs, check_scaler, remove_first_line
from sklearn.utils import check_random_state


__author__ = 'Lisa Ignatyeva, Alex Rogozhnikov, Tatiana Likhomanenko'
__all__ = ['TheanetsBase', 'TheanetsClassifier', 'TheanetsRegressor']

UNSUPPORTED_OPTIMIZERS = {'sample', 'hf'}
# sample has too different interfaces from what we support here
# currently, hf now does not work in theanets, see https://github.com/lmjohns3/theanets/issues/62

# to keep climate from printing anything, uncomment following:
# import os
# import climate
# null_file = open(os.devnull, "w")
# climate.enable_default_logging(default_level='ERROR', stream=null_file)


class TheanetsBase(object):
    """A base class for the estimators from Theanets library.

    :param features: list of features to train model
    :type features: None or list(str)
    :param layers: a sequence of values specifying the **hidden** layer configuration for the network.
        For more information see `Specifying layers <http://theanets.readthedocs.org/en/latest/creating.html#creating-specifying-layers>`_
        in the theanets documentation.
        Note that theanets `layers` parameter includes input and output layers in the sequence as well.
    :type layers: sequence of int, tuple, dict
    :param int input_layer: size of the input layer. If it equals -1, the size is taken from the training dataset
    :param int output_layer: size of the output layer. If it equals -1, the size is taken from the training dataset
    :param str hidden_activation: the name of an activation function to use on the hidden network layers by default
    :param str output_activation: the name of an activation function to use on the output layer by default
    :param float input_noise: standard deviation of desired noise to inject into input
    :param float hidden_noise: standard deviation of desired noise to inject into hidden unit activation output
    :param float input_dropouts: proportion of the input units to randomly set to 0; it ranges [0, 1]
    :param float hidden_dropouts: proportion of hidden unit activations to randomly set to 0; it ranges [0, 1]
    :param int decode_from: any of the hidden layers can be tapped at the output. Just specify a value greater than
        1 to tap the last N hidden layers. The default is 1, which decodes from just the last layer.
    :param scaler: transformer which is applied to the input samples. If it is False, scaling will not be used
    :type scaler: str or sklearn-like transformer or False
    :param trainers: parameters to specify training algorithm(s), for example::

        trainers=[{'algo': sgd, 'momentum': 0.2}, {'algo': 'nag'}]

    :type trainers: list[dict] or None
    :param random_state: state for a pseudo random generator
    :type random_state: None or int or RandomState


    For more information on the available trainers and their parameters see this `page <http://theanets.readthedocs.org/en/latest/training.html>`_.
    """

    __metaclass__ = ABCMeta
    _model_type = None

    def __init__(self,
                 features=None,
                 layers=(10,),
                 input_layer=-1,
                 output_layer=-1,
                 hidden_activation='logistic',
                 output_activation='linear',
                 input_noise=0,
                 hidden_noise=0,
                 input_dropout=0,
                 hidden_dropout=0,
                 decode_from=1,
                 weight_l1=0.01,
                 weight_l2=0.01,
                 scaler='standard',
                 trainers=None,
                 random_state=42, ):
        self.features = list(features) if features is not None else features
        self.layers = list(layers)
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.random_state = random_state

        self.scaler = scaler
        self.trainers = trainers
        self.exp = None

        self.input_noise = input_noise
        self.hidden_noise = hidden_noise
        self.input_dropout = input_dropout
        self.hidden_dropout = hidden_dropout
        self.decode_from = decode_from
        self.weight_l1 = weight_l1
        self.weight_l2 = weight_l2

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def set_params(self, **params):
        """
        Set the parameters of the estimator. Deep parameters of trainers and scaler can be accessed,
        for instance::

                trainers__0 = {'algo': 'sgd', 'learning_rate': 0.3}
                trainers__0_algo = 'sgd'
                layers__1 = 14
                scaler__use_std = True

        :param dict params: parameters to set in the model
        """
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'layers':
                    value = list(value)
                setattr(self, key, value)
            else:
                # accessing deep parameters
                param, sep, param_of_param = key.partition('__')
                if sep != '__':
                    raise ValueError(key + ' is an invalid parameter a Theanets estimator')
                if param == 'trainers':
                    index, sep, param = param_of_param.partition('_')
                    index = int(index)
                    if index >= len(self.trainers):
                        raise ValueError('{} is an invalid parameter for a Theanets estimator: index '
                                         'too big'.format(key))
                    if param == '':
                        # e.g. trainers__0 = {'algo': 'sgd', 'learning_rate': 0.3}
                        self.trainers[index] = value
                    else:
                        # e.g. trainers__0_algo = 'sgd'
                        self.trainers[index][param] = value
                elif param == 'layers':
                    index = int(param_of_param)
                    if index >= len(self.layers):
                        raise ValueError('{} is an invalid parameter for a Theanets estimator: index '
                                         'too big'.format(key))
                    self.layers[index] = value
                elif param == 'scaler':
                    try:
                        self.scaler.set_params(**{param_of_param: value})
                    except Exception:
                        raise ValueError('was unable to set parameter {}={} '
                                         'to scaler {}'.format(param_of_param, value, self.scaler))
                else:
                    raise ValueError(key + ' is an invalid parameter for a Theanets estimator')

    def _transform_data(self, data, y=None):
        """
        Transform input samples by the scaler.

        :param pandas.DataFrame X: input data
        :param y: array-like target
        :return: array-like transformed data
        """
        data_backup = data.copy()
        if not self._is_fitted():
            self.scaler = check_scaler(self.scaler)
            self.scaler.fit(data_backup, y)
        return self.scaler.transform(data_backup)

    def _is_fitted(self):
        """
        Check if the estimator is fitted or not.

        :rtype: bool
        """
        return self.exp is not None

    def fit(self, X, y, sample_weight=None):
        """
        Train a classification/regression model on the data.

        :param pandas.DataFrame X: data of shape [n_samples, n_features]
        :param y: values for samples --- array-like of shape [n_samples]
        :param sample_weight: weights for samples --- array-like of shape [n_samples]
        :return: self
        """
        self.exp = None
        if self.trainers is None:
            # use default trainer with default parameters.
            self.trainers = [{}]

        for trainer in self.trainers:
            if 'algo' in trainer and trainer['algo'] in UNSUPPORTED_OPTIMIZERS:
                raise NotImplementedError(trainer['algo'] + ' is not supported')
            self.partial_fit(X, y, sample_weight=sample_weight, keep_trainer=False, **trainer)
        return self

    @abstractmethod
    def partial_fit(self, X, y, sample_weight=None, keep_trainer=True, **trainer):
        """
        Train the estimator by training the existing estimator again.

        :param pandas.DataFrame X: data of shape [n_samples, n_features]
        :param y: values for samples --- array-like of shape [n_samples]
        :param sample_weight: weights for samples --- array-like of shape [n_samples]
        :param bool keep_trainer: True if the trainer is not stored in self.trainers.
            If True, will add it to the list of the estimators.
        :param dict trainer: parameters of the training algorithm we want to use now
        :return: self
        """
        pass

    def _prepare_for_partial_fit(self, X, y, sample_weight=None, allow_multiple_targets=False, keep_trainer=True,
                                 **trainer):
        """
        Do preparation for fitting which is the same for a classifier and regressor.

        :param pandas.DataFrame X: data of shape [n_samples, n_features]
        :param y: values for samples --- array-like of shape [n_samples]
        :param sample_weight: weights for samples --- array-like of shape [n_samples]
        :param bool allow_multiple_targets: True if target can contain multiple targets
        :param bool keep_trainer: True if the trainer is not stored in self.trainers
        :param dict trainer: parameters of the training algorithm we want to use now
        :return: prepared data and target
        """
        X, y, sample_weight = check_inputs(X, y, sample_weight=sample_weight, allow_none_weights=False,
                                           allow_multiple_targets=allow_multiple_targets)
        X = self._transform_data(self._get_features(X, allow_nans=True), y)
        if keep_trainer:
            self.trainers.append(trainer)
        return X, y, sample_weight

    def _construct_layers(self, input_layer, output_layer):
        """
        Build a layer list including correct input/output layers' sizes.

        :param int input_layer: input layer size taken from the data
        :param int output_layer: output layer size taken from the data
        :return: list of layers
        """
        layers = [self.input_layer] + self.layers + [self.output_layer]
        if layers[0] == -1:
            layers[0] = input_layer
        if layers[-1] == -1:
            layers[-1] = output_layer
        return layers

    def _prepare_network_params(self):
        """
        Prepare simple net parameters.

        :return: prepared dict
        """
        if self.random_state is None:
            seed = 0
        elif isinstance(self.random_state, int):
            seed = self.random_state
        else:
            seed = check_random_state(self.random_state).randint(0, 10000)

        return {'hidden_activation': self.hidden_activation,
                'output_activation': self.output_activation,
                'input_noise': self.input_noise,
                'hidden_noise': self.hidden_noise,
                'input_dropout': self.input_dropout,
                'hidden_dropout': self.hidden_dropout,
                'decode_from': self.decode_from,
                'rng': seed,
                'weight_l1': self.weight_l1,
                'weight_l2': self.weight_l2
        }


class TheanetsClassifier(TheanetsBase, Classifier):
    __doc__ = 'Implements a classification model from the Theanets library. \n' + remove_first_line(TheanetsBase.__doc__)

    _model_type = 'classification'

    def partial_fit(self, X, y, sample_weight=None, keep_trainer=True, **trainer):
        X, y, sample_weight = self._prepare_for_partial_fit(X, y, sample_weight=sample_weight,
                                                            keep_trainer=keep_trainer, **trainer)
        if self.exp is None:
            self._set_classes(y)
            layers = self._construct_layers(X.shape[1], len(self.classes_))
            self.exp = tnt.Experiment(tnt.Classifier, layers=layers, weighted=True)
        params = self._prepare_network_params()
        params.update(**trainer)
        if trainer.get('algo', None) == 'pretrain':
            self.exp.train([X.astype(numpy.float32)], **params)
        else:
            self.exp.train([X.astype(numpy.float32), y.astype(numpy.int32), sample_weight.astype(numpy.float32)],
                           **params)
        return self
    partial_fit.__doc__ = TheanetsBase.partial_fit.__doc__

    def predict_proba(self, X):
        assert self._is_fitted(), 'Classifier wasn`t fitted, please, call `fit` first'
        X = self._transform_data(self._get_features(X, allow_nans=True))
        return self.exp.network.predict_proba(X.astype(numpy.float32))

    predict_proba.__doc__ = Classifier.predict_proba.__doc__

    def staged_predict_proba(self, X):
        """
        .. warning:: This function is not supported in the Theanets (**NotImplementedError** will be thrown)
        """
        raise NotImplementedError("'staged_predict_proba' is not supported by the Theanets classifiers")


class TheanetsRegressor(TheanetsBase, Regressor):
    __doc__ = 'Implements a regression model from the Theanets library. \n' + remove_first_line(TheanetsBase.__doc__)

    _model_type = 'regression'

    def partial_fit(self, X, y, sample_weight=None, keep_trainer=True, **trainer):
        allow_multiple_targets = False if len(numpy.shape(y)) == 1 else True
        X, y, sample_weight = self._prepare_for_partial_fit(X, y, sample_weight=sample_weight,
                                                            allow_multiple_targets=allow_multiple_targets,
                                                            keep_trainer=keep_trainer, **trainer)
        if self.exp is None:
            layers = self._construct_layers(X.shape[1], 1 if len(numpy.shape(y)) == 1 else numpy.shape(y)[1])
            self.exp = tnt.Experiment(tnt.Regressor, layers=layers, weighted=True)
        params = self._prepare_network_params()
        params.update(**trainer)
        if len(numpy.shape(y)) == 1:
            y = y.reshape(len(y), 1)
        if len(numpy.shape(sample_weight)) == 1:
            sample_weight = numpy.repeat(sample_weight, y.shape[1])
            sample_weight = sample_weight.reshape(y.shape)
        if trainer.get('algo') == 'pretrain':
            self.exp.train([X.astype(numpy.float32)], **params)
        else:
            self.exp.train([X.astype(numpy.float32), y, sample_weight.astype(numpy.float32)], **params)
        return self

    partial_fit.__doc__ = TheanetsBase.partial_fit.__doc__

    def predict(self, X):
        assert self._is_fitted(), "Regressor wasn't fitted, please, call `fit` first"
        X = self._transform_data(self._get_features(X, allow_nans=True))
        return self.exp.network.predict(X.astype(numpy.float32))

    predict.__doc__ = Regressor.predict.__doc__

    def staged_predict(self, X):
        """
        .. warning:: This function is not supported in the Theanets (**NotImplementedError** will be thrown)
        """
        raise NotImplementedError("'staged_predict' is not supported by the Theanets regressors")
