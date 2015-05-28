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
import numpy
from abc import abstractmethod
from .interface import Classifier, Regressor
from .utils import check_inputs, check_scaler
from sklearn.utils import check_random_state

import os
import tempfile
from copy import deepcopy

try:
    import theanets as tnt
except ImportError as e:
    raise ImportError("Install theanets before (pip install theanets)")

__author__ = 'Lisa Ignatyeva'

UNSUPPORTED_OPTIMIZERS = ['pretrain', 'sample', 'hf']
# pretrain and sample data formats have too different interface from what we support here
# currently, hf now does not work in theanets, see https://github.com/lmjohns3/theanets/issues/62


class TheanetsBase(object):
    """
    Base class for estimators from Theanets library.

    Parameters:
    -----------
    :param layers: A sequence of values specifying the **hidden** layer configuration for the network.
        For more information please see 'Specifying layers' in theanets documentation:
        http://theanets.readthedocs.org/en/latest/creating.html#creating-specifying-layers
        Note that theanets "layers" parameter included input and output layers in the sequence as well.
    :type layers: sequence of int, tuple, dict
    :param int input_layer: size of the input layer. If equals -1, the size is taken from the training dataset.
    :param int output_layer: size of the output layer. If equals -1, the size is taken from the training dataset.
    :param str hidden_activation: the name of an activation function to use on hidden network layers by default.
    :param str output_activation: The name of an activation function to use on the output layer by default.
    :param int random_state: random seed
    :param float input_noise: Standard deviation of desired noise to inject into input.
    :param float hidden_noise: Standard deviation of desired noise to inject into hidden unit activation output.
    :param input_dropouts: Proportion of input units to randomly set to 0.
    :type input_dropouts: float in [0, 1]
    :param hidden_dropouts: Proportion of hidden unit activations to randomly set to 0.
    :type hidden_dropouts: float in [0, 1]
    :param decode_from: Any of the hidden layers can be tapped at the output. Just specify a value greater than
        1 to tap the last N hidden layers. The default is 1, which decodes from just the last layer.
    :type decode_from: positive int
    :param scaler: scaler used to transform data. If False, scaling will not be used.
    :type scaler: scaler from sklearn.preprocessing or False
    :param list(dict) or None trainers: parameters to specify training algorithm(s)
    example: [{'optimize': sgd, 'momentum': 0.2}, {'optimize': 'nag'}]

    For more information on available trainers and their parameters, see this page
    http://theanets.readthedocs.org/en/latest/training.html?highlight=trainers#gradient-based-methods
    Note that not pretrain, nor sample and hf are not supported.
    """
    def __init__(self,
                 layers,
                 input_layer,
                 output_layer,
                 hidden_activation,
                 output_activation,
                 random_state,
                 input_noise,
                 hidden_noise,
                 input_dropouts,
                 hidden_dropouts,
                 decode_from,
                 scaler,
                 trainers):
        self.layers = list(layers)
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.random_state = random_state
        self.network_params = {'hidden_activation': hidden_activation, 'output_activation': output_activation,
                               'input_noise': input_noise, 'hidden_noise': hidden_noise,
                               'input_dropouts': input_dropouts, 'hidden_dropouts': hidden_dropouts,
                               'decode_from': decode_from}

        self.scaler = scaler
        self.trainers = trainers
        if self.trainers is None:
            # use default trainer with default parameters.
            self.trainers = [{}]
        self.exp = None
        self.features = None

    def __getstate__(self):
        """
        Required for copy, pickle.dump working, because theanets objects can't be pickled by default.

        :return dict result: the dictionary containing all the object, transformed and therefore picklable.
        """
        result = self.__dict__.copy()
        del result['exp']
        if self.exp is None:
            result['dumped_exp'] = None
        else:
            with tempfile.NamedTemporaryFile() as dump:
                self.exp.save(dump.name)
                with open(dump.name, 'rb') as dumpfile:
                    result['dumped_exp'] = dumpfile.read()
        return result

    def __setstate__(self, dictionary):
        """
        Required for pickle.load working, because theanets objects can't be unpickled by default.

        :param dict dictionary: the structure representing a TheanetsClassifier
        """
        self.__dict__ = dictionary
        if dictionary['dumped_exp'] is None:
            self.exp = None
        else:
            with tempfile.NamedTemporaryFile() as dump:
                with open(dump.name, 'wb') as dumpfile:
                    dumpfile.write(dictionary['dumped_exp'])
                assert os.path.exists(dump.name), 'there is no such file: {}'.format(dump.name)
                dummy_layers = [1] + self.layers + [1]
                self.exp = tnt.Experiment(tnt.Classifier, layers=dummy_layers, rng=self._reproducibilize(),
                                          **self.network_params)
                self.exp.load(dump.name)
        del dictionary['dumped_exp']

    def _reproducibilize(self):
        """
        A magic method which makes theanets calls be reproducible.
        Should be called when creating an experiment to pass
        a proper rng value and before running exp.train in order to fix the seed.
        See https://github.com/lmjohns3/theanets/issues/72
        """
        numpy.random.seed(42)
        return check_random_state(self.random_state)

    def set_params(self, **params):
        """
        Set the parameters of this estimator. Deep parameters of trianers and scaler can be accessed,
        for instance:
        trainers__0 = trainers__0 = {'optimize': 'sgd', 'learning_rate': 0.3}
        trainers__0_optimize = 'sgd'
        layers__1 = 14
        scaler__use_std = True

        :param dict params: parameters to set in model
        """
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'layers':
                    value = list(value)
                setattr(self, key, value)
            else:
                if key in self.network_params:
                    self.network_params[key] = value
                else:
                    param, sep, param_of_param = key.partition('__')
                    if sep != '__':
                        raise AttributeError(key + ' is an invalid parameter a Theanets estimator')
                    if param == 'trainers':
                        index, sep, param = param_of_param.partition('_')
                        index = int(index)
                        if index >= len(self.trainers):
                            raise AttributeError('{} is an invalid parameter for a Theanets estimator: index '
                                                 'too big'.format(key))
                        if param == '':
                            # e.g. trainers__0 = {'optimize': 'sgd', 'learning_rate': 0.3}
                            self.trainers[index] = value
                        else:
                            # e.g. trainers__0_optimize = 'sgd'
                            self.trainers[index][param] = value
                    elif param == 'layers':
                        index = int(param_of_param)
                        if index >= len(self.layers):
                            raise AttributeError('{} is an invalid parameter for a Theanets estimator: index '
                                                 'too big'.format(key))
                        self.layers[index] = value
                    elif param == 'scaler':
                        if not self.scaler:
                            raise AttributeError('no scaler')
                        if hasattr(self.scaler, param_of_param):
                            setattr(self.scaler, param_of_param, value)
                        else:
                            raise AttributeError('scaler does not have parameter `{}`'.format(param_of_param))
                    else:
                        raise AttributeError(key + ' is an invalid parameter for a Theanets estimator')

    def get_params(self, deep=True):
        """
        Get parameters of this estimator

        :return dict
        """
        parameters = deepcopy(self.network_params)
        parameters['layers'] = deepcopy(self.layers)
        parameters['input_layer'] = self.input_layer
        parameters['output_layer'] = self.output_layer
        parameters['trainers'] = deepcopy(self.trainers)
        parameters['features'] = deepcopy(self.features)
        parameters['random_state'] = self.random_state
        parameters['scaler'] = self.scaler
        return parameters

    def _transform_data(self, data, y=None):
        """
        Takes the features and transforms data using self.scaler, also fits the scaler if needed.
        :param data: data which should be scaled
        :param y: labels for this data
        :return: transformed data
        """
        data_backup = data.copy()
        if not self._is_fitted():
            self.scaler.fit(data_backup, y)
        return self.scaler.transform(data_backup)

    def _is_fitted(self):
        return self.exp is not None

    def fit(self, X, y):
        """
        Train the estimator from scratch.

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: values - array-like of shape [n_samples]
        :return: self
        """
        self.exp = None
        self.scaler = check_scaler(self.scaler)
        for trainer in self.trainers:
            for optimizer in UNSUPPORTED_OPTIMIZERS:
                if 'optimize' in trainer and trainer['optimize'] == optimizer:
                    raise NotImplementedError(optimizer + ' is not supported')
            self.partial_fit(X, y, new_trainer=False, **trainer)
        return self

    @abstractmethod
    def partial_fit(self, X, y, new_trainer=True, **trainer):
        """
        Train the estimator by training the existing classifier again.

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: values - array-like of shape [n_samples]
        :param bool new_trainer: True if the trainer is not stored in self.trainers.
            If True, will add it to list of classifiers.
        :param dict trainer: parameters of the training algorithm we want to use now
        :return: self
        """
        pass

    def _prepare_for_partial_fit(self, X, y, new_trainer=True, **trainer):
        """
        Does preparation for fitting which is the same for classifier and regressor
        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: values - array-like of shape [n_samples]
        :param bool new_trainer: True if the trainer is not stored in self.trainers
        :param dict trainer: parameters of the training algorithm we want to use now
        :return: prepared data and labels
        """
        X, y, _ = check_inputs(X, y, sample_weight=None)
        if not self._is_fitted():
            self.scaler = check_scaler(self.scaler)

        X = self._transform_data(self._get_train_features(X, allow_nans=True), y)
        if new_trainer:
            self.trainers.append(trainer)
        return X, y

    def _construct_layers(self, input_layer, output_layer):
        """
        Build a layer list, including correct input/output layers' sizes.
        :param int input_layer: input layer size taken from the data
        :param int output_layer: output layer size taken from the data
        :return: list layers
        """
        layers = [self.input_layer] + self.layers + [self.output_layer]
        if layers[0] == -1:
            layers[0] = input_layer
        if layers[-1] == -1:
            layers[-1] = output_layer
        return layers


class TheanetsClassifier(TheanetsBase, Classifier):
    """
    Classifier from Theanets library.

    Parameters:
    -----------
    :param features: list of features to train model
    :type features: None or list(str)
    :param layers: A sequence of values specifying the **hidden** layer configuration for the network.
        For more information please see 'Specifying layers' in theanets documentation:
        http://theanets.readthedocs.org/en/latest/creating.html#creating-specifying-layers
        Note that theanets "layers" parameter included input and output layers in the sequence as well.
    :type layers: sequence of int, tuple, dict
    :param int input_layer: size of the input layer. If equals -1, the size is taken from the training dataset.
    :param int output_layer: size of the output layer. If equals -1, the size is taken from the training dataset.
    :param str hidden_activation: the name of an activation function to use on hidden network layers by default.
    :param str output_activation: The name of an activation function to use on the output layer by default.
    :param int random_state: random seed
    :param float input_noise: Standard deviation of desired noise to inject into input.
    :param float hidden_noise: Standard deviation of desired noise to inject into hidden unit activation output.
    :param input_dropouts: Proportion of input units to randomly set to 0.
    :type input_dropouts: float in [0, 1]
    :param hidden_dropouts: Proportion of hidden unit activations to randomly set to 0.
    :type hidden_dropouts: float in [0, 1]
    :param decode_from: Any of the hidden layers can be tapped at the output. Just specify a value greater than
        1 to tap the last N hidden layers. The default is 1, which decodes from just the last layer.
    :type decode_from: positive int
    :param scaler: scaler used to transform data. If False, scaling will not be used.
    :type scaler: scaler from sklearn.preprocessing or False
    :param list(dict) or None trainers: parameters to specify training algorithm(s)
    example: [{'optimize': sgd, 'momentum': 0.2, }, {'optimize': 'nag'}]

    For more information on available trainers and their parameters, see this page
    http://theanets.readthedocs.org/en/latest/training.html
    Note that not pretrain, sample and hf trainers are not supported.
    """

    def __init__(self,
                 features=None,
                 layers=(10,),
                 input_layer=-1,
                 output_layer=-1,
                 hidden_activation='logistic',
                 output_activation='linear',
                 random_state=42,
                 input_noise=0,
                 hidden_noise=0,
                 input_dropouts=0,
                 hidden_dropouts=0,
                 decode_from=1,
                 scaler='standard',
                 trainers=None):
        TheanetsBase.__init__(self,
                              layers=layers,
                              input_layer=input_layer,
                              output_layer=output_layer,
                              hidden_activation=hidden_activation,
                              output_activation=output_activation,
                              random_state=random_state,
                              input_noise=input_noise,
                              hidden_noise=hidden_noise,
                              input_dropouts=input_dropouts,
                              hidden_dropouts=hidden_dropouts,
                              decode_from=decode_from,
                              scaler=scaler,
                              trainers=trainers)
        Classifier.__init__(self, features=features)

    def partial_fit(self, X, y, new_trainer=True, **trainer):
        """
        Train the classifier by training the existing classifier again.

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: values - array-like of shape [n_samples]
        :param bool new_trainer: True if the trainer is not stored in self.trainers
        :param dict trainer: parameters of the training algorithm we want to use now
        :return: self
        """
        X, y = self._prepare_for_partial_fit(X, y, new_trainer, **trainer)
        # TODO use set_classes?
        self.classes_ = numpy.unique(y)
        if self.exp is None:
            layers = self._construct_layers(X.shape[1], len(self.classes_))
            self.exp = tnt.Experiment(tnt.Classifier, layers=layers,
                                      rng=self._reproducibilize(), **self.network_params)
        self._reproducibilize()
        self.exp.train((X.astype(numpy.float32), y.astype(numpy.int32)),
                       **trainer)
        return self

    def predict_proba(self, X):
        """
        Predict probabilities

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :rtype: numpy.array of shape [n_samples, n_classes] with probabilities
        """
        assert self._is_fitted(), 'Classifier wasn`t fitted, please call `fit` first'
        X = self._transform_data(self._get_train_features(X, allow_nans=True))
        return self.exp.network.predict(X.astype(numpy.float32))

    def staged_predict_proba(self, X):
        """
        Predicts values on each stage

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :return: iterator
        """
        raise NotImplementedError('staged_predict_proba is not supported for theanets classifier')


class TheanetsRegressor(TheanetsBase, Regressor):
    """
    Regressor from Theanets library.

    Parameters:
    -----------
    :param features: list of features to train model
    :type features: None or list(str)
    :param layers: A sequence of values specifying the **hidden** layer configuration for the network.
        For more information please see 'Specifying layers' in theanets documentation:
        http://theanets.readthedocs.org/en/latest/creating.html#creating-specifying-layers
        Note that theanets "layers" parameter included input and output layers in the sequence as well.
    :type layers: sequence of int, tuple, dict
    :param int input_layer: size of the input layer. If equals -1, the size is taken from the training dataset.
    :param int output_layer: size of the output layer. If equals -1, the size is taken from the training dataset.
    :param str hidden_activation: the name of an activation function to use on hidden network layers by default.
    :param str output_activation: The name of an activation function to use on the output layer by default.
    :param int random_state: random seed
    :param float input_noise: Standard deviation of desired noise to inject into input.
    :param float hidden_noise: Standard deviation of desired noise to inject into hidden unit activation output.
    :param input_dropouts: Proportion of input units to randomly set to 0.
    :type input_dropouts: float in [0, 1]
    :param hidden_dropouts: Proportion of hidden unit activations to randomly set to 0.
    :type hidden_dropouts: float in [0, 1]
    :param decode_from: Any of the hidden layers can be tapped at the output. Just specify a value greater than
        1 to tap the last N hidden layers. The default is 1, which decodes from just the last layer.
    :type decode_from: positive int
    :param scaler: scaler used to transform data. If False, scaling will not be used.
    :type scaler: scaler from sklearn.preprocessing or False
    :param list(dict) or None trainers: parameters to specify training algorithm(s)
    example: [{'optimize': sgd, 'momentum': 0.2}, {'optimize': 'nag'}]

    For more information on available trainers and their parameters, see this page
    http://theanets.readthedocs.org/en/latest/training.html?highlight=trainers#gradient-based-methods
    Note that not pretrain, sample and hf are not supported.
    """
    def __init__(self,
                 features=None,
                 layers=(10,),
                 input_layer=-1,
                 output_layer=-1,
                 hidden_activation='logistic',
                 output_activation='linear',
                 random_state=42,
                 input_noise=0,
                 hidden_noise=0,
                 input_dropouts=0,
                 hidden_dropouts=0,
                 decode_from=1,
                 scaler='standard',
                 trainers=None):
        TheanetsBase.__init__(self,
                              layers=layers,
                              input_layer=input_layer,
                              output_layer=output_layer,
                              hidden_activation=hidden_activation,
                              output_activation=output_activation,
                              random_state=random_state,
                              input_noise=input_noise,
                              hidden_noise=hidden_noise,
                              input_dropouts=input_dropouts,
                              hidden_dropouts=hidden_dropouts,
                              decode_from=decode_from,
                              scaler=scaler,
                              trainers=trainers)
        Regressor.__init__(self, features=features)

    def partial_fit(self, X, y, sample_weight=None, new_trainer=True, **trainer):
        """
        Train the regressor by training the existing regressor again.

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: values - array-like of shape [n_samples] (or [n_samples, n_targets])
        :param bool new_trainer: True if the trainer is not stored in self.trainers
        :param dict trainer: parameters of the training algorithm we want to use now
        :return: self
        """
        X, y = self._prepare_for_partial_fit(X, y, new_trainer, **trainer)

        if self.exp is None:
            layers = self._construct_layers(X.shape[1], 1)
            self.exp = tnt.Experiment(tnt.feedforward.Regressor, layers=layers,
                                      rng=self._reproducibilize(), **self.network_params)
        self._reproducibilize()
        if len(numpy.shape(y)) == 1:
            y = y.reshape(len(y), 1)
        self.exp.train([X.astype(numpy.float32), y], **trainer)
        return self

    def predict(self, X):
        """
        Predict probabilities

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :rtype: numpy.array of shape [n_samples, n_classes] with probabilities
        """
        assert self._is_fitted(), "Regressor wasn't fitted, please call `fit` first"
        X = self._transform_data(self._get_train_features(X, allow_nans=True))
        return self.exp.network.predict(X.astype(numpy.float32))

    def staged_predict(self, X):
        """
        Predicts values on each stage

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :return: iterator
        """
        raise NotImplementedError('staged_predict is not supported for theanets regressor')
