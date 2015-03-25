from __future__ import division, print_function, absolute_import
import numpy

from .interface import Classifier
from .utils import check_inputs
from sklearn.preprocessing import MinMaxScaler, Imputer
import os
import tempfile

try:
    import theanets as tnt
except ImportError as e:
    raise ImportError("Install theanets before")

__author__ = 'Lisa Ignatyeva'


class TheanetsClassifier(Classifier):
    # TODO: fix the doc
    """
    Implements classification from Theanets library.

    Parameters:
    -----------
    :param layers: A sequence of values specifying the layer configuration for the network. For more information
        please see 'Specifying layers' in theanets documentation:
        http://theanets.readthedocs.org/en/latest/creating.html#creating-specifying-layers
    :type layers: sequence of int, tuple, dict
    :param hidden_activation: the name of an activation function to use on hidden network layers by default.
        Defaults to 'logistic'
    :type hidden_activation: str
    :param output_activation: The name of an activation function to use on the output layer by default.
        Defaults to 'linear'
    :type output_activation: str
    :param rng: Use a specific Theano random number generator. A new one will be created if this is None.
    :type rng: theano RandomStreams object
    :param input_noise: Standard deviation of desired noise to inject into input.
    :type input_noise: float
    :param hidden_noise: Standard deviation of desired noise to inject into hidden unit activation output.
    :type hidden_noise: float
    :param input_dropouts: Proportion of input units to randomly set to 0.
    :type input_dropouts: float in [0, 1]
    :param hidden_dropouts: Proportion of hidden unit activations to randomly set to 0.
    :type hidden_dropouts: float in [0, 1]
    :param decode_from: Any of the hidden layers can be tapped at the output. Just specify a value greater than
        1 to tap the last N hidden layers. The default is 1, which decodes from just the last layer.
    :type decode_from: positive int
    :param features: list of features to train model
    :type features: None or list(str)
    :param list(dict) or None trainers: parameters to specify training algorithm
    """
    def __init__(self, 
                 layers,
                 hidden_activation='logistic',
                 output_activation='linear',
                 rng=None,
                 input_noise=0,
                 hidden_noise=0,
                 input_dropouts=0,
                 hidden_dropouts=0,
                 decode_from=1,
                 features=None,
                 trainers=None):
        self.layers = layers
        self.network_params = {'hidden_activation': hidden_activation, 'output_activation': output_activation,
                               'rng': rng, 'input_noise': input_noise, 'hidden_noise': hidden_noise,
                               'input_dropouts': input_dropouts, 'hidden_dropouts': hidden_dropouts,
                               'decode_from': decode_from}
        # TODO: Do something with rng!
        self.trainers = trainers  # should it really be None?
        if self.trainers is None:
            self.trainers = [{}]
        self.exp = None
        Classifier.__init__(self, features=features)

    def __getstate__(self):
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
        self.__dict__ = dictionary
        if dictionary['dumped_exp'] is None:
            self.exp = None
        else:
            with tempfile.NamedTemporaryFile() as dump:
                with open(dump.name, 'wb') as dumpfile:
                    dumpfile.write(dictionary['dumped_exp'])
                assert os.path.exists(dump.name), 'there is no such file: {}'.format(dump.name)
                self.exp = tnt.Experiment.load(dump.name)
        del dictionary['dumped_exp']

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        :param dict params: parameters to set in model
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                if key in self.network_params:
                    self.network_params[key] = value
                else:
                    # TODO: if there is only one trainer, parameters names should be allowed to be simpler
                    trainer_num, sep, param = key.partition('_')
                    if not sep:
                        raise AttributeError(key + ' is an invalid parameter for a NN with multiple training')
                    trainer_num = int(trainer_num[7:])
                    # resize if needed
                    self.trainers[trainer_num][param] = value

    def get_params(self, deep=True):
        """
        Get parameters of this estimator

        :return dict
        """
        parameters = {'layers': self.layers,
                      'network_params': self.network_params,
                      'trainers': self.trainers,
                      'features': self.features,
                      'exp': self.exp}
        return parameters

    def _transform_data(self, data):
        return MinMaxScaler().fit_transform(Imputer().fit_transform(self._get_train_features(data)))

    def _check_fitted(self):
        assert self.exp is not None, 'Classifier wasn`t fitted, please call `fit` first'

    def fit(self, X, y, sample_weight=None):
        """
        Train the classifier from scratch.

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: values - array-like of shape [n_samples]
        :param sample_weight: weight of events,
               array-like of shape [n_samples] or None if all weights are equal
        :return: self
        """
        self.exp = None
        for trainer in self.trainers:
            self.partial_fit(X, y, new_trainer=False, **trainer)
        return self

    def partial_fit(self, X, y, sample_weight=None, new_trainer=True,  **trainer):
        """
        Train the classifier by training the existing classifier again.

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: values - array-like of shape [n_samples]
        :param sample_weight: weight of events,
               array-like of shape [n_samples] or None if all weights are equal
        :param bool new_trainer: True if the trainer is not stored in self.trainers
        :param dict trainer: parameters of the training algorithm we want to use now
        :return: self
        """
        if sample_weight is not None:
            # https://github.com/lmjohns3/theanets/issues/58
            raise NotImplementedError('sample_weight is not supported for theanets')
        X, y, sample_weight = check_inputs(X, y, sample_weight)
        X = self._transform_data(X)
        self.classes_ = numpy.unique(y)
        if self.exp is None:
            # initialize experiment
            if self.layers[0] == -1:
                self.layers = (X.shape[1],) + tuple(self.layers[1:])
            if self.layers[-1] == -1:
                self.layers = self.layers[:-1] + (len(self.classes_),)
            self.exp = tnt.Experiment(tnt.Classifier, layers=self.layers, **self.network_params)
        if new_trainer:
            self.trainers.append(trainer)
        self.exp.train((X.astype(numpy.float32), y.astype(numpy.int32)),
                       **trainer)
        return self

    def predict_proba(self, X):
        """
        Predict probabilities

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :rtype: numpy.array of shape [n_samples, n_classes] with probabilities
        """
        self._check_fitted()
        X = self._transform_data(X)
        return self.exp.network.predict(X.astype(numpy.float32))

    def staged_predict_proba(self, X):
        """
        Predicts values on each stage

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :return: iterator
        """
        raise NotImplementedError('staged_predict_proba is not supported for theanets')