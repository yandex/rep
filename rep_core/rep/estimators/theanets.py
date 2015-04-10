from __future__ import division, print_function, absolute_import
import numpy
from abc import abstractmethod
from .interface import Classifier, Regressor
from .utils import check_inputs
from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.utils import check_random_state
import os
import tempfile

try:
    import theanets as tnt
except ImportError as e:
    raise ImportError("Install theanets before")

__author__ = 'Lisa Ignatyeva'

UNSUPPORTED_OPTIMIZERS = ['pretrain', 'sample', 'hf']
# pretrain and sample data formats are too different from what we support here
# hf now does not work in theanets, see https://github.com/lmjohns3/theanets/issues/62

class TheanetsBase(object):
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
        self.imputer = Imputer()
        self.trainers = trainers
        if self.trainers is None:
            self.trainers = [{}]
        self.exp = None

    def __getstate__(self):
        """
        Required for pickle.dump working, because theanets objects can't be pickled by default.

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
                layers = [self.input_layer] + self.layers + [self.output_layer]
                self.exp = tnt.Experiment(tnt.Classifier, layers=layers, rng=self._get_rng(), **self.network_params)
                self.exp.load(dump.name)
        del dictionary['dumped_exp']

    def _get_rng(self):
        numpy.random.seed(42)
        return check_random_state(self.random_state)

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
                    if not sep or trainer_num[:7] != 'trainer' or len(trainer_num) <= 7:
                        raise AttributeError(key + ' is an invalid parameter for a NN with multiple training')
                    trainer_num = int(trainer_num[7:])
                    # resize if needed
                    self.trainers[trainer_num][param] = value

    def get_params(self, deep=True):
        """
        Get parameters of this estimator

        :return dict
        """
        parameters = self.network_params.copy()
        parameters['layers'] = self.layers
        parameters['input_layer'] = self.input_layer
        parameters['output_layer'] = self.output_layer
        parameters['trainers'] = self.trainers
        parameters['features'] = self.features
        parameters['random_state'] = self.random_state
        return parameters

    def _transform_data(self, data):
        data_backup = data
        data_backup = (self._get_train_features(data_backup, allow_nans=True))
        if self.scaler is None:
            if self._is_fitted():
                return self.imputer.transfrom(data_backup)
            else:
                return self.imputer.fit_transform(data_backup)

        if self._is_fitted():
            return self.scaler.transform(self.imputer.transfrom(data_backup))
        return self.scaler.fit_transform(self.imputer.fit_transform(data_backup))

    def _is_fitted(self):
        return self.exp is not None

    def fit(self, X, y, sample_weight=None):
        """
        Train the estimator from scratch.

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: values - array-like of shape [n_samples]
        :param sample_weight: weight of events,
               array-like of shape [n_samples] or None if all weights are equal
        :return: self
        """
        self.exp = None
        for trainer in self.trainers:
            for optimizer in UNSUPPORTED_OPTIMIZERS:
                if 'optimize' in trainer and trainer['optimize'] == optimizer:
                    raise NotImplementedError(optimizer + ' is not supported')
            self.partial_fit(X, y, new_trainer=False, **trainer)
        return self

    @abstractmethod
    def partial_fit(self, X, y, sample_weight=None, new_trainer=True,  **trainer):
        """
        Train the estimator by training the existing classifier again.

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: values - array-like of shape [n_samples]
        :param sample_weight: weight of events,
               array-like of shape [n_samples] or None if all weights are equal
        :param bool new_trainer: True if the trainer is not stored in self.trainers
        :param dict trainer: parameters of the training algorithm we want to use now
        :return: self
        """
        pass




class TheanetsClassifier(TheanetsBase, Classifier):
    """
    Implements classification from Theanets library.

    Parameters:
    -----------
    :param layers: A sequence of values specifying the hidden layer configuration for the network. For more information
        please see 'Specifying layers' in theanets documentation:
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
    :param scaler: scaler used to transform data
    :type scaler: scaler from sklearn.preprocessing or None
    :param features: list of features to train model
    :type features: None or list(str)
    :param list(dict) or None trainers: parameters to specify training algorithm
    """
    def __init__(self, 
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
                 scaler=MinMaxScaler(),
                 features=None,
                 trainers=None):
        TheanetsBase.__init__(self, layers=layers,
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
            if self.input_layer == -1:
                self.input_layer = X.shape[1]
            if self.output_layer == -1:
                self.output_layer = len(self.classes_)
            layers = [self.input_layer] + self.layers + [self.output_layer]
            print(layers)
            self.exp = tnt.Experiment(tnt.Classifier, layers=layers, rng=self._get_rng(), **self.network_params)
        if new_trainer:
            self.trainers.append(trainer)
        numpy.random.seed(42)
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
        X = self._transform_data(X)
        return self.exp.network.predict(X.astype(numpy.float32))

    def staged_predict_proba(self, X):
        """
        Predicts values on each stage

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :return: iterator
        """
        raise NotImplementedError('staged_predict_proba is not supported for theanets classifier')


class TheanetsRegressor(TheanetsBase, Regressor):

    def __init__(self,
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
                 scaler=MinMaxScaler(),
                 features=None,
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

    def partial_fit(self, X, y, sample_weight=None, new_trainer=True,  **trainer):
        """
        Train the regressor by training the existing regressor again.

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
        if self.exp is None:
            # initialize experiment
            if self.input_layer == -1:
                self.input_layer = X.shape[1]
            if self.output_layer == -1:
                self.output_layer = 1
            layers = [self.input_layer] + self.layers + [self.output_layer]
            print(layers)
            self.exp = tnt.Experiment(tnt.Regressor, layers=layers, rng=self._get_rng(), **self.network_params)
        if new_trainer:
            self.trainers.append(trainer)
        numpy.random.seed(42)
        self.exp.train((X.astype(numpy.float32), y.astype(numpy.int32)),
                       **trainer)
        return self

    def predict(self, X):
        """
        Predict probabilities

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :rtype: numpy.array of shape [n_samples, n_classes] with probabilities
        """
        assert self._is_fitted(), 'Regressor wasn`t fitted, please call `fit` first'
        X = self._transform_data(X)
        return self.exp.network.predict(X.astype(numpy.float32))

    def staged_predict(self, X):
        """
        Predicts values on each stage

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :return: iterator
        """
        raise NotImplementedError('staged_predict is not supported for theanets regressor')
