from __future__ import division, print_function, absolute_import
import numpy

from .interface import Classifier
from .utils import check_inputs

try:
    import theanets as tnt
except ImportError as e:
    raise ImportError("Install theanets before")

__author__ = 'Ignatyeva Lisa'

# TODO: make printing TheanetsClassifier informative (include training method & its parameters)
# TODO: input and output layers' sizes can be pulled from the dataset automatically
#       and what should we do if, pulling it automatically, we get a dataset with another output size, which is
#       quite possible?


class TheanetsClassifier(Classifier):
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
    :param dict kwargs: parameters to specify the training algorithms
    """
    def __init__(self, 
                 layers,
                 hidden_activation=None,
                 output_activation=None,
                 rng=None,
                 input_noise=None,
                 hidden_noise=None,
                 input_dropouts=None,
                 hidden_dropouts=None,
                 decode_from=None,
                 features=None,
                 **kwargs):
        self.network_params_names = {'hidden_activation', 'output_activation', 'rng',
                                     'input_noise', 'hidden_noise', 'input_dropouts',
                                     'hidden_dropouts', 'decode_from'}
        self.layers = layers
        self.network_params = {}
        #self.classes_ = None
        # TODO: rewrite these ifs into something prettier
        if hidden_activation is not None:
            self.network_params['hidden_activation'] = hidden_activation
        if output_activation is not None:
            self.network_params['output_activation'] = output_activation
        # TODO: actually, None is a valid value for rng to be passed...
        if rng is not None:
            self.network_params['rng'] = rng
        if input_noise is not None:
            self.network_params['input_noise'] = input_noise
        if hidden_noise is not None:
            self.network_params['hidden_noise'] = hidden_noise
        if input_dropouts is not None:
            self.network_params['input_dropouts'] = input_dropouts
        if hidden_dropouts is not None:
            self.network_params['hidden_dropouts'] = hidden_dropouts
        if decode_from is not None:
            self.network_params['decode_from'] = decode_from
        self.optimize_params = kwargs
        self.exp = None
        Classifier.__init__(self, features=features)

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        :param dict params: parameters to set in model
        """
        for key, value in params.items():
            if key in self.network_params_names:
                self.network_params[key] = value
            else:
                self.optimize_params[key] = value

    def get_params(self):
        """
        Get parameters of this estimator

        :return dict which holds:
            'optimize_params': dict or (if additional_fit was called at least once) list(dict)
                each subdict here contains all the parameters which were used in a fit/additional_fit call
            'network_params': a dict with network parameters
            'features': a list of features used (None if all were used).
        """
        parameters = {'optimize_params': self.optimize_params,
                      'network_params': self.network_params,
                      'features': self.features}
        return parameters

    def fit(self, X, y, sample_weight=None):
        """
        Train the classifier

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: values - array-like of shape [n_samples]
        :param sample_weight: weight of events,
               array-like of shape [n_samples] or None if all weights are equal
        :return: self
        """
        if sample_weight is not None:
            raise NotImplementedError('sample_weight is not supported yet for theanets')
        self.classes_ = numpy.unique(y)
        X, y, sample_weight = check_inputs(X, y, sample_weight)
        X = self._get_train_features(X)
        if self.layers[0] == -1:
            self.layers = (X.shape[1],) + self.layers[1:]
        if self.layers[-1] == -1:
            self.layers = self.layers[:-1] + (len(self.classes_),)
        self.exp = tnt.Experiment(tnt.Classifier, layers=self.layers, **self.network_params)

        self.exp.train((X.values.astype(numpy.float32), y.astype(numpy.int32)),
                       **self.optimize_params)

    def additional_fit(self, X, y, sample_weight=None, **kwargs):
        """
        Train the classifier again

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: values - array-like of shape [n_samples]
        :param sample_weight: weight of events,
               array-like of shape [n_samples] or None if all weights are equal
        :param dict kwargs: parameters of the training algorithm we want to use now
        :return: self
        """
        if self.exp is None:
            raise AttributeError('you cannot call additional_fit before calling fit')
        if isinstance(self.optimize_params, list):
            self.optimize_params.append(kwargs)
        else:
            self.optimize_params = [self.optimize_params, kwargs]
        X, y, sample_weight = check_inputs(X, y, sample_weight)
        X = self._get_train_features(X)
        self.exp.train((X.values.astype(numpy.float32), y.astype(numpy.int32)),
                       **self.params)

    def predict_proba(self, X):
        """
        Predict probabilities

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :rtype: numpy.array of shape [n_samples, n_classes] with probabilities
        """
        X = self._get_train_features(X)
        return self.exp.network.predict(X.values.astype(numpy.float32))

    def staged_predict_proba(self, X):
        """
        Predicts values on each stage

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :return: iterator
        """
        raise NotImplementedError('staged_predict_proba is not supported yet for theanets')