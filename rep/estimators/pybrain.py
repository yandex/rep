"""
These classes are wrappers for the `PyBrain library <http://pybrain.org/docs/>`_ --- a neural network python library.

.. warning:: pybrain training isn't reproducible
    (training with the same parameters produces different neural network each time)


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

import numpy
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain import structure

from .interface import Classifier, Regressor
from .utils import check_inputs, check_scaler, one_hot_transform, remove_first_line

__author__ = 'Artem Zhirokhov, Alex Rogozhnikov, Tatiana Likhomanenko'
__all__ = ['PyBrainBase', 'PyBrainClassifier', 'PyBrainRegressor']

LAYER_CLASS = {'BiasUnit': structure.BiasUnit,
               'LinearLayer': structure.LinearLayer,
               'MDLSTMLayer': structure.MDLSTMLayer,
               'SigmoidLayer': structure.SigmoidLayer,
               'SoftmaxLayer': structure.SoftmaxLayer,
               'TanhLayer': structure.TanhLayer}


class PyBrainBase(object):
    """A base class for the estimator from the PyBrain.

    :param features: features used in training.
    :type features: list[str] or None
    :param scaler: transformer which is applied to the input samples. If it is False, scaling will not be used
    :type scaler: str or sklearn-like transformer or False
    :param bool use_rprop: flag to indicate whether we should use Rprop or SGD trainer
    :param bool verbose: print train/validation errors.
    :param random_state: it is ignored parameter, pybrain training is not reproducible

    **Net parameters:**

    :param list[int] layers: indicate how many neurons in each hidden(!) layer; default is 1 hidden layer with 10 neurons
    :param list[str] hiddenclass: classes of the hidden layers; default is `'SigmoidLayer'`
    :param dict params: other net parameters:

        * `bias` and `outputbias` (boolean) flags to indicate whether the network should have the corresponding biases,
          both default to True;
        * `peepholes` (boolean);
        * `recurrent` (boolean): if the `recurrent` flag is set, a :class:`RecurrentNetwork` will be created,
          otherwise a :class:`FeedForwardNetwork`

    **Gradient descent trainer parameters:**

    :param float learningrate: gives the ratio of which parameters are changed into the direction of the gradient
    :param float lrdecay: the learning rate decreases by lrdecay, which is used to multiply the learning rate after each training step
    :param float momentum: the ratio by which the gradient of the last time step is used
    :param boolean batchlearning: if set, the parameters are updated only at the end of each epoch. Default is False
    :param float weightdecay: corresponds to the `weightdecay` rate, where 0 is no weight decay at all

    **Rprop trainer parameters:**

    :param float etaminus: factor by which a step width is decreased when overstepping (default=0.5)
    :param float etaplus: factor by which a step width is increased when following gradient (default=1.2)
    :param float delta: step width for each weight
    :param float deltamin: minimum step width (default=1e-6)
    :param float deltamax: maximum step width (default=5.0)
    :param float delta0: initial step width (default=0.1)

    **Training termination parameters**

    :param int epochs: number of iterations in training; if < 0 then estimator trains until converge
    :param int max_epochs: maximum number of epochs the trainer should train if it is given
    :param int continue_epochs: each time validation error decreases, try for `continue_epochs` epochs to find a better one
    :param float validation_proportion: the ratio of the dataset that is used for the validation dataset

    .. note::

        Details about parameters `here <http://pybrain.org/docs/>`_.
    """
    __metaclass__ = ABCMeta
    # to be overriden in descendants.
    _model_type = None

    def __init__(self,
                 features=None,
                 layers=(10,),
                 hiddenclass=None,
                 epochs=10,
                 scaler='standard',
                 use_rprop=False,
                 learningrate=0.01,
                 lrdecay=1.0,
                 momentum=0.,
                 verbose=False,
                 batchlearning=False,
                 weightdecay=0.,
                 etaminus=0.5,
                 etaplus=1.2,
                 deltamin=1.0e-6,
                 deltamax=0.5,
                 delta0=0.1,
                 max_epochs=None,
                 continue_epochs=3,
                 validation_proportion=0.25,
                 random_state=None,
                 **params):
        self.features = list(features) if features is not None else features
        self.epochs = epochs
        self.scaler = scaler
        self.use_rprop = use_rprop

        # net options
        self.layers = list(layers)
        self.hiddenclass = hiddenclass
        self.params = params

        # SGD trainer options
        self.learningrate = learningrate
        self.lrdecay = lrdecay
        self.momentum = momentum
        self.verbose = verbose
        self.batchlearning = batchlearning
        self.weightdecay = weightdecay

        # Rprop trainer
        self.etaminus = etaminus
        self.etaplus = etaplus
        self.deltamin = deltamin
        self.deltamax = deltamax
        self.delta0 = delta0

        # trainUntilConvergence options
        self.max_epochs = max_epochs
        self.continue_epochs = continue_epochs
        self.validation_proportion = validation_proportion

        self.random_state = random_state
        self.net = None

    def _check_params(self):
        """
        Checks the input of __init__.
        """
        if self.hiddenclass is not None:
            assert len(self.layers) == len(
                    self.hiddenclass), 'Number of hidden layers does not match number of hidden classes'
            if self.hiddenclass[0] == 'BiasUnit':
                raise ValueError('BiasUnit should not be the first unit class')

            for hid_class in self.hiddenclass:
                if hid_class not in LAYER_CLASS:
                    raise ValueError('Wrong class name ' + hid_class)

    def fit(self, X, y):
        """
        Train a classification/regression model on the data.

        :param pandas.DataFrame X: data of shape [n_samples, n_features]
        :param y: values for samples --- array-like of shape [n_samples]
        :return: self
        """
        self.net = None
        return self.partial_fit(X, y)

    def partial_fit(self, X, y):
        """
        Additional training of the classification/regression model.

        :param pandas.DataFrame X: data of shape [n_samples, n_features]
        :param y: values for samples, array-like of shape [n_samples]
        :return: self
        """
        dataset = self._prepare_dataset(X, y, self._model_type)

        if not self._is_fitted():
            self._prepare_net(dataset=dataset, model_type=self._model_type)

        if self.use_rprop:
            trainer = RPropMinusTrainer(self.net,
                                        etaminus=self.etaminus,
                                        etaplus=self.etaplus,
                                        deltamin=self.deltamin,
                                        deltamax=self.deltamax,
                                        delta0=self.delta0,
                                        dataset=dataset,
                                        learningrate=self.learningrate,
                                        lrdecay=self.lrdecay,
                                        momentum=self.momentum,
                                        verbose=self.verbose,
                                        batchlearning=self.batchlearning,
                                        weightdecay=self.weightdecay)
        else:
            trainer = BackpropTrainer(self.net,
                                      dataset,
                                      learningrate=self.learningrate,
                                      lrdecay=self.lrdecay,
                                      momentum=self.momentum,
                                      verbose=self.verbose,
                                      batchlearning=self.batchlearning,
                                      weightdecay=self.weightdecay)

        if self.epochs < 0:
            trainer.trainUntilConvergence(maxEpochs=self.max_epochs,
                                          continueEpochs=self.continue_epochs,
                                          verbose=self.verbose,
                                          validationProportion=self.validation_proportion)
        else:
            trainer.trainEpochs(epochs=self.epochs, )
        return self

    def _is_fitted(self):
        """
        Check if the estimator is fitted or not.

        :rtype: bool
        """
        return self.net is not None

    def set_params(self, **params):
        """
        Set the parameters of the estimator.

        Names of the parameters are the same as in the constructor.
        """
        for name, value in params.items():
            if hasattr(self, name):
                setattr(self, name, value)
            else:
                if name.startswith('layers__'):
                    index = int(name[len('layers__'):])
                    self.layers[index] = value
                elif name.startswith('hiddenclass__'):
                    index = int(name[len('hiddenclass__'):])
                    self.hiddenclass[index] = value
                elif name.startswith('scaler__'):
                    scaler_params = {name[len('scaler__'):]: value}
                    self.scaler.set_params(**scaler_params)
                else:
                    self.params[name] = value

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
        # most of transformers there modify X if it is pandas.DataFrame.
        data_temp = numpy.copy(X)
        if fit:
            self.scaler = check_scaler(self.scaler)
            self.scaler.fit(data_temp, y)
        return self.scaler.transform(data_temp)

    def _prepare_dataset(self, X, y, model_type):
        """
        Prepare data in pybrain format.

        :param pandas.DataFrame X: data of shape [n_samples, n_features]
        :param y: values for samples --- array-like of shape [n_samples]
        :param str model_type: classification or regression label
        :return: self
        """
        X, y, sample_weight = check_inputs(X, y, sample_weight=None, allow_none_weights=True,
                                           allow_multiple_targets=model_type == 'regression')
        X = self._transform_data(X, y, fit=not self._is_fitted())

        if model_type == 'classification':
            if not self._is_fitted():
                self._set_classes(y)
            target = one_hot_transform(y, n_classes=len(self.classes_))
        elif model_type == 'regression':
            if len(y.shape) == 1:
                target = y.reshape((len(y), 1))
            else:
                # multi regression
                target = y

            if not self._is_fitted():
                self.n_targets = target.shape[1]
        else:
            raise ValueError('Wrong model type')

        dataset = SupervisedDataSet(X.shape[1], target.shape[1])
        dataset.setField('input', X)
        dataset.setField('target', target)

        return dataset

    def _prepare_net(self, dataset, model_type):
        """
        Prepare net for training.

        :param pybrain.datasets.SupervisedDataSet dataset: dataset in pybrain format
        :param str model_type: classification or regression label
        """
        self._check_params()

        if self.hiddenclass is None:
            self.hiddenclass = ['SigmoidLayer'] * len(self.layers)

        net_options = {'bias': True,
                       'outputbias': True,
                       'peepholes': False,
                       'recurrent': False,
                       }
        for key in self.params:
            if key not in net_options.keys():
                raise ValueError('Unexpected parameter: {}'.format(key))
            net_options[key] = self.params[key]
        # This flag says to use native python implementation, not arac.
        net_options['fast'] = False

        if model_type == 'classification':
            net_options['outclass'] = structure.SoftmaxLayer
        else:
            net_options['outclass'] = structure.LinearLayer

        layers_for_net = [dataset.indim] + self.layers + [dataset.outdim]
        self.net = buildNetwork(*layers_for_net, **net_options)

        for layer_id in range(1, len(self.layers)):
            hid_layer = LAYER_CLASS[self.hiddenclass[layer_id]](self.layers[layer_id])
            self.net.addModule(hid_layer)
        self.net.sortModules()

    def _activate_on_dataset(self, X):
        """
        Predict data.

        :param pandas.DataFrame X: data to be predicted
        :return: array-like predictions [n_samples, n_targets]
        """
        assert self._is_fitted(), "Net isn't fitted, please call 'fit' first"

        X = self._transform_data(X, fit=False)
        y_test_dummy = numpy.zeros((len(X), 1))

        ds = SupervisedDataSet(X.shape[1], y_test_dummy.shape[1])
        ds.setField('input', X)
        ds.setField('target', y_test_dummy)

        return self.net.activateOnDataset(ds)

    def __setstate__(self, dict):
        # resolve pickling issue with pyBrain http://stackoverflow.com/questions/4334941/
        self.__dict__ = dict
        if self.net is not None:
            self.net.sorted = False
            self.net.sortModules()


class PyBrainClassifier(PyBrainBase, Classifier):
    __doc__ = "Implements a classification model from the PyBrain library. \n" + remove_first_line(PyBrainBase.__doc__)
    _model_type = 'classification'

    def predict_proba(self, X):
        return self._activate_on_dataset(X=X)

    predict_proba.__doc__ = Classifier.predict_proba.__doc__

    def staged_predict_proba(self, X):
        """
        .. warning:: This function is not supported for PyBrain (**AttributeError** will be thrown).
        """
        raise AttributeError("'staged_predict_proba' is not supported by the PyBrain networks")


class PyBrainRegressor(PyBrainBase, Regressor):
    __doc__ = "Implements a regression model from the PyBrain library. \n" + remove_first_line(PyBrainBase.__doc__)
    _model_type = 'regression'

    def predict(self, X):
        predictions = self._activate_on_dataset(X)
        if self.n_targets == 1:
            predictions = predictions.flatten()
        return predictions

    predict.__doc__ = Classifier.predict.__doc__

    def staged_predict(self, X):
        """
        .. warning:: This function is not supported for PyBrain (**AttributeError** will be thrown).
        """
        raise AttributeError("'staged_predict' is not supported by the PyBrain networks")
