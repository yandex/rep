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
from .utils import check_inputs, check_scaler, one_hot_transform, remove_first_line

import numpy
import pandas

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain import structure


__author__ = 'Artem Zhirokhov'

LAYER_CLASS = {'BiasUnit': structure.BiasUnit,
               'LinearLayer': structure.LinearLayer,
               'MDLSTMLayer': structure.MDLSTMLayer,
               'SigmoidLayer': structure.SigmoidLayer,
               'SoftmaxLayer': structure.SoftmaxLayer,
               'TanhLayer': structure.TanhLayer}

_PASS_PARAMETERS = {'random_state'}

__all__ = ['PyBrainBase', 'PyBrainClassifier', 'PyBrainRegressor']


class PyBrainBase(object):
    """Base class for estimator from PyBrain.

    Parameters:
    -----------
    :param features: features used in training.
    :type features: list[str] or None
    :param scaler: scaler used to transform data; default is StandardScaler.
    :type scaler: transformer from sklearn.preprocessing or str or False
    :param bool use_rprop: flag to indicate whether we should use Rprop or SGD trainer.
    :param bool verbose: print train/validation errors.
    **Net parameters:**

    :param layers: indicate how many neurons in each hidden(!) layer; default is 1 hidden layer with 10 neurons.
    :type layers: list[int]
    :param hiddenclass: classes of the hidden layers; default is 'SigmoidLayer'.
    :type hiddenclass: list[str]
    :param dict params: other net parameters:
        bias and outputbias (boolean) flags to indicate whether the network should have the corresponding biases,
        both default to True;
        peepholes (boolean);
        recurrent (boolean) if the `recurrent` flag is set, a :class:`RecurrentNetwork` will be created,
        otherwise a :class:`FeedForwardNetwork`.
    **Gradient descent trainer parameters:**

    :param float learningrate: gives the ratio of which parameters are changed into the direction of the gradient.
    :param float lrdecay: the learning rate decreases by lrdecay, which is used to multiply the learning rate after each training step.
    :param float momentum: the ratio by which the gradient of the last timestep is used.
    :param boolean batchlearning: if set, the parameters are updated only at the end of each epoch. Default is False.
    :param float weightdecay: corresponds to the weightdecay rate, where 0 is no weight decay at all.
    **Rprop trainer parameters:**

    :param float etaminus: factor by which step width is decreased when overstepping (0.5).
    :param float etaplus: factor by which step width is increased when following gradient (1.2).
    :param float delta: step width for each weight.
    :param float deltamin: minimum step width (1e-6).
    :param float deltamax: maximum step width (5.0).
    :param float delta0: initial step width (0.1).
    **Training termination parameters**

    :param int epochs: number of iterations of training; if < 0 then classifier trains until convergence.
    :param int max_epochs: if is given, at most that many epochs are trained.
    :param int continue_epochs: each time validation error decreases, try for continue_epochs epochs to find a better one.
    :param float validation_proportion: the ratio of the dataset that is used for the validation dataset.

    Details about parameters: http://pybrain.org/docs/
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
                 **params):
        self.features = features
        self.epochs = epochs
        self.scaler = scaler
        self.use_rprop = use_rprop

        # net options
        self.layers = layers
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

        self._fitted = False

    def fit(self, X, y):
        """
        Trains the estimator on data.
        """
        self.scaler = check_scaler(self.scaler)
        self.partial_fit(X, y)
        self._fitted = True
        return self

    def partial_fit(self, X, y):
        dataset = self._prepare_net_and_dataset(X, y, self._model_type)

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

    def _check_init_input(self, layers, hiddenclass):
        """
        Checks the input of __init__.
        """
        if layers is not None and hiddenclass is not None and len(layers) != len(hiddenclass):
            raise ValueError('Number of hidden layers does not match number of hidden classes')

        if hiddenclass is not None:
            if hiddenclass[0] == 'BiasUnit':
                raise ValueError('BiasUnit should not be the first unit class')

            for hid_class in hiddenclass:
                if hid_class not in LAYER_CLASS:
                    raise ValueError('Wrong class name ' + hid_class)

    def _is_fitted(self):
        return self._fitted

    def set_params(self, **params):
        """
        Change estimator's parameters.
        Names of parameters are the same as in constructor.
        """
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                if k in _PASS_PARAMETERS:
                    continue

                if k.startswith('layers__'):
                    index = int(k[len('layers__'):])
                    self.layers[index] = v
                elif k.startswith('hiddenclass__'):
                    index = int(k[len('hiddenclass__'):])
                    self.hiddenclass[index] = v
                elif k.startswith('scaler__'):
                    scaler_params = {k[len('scaler__'):]: v}
                    self.scaler.set_params(**scaler_params)
                else:
                    self.params[k] = v

    def _transform_data(self, X, y=None, fit=True):
        X = self._get_train_features(X)
        data_temp = numpy.copy(X)
        if fit:
            self.scaler.fit(data_temp, y)
        return self.scaler.transform(data_temp)

    def _prepare_net_and_dataset(self, X, y, model_type):
        X, y, sample_weight = check_inputs(X, y, sample_weight=None, allow_none_weights=True,
                                           allow_multiple_targets=model_type == 'regression')
        self._check_init_input(self.layers, self.hiddenclass)
        X = self._transform_data(X, y, fit=True)

        self.layers = list(self.layers)

        if self.hiddenclass is None:
            self.hiddenclass = ['SigmoidLayer' for layer_size in self.layers]

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
            self._set_classes(y)
            target = one_hot_transform(y)

        elif model_type == 'regression':
            net_options['outclass'] = structure.LinearLayer
            if len(y.shape) == 1:
                target = y.reshape((len(y), 1))
            else:
                # multi regression
                target = y
            self.n_targets = target.shape[1]
        else:
            raise ValueError('Wrong model type')

        layers_for_net = [X.shape[1]] + self.layers + [target.shape[1]]
        dataset = SupervisedDataSet(X.shape[1], target.shape[1])
        dataset.setField('input', X)
        dataset.setField('target', target)

        self.net = buildNetwork(*layers_for_net, **net_options)

        for layer_id in range(1, len(self.layers)):
            hid_layer = LAYER_CLASS[self.hiddenclass[layer_id]](self.layers[layer_id])
            self.net.addModule(hid_layer)
        self.net.sortModules()

        return dataset


class PyBrainClassifier(PyBrainBase, Classifier):
    __doc__ = "Implements classification from PyBrain library \n" + remove_first_line(PyBrainBase.__doc__)
    _model_type = 'classification'

    def predict_proba(self, X):
        """
        Predict probabilities

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :rtype: numpy.array of shape [n_samples, n_classes] with probabilities
        """
        assert self._is_fitted(), "classifier isn't fitted, please call 'fit' first"

        X = self._transform_data(X, fit=False)
        proba = []
        for values in X:
            pred = self.net.activate(list(values))
            np_pred = numpy.asarray(pred)
            proba.append(np_pred)

        return numpy.asarray(proba)

    def staged_predict_proba(self, X):
        """
        Predicts probabilities on each stage.

        :param pandas.DataFrame X: data shape [n_samples, n_features].
        :return: iterator

        .. warning:: Isn't supported for PyBrain (**AttributeError** will be thrown).
        """
        raise AttributeError("Staged predict_proba not supported for PyBrain")


class PyBrainRegressor(PyBrainBase, Regressor):
    __doc__ = "Implements regression from PyBrain library \n" + remove_first_line(PyBrainBase.__doc__)
    _model_type = 'regression'

    def predict(self, X):
        """
        Predict values for all events in dataset.

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :rtype: numpy.array of shape [n_samples] with predicted values
        """
        assert self._is_fitted(), "regressor isn't fitted, please call 'fit' first"

        X = self._transform_data(X, fit=False)
        y_test_dummy = numpy.zeros((len(X), 1))

        ds = SupervisedDataSet(X.shape[1], y_test_dummy.shape[1])
        ds.setField('input', X)
        ds.setField('target', y_test_dummy)

        predictions = self.net.activateOnDataset(ds)
        if self.n_targets == 1:
            predictions = predictions.flatten()
        return predictions

    def staged_predict(self, X):
        """
        Predicts values on each stage.

        :param X: pandas.DataFrame of shape [n_samples, n_features].
        :rtype: iterator

        .. warning:: Isn't supported for PyBrain (**AttributeError** will be thrown).
        """
        raise AttributeError("Staged predict not supported for PyBrain")
