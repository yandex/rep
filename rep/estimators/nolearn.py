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

from .interface import Classifier
from .utils import check_inputs

from nolearn.dbn import DBN
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone, BaseEstimator
import gnumpy as gnp

__author__ = 'Alexey Berdnikov'


LIST_PARAMS = {'layers', 'scales', 'fan_outs', 'uniforms', 'learn_rates', 'learn_rate_decays', 'learn_rate_minimums',
               'l2_costs', 'dropouts', 'rms_lims', 'learn_rates_pretrain', 'l2_costs_pretrain', 'epochs_pretrain'}


class NolearnClassifier(Classifier):
    """
    Classifier based on :class:`DBN` from :mod:`nolearn.dbn`.

    Parameters:
    -----------
    :param features: Features used in training.
    :type features: `list[str]` or None
    :param layers: A list of ints of the form `[n_hid_units_1, n_hid_units_2, ...]`, where `n_hid_units_i` is the number
        of units in i-th hidden layer. The number of units in the input layer and the output layer will be set
        automatically. Default value is `[10]` which means one hidden layer containing 10 units.
    :param scaler: A scikit-learn transformer to apply to the input objects. If `None` (which is default),
        `StandardScaler()` from :mod:`sklearn.preprocessing` will be used. If you do not want to use any transformer,
         set `False`.
    :param scales: Scale of the randomly initialized weights. A list of floating point values. When you find good values
        for the scale of the weights you can speed up training a lot, and also improve performance. Defaults to `0.05`.
    :param fan_outs: Number of nonzero incoming connections to a hidden unit. Defaults to `None`, which means that all
        connections have non-zero weights.
    :param output_act_funct: Output activation function. Instance of type :class:`gdbn.activationFunctions.Sigmoid`,
        :class:`gdbn.activationFunctions.Linear`, :class:`gdbn.activationFunctions.Softmax` from the
        :mod:`gdbn.activationFunctions` module.  Defaults to :class:`gdbn.activationFunctions.Softmax`.
    :param real_valued_vis: Set `True` (the default) if visible units are real-valued.
    :param use_re_lu: Set `True` to use rectified linear units. Defaults to `False`.
    :param uniforms: Not documented at this time.
    :param learn_rates: A list of learning rates, one entry per weight layer.
    :param learn_rate_decays: The number with which the `learn_rate` is multiplied after each epoch of fine-tuning.
    :param learn_rate_minimums: The minimum `learn_rates`; after the learn rate reaches the minimum learn rate, the
        `learn_rate_decays` no longer has any effect.
    :param momentum: Momentum.
    :param l2_costs: L2 costs per weight layer.
    :param dropouts: Dropouts per weight layer.
    :param nesterov: Not documented at this time.
    :param nest_compare: Not documented at this time.
    :param rms_lims: Not documented at this time.
    :param learn_rates_pretrain: A list of learning rates similar to `learn_rates`, but used for pretraining. Defaults
        to value of `learn_rates` parameter.
    :param momentum_pretrain: Momentum for pre-training. Defaults to value of `momentum` parameter.
    :param l2_costs_pretrain: L2 costs per weight layer, for pre-training.  Defaults to the value of `l2_costs`
        parameter.
    :param nest_compare_pretrain: Not documented at this time.
    :param epochs: Number of epochs to train (with backprop).
    :param epochs_pretrain: Number of epochs to pre-train (with CDN).
    :param loss_funct: A function that calculates the loss. Used for displaying learning progress.
    :param minibatch_size: Size of a minibatch.
    :param minibatches_per_epoch: Number of minibatches per epoch. The default is to use as many as fit into our
        training set.
    :param pretrain_callback: An optional function that takes as arguments the :class:`nolearn.dbn.DBN` instance, the
        epoch and the layer index as its argument, and is called for each epoch of pretraining.
    :param fine_tune_callback: An optional function that takes as arguments the :class:`nolearn.dbn.DBN` instance and
        the epoch, and is called for each epoch of fine tuning.
    :param verbose: Debugging output.
    .. warning::
        nolearn doesn't support `staged_predict_proba()`, `feature_importances__` and sample weights.
    .. warning::
        The `random_state` parameter is not implemented in this wrapper because nolearn uses this parameter
        in a way that is incompatible with scikit-learn.

    """
    def __init__(self, features=None,
                 layers=(10,),
                 scaler=None,
                 scales=0.05,
                 fan_outs=None,
                 output_act_funct=None,
                 real_valued_vis=True,
                 use_re_lu=True,
                 uniforms=False,
                 learn_rates=0.1,
                 learn_rate_decays=1.0,
                 learn_rate_minimums=0.0,
                 momentum=0.9,
                 l2_costs=0.0001,
                 dropouts=0,
                 nesterov=True,
                 nest_compare=True,
                 rms_lims=None,
                 learn_rates_pretrain=None,
                 momentum_pretrain=None,
                 l2_costs_pretrain=None,
                 nest_compare_pretrain=None,
                 epochs=10,
                 epochs_pretrain=0,
                 loss_funct=None,
                 minibatch_size=64,
                 minibatches_per_epoch=None,
                 pretrain_callback=None,
                 fine_tune_callback=None,
                 verbose=0):
        Classifier.__init__(self, features=features)

        self.layers = layers
        self.scaler = scaler
        self.scales = scales
        self.fan_outs = fan_outs
        self.output_act_funct = output_act_funct
        self.real_valued_vis = real_valued_vis
        self.use_re_lu = use_re_lu
        self.uniforms = uniforms

        self.learn_rates = learn_rates
        self.learn_rate_decays = learn_rate_decays
        self.learn_rate_minimums = learn_rate_minimums
        self.momentum = momentum
        self.l2_costs = l2_costs
        self.dropouts = dropouts
        self.nesterov = nesterov
        self.nest_compare = nest_compare
        self.rms_lims = rms_lims

        self.learn_rates_pretrain = learn_rates_pretrain
        self.momentum_pretrain = momentum_pretrain
        self.l2_costs_pretrain = l2_costs_pretrain
        self.nest_compare_pretrain = nest_compare_pretrain

        self.epochs = epochs
        self.epochs_pretrain = epochs_pretrain
        self.loss_funct = loss_funct
        self.minibatch_size = minibatch_size
        self.minibatches_per_epoch = minibatches_per_epoch

        self.pretrain_callback = pretrain_callback
        self.fine_tune_callback = fine_tune_callback

        self.verbose = verbose

    def _transform_data(self, X, y=None, fit=False):
        X = self._get_train_features(X).values

        if fit:
            self._fit_scaler(X, y)

        if self.scaler_fitted_ is not None:
            X = self.scaler_fitted_.transform(X)

        return X

    def _fit_scaler(self, X, y):
        if self.scaler is None:
            self.scaler_fitted_ = StandardScaler()
        elif self.scaler == False:
            self.scaler_fitted_ = None
        else:
            self.scaler_fitted_ = clone(self.scaler)

        if self.scaler_fitted_ is not None:
            self.scaler_fitted_.fit(X,y)

    def _build_net(self):
        net_params = self.get_params(deep=False)
        del net_params["features"]
        del net_params["layers"]
        del net_params["scaler"]

        layers = self._check_list_parameter('layers')
        net_params["layer_sizes"] = [-1] + layers + [-1]

        net = DBN(**net_params)

        # Black magic. For some reason this makes results of prediction to be reproducible.
        gnp.seed_rand(42)

        return net

    def _check_is_fitted(self):
        if not hasattr(self, 'net_'):
            raise AttributeError("estimator not fitted, call 'fit' before making predictions")

    def _check_list_parameter(self, name):
        parameter = getattr(self, name)
        if not hasattr(parameter, '__iter__'):
            raise ValueError("cannot convert '{}' parameter to list".format(name))
        return list(parameter)

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        The method works also on nested objects (such as pipelines). Use parameter of the form
        ``<component>__<parameter>`` to update the parameter of the component. Use ``<parameter>__<number>`` to change
        an element of parameter which is presented by a list (e. g. `layers__0`).

        Returns
        -------
        self

        """
        skipped_params = {}
        for key, value in params.items():
            split = key.split('__', 1)
            param_name = split[0]
            if len(split) > 1 and param_name in LIST_PARAMS:
                param_value = self._check_list_parameter(param_name)
                k = int(split[1])
                param_value[k] = value
                setattr(self, param_name, param_value)
            else:
                skipped_params[key] = value
        BaseEstimator.set_params(self, **skipped_params)

    def fit(self, X, y):
        """
        Train the classifier.

        :param pandas.DataFrame | numpy.ndarray X: Data shape `[n_samples, n_features]`.
        :param list | numpy.array y: Values - array-like of shape `[n_samples]`.
        :return: self
        .. warning::
            Sample weights are not supported for nolearn.

        """
        X, y, sample_weight = check_inputs(X, y, sample_weight=None, allow_none_weights=True)

        X = self._transform_data(X, y, fit=True)
        self._set_classes(y)

        self.net_ = self._build_net()
        self.net_.fit(X, y)

        return self

    def predict(self, X):
        """
        Predict data.

        :param pandas.DataFrame | numpy.ndarray X: Data shape `[n_samples, n_features]`.
        :return: Predicted values of shape `n_samples`.

        """
        self._check_is_fitted()
        X = self._transform_data(X)
        return self.net_.predict(X)

    def predict_proba(self, X):
        """
        Predict data.

        :param pandas.DataFrame | numpy.ndarray X: Data shape `[n_samples, n_features]`.
        :return: A `numpy.array` of shape `[n_samples, n_classes]` with probabilities.

        """
        self._check_is_fitted()
        X = self._transform_data(X)
        return self.net_.predict_proba(X)

    def staged_predict_proba(self, X):
        """
        Predict values on each stage.

        :param pandas.DataFrame | numpy.ndarray X: Data shape `[n_samples, n_features]`.
        :return: iterator
        .. warning::
            Doesn't support for nolearn (**AttributeError** will be thrown).

        """
        raise AttributeError("'staged_predict_proba' is not supported for nolearn")
